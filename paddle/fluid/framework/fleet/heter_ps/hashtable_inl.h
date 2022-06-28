/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/heter_util.h"

namespace paddle {
namespace framework {

template <typename value_type>
struct ReplaceOp {
  __host__ __device__ value_type operator()(value_type new_value,
                                            value_type old_value) {
    return new_value;
  }
};

template <typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              size_t len, char* pool, size_t feature_value_size,
                              int start_index) {
  ReplaceOp<typename Table::mapped_type> op;
  thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
    kv.first = keys[i];
    uint64_t offset = uint64_t(start_index + i) * feature_value_size;
    kv.second = (Table::mapped_type)(pool + offset);
    auto it = table->insert(kv, op);
    assert(it != table->end() && "error: insert fails: table is full");
  }
}

template <typename Table, typename ValType>
__global__ void search_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    ValType* vals, size_t len,
                                    size_t pull_feature_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    char* d_value = (char*)(vals);
    if (it != table->end()) {
      uint64_t offset = i * pull_feature_value_size;
      ValType* cur = (ValType*)(d_value + offset);
      ValType& input = *(ValType*)(it->second);
      *cur = input;
    } else {
      if (keys[i] != 0) printf("pull miss key: %llu", keys[i]);
      ValType* cur = (ValType*)(d_value + i * pull_feature_value_size);
      *cur = ValType();
    }
  }
}

template <typename Table, typename GradType, typename Sgd>
__global__ void update_kernel(Table* table,
                                    const typename Table::key_type* const keys,
                                    const GradType* grads, curandState* p_state, size_t len,
                                    Sgd sgd, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    auto it = table->find(keys[i]);
    if (it != table->end()) {
      char* grads_tmp = (char*)(grads);
      GradType* cur = (GradType*)(grads_tmp + i * grad_value_size);
      sgd.update_value((it.getter())->second, *cur, p_state[i]);
    } else {
      if (keys[i] != 0) printf("pull miss key: %llu", keys[i]);
    }
  }
}

__global__ void curand_init_kernel(curandState* p_value, int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    curand_init(clock64(), i, 0, p_value + i);
  }
}

class CuRandState {
 public:
  CuRandState() = default;
  CuRandState(const CuRandState&) = delete;
  CuRandState(CuRandState&&) = delete;
  ~CuRandState() { CHECK(cudaFree(states_) == cudaSuccess); }
  curandState* get(size_t size, gpuStream_t stream) {
    if (size > size_) {
      size_t new_size = size * 2;
      curandState* new_state = nullptr;
      CHECK(cudaMalloc(reinterpret_cast<void**>(&new_state),
                       new_size * sizeof(curandState)) == cudaSuccess);
      if (size_ > 0) {
        CHECK(cudaMemcpyAsync(new_state, states_, size_ * sizeof(curandState),
                              cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
      }
      int init_len = new_size - size_;
      const int BLOCK_SIZE_{256};
      const int init_kernel_grid = (init_len - 1) / BLOCK_SIZE_ + 1;
      curand_init_kernel<<<init_kernel_grid, BLOCK_SIZE_, 0, stream>>>(
          new_state + size_, init_len);
      if (size_ != 0) {
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(cudaFree(states_) == cudaSuccess);
      }
      states_ = new_state;
      size_ = new_size;
    }
    return states_;
  }

  static HeterObjectPool<CuRandState>& pool() {
    static HeterObjectPool<CuRandState> p;
    return p;
  }

  static std::shared_ptr<CuRandState> get() { return pool().Get(); }

  static void CUDART_CB pushback_cu_rand_state(void* data) {
    auto state = static_cast<std::shared_ptr<CuRandState>*>(data);
    pool().Push(std::move(*state));
    delete state;
  }

  static void push(std::shared_ptr<CuRandState> state, gpuStream_t stream) {
    CHECK(cudaLaunchHostFunc(stream, pushback_cu_rand_state,
                             new std::shared_ptr<CuRandState>(
                                 std::move(state))) == cudaSuccess);
  }

 private:
  size_t size_ = 0;
  curandState* states_ = nullptr;
};

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::HashTable(size_t capacity) {
  container_ = new TableContainer<KeyType, ValType>(capacity);
  rwlock_.reset(new phi::RWLock);
}

template <typename KeyType, typename ValType>
HashTable<KeyType, ValType>::~HashTable() {
  delete container_;
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::show() {
  container_->print();
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::get(const KeyType* d_keys, ValType d_vals, size_t len, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_vals, len, pull_feature_value_size_);
}

template <typename KeyType, typename ValType>
void HashTable<KeyType, ValType>::insert(const KeyType* d_keys, size_t len,
                                         char* pool, size_t feature_value_size,
                                         size_t start_index,
                                         gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  if (pool == NULL) {
    return;
  }
  insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, len, pool, feature_value_size, start_index);
}

template <typename KeyType, typename ValType>
template <typename GradType, typename Sgd>
void HashTable<KeyType, ValType>::update(const KeyType* d_keys,
                                         const GradType* d_grads, size_t len,
                                         Sgd sgd, gpuStream_t stream) {
  if (len == 0) {
    return;
  }
  const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
  auto state = CuRandState::get();
  auto d_state = state->get(len, stream);
  update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      container_, d_keys, d_grads, d_state, len, sgd, push_grad_value_size_);
  CuRandState::push(state, stream);
}

}  // end namespace framework
}  // end namespace paddle
#endif
