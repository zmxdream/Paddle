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

#pragma once
#include <thread>
#include <vector>
#include "paddle/fluid/framework/barrier.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "hashtable.h"       // NOLINT
#include "heter_resource.h"  // NOLINT
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/place.h"
#include "thrust/pair.h"
#include "paddle/fluid/platform/timer.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

Barrier h_barrier;

// #define TYPEALIGN(ALIGNVAL, LEN) \
//   (((uint64_t)(LEN) + ((ALIGNVAL)-1)) & ~((uint64_t)((ALIGNVAL)-1)))

struct CustomGradMerger {
  // template <typename T>
  // CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  // operator()(const T& a, const T& b) const {
  //   T out;
  //   out.slot = a.slot;
  //   out.mf_dim = a.mf_dim;
  //   out.show = a.show + b.show;
  //   out.clk = a.clk + b.clk;
  //   out.lr_g = a.lr_g + b.lr_g;
  //   for (int i = 0; i < out.mf_dim; ++i) {
  //     //printf("mf_g: %f\n", a.mf_g[0]);
  //     // a.mf_g[0] = b.mf_g[0];
  //     //((float*)out.mf_g)[i] = ((float*)a.mf_g)[i] + ((float*)b.mf_g)[i]; //
  //     // for local test
  //     out.mf_g[i] = a.mf_g[i] + b.mf_g[i];
  //   }

  //   return out;
  // }
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__ T
  operator()(const T& a, const T& b) const {
    T out;
    out.slot = a.slot;
    out.mf_dim = a.mf_dim;
    out.show = a.show + b.show;
    out.clk = a.clk + b.clk;
    out.lr_g = a.lr_g + b.lr_g;
    return out;
  }

  template <typename GPUAccessor>
  __device__ __forceinline__
  void init_field(float* output, const float* input, GPUAccessor& gpu_accessor) {
      gpu_accessor.PushValueFill(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__
  void copy_basic_field(float* output, const float* input, GPUAccessor& gpu_accessor) {
      gpu_accessor.PushValueFillBasic(output, input);
  }

  template <typename GPUAccessor>
  __device__ __forceinline__
  void add_basic_field(float* output, const float* input, GPUAccessor& gpu_accessor) {
      gpu_accessor.MergePushValueBasic(output, input);
  }


  template <typename GPUAccessor>
  __device__ __forceinline__
  void copy_embedx_field(float* output, const float* input, size_t embedx_id, GPUAccessor& gpu_accessor) {
       gpu_accessor.PushValueFillEmbedx(output, input, embedx_id);
  }


  template <typename GPUAccessor>
  __device__ __forceinline__
  void add_embedx_field(float* output, const float* input, size_t embedx_id, GPUAccessor& gpu_accessor) {
       gpu_accessor.MergePushValueEmbedx(output, input, embedx_id);
  }

};

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
class HeterComm {
  using HeterCommType = HeterComm<KeyType, ValType, GradType, GPUAccessor>;
  static const int COPY_KEY = 0x01;
  static const int COPY_VAL = 0x02;
  static const int COPY_ALL = COPY_KEY | COPY_VAL;

 public:
  HeterComm(size_t capacity, std::shared_ptr<HeterPsResource> resource, GPUAccessor& gpu_accessor);
  virtual ~HeterComm();
  HeterComm(const HeterComm&) = delete;
  HeterComm& operator=(const HeterComm&) = delete;

/*
 // ======== all2all ========
  // 和pslib分机器对齐
  // copied from pslib
  __device__ int32_t sparse_local_shard_num(uint32_t shard_num, uint32_t server_num) {
    if (shard_num % server_num == 0) {
      return shard_num / server_num;
    }
    size_t local_shard_num = shard_num / server_num + 1;
    return local_shard_num;
  }

  // shard分机器
  __device__ int PartitionShardForRank(const uint64_t& shard_id) {
    int rank_id = shard_id  / sparse_local_shard_num(thread_keys_shard_num_, node_size_);
    return rank_id;
  }

  // feasign分机器
  __device__ int PartitionKeyForRank(const uint64_t& key) {
    int shard_id = key % thread_keys_shard_num_;
    int rank_id = shard_id  / sparse_local_shard_num(thread_keys_shard_num_, node_size_);
    return rank_id;
  }
*/

  void set_thread_keys_shard_num(const int& thread_keys_shard_num) {
    thread_keys_shard_num_ = thread_keys_shard_num;
  }
  // ======== all2all ========
  size_t merge_keys(const int gpu_num,
                    const KeyType* d_keys,
                    const size_t& len,
                    KeyType* d_sorted_keys,
                    KeyType* d_merged_keys,
                    uint32_t* d_restore_idx,
                    const cudaStream_t& stream);

  void split_input_to_shard(KeyType* d_keys, int* d_idx_ptr, size_t len,
                            int* left, int* right, int gpu_num);

  void split_input_by_mfdim(
    GradType* d_grads, int* d_idx_ptr, size_t len, int* left, int* right, int gpu_num);

  void reorder_input_by_mfdim(KeyType* d_keys, GradType* d_grads, size_t len, int* lens, int gpu_num, size_t& reorder_grad_len);

  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
                  int& uniq_len);  // NOLINT
  void merge_grad(int gpu_num, KeyType* d_keys, GradType* d_grads, float* mf,
                  size_t len, int& uniq_len);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  // void pull_sparse_multi_node(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void build_ps(int num, KeyType* h_keys, char* pool, size_t len, size_t feature_value_size,
                size_t chunk_size, int stream_num);
  void dump();
  void show_one_table(int gpu_num);
  int get_index_by_devid(int devid);
 
  template <typename Sgd>
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                   Sgd& sgd);  // NOLINT

  // template <typename Sgd>
  // void push_sparse_multi_node(int num, KeyType* d_keys, GradType* d_grads,
  //                             size_t len, Sgd& sgd);  // NOLINT

  template <typename Sgd>
  void update_one_table(int num, KeyType* d_keys, GradType* d_grads, size_t len,
                        Sgd& sgd);  // NOLINT


/*
  int gather_one_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                           int len);
  
  int gather_one_node_grad_v2(int num, KeyType* d_keys, GradType* d_grads,
                           int len);


  int gather_multi_node_grad(int num, KeyType* d_keys, GradType* d_grads,
                             int len);

  int gather_multi_node_grad_v2(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len);
  int gather_multi_node_grad_v3(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len);
  int gather_multi_node_grad_v4(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len);
  int gather_multi_node_grad_v5(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len);
*/


  void set_sparse_sgd(const OptimizerConfig& optimizer_config);
  void set_embedx_sgd(const OptimizerConfig& optimizer_config);

  int log2i(int x);

  void set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                              const std::vector<ncclComm_t>& inter_comms,
                              int comm_size,
                              int rank_id) {
    nccl_inner_comms_ = inner_comms;
    nccl_inter_comms_ = inter_comms;
    node_size_ = comm_size;
    rank_id_ = rank_id;
  }
  

 

  void set_trans_inter_comm(const std::vector<ncclComm_t>& trans_inter_comms) {
    nccl_trans_inter_comms_ = trans_inter_comms;
  }
  
  void set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
    
    multi_mf_dim_ = multi_mf_dim;
    max_mf_dim_ = max_mf_dim;
    VLOG(3) << "heter comm set multi multi_mf_dim_: " << multi_mf_dim_ << " max_mf_dim_: " << max_mf_dim_;
  }

  bool need_transfer(int send_id, int receive_id) {
    return ((send_id / 4 != receive_id / 4) && (send_id + 4) % 8 != receive_id);
  }

  // void dump_to_cpu(int index);

  void end_pass();

  int dedup_keys_and_fillidx(const int gpu_id,
                             const int total_fea_num,
                             const KeyType* d_keys,   // input
                             KeyType* d_merged_keys,  // output
                             KeyType* d_sorted_keys,
                             uint32_t* d_restore_idx,
                             uint32_t* d_sorted_idx,
                             uint32_t* d_offset,
                             uint32_t* d_merged_cnts,
                             bool filter_zero,
                             cudaStream_t stream = 0);

  void fill_restore_idx(bool filter_zero,
                        const size_t total_num,
                        const size_t merge_size,
                        const KeyType* d_keys,
                        const uint32_t* d_sorted_idx,
                        const uint32_t* d_offset,
                        const uint32_t* d_merged_cnts,
                        uint32_t* d_restore_idx,
                        const cudaStream_t& stream);

  int get_transfer_devid(int send_id) { return (send_id + 4) % 8; }


template <typename T>
  void split_idx_to_shard(KeyType* d_keys,
                          T* d_idx_ptr,
                          size_t len,
                          T* left,
                          T* right,
                          int gpu_num,
                          const cudaStream_t& stream);


  struct Node {
    cudaStream_t in_stream;
    cudaStream_t out_stream;
    char* key_storage;
    char* val_storage;
    int sync;
    size_t key_bytes_len;
    size_t val_bytes_len;
    int gpu_num;
  };

  struct Path {
    std::vector<Node> nodes_;
  };

  struct CopyTask {
    Path* path;
    int step;
    CopyTask(Path* path_, int step_) : path(path_), step(step_) {}
  };

 struct InnerResource {
    uint32_t* d_idx = nullptr;
    size_t* h_part_sizes = nullptr;
    std::vector<size_t> h_offsets;
    uint32_t* d_offset_ptr = nullptr;

    KeyType* d_keys_parted = nullptr;
    char* d_vals_parted = nullptr;
    std::vector<KeyType*> d_remote_keys;
    std::vector<char*> d_remote_vals;
    KeyType* d_trans_keys = nullptr;
    char* d_trans_vals = nullptr;

    // resize vector
    void resize(const int num_gpu) {
      h_offsets.resize(num_gpu);
      d_remote_keys.resize(num_gpu);
      d_remote_vals.resize(num_gpu);
    }
  };
  // Resource for partition shard Key by nodes
  struct ShardResource {
    uint32_t* d_local_idx_parted = nullptr;  // uint32_t for multisplit
    std::vector<size_t> h_local_part_sizes;
    std::vector<size_t> h_local_part_offsets;
    std::vector<size_t> h_remote_part_sizes;
    std::vector<size_t> h_remote_part_offsets;
    uint32_t* d_node_size_ptr = nullptr;
    std::vector<uint32_t> h_push_fea_sizes;
    // shard part
    void resize_part_size(const int node_size) {
      if (h_local_part_sizes.size() >= static_cast<size_t>(node_size)) {
        return;
      }
      h_local_part_sizes.resize(node_size);
      h_local_part_offsets.resize(node_size + 1);
      h_remote_part_sizes.resize(node_size);
      h_remote_part_offsets.resize(node_size + 1);
      h_push_fea_sizes.resize(node_size * node_size);
    }
  };
  // pull parition shard key by devices
  struct PullResource {
    size_t h_recv_fea_num = 0;
    uint32_t* d_restore_keys_idx = nullptr;
  };

  struct LocalStorage {
    LocalStorage() {}

    // void init(size_t size, int dev_id) {
    //  place_ = platform::CUDAPlace(dev_id);
    //  alloc(size, true);
    // }

    void init(int device_num, int dev_id, phi::Stream stream) {
      place_ = platform::CUDAPlace(dev_id);
      h_recv_offsets.resize(device_num);
      h_fea_sizes.resize(device_num);
      stream_ = stream;
    }

/*
    void reset() {
      all_keys_mem.reset();
      all_grads_mem.reset();
      local_keys_mem.reset();
      local_grads_mem.reset();

    }


    // int feanum_{1800 * 2048};
    // storage_[i].init(feanum_, resource_->dev_id(i), grad_type_size_);
    void init(size_t size, int dev_id, size_t grad_type_size) {
      place_ = platform::CUDAPlace(dev_id);
      grad_type_size_ = grad_type_size;
      alloc(size, true);
    }

    void alloc(size_t size, bool force = false) {

      // VLOG(0) << "yxf local storage alloc size: " << size << " force: " << force;
      // VLOG(0) << "yxf LocalStorage alloc grad size: " << grad_type_size_;

      if (force || size > all_keys_mem->size()) {
        // VLOG(0) << "yxf11 LocalStorage alloc grad size: " << grad_type_size_;
        all_keys_mem.reset();
        // VLOG(0) << "yxf22 LocalStorage alloc grad size: " << grad_type_size_;
        all_grads_mem.reset();
        // VLOG(0) << "yxf33 LocalStorage alloc grad size: " << grad_type_size_;
        all_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));

        // VLOG(0) << "yxf LocalStorage alloc grad size: " << grad_type_size_;
        all_grads_mem = memory::AllocShared(place_, size * grad_type_size_);

        all_keys = reinterpret_cast<KeyType*>(all_keys_mem->ptr());

        all_grads = reinterpret_cast<GradType*>(all_grads_mem->ptr());

      }
      if (force || size > local_keys_mem->size()) {

        local_keys_mem.reset();
        local_grads_mem.reset();

        local_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));

        // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
        local_grads_mem = memory::AllocShared(place_, size * grad_type_size_);

        local_keys = reinterpret_cast<KeyType*>(local_keys_mem->ptr());
        local_grads = reinterpret_cast<GradType*>(local_grads_mem->ptr());

      }

    }
*/

    template <typename T>
    T* alloc_cache(const size_t& len,
                   std::shared_ptr<memory::Allocation>& alloc,  // NOLINT
                   bool need_copy = false) {
      size_t need_mem = len * sizeof(T);
      if (alloc.get() == nullptr) {
        alloc = memory::AllocShared(place_, need_mem);
      } else if (need_mem > alloc->size()) {
        if (need_copy) {
          std::shared_ptr<memory::Allocation> tmp =
              memory::AllocShared(place_, need_mem);
#if defined(PADDLE_WITH_CUDA)
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaMemcpyAsync(tmp->ptr(),  // output
                              alloc->ptr(),
                              alloc->size(),
                              cudaMemcpyDeviceToDevice,
                              reinterpret_cast<cudaStream_t>(stream_.id())));
          PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(
              reinterpret_cast<cudaStream_t>(stream_.id())));
#endif
          alloc.reset();
          alloc = tmp;
        } else {
          alloc.reset();
          alloc = memory::AllocShared(place_, need_mem);
        }
      }
      return reinterpret_cast<T*>(alloc->ptr());
    }
/*
    template <typename T>
    T* alloc_cache(const size_t& len,
                   std::shared_ptr<memory::Allocation>& alloc,  // NOLINT
                   bool need_copy = false) {
      size_t need_mem = len * sizeof(T);
      if (alloc.get() == nullptr) {
        alloc = memory::Alloc(place_, need_mem);
      } else if (need_mem > alloc->size()) {
        if (need_copy) {
          std::shared_ptr<memory::Allocation> tmp =
              memory::Alloc(place_, need_mem);
          cudaMemcpy(tmp->ptr(),
                     alloc->ptr(),
                     alloc->size(),
                     cudaMemcpyDeviceToDevice);
          alloc.reset();
          alloc = tmp;
        } else {
          alloc.reset();
          alloc = memory::Alloc(place_, need_mem);
        }
      }
      return reinterpret_cast<T*>(alloc->ptr());
    }
*/

    void alloc(const size_t& len,
               const size_t& value_bytes = sizeof(GradType),
               const int copy_mode = 0) {
      all_keys =
          alloc_cache<KeyType>(len, all_keys_mem, (copy_mode & COPY_KEY));
      all_grads = alloc_cache<char>(
          len * value_bytes, all_grads_mem, (copy_mode & COPY_VAL));
      local_keys =
          alloc_cache<KeyType>(len, local_keys_mem, (copy_mode & COPY_KEY));
      local_grads = alloc_cache<char>(
          len * value_bytes, local_grads_mem, (copy_mode & COPY_VAL));

      d_merged_keys = all_keys;
      d_merged_push_keys = local_keys;
      d_merged_vals = all_grads;
      d_merged_push_vals = local_grads;
    }
/*
    void alloc_for_inter_copy(size_t size, bool force = false) {
      // VLOG(0) << "yxf LocalStorage222 alloc grad size: " << size << " local mem size: " << local_keys_mem->size();

      if (force || size * sizeof(KeyType) > local_keys_mem->size()) {

        local_keys_mem.reset();
        local_grads_mem.reset();

        local_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));

        VLOG(0) << "yxf LocalStorage111 alloc grad size: " << size * grad_type_size_;

        local_grads_mem = memory::AllocShared(place_, size * grad_type_size_);

        local_keys = reinterpret_cast<KeyType*>(local_keys_mem->ptr());
        local_grads = reinterpret_cast<GradType*>(local_grads_mem->ptr());

      }

    }

    void alloc_for_multi_node_nccl(size_t size, bool force = false) {
      if (force || size > all_keys_mem->size()) {
        all_keys_mem.reset();
        all_grads_mem.reset();
        all_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));
        // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
        all_grads_mem = memory::AllocShared(place_, size * grad_type_size_);
        all_keys = reinterpret_cast<KeyType*>(all_keys_mem->ptr());
        all_grads = reinterpret_cast<GradType*>(all_grads_mem->ptr());
      }
    }

    void alloc_for_data_transfer(size_t size, bool force = false) {
      if (force || size > all_keys_mem->size()) {
        all_keys_mem.reset();
        all_grads_mem.reset();
        all_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));
        // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
        all_grads_mem = memory::AllocShared(place_, size * grad_type_size_);
        all_keys = reinterpret_cast<KeyType*>(all_keys_mem->ptr());
        all_grads = reinterpret_cast<GradType*>(all_grads_mem->ptr());
      }
    }

    void alloc_for_data_transfer_nccl(size_t size, bool force = false) {
      if (force || size > local_keys_mem->size()) {
        local_keys_mem.reset();
        local_grads_mem.reset();
        local_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));
        // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
        local_grads_mem = memory::AllocShared(place_, size * grad_type_size_);
        local_keys = reinterpret_cast<KeyType*>(local_keys_mem->ptr());
        local_grads = reinterpret_cast<GradType*>(local_grads_mem->ptr());
      }
    }

    void alloc_in_transfer(size_t size, bool force = false) {
      VLOG(0) << "yxf::alloc in trans: size: " << size << " mem size: " << trans_all_keys_mem->size();
      if (force || size > trans_all_keys_mem->size()) {
        trans_all_keys_mem.reset();
        trans_all_grads_mem.reset();
        trans_all_keys_mem = memory::AllocShared(place_, size * sizeof(KeyType));
        // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
        trans_all_grads_mem = memory::AllocShared(place_, size * grad_type_size_);
        trans_all_keys = reinterpret_cast<KeyType*>(trans_all_keys_mem->ptr());
        trans_all_grads = reinterpret_cast<GradType*>(trans_all_grads_mem->ptr());
      }
    }

*/
    // === all2all ===
/*
    void init_pull(const size_t& len) {
      pull_res.h_recv_fea_num = len;
      local_pull_idx = memory::AllocShared(place_, len * sizeof(uint32_t));
      pull_res.d_restore_keys_idx = reinterpret_cast<uint32_t*>(local_pull_idx->ptr());
      
    }
    void init_shard(const size_t& len, const size_t& node_size) {
      local_shard_idx = memory::AllocShared(place_, len * sizeof(uint32_t));
      shard_res.d_local_idx_parted = reinterpret_cast<uint32_t*>(local_shard_idx->ptr());
      d_node_size_buf = memory::AllocShared(place_, node_size * node_size * sizeof(uint32_t));
      shard_res.d_node_size_ptr = reinterpret_cast<uint32_t*>(d_node_size_buf->ptr());
      shard_res.resize_part_size(node_size);
    }
    void init_inner(const size_t& len, const int& device_num) {
      local_inner_idx = memory::AllocShared(place_, len * sizeof(uint32_t));
      inner_res.d_idx = reinterpret_cast<uint32_t*>(local_inner_idx->ptr());
      inner_offset = memory::AllocShared(place_, device_num * 2 * sizeof(uint32_t));
      inner_res.d_offset_ptr = reinterpret_cast<uint32_t*>(inner_offset->ptr());
      inner_res.resize(device_num);
    }
*/
   void init_pull(const size_t& len) {
      pull_res.h_recv_fea_num = len;
      pull_res.d_restore_keys_idx = alloc_cache<uint32_t>(len, local_pull_idx);
    }
    void init_shard(const size_t& len, const size_t& node_size) {
      shard_res.d_local_idx_parted =
          alloc_cache<uint32_t>(len, local_shard_idx);
      shard_res.d_node_size_ptr =
          alloc_cache<uint32_t>(node_size * node_size, d_node_size_buf);
      shard_res.resize_part_size(node_size);
    }
    void init_inner(const size_t& len, const int& device_num) {
      inner_res.d_idx = alloc_cache<uint32_t>(len, local_inner_idx);
      inner_res.d_offset_ptr =
          alloc_cache<uint32_t>(device_num * 2, inner_offset);
      inner_res.resize(device_num);
    }
    // void init_trans(const size_t& fea_num, const size_t& value_bytes) {
    //   d_merged_trans_keys = alloc_cache<KeyType>(fea_num * 2, trans_keys_buff);
    //   d_merged_push_trans_keys = &d_merged_trans_keys[fea_num];
    //   d_merged_trans_vals =
    //       alloc_cache<char>(fea_num * 2 * value_bytes, trans_vals_buff);
    //   d_merged_push_trans_vals = &d_merged_trans_vals[fea_num * value_bytes];
    // }



    std::shared_ptr<memory::Allocation> local_inner_idx = nullptr;
    std::shared_ptr<memory::Allocation> local_pull_idx = nullptr;
    std::shared_ptr<memory::Allocation> local_shard_idx = nullptr;
    std::shared_ptr<memory::Allocation> inner_offset = nullptr;
    std::shared_ptr<memory::Allocation> d_node_size_buf = nullptr;

    InnerResource inner_res;
    ShardResource shard_res;
    PullResource pull_res;

    KeyType* d_merged_keys = nullptr;
    char* d_merged_vals = nullptr;
    KeyType* d_merged_push_keys = nullptr;
    char* d_merged_push_vals = nullptr;

    std::vector<size_t> h_recv_offsets;
    std::vector<size_t> h_fea_sizes;
    // ==== all2all ====
  

    platform::CUDAPlace place_;
    phi::Stream stream_{0};
    std::shared_ptr<memory::Allocation> all_keys_mem;
    std::shared_ptr<memory::Allocation> all_grads_mem;

    KeyType* all_keys;
    char* all_grads;

    std::shared_ptr<memory::Allocation> local_keys_mem;
    std::shared_ptr<memory::Allocation> local_grads_mem;

    KeyType* local_keys;
    char* local_grads;

    std::shared_ptr<memory::Allocation> trans_all_keys_mem;
    std::shared_ptr<memory::Allocation> trans_all_grads_mem;

    KeyType* trans_all_keys;
    GradType* trans_all_grads;

    size_t grad_type_size_;

    // nccl shard info
    int* h_local_left;
    int* h_local_right;

    int* h_merge_offset;
    int* h_merge_len;

    KeyType* tmp_local_keys;
    char* tmp_local_grads;

    size_t gather_one_node_len;

  };

  void init_path();

  void create_storage(int start_index, int end_index, size_t keylen, size_t vallen);
  void destroy_storage(int start_index, int end_index);
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyType* src_key, GradType* src_val);
  void walk_to_dest(int start_index, int gpu_num, int* h_left, int* h_right,
                    KeyType* src_key, char* src_val, size_t val_size);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   ValType* src_val);
  void walk_to_src(int start_index, int gpu_num, int* h_left, int* h_right,
                   char* src_val, size_t val_size);


  void shard_inner_keys(const size_t& total_fea_num,
                        const KeyType* d_keys,
                        const int& gpu_id,
                        const int& gpu_num,
                        InnerResource* res,
                        const cudaStream_t& stream);
  void gather_inner_keys_p2p(const size_t& total_fea_num,
                             const KeyType* d_keys,
                             InnerResource& res,  // NOLINT
                             const int& gpu_id,
                             const int& gpu_num,
                             const int& trans_id,
                             const cudaStream_t& stream);
  size_t gather_inter_keys_by_copy(const int& gpu_id,
                                   const size_t& fea_size,
                                   const KeyType* d_keys,
                                   const cudaStream_t& stream);
  void partition_shard_keys(const int& gpu_id,
                            const size_t& total_fea_num,
                            const KeyType* d_keys,
                            uint32_t* d_idx_parted,
                            KeyType* d_keys_parted,
                            size_t* h_part_sizes,
                            const int& shard_num,
                            const cudaStream_t& stream);
  size_t send_data_by_all2all(const int& gpu_id,
                              const int& nccl_node_size,
                              const int& nccl_rank_id,
                              const int& value_bytes,
                              const size_t* h_send_part_sizes,
                              const size_t* h_send_part_offsets,
                              const size_t* h_recv_part_sizes,
                              const size_t* h_recv_part_offsets,
                              const char* d_send_buff,
                              char* d_rev_buff,
                              const cudaStream_t& stream);

  // size_t gather_sparse_keys_by_all2all(const int& gpu_id,
  //                                     const size_t& fea_size,
  //                                     const KeyType* d_in_keys,
  //                                     const cudaStream_t& stream);

 size_t gather_sparse_keys_by_all2all(const int& gpu_id,
                                       const size_t& fea_size,
                                       const KeyType* d_in_keys,
                                       KeyType* d_out_keys,
                                       KeyType* d_tmp_keys,
                                       const cudaStream_t& stream);


  template <typename T>
  void gather_keys(KeyType* d_shard_keys,
                   const KeyType* d_keys,
                   T* idx,
                   size_t len,
                   const cudaStream_t& stream);

  template <typename T>
  void scatter_keys(const KeyType* d_shard_keys,
                    KeyType* d_keys,
                    T* idx,
                    size_t len,
                    const cudaStream_t& stream);

  template <typename T>
  void gather_vals(float* d_shard_vals,
                  const float* d_vals,
                  T* idx,
                  size_t len,
                  size_t value_bytes,
                  const cudaStream_t& stream);

template <typename T>
void scatter_vals(const float* d_shard_vals,
                  float* d_vals,
                  T* idx,
                  size_t len,
                  size_t value_bytes,
                  const cudaStream_t& stream);
void scale_grad(const size_t& len,
                char* grads,
                const size_t& value_bytes,
                const size_t& max_mf_dim,
                const cudaStream_t& stream,
                const GPUAccessor& gpu_accessor);

  void merge_gradient(const KeyType* d_shard_keys,
                      const uint32_t* offset,
                      const uint32_t* fea_num,
                      const uint32_t* index,
                      const char* input,
                      char* output,
                      int n,
                      size_t grad_dim,
                      size_t grad_value_size,
                      CustomGradMerger& merger,
                      const cudaStream_t& stream,
                      GPUAccessor& gpu_accessor);

  void scatter_sparse_vals_by_all2all(const int& gpu_id,
                                      const size_t& fea_size,
                                      const char* d_in_vals,
                                      void* d_out_vals,
                                      const size_t& value_bytes,
                                      void* d_tmp_vals,
                                      const cudaStream_t& stream);
  void scatter_inner_vals_p2p(const size_t& total_fea_num,
                              void* d_out_vals,
                              InnerResource& res,  // NOLINT
                              const int& gpu_id,
                              const int& gpu_num,
                              const int& trans_id,
                              const size_t& value_bytes,
                              const cudaStream_t& stream);
  void scatter_inter_vals_by_copy(const int& gpu_id,
                                  const size_t& fea_size,
                                  const char* d_in_vals,
                                  void* d_out_vals,
                                  const size_t& value_bytes,
                                  const cudaStream_t& stream);
  void gather_inner_data_p2p(const size_t& total_fea_num,
                             const KeyType* d_keys,
                             const void* d_vals,
                             InnerResource& res,  // NOLINT
                             const int& gpu_id,
                             const int& gpu_num,
                             const int& trans_id,
                             const size_t& value_bytes,
                             const cudaStream_t& stream);
  template <typename Sgd>
  void push_sparse_all2all(const int& gpu_id,
                           KeyType* d_keys,
                           GradType* d_grads,
                           const size_t& len,
                           Sgd& sgd);  // NOLINT
  size_t merge_grads(const int& gpu_id,
                    const size_t& len,
                    const KeyType* d_in_keys,
                    KeyType* d_out_keys,
                    const void* d_in_grads,
                    void* d_out_grads,
                    const cudaStream_t& stream);
  size_t gather_inter_gradient_by_copy(const int& gpu_id,
                                       const size_t& push_size,
                                       KeyType* d_keys,
                                       void* d_push_vals,
                                       const size_t& value_bytes,
                                       const cudaStream_t& stream);
  size_t gather_sparse_gradient_by_all2all(const int& gpu_id,
                                           const size_t& push_size,
                                           const KeyType* d_keys,
                                           const char* d_push_vals,
                                           const size_t& value_bytes,
                                           KeyType* d_out_keys,
                                           KeyType* d_tmp_keys,
                                           char* d_out_vals,
                                           char* d_tmp_vals,
                                           const cudaStream_t& stream);


 void pull_one_table(const int gpu_id,
                      KeyType* d_keys,
                      ValType* d_vals,
                      const size_t& len,
                      const cudaStream_t& stream);

  // node all2all pull
  void pull_sparse_all2all(const int& gpu_id,
                           KeyType* d_keys,
                           ValType* d_vals,
                           const size_t& len);


 protected:
  GPUAccessor gpu_accessor_; 
  using Table = HashTable<KeyType, ValType>;
  using PtrTable = HashTable<KeyType, ValType*>;
  std::vector<Table*> tables_;
  std::vector<PtrTable*> ptr_tables_;
  std::shared_ptr<HeterPsResource> resource_;
  std::vector<std::vector<Path>> path_;
  float load_factor_{0.75};
  int block_size_{256};
  int direct_access_ = 1;
  int test_batch = 0;

 private:

  std::vector<LocalStorage> storage_;
  // std::vector<LocalStorage> pull_storage_;

  CustomGradMerger merger_;

  int topo_aware_{0};

  int feanum_{1800 * 2048};
  int device_num_ = 8;
  int multi_node_{1};
  int rank_id_{0};
  std::vector<ncclComm_t> nccl_inner_comms_;
  std::vector<ncclComm_t> nccl_inter_comms_;
  std::vector<ncclComm_t> nccl_trans_inter_comms_;
/*
  std::vector<double> mg_time_1;
  std::vector<double> mg_time_2;
  std::vector<double> mg_time_3;
  std::vector<double> mg_time_4;
  std::vector<double> mg_time_5;
  std::vector<double> mg_time_6;
  std::vector<double> mg_time_7;
  std::vector<double> mg_time_8;
  std::vector<double> mg_time_9;
  std::vector<double> mg_time_10;
  std::vector<double> mg_time_11;
  std::vector<double> mg_time_12;
  std::vector<double> mg_time_13;
  std::vector<double> mg_time_14;
  std::vector<double> mg_time_15;
*/

  int node_size_ = 1;
  int thread_keys_shard_num_ = 37;
  std::vector<std::shared_ptr<cub::CachingDeviceAllocator>> allocators_;

  int multi_mf_dim_{8};
  int max_mf_dim_ = 8;
  size_t val_type_size_;
  size_t grad_type_size_;
  size_t pull_type_size_;
  size_t max_type_size_;
  std::unordered_map<int,std::shared_ptr<memory::Allocation>> trans_keys;
  std::unordered_map<int, std::shared_ptr<memory::Allocation>> trans_grads;

  std::vector<int> trans_ids = {2, 3, 4, 5};

 };

}  // end namespace framework
}  // end namespace paddle
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_inl.h"
#endif
