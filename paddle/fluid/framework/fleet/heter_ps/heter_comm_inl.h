/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
ou may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_HETERPS
//#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_utils.h"
#include <queue>

DECLARE_int32(gpups_dedup_pull_push_mode);

namespace paddle {
namespace framework {



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
  // __device__ int PartitionShardForRank(const uint64_t& shard_id, const int thread_keys_shard_num, const int node_size) {
  //  int rank_id = shard_id  / sparse_local_shard_num(thread_keys_shard_num, node_size);
  //  return rank_id;
  //}

  // feasign分机器
  __device__ int PartitionKeyForRank(const uint64_t& key, int thread_keys_shard_num, int node_size) {
    int shard_id = key % thread_keys_shard_num;
    int rank_id = shard_id  / sparse_local_shard_num(thread_keys_shard_num, node_size);
    return rank_id;
  }
// ======== all2all ========


template <typename T>
__global__ void fill_idx(T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

// template <typename T>
// void show_tensor(T* input, size_t len, gpuStream_t stream, std::string name) {
//  T tmp[len];  // NOLINT
//  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost, stream);
//  cudaStreamSynchronize(stream);
//  std::cout << name;
//  for (int i = 0; i < len; ++i) {
//    std::cout << ":" << tmp[i];
//  }
//  std::cout << std::endl;
//}

template <typename T>
__global__ void calc_shard_offset(T* idx, T* left, T* right, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

template <typename TUnit, typename T>
__global__ void gather_keys_kernel(TUnit* d_dest_vals,
                            const TUnit* d_src_vals,
                            T* idx,
                            size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_dest_vals[i] = d_src_vals[idx[i]];
  }
}

template <typename TUnit, typename T>
__global__ void scatter_keys_kernel(TUnit* d_dest_vals,
                                    const TUnit* d_src_vals,
                                    T* idx,
                                    size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_dest_vals[idx[i]] = d_src_vals[i];
  }
}


template <typename TUnit, typename T>
__global__ void gather_dvals_by_unit_kernel(TUnit* d_dest_vals,
                                            const TUnit* d_src_vals,
                                            T* idx,
                                            size_t len,
                                            const size_t val_size_unit) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t pos = idx[i / val_size_unit] * val_size_unit + (i % val_size_unit);
    d_dest_vals[i] = d_src_vals[pos];
  }
}

template <typename TUnit, typename T>
__global__ void scatter_dvals_by_unit_kernel(TUnit* d_dest_vals,
                                             const TUnit* d_src_vals,
                                             T* idx,
                                             size_t len,
                                             const size_t val_size_unit) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    size_t pos = idx[i / val_size_unit] * val_size_unit + (i % val_size_unit);
    d_dest_vals[pos] = d_src_vals[i];
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename T>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_keys(KeyType* d_shard_keys,
                                                                    const KeyType* d_keys,
                                                                    T* idx,
                                                                    size_t len,
                                                                    const cudaStream_t& stream) {
  size_t N = len;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_keys -> d_shard_keys
  gather_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, idx, N);
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename T>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::scatter_keys(const KeyType* d_shard_keys,
                                                                      KeyType* d_keys,
                                                                      T* idx,
                                                                      size_t len,
                                                                      const cudaStream_t& stream) {
  size_t N = len;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_shard_keys -> d_keys
  scatter_keys_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, d_shard_keys, idx, N);
}

template <typename KeyType, typename T>
__global__ void calc_shard_index(KeyType* d_keys, size_t len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

// adapt for pslib shard rules 
template <typename KeyType, typename T>
__global__ void calc_node_shard_index(KeyType* d_keys,
                                      const size_t len,
                                      T* shard_index,
                                      const int thread_keys_shard_num,
                                      const int node_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    // shard_index[i] = (d_keys[i] / total_gpu) % node_num;
    shard_index[i] = PartitionKeyForRank(d_keys[i], thread_keys_shard_num, node_size);
  }
}

/*
template <typename GradType, typename T>
__global__ void calc_mfdim_index(GradType* d_grads, size_t len, T* mfdim,
                                 int total_gpu, size_t grad_value_size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    FeaturePushValue* cur = (FeaturePushValue*)((char*)d_grads + i * grad_value_size);
    if (cur->mf_dim == 8) {
      mfdim[i] = 0;
    } else {
      mfdim[i] = 1;
    }
    // mfdim[i] = cur->mf_dim;
  }
}
*/


// template <typename T, typename GPUAccessor>
// __global__ void calc_mfdim_index(float* d_grads,
//                                  size_t len,
//                                  T* mfdim,
//                                  int total_gpu,
//                                  size_t grad_value_size,
//                                  GPUAccessor gpu_accessor) {
//   const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < len) {
//     // FeaturePushValue* cur = (FeaturePushValue*)((char*)d_grads + i * grad_value_size);
//     float* cur = (float*)((char*)d_grads + i * grad_value_size);
//     int mf_dim = cur[gpu_accessor->get_MfDim_index()];
//     if (mf_dim == 8) {
//       mfdim[i] = 0;
//     } else {
//       mfdim[i] = 1;
//     }
//     // mfdim[i] = cur->mf_dim;
//   }
// }

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                               size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                 GradType* d_shard_grads, GradType* d_grads,
                                 T* idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

// check
// template <typename GradType>
// __global__ void dy_mf_fill_mfdim_grads(GradType* d_shard_grads,
//                                        GradType* d_grads, int* idx, size_t len,
//                                        size_t grad_value_size, size_t offset, size_t max_value_size) {
//   const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < len) {
//     *(GradType*)((char*)d_shard_grads + i * grad_value_size) =
//         *(GradType*)((char*)d_grads + uint64_t(idx[i + offset]) * max_value_size);
//   }
// }

template <typename KeyType, typename T, typename GPUAccessor>
__global__ void dy_mf_fill_shard_grads(KeyType* d_shard_keys,
                                       KeyType* d_keys,
                                       float* d_shard_grads,
                                       float* d_grads,
                                       T* idx,
                                       size_t len,
                                       size_t grad_value_size,
                                       GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    float* cur = (float*)((char*)d_shard_grads + i * grad_value_size);
    float* shard_val =
        (float*)((char*)d_grads + uint64_t(idx[i]) * grad_value_size);
    gpu_accessor.PushValueFill(cur, shard_val);
  }
}


template <typename KeyType, typename T, typename GPUAccessor>
__global__ void dy_mf_fill_shard_vals(KeyType* d_shard_keys,
                                      KeyType* d_keys,
                                      float* d_shard_vals,
                                      float* d_vals,
                                      T* idx,
                                      size_t len,
                                      size_t pull_value_size,
                                      GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    float* cur = (float*)((char*)d_shard_vals + i * pull_value_size);
    float* shard_val =
        (float*)((char*)d_vals + uint64_t(idx[i]) * pull_value_size);
    gpu_accessor.Pull2PullValueFill(cur, shard_val);
  }
}


// optimized version
template <>
__global__ void dy_mf_fill_shard_grads<FeatureKey, int, CommonFeatureValueAccessor>(FeatureKey* d_shard_keys,
                                                                                    FeatureKey* d_keys,
                                                                                    float* d_shard_grads,
                                                                                    float* d_grads,
                                                                                    int* idx,
                                                                                    size_t len,
                                                                                    size_t grad_value_size,
                                                                                    CommonFeatureValueAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t k = threadIdx.x;

  if (i < len) {
    if (k == 0) {
      d_shard_keys[i] = d_keys[idx[i]];
    }
    float* cur = (float*)((char*)d_shard_grads + i * grad_value_size);
    float* input = (float*)((char*)d_grads + uint64_t(idx[i]) * grad_value_size);
    gpu_accessor.FillShardGrads(cur, input, blockDim.x, k);
  }
}

// optimized version
template <>
__global__ void dy_mf_fill_shard_vals<FeatureKey, int, CommonFeatureValueAccessor>(FeatureKey* d_shard_keys,
                                                                                   FeatureKey* d_keys,
                                                                                   float* d_shard_vals,
                                                                                   float* d_vals,
                                                                                   int* idx,
                                                                                   size_t len,
                                                                                   size_t pull_value_size,
                                                                                   CommonFeatureValueAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t k = threadIdx.x;

  if (i < len) {
    if (k == 0) {
      d_shard_keys[i] = d_keys[idx[i]];
    }
    float* cur = (float*)((char*)d_shard_vals + i * pull_value_size);
    float* input = (float*)((char*)d_vals + uint64_t(idx[i]) * pull_value_size);
    gpu_accessor.FillPull2PullDvals(cur, input, blockDim.x, k);
  }
}

// merge gradient
template <typename GPUAccessor>
__global__ void merge_gradient_basic_kernel(const uint64_t* keys,
                                            const uint32_t* offset,
                                            const uint32_t* fea_num,
                                            const uint32_t* index,
                                            const char* input,
                                            char* output,
                                            int n,
                                            size_t grad_value_size,
                                            CustomGradMerger& merger,
                                            GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
/*
    if (keys[i] != 0) {
      uint32_t start = offset[i];
      uint32_t num = fea_num[i];
      int ori_index = index[start];

      FeaturePushValue& lhs = *(FeaturePushValue*)(output + i * grad_value_size);
      FeaturePushValue& in =
          *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
      merger.copy_basic_field(lhs, in);
      
      for (int j = 1; j < num; ++j) {
        ori_index = index[start + j];
        FeaturePushValue& rhs = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
        merger.add_basic_field(lhs, rhs);
      }
    } else if (keys[i] == 0) {
      FeaturePushValue& lhs = *(FeaturePushValue*)(output + i * grad_value_size);
      lhs.show = 0;
      lhs.clk = 0;
      lhs.slot = -1;
      lhs.lr_g = 0;
      // 
      lhs.mf_dim = 64;
      for (int j = 0; j < 64; ++j) {
        lhs.mf_g[j] = 0;
      }
*/
    if (keys[i] != 0) {
      uint32_t start = offset[i];
      uint32_t num = fea_num[i];
      int ori_index = index[start];
      float* lhs = (float*)(output + i * grad_value_size);
      float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
      merger.copy_basic_field(lhs, rhs, gpu_accessor);
      for (int j = 1; j < num; ++j) {
        ori_index = index[start + j];
        rhs = (float*)(input + size_t(ori_index) * grad_value_size);
        merger.add_basic_field(lhs, rhs, gpu_accessor);
      }
    } else if (keys[i] == 0) {
      uint32_t start = offset[i];
      int ori_index = index[start];
      float* lhs = (float*)(output + i * grad_value_size);
      float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
      merger.init_field(lhs, rhs, gpu_accessor);
    }
  }
}

// __global__ void merge_gradient_embedx_kernel(const uint64_t* keys, const uint32_t* offset,
//                                       const uint32_t* fea_num,
//                                       const uint32_t* index, const char* input,
//                                       char* output, int n,
//                                       size_t grad_dim,
//                                       size_t grad_value_size,
//                                       CustomGradMerger& merger) {
template <typename GPUAccessor>
__global__ void merge_gradient_embedx_kernel(const uint64_t* keys,
                                             const uint32_t* offset,
                                             const uint32_t* fea_num,
                                             const uint32_t* index,
                                             const char* input,
                                             char* output,
                                             int n,
                                             size_t grad_dim,
                                             size_t grad_value_size,
                                             CustomGradMerger& merger,
                                             GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t value_idx = i / grad_dim;
    if (keys[value_idx] > 0) {
/*
      size_t field_idx = i % grad_dim;

      uint32_t start = offset[value_idx];
      uint32_t num = fea_num[value_idx];
      int ori_index = index[start];

      FeaturePushValue& in = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
      FeaturePushValue& lhs = *(FeaturePushValue*)(output + value_idx * grad_value_size);

      merger.copy_embedx_field(lhs, in, field_idx);

      for (int j = 1; j < num; ++j) {
        int ori_index = index[start + j];
        FeaturePushValue& rhs = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
        merger.add_embedx_field(lhs, rhs, field_idx);
      }
*/
      size_t field_idx = i % grad_dim;
      uint32_t start = offset[value_idx];
      uint32_t num = fea_num[value_idx];
      int ori_index = index[start];
      float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
      float* lhs = (float*)(output + value_idx * grad_value_size);
      merger.copy_embedx_field(lhs, rhs, field_idx, gpu_accessor);
      for (int j = 1; j < num; ++j) {
        int ori_index = index[start + j];
        float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
        merger.add_embedx_field(lhs, rhs, field_idx, gpu_accessor);
      }
    }
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                           size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

// ===== hbm optimized ======
template <typename T, typename GPUAccessor>
__global__ void dy_mf_fill_dvals(float* d_shard_vals,
                                 float* d_vals,
                                 T* idx,
                                 size_t len,
                                 size_t val_size,
                                 GPUAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint64_t new_offset = uint64_t(idx[i]) * val_size;
    float* cur = (float*)((char*)d_vals + new_offset);
    float* shard_val = (float*)((char*)d_shard_vals + uint64_t(i) * val_size);
    gpu_accessor.Pull2PullValueFill(cur, shard_val);
  }
}

// optimized version
template <>
__global__ void dy_mf_fill_dvals<int, CommonFeatureValueAccessor>(float* d_shard_vals,
                                                                  float* d_vals,
                                                                  int* idx,
                                                                  size_t len,
                                                                  size_t val_size,
                                                                  CommonFeatureValueAccessor gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t k = threadIdx.x;
  if (i < len) {
    uint64_t new_offset = uint64_t(idx[i]) * val_size;
/*
    FeatureValue* cur = (FeatureValue*)((char*)d_vals + new_offset);
    FeatureValue& input = *(FeatureValue*)((char*)d_shard_vals + i * val_size);
    char* cur_p = (char*)cur;
    char* input_p = (char*)(&input);
    int len = 9 + input.mf_dim + 1;
    if (len == 18 || len == 74) {

    if (k == 3 || k == 6 || k == 7) *(int*)(cur_p + k * 4) = *(int*)(input_p + k * 4);
    else if (k < 8) *(float*)(cur_p + k * 4) = *(float*)(input_p + k * 4);
    else if (k == 8) { 
      *(uint64_t*)(cur_p + k * 4) = *(uint64_t*)(input_p + k * 4);
    } else {
      if (k - 9 < input.mf_dim + 1) {
        for (int j = k-9; j < input.mf_dim + 1; j += blockDim.x - 9) {
          cur->mf[j] = input.mf[j];
        }
      }
        // int len_per_thread = (len - 9) / (blockDim.x - 9);
        // int remain = (len - 9) % (blockDim.y - 9);
        // int real_len = len_per_thread;
        // if ((k - 9) < remain) real_len++;
        // int left = -1, right = -1;
        // if ((k - 9) < remain) {
        //   left = 9 + (k - 9) * (len_per_thread + 1);
        //   right = left + real_len;
        // } else {
        //   left = 9 + remain * (len_per_thread + 1) + (k - 9 - remain) * len_per_thread;
        //   right = left + real_len;
        // }
        // for(int j = left; j < right; j++) *(float*)(cur_p + (j + 1) * 4) = *(float*)(input_p + (j + 1) * 4);

    }

    }
*/  
    float* cur = (float*)((char*)d_vals + new_offset);
    float* input = (float*)((char*)d_shard_vals + i * val_size);
    gpu_accessor.FillPull2PullDvals(cur, input, blockDim.x, k);
  }
}

// ===== hbm optimized ======

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::HeterComm(
    size_t capacity, std::shared_ptr<HeterPsResource> resource, GPUAccessor& gpu_accessor) {

  VLOG(1) << "Construct new HeterComm";
  resource_ = resource;
  // rank_id_ = resource_->node_rank();
  device_num_ = resource_->total_gpu();
  storage_.resize(device_num_);
  h_barrier.reset(device_num_);
  
  multi_mf_dim_ = resource->multi_mf();
  gpu_accessor_ = gpu_accessor;

  multi_node_ = resource_->multi_node();
  max_mf_dim_ = resource->max_mf_dim();
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  // size_t val_type_size =
  //    accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
  // size_t grad_type_size =
  //    accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
  val_type_size_ =
      accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
  grad_type_size_ =
      accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
  pull_type_size_ =
      accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

  VLOG(3) << " HeterComm init, max feature_value_size:" << val_type_size_
          << ", pull_value_size:" << pull_type_size_
          << ", feature_value_push_size:" << grad_type_size_;

  max_type_size_ = std::max(pull_type_size_, grad_type_size_);

  for (int i = 0; i < resource_->total_gpu(); ++i) {
    auto stream = resource_->local_stream(i, 0);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
    //     2, 1, 20, (size_t)-1, false, false));  // NOLINT

    // 不是direct accesss就用不到了
    allocators_.push_back(std::make_shared<cub::CachingDeviceAllocator>(
        8, 1, (unsigned int)-1, (size_t)-1, false, false));

/*
      max_mf_dim_ = resource->max_mf_dim();
      auto accessor_wrapper_ptr =
          GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

      // size_t val_type_size =
      //    accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
      // size_t grad_type_size =
      //    accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
      val_type_size_ =
          accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
      grad_type_size_ =
          accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
      pull_type_size_ =
          accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

      VLOG(3) << " HeterComm init, max feature_value_size:" << val_type_size_
              << ", pull_value_size:" << pull_type_size_
              << ", feature_value_push_size:" << grad_type_size_;

      max_type_size_ = std::max(pull_type_size_, grad_type_size_);
*/

      auto ptr_table = new PtrTable(capacity / load_factor_);
      ptr_table->set_feature_value_size(pull_type_size_, grad_type_size_);
      ptr_tables_.push_back(ptr_table);

    if (multi_node_) {
      storage_[i].init(device_num_, resource_->dev_id(i),
                       phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    } // else {
    //   storage_[i].init(feanum_, resource_->dev_id(i), grad_type_size_);
    // }
    
  }

  // std::vector<int> trans_ids = {2, 3, 4, 5};
  // trans_keys.resize(4);
  // trans_grads.resize(4);
  // 这4张卡用来干嘛？？
  // gather_multi_node_grad_v4用的
  // for (auto id : trans_ids) {
  //
  //  platform::CUDAPlace place = platform::CUDAPlace(id);
  //  platform::CUDADeviceGuard guard(id);
  //
  //  // std::unordered_map<int,std::shared_ptr<memory::Allocation>> trans_keys;
  //   // std::unordered_map<int, std::shared_ptr<memory::Allocation>> trans_grads;
  //
  //   // trans_keys[id] = memory::Alloc(place, feanum_ * sizeof(KeyType));
  //  // trans_grads[id] = memory::Alloc(place, feanum_ * grad_type_size_);
  //   trans_keys[id] = nullptr;
  //   trans_grads[id] = nullptr;
  // }

/*
  mg_time_1 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_2 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_3 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_4 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_5 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_6 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_7 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_8 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_9 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_10 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_11 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_12 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_13 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_14 = std::vector<double>(resource_->total_gpu(), 0.0);
  mg_time_15 = std::vector<double>(resource_->total_gpu(), 0.0);
*/



  init_path();
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::init_path() {
  int total_gpu = resource_->total_gpu();
  path_.resize(total_gpu);

  if (!topo_aware_) {
    VLOG(0) << "init path without topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        nodes.resize(1);
        nodes[0].in_stream = resource_->comm_stream(i, j);
        nodes[0].out_stream = resource_->comm_stream(i, j);
        nodes[0].key_storage = NULL;
        nodes[0].val_storage = NULL;
        nodes[0].sync = 0;
        nodes[0].gpu_num = j;
      }
    }
  } else {
    VLOG(0) << "init path with topo aware";
    for (int i = 0; i < total_gpu; ++i) {
      path_[i].resize(total_gpu);
      for (int j = 0; j < total_gpu; ++j) {
        auto& nodes = path_[i][j].nodes_;
        int from = resource_->dev_id(i);
        int to = resource_->dev_id(j);
        int transfer_id = i;
        if (need_transfer(from, to)) {
          transfer_id = resource_->get_index_by_devid(get_transfer_devid(from));
          nodes.push_back(Node());
          Node& node = nodes.back();
          node.in_stream = resource_->comm_stream(i, transfer_id);
          node.out_stream = resource_->comm_stream(transfer_id, i);
          node.key_storage = NULL;
          node.val_storage = NULL;
          node.sync = 1;
          node.gpu_num = transfer_id;
        }
        nodes.push_back(Node());
        Node& node = nodes.back();
        node.in_stream = resource_->comm_stream(i, transfer_id);
        node.out_stream = resource_->comm_stream(transfer_id, i);
        node.key_storage = NULL;
        node.val_storage = NULL;
        node.sync = 0;
        node.gpu_num = j;
      }
    }
  }
  VLOG(0) << "HeterComm init_path done";
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::create_storage(int start_index,
                                                                       int end_index,
                                                                       size_t keylen,
                                                                       size_t vallen) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].gpu_num),
        (void**)&(nodes[i].key_storage),  // NOLINT
        keylen, resource_->remote_stream(nodes[i].gpu_num, start_index)));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceAllocate(
        resource_->dev_id(nodes[i].gpu_num),
        (void**)&(nodes[i].val_storage),  // NOLINT
        vallen, resource_->remote_stream(nodes[i].gpu_num, start_index)));

    nodes[i].key_bytes_len = keylen;
    nodes[i].val_bytes_len = vallen;
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::destroy_storage(int start_index,
                                                                        int end_index) {
  auto& allocator = allocators_[start_index];
  auto& nodes = path_[start_index][end_index].nodes_;
  for (size_t i = 0; i < nodes.size(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(nodes[i].gpu_num));

    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                          nodes[i].key_storage));
    PADDLE_ENFORCE_GPU_SUCCESS(allocator->DeviceFree(resource_->dev_id(nodes[i].gpu_num),
                          nodes[i].val_storage));
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyType* src_key,
    GradType* src_val) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_dest(
    int start_index, int gpu_num, int* h_left, int* h_right, KeyType* src_key,
    char* src_val, size_t val_size) {
  int need_copy_val = 0;
  if (src_val) {
    need_copy_val = 1;
  }
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int size = path_[start_index][i].nodes_.size();
    auto& node = path_[start_index][i].nodes_[0];
    CopyTask t(&path_[start_index][i], 0);
    que.push(t);
    cudaMemcpyAsync(node.key_storage,
                    reinterpret_cast<char*>(src_key + h_left[i]),
                    node.key_bytes_len, cudaMemcpyDefault, node.in_stream);
    if (need_copy_val) {
      cudaMemcpyAsync(node.val_storage,
                      src_val + uint64_t(h_left[i]) * uint64_t(val_size),
                      node.val_bytes_len, cudaMemcpyDefault, node.in_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    if (cur_task.path->nodes_[cur_task.step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_task.step].in_stream);
    }
    if (cur_task.step != cur_task.path->nodes_.size() - 1) {
      int cur_step = cur_task.step;
      CopyTask c(cur_task.path, cur_step + 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].key_storage,
                      cur_task.path->nodes_[cur_step].key_storage,
                      cur_task.path->nodes_[cur_step + 1].key_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step + 1].in_stream);
      if (need_copy_val) {
        cudaMemcpyAsync(cur_task.path->nodes_[cur_step + 1].val_storage,
                        cur_task.path->nodes_[cur_step].val_storage,
                        cur_task.path->nodes_[cur_step + 1].val_bytes_len,
                        cudaMemcpyDefault,
                        cur_task.path->nodes_[cur_step + 1].in_stream);
      }
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, ValType* src_val) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[i]),
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[end_index]),
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::walk_to_src(
    int start_index, int gpu_num, int* h_left, int* h_right, char* src_val, size_t val_size) {
  std::queue<CopyTask> que;
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      cudaMemcpyAsync(src_val + uint64_t(h_left[i]) * val_size,
                      node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                      node.out_stream);
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[cur_step - 1].out_stream);
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      cudaMemcpyAsync(src_val + uint64_t(h_left[end_index]) * val_size,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
HeterComm<KeyType, ValType, GradType, GPUAccessor>::~HeterComm() {

//  if (!multi_mf_dim_) {
//    for (auto& table : tables_) {
//      delete table;
//      table = nullptr;
//    }
//  } else {

    //VLOG(0) << "yxf:: heter comm ~hetercomm";
    for (auto& table : ptr_tables_) {
      delete table;
      table = nullptr;
    }

  
    // 有必要吗，
    for (auto& trans_pair: trans_keys) {
      if (trans_pair.second != nullptr) {
        trans_pair.second.reset();
      }
    } 
    for (auto& trans_pair: trans_grads) {
      if (trans_pair.second != nullptr) {
        trans_pair.second.reset();
      }
    }
    // 
    // for (auto& local_storage: storage_) {
    //   local_storage.reset();
    // }
    // for (auto& table : tables_) {
    //  delete table;
    //  table = nullptr;
    //}

    /*
    for (size_t i = 1; i < mg_time_1.size(); i++) {
      mg_time_1[0] += mg_time_1[i];
      mg_time_2[0] += mg_time_2[i];
      mg_time_3[0] += mg_time_3[i];
      mg_time_4[0] += mg_time_4[i];
      mg_time_5[0] += mg_time_5[i];
      mg_time_6[0] += mg_time_6[i];
      mg_time_7[0] += mg_time_7[i];
      mg_time_8[0] += mg_time_8[i];
      mg_time_9[0] += mg_time_9[i];
      mg_time_10[0] += mg_time_10[i];
      mg_time_11[0] += mg_time_11[i];
      mg_time_12[0] += mg_time_12[i];
      mg_time_13[0] += mg_time_13[i];
      mg_time_14[0] += mg_time_14[i];
      mg_time_15[0] += mg_time_15[i];
    }
    VLOG(0) << "yxf::mg_1::merge: " << mg_time_1[0];
    VLOG(0) << "yxf::mg_2:pull: " << mg_time_2[0];
    VLOG(0) << "yxf::mg_3:push: " << mg_time_3[0];
    VLOG(0) << "yxf::mg_4:gather one node before nccl: " << mg_time_14[0];
    VLOG(0) << "yxf::mg_4:gather one node nccl len: " << mg_time_4[0];
    VLOG(0) << "yxf::mg_5:gather one node nccl grads: " << mg_time_5[0];
    VLOG(0) << "yxf::mg_6:gather one node fill grads: " << mg_time_6[0];
    VLOG(0) << "yxf::mg_7:gather multi node before nccl: " << mg_time_15[0];
    VLOG(0) << "yxf::mg_7:gather multi node nccl len: " << mg_time_7[0];
    VLOG(0) << "yxf::mg_8:gather multi node nccl grads: " << mg_time_8[0];
    VLOG(0) << "yxf::mg_9:gather multi node fill grads: " << mg_time_9[0];
    VLOG(0) << "yxf::mg_10:push gather one node: " << mg_time_10[0];
    VLOG(0) << "yxf::mg_11:push gather multi node: " << mg_time_11[0];
    VLOG(0) << "yxf::mg_12:push update one table: " << mg_time_12[0];
    VLOG(0) << "yxf::mg_13:push merge_grad: " << mg_time_13[0];
    */
   // }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::show_one_table(int gpu_num) {
  if (!multi_mf_dim_) {
    tables_[gpu_num]->show();
  } else {
    // ptr_tables_[gpu_num]->show();
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::log2i(int x) {
  unsigned res = 0;
  while (x >>= 1) {
    ++res;
  }
  return res;
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::get_index_by_devid(int devid) {
  return resource_->get_index_by_devid(devid);
}

// ===== hbm optimized ======
template <typename KeyType, typename T>
__global__ void kernel_fill_restore_idx_filter_zero(const size_t N,  // merged size
                                                    const KeyType* d_keys,  // 去重以后的key
                                                    const T* d_sorted_idx,  // 每个key原来的idx
                                                    const T* d_offset,      // 前缀和 
                                                    const T* d_merged_cnts, // 每个key的数量
                                                    T* d_restore_idx) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (d_keys[i] == 0) { // 不管feasign 0
      return;
    }
    const T& off = d_offset[i]; // 前面有几个feasign，offset
    const T& num = d_merged_cnts[i]; // 当前feasign数
    for (size_t k = 0; k < num; ++k) { // 
      d_restore_idx[d_sorted_idx[off + k]] = i;  // d_sorted_idx[off + k]表示原idx, 那么d_restore_idx保存的是
                                                 // 每个key排序去重以后是第几个key..
    }
  }
}

template <typename T>
__global__ void kernel_fill_restore_idx(const size_t N,
                                        const T* d_sorted_idx,
                                        const T* d_offset,
                                        const T* d_merged_cnts,
                                        T* d_restore_idx) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const T& off = d_offset[i];
    const T& num = d_merged_cnts[i];
    for (size_t k = 0; k < num; ++k) {
      d_restore_idx[d_sorted_idx[off + k]] = i;
    }
  }
}

template <typename T>
__global__ void kernel_fill_restore_idx_by_search(const size_t N,
                                                  const T* d_sorted_idx,
                                                  const size_t merge_num,
                                                  const T* d_offset,
                                                  T* d_restore_idx) { // 输出
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) { // 对排序以后的feasign数组进行遍历,
    if (i < d_offset[1]) { // d_offset[1]是第一个feasign的数量, 也就是对于第一个feasign特殊处理
      d_restore_idx[d_sorted_idx[i]] = 0; // 根据d_sorted_idx[i]拿到原来的idx..., 那么对应的d_restore_idx[idx] = 0
      return;
    }
    int high = merge_num - 1;
    int low = 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < d_offset[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    d_restore_idx[d_sorted_idx[i]] = low;
  }
}

/*
  fill_restore_idx(filter_zero,
                   total_fea_num,
                   merged_size,   // 去重以后的feasign数
                   d_merged_keys, // 去重以后的key
                   d_sorted_idx,  // 排序以后，保存每个key原来的idx 
                   d_offset,      // 保存每个feasign的数量的数组的前缀和数组
                   d_merged_cnts, // 每个feasign的数量
                   d_restore_idx, // 输出
                   stream);
*/

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::fill_restore_idx(bool filter_zero,
                                                                          const size_t total_num,
                                                                          const size_t merge_size,
                                                                          const KeyType* d_keys,
                                                                          const uint32_t* d_sorted_idx,
                                                                          const uint32_t* d_offset,
                                                                          const uint32_t* d_merged_cnts,
                                                                          uint32_t* d_restore_idx,
                                                                          const cudaStream_t& stream) {
  // fill restore idx [1,3,5,2,4,6] = [1,2,1,3,2,1]
  if (merge_size * 3 > total_num) { // 重复率 < 3
    // repetition rate is not very high
    size_t grid_size = (merge_size - 1) / block_size_ + 1;
    if (filter_zero) {
      kernel_fill_restore_idx_filter_zero<<<grid_size,
                                            block_size_,
                                            0,
                                            stream>>>(merge_size, // 去重以后的数量
                                                      d_keys,     // 去重以后的key 
                                                      d_sorted_idx, // 每个key原来的idx 
                                                      d_offset,     // 前缀和
                                                      d_merged_cnts, // 每个feasign的数量
                                                      d_restore_idx); // 输出
    } else {
      kernel_fill_restore_idx<<<grid_size, block_size_, 0, stream>>>(
          merge_size, d_sorted_idx, d_offset, d_merged_cnts, d_restore_idx);
    }
  } else {
    size_t grid_size = (total_num - 1) / block_size_ + 1;
    // mid search
    kernel_fill_restore_idx_by_search<<<grid_size, block_size_, 0, stream>>>(
        total_num, d_sorted_idx, merge_size, d_offset, d_restore_idx);  // total_num 所有的feasign数量
                                                                        // d_sorted_idx 排序以后每个key原来的idx
                                                                        // merge_size merge以后key的数量
                                                                        // d_offset 数组
                                                                        // d_restore_idx kernel输出
  }
}
// ====== hbm optimized =============

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_sparse_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_sparse_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_sparse_sgd(optimizer_config);
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::set_embedx_sgd(
    const OptimizerConfig& optimizer_config) {
  for (int i = 0; i < resource_->total_gpu(); ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (!multi_mf_dim_) {
      tables_[i]->set_embedx_sgd(optimizer_config);
    } else {
      ptr_tables_[i]->set_embedx_sgd(optimizer_config);
    }
  }
}

/*
template <typename KeyType, typename ValType, typename GradType, typename FVAccessor>
void HeterComm<KeyType, ValType, GradType, FVAccessor>::build_ps(int num, KeyType* h_keys,
                                                                 ValType* h_vals,
                                                                 size_t len,
                                                                 size_t chunk_size,
                                                                 int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  std::vector<memory::allocation::AllocationPtr> d_key_bufs;
  std::vector<memory::allocation::AllocationPtr> d_val_bufs;

  gpuStream_t streams[stream_num];  // NOLINT
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    auto d_v_buf = memory::Alloc(place, chunk_size * sizeof(ValType));
    d_key_bufs.push_back(std::move(d_k_buf));
    d_val_bufs.push_back(std::move(d_v_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_val_bufs[cur_stream]->ptr(), h_vals + cur_len,
                        sizeof(ValType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<ValType*>(d_val_bufs[cur_stream]->ptr()), tmp_len,
        streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}
*/

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::build_ps(int num,
                                                                 KeyType* h_keys,
                                                                 char* pool,
                                                                 size_t len,
                                                                 size_t feature_value_size,
                                                                 size_t chunk_size,
                                                                 int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  // use hbm pool
  std::vector<memory::allocation::AllocationPtr> d_key_bufs;

  gpuStream_t streams[stream_num];
  for (int i = 0; i < stream_num; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreate(&(streams[i])));
    auto d_k_buf = memory::Alloc(place, chunk_size * sizeof(KeyType));
    d_key_bufs.push_back(std::move(d_k_buf));
  }

  int cur_len = 0;
  int cur_stream = 0;

  while (cur_len < len) {
    cur_stream = cur_stream % stream_num;
    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(d_key_bufs[cur_stream]->ptr(), h_keys + cur_len,
                        sizeof(KeyType) * tmp_len, cudaMemcpyHostToDevice,
                        streams[cur_stream]));
    ptr_tables_[num]->insert(
        reinterpret_cast<KeyType*>(d_key_bufs[cur_stream]->ptr()), tmp_len,
        pool, feature_value_size, cur_len, streams[cur_stream]);
    cur_stream += 1;
    cur_len += tmp_len;
  }

  for (int i = 0; i < stream_num; ++i) {
    cudaStreamSynchronize(streams[i]);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(streams[i]));
  }
}
/*
template <typename KeyType, typename ValType, typename GradType, typename FVAccessor>
void HeterComm<KeyType, ValType, GradType, FVAccessor>::merge_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    int& uniq_len) {  // NOLINT
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * sizeof(GradType));
  GradType* d_merge_grads_ptr =
      reinterpret_cast<GradType*>(d_merge_grads->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_grads,
      d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_grads, d_merge_grads_ptr, len, 0, 8 * sizeof(KeyType), stream, false));
  temp_storage_bytes = 0;

  auto d_num_runs_out_mem = memory::Alloc(place, sizeof(int));
  int* d_num_runs_out = reinterpret_cast<int*>(d_num_runs_out_mem->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_merge_grads_ptr,
      d_grads, d_num_runs_out, merger_, len, stream, false));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceReduce::ReduceByKey(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_merge_grads_ptr, d_grads, d_num_runs_out, merger_, len, stream, false));

  cudaMemcpyAsync(&uniq_len, d_num_runs_out, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}
*/

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_grad(int gpu_num,
                                                                   KeyType* d_keys,
                                                                   GradType* d_grads,
                                                                   float* mf, size_t len,
                                                                   int& uniq_len) {
  platform::Timer timeline;
  timeline.Start();

  int dev_id = resource_->dev_id(gpu_num);

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  auto d_merge_grads = memory::Alloc(place, len * grad_value_size);
  float* d_merge_grads_ptr =
      reinterpret_cast<float*>(d_merge_grads->ptr());

  auto d_fea_num_info =
      memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));

  uint32_t* d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());

  uint32_t* d_index = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_idx = (uint32_t*)&d_index[len];

  int* d_merged_size = (int*)&d_idx[len];

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx, len);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_idx, d_index, len,
      0, 8 * sizeof(KeyType), stream));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_idx, d_index, len, 0, 8 * sizeof(KeyType), stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  // timeline.Pause();
  // mg_time_4[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  /*
  char* test_grad_values_1 =
      (char*)malloc(sizeof(KeyType) * len);
  char* test_grad_keys_1 =
      (char*)malloc(sizeof(KeyType) * len);
  cudaMemcpy(test_grad_values_1, d_merge_keys_ptr,
            sizeof(KeyType) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_grad_keys_1, d_index,
            sizeof(uint32_t) * len, cudaMemcpyDeviceToHost);
  for (int i = 0; i < len; i++) {
    VLOG(0) << "yxfpush22222:: i: " << i << " key: " << ((uint64_t*)test_grad_values_1)[i] << " cur index: " << ((uint32_t*)test_grad_keys_1)[i];
  }
  */
  timeline.Pause();

  timeline.Start();

  temp_storage_bytes = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_fea_num_info_ptr,
      d_merged_size, len, stream));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

  // d_keys保存所有去重后的key,d_fea_num_info_ptr保存每个key的次数
  // d_merged_size保存key的数量
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_fea_num_info_ptr, d_merged_size, len, stream));

  cudaMemcpyAsync((void*)&uniq_len, d_merged_size, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // if (test_batch == 0) {
  //   char* test_count =
  //     (char*)malloc(sizeof(uint32_t) * uniq_len);
  //   cudaMemcpy(test_count, d_fea_num_info_ptr,
  //           sizeof(uint32_t) * uniq_len, cudaMemcpyDeviceToHost);
  //   char* test_key =
  //     (char*)malloc(sizeof(uint64_t) * uniq_len);
  //   cudaMemcpy(test_key, d_keys,
  //           sizeof(uint64_t) * uniq_len, cudaMemcpyDeviceToHost);
  //   for (int i = 0; i < 50; i++) {
  //     VLOG(0) << "yxfpush22222:: i: " << i << " count out: " << ((uint32_t*)test_count)[i] << " key: " << ((uint64_t*)test_key)[i];
  //   }
  //   free(test_count);
  //   free(test_key);
  //   test_batch += 1;
  // }
  // timeline.Pause();
  // mg_time_5[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  assert(d_merged_size > 0);
  uint32_t* d_offset = (uint32_t*)&d_index[len];

  // 传入的是每个key的次数,d_offset保存前缀和
  temp_storage_bytes = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_fea_num_info_ptr, d_offset, uniq_len,
      stream));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_fea_num_info_ptr, d_offset,
      uniq_len, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // timeline.Pause();
  // mg_time_6[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  // tmp try to delete 0 when merge grad
  timeline.Pause();
  timeline.Start();
  grid_size = (uniq_len - 1) / block_size_ + 1;

  merge_gradient_basic_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_keys, d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads,
      (char*)d_merge_grads_ptr, uniq_len, grad_value_size, merger_, gpu_accessor_);

  const size_t grad_dim = max_mf_dim_;

  if (grad_dim > 0) {
    int grid_size2 = (uniq_len * grad_dim - 1) / block_size_ + 1;
    merge_gradient_embedx_kernel<<<grid_size2, block_size_, 0, stream>>>(
            d_keys, d_offset, d_fea_num_info_ptr, d_index, (char*)d_grads, (char*)d_merge_grads_ptr, uniq_len * grad_dim, grad_dim, grad_value_size, merger_, gpu_accessor_);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  //VLOG(0) << "yxf333";
  // timeline.Pause();
  // mg_time_7[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();
  timeline.Pause();

  timeline.Start();
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(d_grads, d_merge_grads_ptr, grad_value_size * uniq_len,
                      cudaMemcpyDeviceToDevice, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
 
  

  timeline.Pause();
}

/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_key(int gpu_num,
                                                                   KeyType* d_keys,
                                                                   uint32_t* d_fea_num_info_ptr,
                                                                   size_t len) {
  platform::Timer timeline;
  timeline.Start();

  int dev_id = resource_->dev_id(gpu_num);

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  size_t temp_storage_bytes;
  // auto accessor_wrapper_ptr =
  //     GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  // size_t pull_value_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
  size_t pull_value_size = pull_type_size_;

  auto d_merge_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_merge_keys_ptr = reinterpret_cast<KeyType*>(d_merge_keys->ptr());

  // d_merge_vals_ptr = nullptr;
  // if (d_vals) {
  //   auto d_merge_vals = memory::Alloc(place, len * pull_value_size);
  //  float* d_merge_vals_ptr =
  //        reinterpret_cast<float*>(d_merge_vals->ptr());
  // }
  // auto d_fea_num_info =
  //    memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));

  // d_fea_num_info_ptr =
  //    reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());

  uint32_t* d_index = (uint32_t*)&d_fea_num_info_ptr[len];
  uint32_t* d_idx = (uint32_t*)&d_index[len];

  int* d_merged_size = (int*)&d_idx[len];

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx, len);

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_keys, d_merge_keys_ptr, d_idx, d_index, len,
      0, 8 * sizeof(KeyType), stream));

  void* d_buff = NULL;
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_keys, d_merge_keys_ptr,
      d_idx, d_index, len, 0, 8 * sizeof(KeyType), stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  // timeline.Pause();
  // mg_time_4[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  timeline.Pause();

  timeline.Start();

  temp_storage_bytes = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, d_merge_keys_ptr, d_keys, d_fea_num_info_ptr,
      d_merged_size, len, stream));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }

  // d_keys保存所有去重后的key,d_fea_num_info_ptr保存每个key的次数
  // d_merged_size保存key的数量
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage->ptr(), temp_storage_bytes, d_merge_keys_ptr, d_keys,
      d_fea_num_info_ptr, d_merged_size, len, stream));

  int uniq_len;
  cudaMemcpyAsync((void*)&uniq_len, d_merged_size, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // if (test_batch == 0) {
  //   char* test_count =
  //     (char*)malloc(sizeof(uint32_t) * uniq_len);
  //   cudaMemcpy(test_count, d_fea_num_info_ptr,
  //           sizeof(uint32_t) * uniq_len, cudaMemcpyDeviceToHost);
  //   char* test_key =
  //     (char*)malloc(sizeof(uint64_t) * uniq_len);
  //   cudaMemcpy(test_key, d_keys,
  //           sizeof(uint64_t) * uniq_len, cudaMemcpyDeviceToHost);
  //   for (int i = 0; i < 50; i++) {
  //     VLOG(0) << "yxfpush22222:: i: " << i << " count out: " << ((uint32_t*)test_count)[i] << " key: " << ((uint64_t*)test_key)[i];
  //   }
  //   free(test_count);
  //   free(test_key);
  //   test_batch += 1;
  // }
  // timeline.Pause();
  // mg_time_5[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  assert(d_merged_size > 0);
  uint32_t* d_offset = (uint32_t*)&d_index[len];

  // 传入的是每个key的次数,d_offset保存前缀和
  temp_storage_bytes = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_fea_num_info_ptr, d_offset, uniq_len,
      stream));

  if (d_temp_storage->size() < temp_storage_bytes) {
    d_temp_storage = NULL;
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_fea_num_info_ptr, d_offset,
      uniq_len, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // timeline.Pause();
  // mg_time_6[gpu_num] += timeline.ElapsedSec();
  // timeline.Start();

  // tmp try to delete 0 when merge grad
  timeline.Pause();
  return uniq_len;
}
*/



/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::reorder_input_by_mfdim(KeyType* d_keys, GradType* d_grads, size_t len, int* lens, int gpu_num, size_t& reorder_grad_len) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * multi_mf_dim_);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * multi_mf_dim_);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  auto d_left = memory::Alloc(place, multi_mf_dim_ * sizeof(int));
  auto d_right = memory::Alloc(place, multi_mf_dim_ * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  split_input_by_mfdim(d_grads, d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_num);
  cudaMemcpyAsync(h_left, d_left_ptr, multi_mf_dim_ * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, multi_mf_dim_ * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // VLOG(0) << "yxffff00";
  // size_t dim8_grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (8 * sizeof(float)));
  //size_t dim64_grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (64 * sizeof(float)));

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t dim8_grad_value_size =
      accessor_wrapper_ptr->GetPushValueSize(8);
  size_t dim64_grad_value_size =
      accessor_wrapper_ptr->GetPushValueSize(64);

  size_t dim8_len = h_right[0] - h_left[0] + 1;
  size_t dim64_len = h_right[1] - h_left[1] + 1;
  auto d_reorder_grads_mem = memory::Alloc(place, dim8_len * dim8_grad_value_size + dim64_len * dim64_grad_value_size);
  GradType* d_reorder_grads = reinterpret_cast<GradType*>(d_reorder_grads_mem->ptr());
  int grid_size = (dim8_len - 1) / block_size_ + 1;
  dy_mf_fill_mfdim_grads<<<grid_size, block_size_, 0, stream>>>(d_reorder_grads, d_grads, d_idx_ptr, dim8_len, dim8_grad_value_size, size_t(0), dim64_grad_value_size);
  GradType* d_reorder_grads_dim64 = (GradType*)((char*)(d_reorder_grads) + dim8_len * dim8_grad_value_size);
  grid_size = (dim64_len - 1) / block_size_ + 1;
  dy_mf_fill_mfdim_grads<<<grid_size, block_size_, 0, stream>>>(d_reorder_grads_dim64, d_grads, d_idx_ptr, dim64_len, dim64_grad_value_size, size_t(h_left[1]), dim64_grad_value_size);
  cudaStreamSynchronize(stream);
  // VLOG(0) << "yxffff11";
  cudaMemcpyAsync((char*)d_grads, (char*)d_reorder_grads, dim8_len * dim8_grad_value_size + dim64_len * dim64_grad_value_size,
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  // VLOG(0) << "yxffff22";
  lens[0] = dim8_len;
  lens[1] = dim64_len;
  reorder_grad_len = dim8_len * dim8_grad_value_size + dim64_len * dim64_grad_value_size;
  // VLOG(0) << "yxffff33";
  return;
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_input_by_mfdim(
    GradType* d_grads, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);
  calc_mfdim_index<<<grid_size, block_size_, 0, stream>>>(
      d_grads, len, d_shard_index_tmp_ptr, total_gpu, grad_type_size_);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}
*/

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_input_to_shard(
    KeyType* d_keys, int* d_idx_ptr, size_t len, int* left, int* right,
    int gpu_num) {

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  auto stream = resource_->local_stream(gpu_num, 0);

  auto d_idx_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_idx_tmp_ptr = reinterpret_cast<int*>(d_idx_tmp->ptr());

  auto d_shard_index = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_ptr = reinterpret_cast<int*>(d_shard_index->ptr());

  auto d_shard_index_tmp = memory::Alloc(place, len * sizeof(int));
  int* d_shard_index_tmp_ptr = reinterpret_cast<int*>(d_shard_index_tmp->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);

  // d_shard_index_tmp_ptr保存每个key的shard id
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_gpu);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_gpu);

  // 对d_shard_index_tmp_ptr进行排序，将结果保存到d_shard_index_ptr
  // d_idx_ptr保存每个key原来的顺序
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes, d_shard_index_tmp_ptr, d_shard_index_ptr,
      d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceRadixSort::SortPairs(
      d_temp_storage->ptr(), temp_storage_bytes, d_shard_index_tmp_ptr,
      d_shard_index_ptr, d_idx_tmp_ptr, d_idx_ptr, len, 0, num_bits, stream));

  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(d_shard_index_ptr,
                                                           left, right, len);
  cudaStreamSynchronize(stream);
}

// ==== add for all2all ===
// 每张卡会处理一部分feasign, 那有些feasign可能是当前机器的，有些feasign不是当前机器的
// 但是因为feasign在每个机器上的分卡逻辑是一样的

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename T>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::split_idx_to_shard(
    KeyType *d_keys,
    T *d_idx_ptr,
    size_t len,
    T *left,
    T *right,
    int gpu_num,
    const cudaStream_t& stream) {
  int total_device = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  auto d_idx_tmp =
      memory::Alloc(place,
                    3 * len * sizeof(T),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));


  T *d_idx_tmp_ptr = reinterpret_cast<T *>(d_idx_tmp->ptr());
  T *d_shard_index_ptr = reinterpret_cast<T *>(&d_idx_tmp_ptr[len]);
  T *d_shard_index_tmp_ptr = reinterpret_cast<T *>(&d_shard_index_ptr[len]);

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);

  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, total_device);

  // heter_comm_kernel_->fill_idx(d_idx_tmp_ptr, len, stream);
  // heter_comm_kernel_->calc_shard_index(
  //    d_keys, len, d_shard_index_tmp_ptr, total_device, stream);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(total_device);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_shard_index_tmp_ptr,
                                      d_shard_index_ptr,

                                      d_idx_tmp_ptr,
                                      d_idx_ptr,
                                      len,
                                      0,
                                      num_bits,
                                      stream));

  auto d_temp_storage =
      memory::Alloc(place,
                    temp_storage_bytes);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(),
                                      temp_storage_bytes,
                                      d_shard_index_tmp_ptr,
                                      d_shard_index_ptr,
                                      d_idx_tmp_ptr,
                                      d_idx_ptr,
                                      len,
                                      0,
                                      num_bits,
                                      stream));

  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(
      d_shard_index_ptr, left, right, len);
  // heter_comm_kernel_->calc_shard_offset(
  //     d_shard_index_ptr, left, right, len, total_device, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::shard_inner_keys(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    const int &gpu_id,
    const int &gpu_num,
    HeterCommType::InnerResource *res,
    const cudaStream_t &stream) {
  std::vector<uint32_t> h_offsets(gpu_num * 2);  // NOLINT
  uint32_t *d_left_ptr = res->d_offset_ptr;
  cudaMemsetAsync(d_left_ptr, -1, gpu_num * 2 * sizeof(int), stream);

  uint32_t *d_right_ptr = &d_left_ptr[gpu_num];
  split_idx_to_shard<uint32_t>((KeyType *)d_keys,
                                             res->d_idx,
                                             total_fea_num,
                                             d_left_ptr,
                                             d_right_ptr,
                                             gpu_id,
                                             stream);

  cudaMemcpyAsync(&h_offsets[0],
                  d_left_ptr,
                  gpu_num * 2 * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  for (int i = 0; i < gpu_num; ++i) {
    uint32_t &h_right = h_offsets[gpu_num + i];
    uint32_t &h_left = h_offsets[i];
    if (static_cast<int>(h_right) == -1 || static_cast<int>(h_left) == -1) {
      res->h_part_sizes[i] = 0;
    } else {
      res->h_part_sizes[i] = h_right - h_left + 1;
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inner_keys_p2p(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    HeterCommType::InnerResource &res,
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const cudaStream_t &stream) {

  // gather all datas
  // 把d_keys中的数据按照shard id排列放到res.d_keys_parted
  // 
  gather_keys(
      res.d_keys_parted, d_keys, res.d_idx, total_fea_num, stream);
  
  // 把数据拷贝到对应的卡上,也就是每个storage的d_merged_keys
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    return;
  }

  /*
  // need transfer
  for (int i = 0; i < gpu_num; ++i) {
    size_t data_len = res.h_part_sizes[i];
    if (data_len == 0) {
      continue;
    }
    size_t &offset = res.h_offsets[i];
    // printf("[%d->%d->%d]send keys offset: %ld, len: %ld\n", gpu_id,
    // trans_id, i, offset, data_len);
    if (!need_transfer(gpu_id, i)) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_keys,
                                                   trans_id,
                                                   &res.d_keys_parted[offset],
                                                   gpu_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                   i,
                                                   res.d_trans_keys,
                                                   trans_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  */

}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_keys(
    const int gpu_num,
    const KeyType *d_keys,
    const size_t &len,       // input
    KeyType *d_sorted_keys,  // output
    KeyType *d_merged_keys,  // output
    uint32_t *d_restore_idx,
    const cudaStream_t &stream) {
#if defined(PADDLE_WITH_CUDA)
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);

  auto d_fea_num_info =
      memory::Alloc(place,
                    sizeof(uint32_t) * (len * 3));

  uint32_t *d_offset = reinterpret_cast<uint32_t *>(d_fea_num_info->ptr());
  uint32_t *d_merged_cnts = (uint32_t *)&d_offset[len];
  uint32_t *d_sorted_idx = (uint32_t *)&d_merged_cnts[len];

  


  return dedup_keys_and_fillidx(gpu_num,
                                len,
                                d_keys,         // input
                                d_merged_keys,  // output
                                d_sorted_keys,
                                d_restore_idx,
                                d_sorted_idx,
                                d_offset,
                                d_merged_cnts,
                                true,
                                stream);
#else
  return 0;
#endif
}


template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inter_keys_by_copy(
    const int &gpu_id,
    const size_t &fea_size,
    const KeyType *d_keys,
    const cudaStream_t &stream) {


  auto &my_cache = storage_[gpu_id];
  auto &res = my_cache.inner_res;
  my_cache.init_inner(fea_size, device_num_);
  res.h_part_sizes = &my_cache.h_fea_sizes[0];
  // shard inner keys
  shard_inner_keys(fea_size, d_keys, gpu_id, device_num_, &res, stream);
  // barrier wait all gpu done
  h_barrier.wait();

  size_t max_part_size = 0;
  size_t shard_recv_offset = 0;
  size_t shard_send_offset = 0;

  // res.h_part_sizes保存分配到每张卡上的key size
  // 
  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    my_cache.h_recv_offsets[i] = shard_recv_offset;
    shard_recv_offset += cache.h_fea_sizes[gpu_id];
    res.h_offsets[i] = shard_send_offset;
    shard_send_offset += res.h_part_sizes[i];
    if (max_part_size < res.h_part_sizes[i]) {
      max_part_size = res.h_part_sizes[i];
    }
  }
  // shard_recv_offset保存的是每张卡分配到gpuid卡上的key数量之和
  CHECK(shard_send_offset == static_cast<size_t>(fea_size));

  size_t trans_need_size =
      std::max(shard_recv_offset, static_cast<size_t>(fea_size));
  int trans_id = -1;

  // if (topo_aware_ && device_num_ > 4) {
  //   trans_id = get_transfer_devid(gpu_id);
  //   storage_[trans_id].h_trans_size = max_part_size;
  //   // barrier wait all set trans length [0-4, 1-5, 3-7, 2-6]
  //   h_barrier.wait();
  //   my_cache.h_trans_offset = trans_need_size;
  //   trans_need_size += my_cache.h_trans_size;
  // }
  my_cache.alloc(trans_need_size, max_type_size_);

  // barrier wait all hbm malloc size
  h_barrier.wait();

  // 每个gpuid 卡上的h_recv_offsets保存的是每张卡上分配到gpuid卡的key offsets
  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id]; // gpu_id卡分配到第i张卡的key的起始地址
    res.d_remote_keys[i] = &cache.d_merged_keys[recv_offset];
    // if (trans_id >= 0) {
    //   // set transfer buffer
    //   auto &trans_cache = storage_[trans_id];
    //   res.d_trans_keys = &trans_cache.d_merged_keys[trans_cache.h_trans_offset];
    // }
  }
  // 那我们的目标就是填充res.d_remote_keys
 
  res.d_keys_parted = my_cache.d_merged_push_keys;
  // barrier wait set buffer ptr
  h_barrier.wait();



  
  gather_inner_keys_p2p(
      fea_size, d_keys, res, gpu_id, device_num_, trans_id, stream);

  // barrier wait all gpu aync memcpy data
  h_barrier.wait();


  // my_cache的d_merged_keys已经接收了其他卡拷贝过来的key 
  // 
  my_cache.init_pull(shard_recv_offset);

  size_t uniq_len = merge_keys(gpu_id,
                               my_cache.d_merged_keys,  // in keys
                               shard_recv_offset,
                               my_cache.d_merged_push_keys,  // sort keys
                               my_cache.d_merged_keys,       // out merge keys
                               my_cache.pull_res.d_restore_keys_idx,
                               stream);

  return uniq_len;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::partition_shard_keys(
    const int &gpu_id,
    const size_t &len,
    const KeyType *d_keys,
    uint32_t *d_idx_parted,
    KeyType *d_keys_parted,
    size_t *h_part_sizes,
    const int &shard_num,
    const cudaStream_t &stream) {


  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  std::vector<uint32_t> h_offsets(shard_num * 2);
  auto d_offset_tmp =
      memory::Alloc(place,
                    (len * 3 + shard_num * 2) * sizeof(int));

  uint32_t *d_left = reinterpret_cast<uint32_t *>(d_offset_tmp->ptr());
  uint32_t *d_right = &d_left[shard_num];
  // init
  cudaMemsetAsync(d_left, -1, shard_num * 2 * sizeof(int), stream);

  uint32_t *d_idx_tmp_ptr = reinterpret_cast<uint32_t *>(&d_right[shard_num]);
  uint32_t *d_shard_index_ptr = &d_idx_tmp_ptr[len];
  uint32_t *d_shard_index_tmp_ptr = &d_shard_index_ptr[len];

  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_idx_tmp_ptr, len);

  calc_node_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, d_shard_index_tmp_ptr, thread_keys_shard_num_, node_size_);

  size_t temp_storage_bytes;
  const int num_bits = 1 + log2i(shard_num);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_shard_index_tmp_ptr,
                                      d_shard_index_ptr,
                                      d_idx_tmp_ptr,
                                      d_idx_parted,
                                      len, 
                                      0,
                                      num_bits,
                                      stream));

  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);
  

  
  // auto d_temp_storage =
  //    memory::Alloc(place,
  //                  temp_storage_bytes,
  //                  phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_temp_storage->ptr(),
                                      temp_storage_bytes,
                                      d_shard_index_tmp_ptr,
                                      d_shard_index_ptr,
                                      d_idx_tmp_ptr,
                                      d_idx_parted,
                                      len,
                                      0,
                                      num_bits,
                                      stream));

  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(
      d_shard_index_ptr, d_left, d_right, len);

  // d_keys的数据按照node id排序后保存到d_keys_parted 
  gather_keys(
      d_keys_parted, d_keys, d_idx_parted, len, stream);

  cudaMemcpyAsync(&h_offsets[0],
                  d_left,
                  shard_num * 2 * sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  for (int i = 0; i < shard_num; ++i) {
    uint32_t &h_right = h_offsets[shard_num + i];
    uint32_t &h_left = h_offsets[i];
    if (static_cast<int>(h_right) == -1 || static_cast<int>(h_left) == -1) {
      h_part_sizes[i] = 0;
    } else {
      h_part_sizes[i] = h_right - h_left + 1;
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::send_data_by_all2all(
    const int &gpu_id,
    const int &nccl_node_size,
    const int &nccl_rank_id,
    const int &value_bytes,
    const size_t *h_send_part_sizes,
    const size_t *h_send_part_offsets,
    const size_t *h_recv_part_sizes,
    const size_t *h_recv_part_offsets,
    const char *d_send_buff,
    char *d_rev_buff,
    const cudaStream_t &stream) {
  auto &comm = nccl_inter_comms_[gpu_id];
  const size_t &send_size = h_send_part_sizes[nccl_rank_id];
  size_t send_offset = h_send_part_offsets[nccl_rank_id] * value_bytes;
  size_t recv_offset = h_recv_part_offsets[nccl_rank_id] * value_bytes;
  // 自己发给自己的直接拷贝
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
      reinterpret_cast<void *>(&d_rev_buff[recv_offset]),  // output
      (const void *)&d_send_buff[send_offset],
      send_size * value_bytes,
      cudaMemcpyDeviceToDevice,
      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  CHECK(send_size == h_recv_part_sizes[nccl_rank_id])
      << "gpu id=" << gpu_id << ", rank_id=" << nccl_rank_id
      << ", node_size=" << nccl_node_size << ", send_size=" << send_size
      << ", recv_size=" << h_recv_part_sizes[nccl_rank_id];

  size_t total_fea_num = 0;
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  for (int i = 0; i < nccl_node_size; i++) {
    if (i == nccl_rank_id) {
      continue;
    }
    const size_t &send_size = h_send_part_sizes[i];
    if (send_size > 0) {
      send_offset = h_send_part_offsets[i] * value_bytes;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclSend((const void *)&d_send_buff[send_offset],
                                      send_size * value_bytes,
                                      ncclInt8,
                                      i,
                                      comm,
                                      nccl_stream));
      total_fea_num += send_size;
    }
    const size_t &recv_size = h_recv_part_sizes[i];
    if (recv_size > 0) {
      recv_offset = h_recv_part_offsets[i] * value_bytes;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          reinterpret_cast<void *>(&d_rev_buff[recv_offset]),
          recv_size * value_bytes,
          ncclInt8,
          i,
          comm,
          nccl_stream));
      total_fea_num += recv_size;
    }
  }
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(nccl_stream));

  return total_fea_num;
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_sparse_keys_by_all2all(
       const int &gpu_id, const size_t &fea_size, const KeyType *d_in_keys, KeyType *d_out_keys,
           KeyType *d_tmp_keys, const cudaStream_t &stream) {

  auto &cache = storage_[gpu_id];
  cache.init_shard(fea_size, node_size_);
  auto &res = cache.shard_res;

  size_t *h_local_part_sizes = res.h_local_part_sizes.data();
  size_t *h_local_part_offsets = res.h_local_part_offsets.data();
  uint32_t *h_push_fea_sizes = res.h_push_fea_sizes.data();

  // partition keys
  partition_shard_keys(gpu_id,
                       fea_size,
                       d_in_keys,
                       res.d_local_idx_parted,
                       d_tmp_keys,
                       h_local_part_sizes,
                       node_size_,
                       stream);


  int all_shard_part_size = node_size_ * node_size_;
  int rank_offset = rank_id_ * node_size_;
  h_local_part_offsets[0] = 0;

  for (int i = 0; i < node_size_; ++i) {
    h_push_fea_sizes[rank_offset + i] = h_local_part_sizes[i];
    h_local_part_offsets[i + 1] =
        h_local_part_offsets[i] + h_local_part_sizes[i];
  }
  CHECK(fea_size == h_local_part_offsets[node_size_])
      << "gpu id=" << gpu_id << ", fea_size=" << fea_size
      << ", offset size=" << h_local_part_offsets[node_size_];
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&res.d_node_size_ptr[rank_offset],
                                             &h_push_fea_sizes[rank_offset],
                                             node_size_ * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


  auto &comm = nccl_inter_comms_[gpu_id];
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (const void *)&res.d_node_size_ptr[rank_offset],
      reinterpret_cast<void *>(res.d_node_size_ptr),
      node_size_,
      ncclInt,
      comm,
      nccl_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(nccl_stream));


  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&h_push_fea_sizes[0],
                                             res.d_node_size_ptr,
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


  size_t *h_remote_part_sizes = res.h_remote_part_sizes.data();
  size_t *h_remote_part_offsets = res.h_remote_part_offsets.data();
  h_remote_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = node_size_ * i + rank_id_;
    h_remote_part_sizes[i] = h_push_fea_sizes[offset];
    h_remote_part_offsets[i + 1] =
        h_remote_part_offsets[i] + h_remote_part_sizes[i];
  }
  // h_remote_part_sizes 保存gpuid上，每个机器发送到当前机器的key数量,按照机器id排列
  size_t &remote_size = h_remote_part_offsets[node_size_];
  // 为什么要用max_type_size_而不是grad_type_size_
  // 这个是分配d_merged_keys, d_merged_push_keys,d_merged_vals,d_merged_push_vals
  cache.alloc(remote_size, max_type_size_, HeterCommType::COPY_KEY);

  size_t total_fea_num = 0;
  // if (rdma_checker_->need_rdma_trans()) {
  //  total_fea_num = send_keys_by_all2all_trans(
  //      gpu_id, rank_id_, node_size_, fea_size, d_tmp_keys, d_out_keys, stream);
  // } else {
  total_fea_num = send_data_by_all2all(gpu_id,
                                       node_size_,
                                       rank_id_,
                                       sizeof(KeyType),
                                       h_local_part_sizes,
                                       h_local_part_offsets,
                                       h_remote_part_sizes,
                                       h_remote_part_offsets,
                                       (const char *)(d_tmp_keys),
                                       (char *)(d_out_keys),
                                       stream);
  // }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return remote_size;
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_one_table(
    const int gpu_id, KeyType *d_keys, ValType *d_vals, const size_t &len, const cudaStream_t &stream) {
  ptr_tables_[gpu_id]->rwlock_->RDLock();
  ptr_tables_[gpu_id]->get(d_keys, reinterpret_cast<char *>(d_vals), len, stream, gpu_accessor_);
  ptr_tables_[gpu_id]->rwlock_->UNLock();
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename T>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::scatter_vals(const float* d_shard_vals,
                                                                      float* d_vals,
                                                                      T* idx,
                                                                      size_t len,
                                                                      size_t value_bytes,
                                                                      const cudaStream_t& stream) {
  const size_t val_size_float = size_t(value_bytes / sizeof(float));
  CHECK((value_bytes % sizeof(float)) == 0);
  size_t N = len * val_size_float;
  const int grid_size = (N - 1) / block_size_ + 1;
  // fill by float, d_shard_vals to d_vals
  scatter_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_vals, d_shard_vals, idx, N, val_size_float);
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename T>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_vals(float* d_shard_vals,
                                                                     const float* d_vals,
                                                                     T* idx,
                                                                     size_t len,
                                                                     size_t value_bytes,
                                                                     const cudaStream_t& stream) {
  const size_t value_size_float = size_t(value_bytes / sizeof(float));
  size_t N = len * value_size_float;
  int grid_size = (N - 1) / block_size_ + 1;
  // d_vals -> d_shard_vals
  gather_dvals_by_unit_kernel<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals, d_vals, idx, N, value_size_float);
}

// ===== all2all ======
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    scatter_sparse_vals_by_all2all(const int &gpu_id,
                                   const size_t &fea_size,
                                   const char *d_in_vals,
                                   void *d_out_vals,
                                   const size_t &value_bytes,
                                   void *d_tmp_vals,
                                   const cudaStream_t &stream) {
  auto &cache = storage_[gpu_id];
  auto &res = cache.shard_res;

  // 把属于不同机器的key发送回去
  auto h_local_part_sizes = res.h_local_part_sizes.data();
  auto h_local_part_offsets = res.h_local_part_offsets.data();
  auto h_remote_part_sizes = res.h_remote_part_sizes.data();
  auto h_remote_part_offsets = res.h_remote_part_offsets.data();

  size_t total_fea_num = 0;
  // if (rdma_checker_->need_rdma_trans()) {
  //   total_fea_num =
  //       send_vals_by_all2all_trans(gpu_id,
  //                                 rank_id_,
  //                                  node_size_,
  //                                  d_in_vals,
  //                                  reinterpret_cast<char *>(d_tmp_vals),
  //                                  value_bytes,
  //                                  stream);
  // } else {
    // send local device
    // 把数据发送回原来的机器,接收后的数据存储在d_tmp_vals
    total_fea_num = send_data_by_all2all(gpu_id,
                                         node_size_,
                                         rank_id_,
                                         value_bytes, // pull_type_size
                                         h_remote_part_sizes,
                                         h_remote_part_offsets,
                                         h_local_part_sizes,
                                         h_local_part_offsets,
                                         d_in_vals,
                                         reinterpret_cast<char *>(d_tmp_vals),
                                         stream);
  // }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // fill vals
  scatter_vals(
      (const float *)(d_tmp_vals),            // in
      reinterpret_cast<float *>(d_out_vals),  // out
      res.d_local_idx_parted,
      fea_size,
      value_bytes,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // CUDA_CHECK(cudaStreamSynchronize(stream));
}
// ===== all2all ======

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::scatter_inner_vals_p2p(
    const size_t &total_fea_num,
    void *d_out_vals,
    HeterCommType::InnerResource &res,  // NOLINT
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const size_t &value_bytes,
    const cudaStream_t &stream) {
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              res.d_remote_vals[i],
                              i,
                              data_len * value_bytes,
                              stream));
    }
  } else {
    /*
    // need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      // printf("[%d<-%d<-%d]recv vals offset: %ld, len: %ld\n", gpu_id,
      // trans_id, i, offset, data_len);
      if (!need_transfer(gpu_id, i)) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                                gpu_id,
                                res.d_remote_vals[i],
                                i,
                                data_len * value_bytes,
                                stream));
        continue;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_vals,
                                                     trans_id,
                                                     res.d_remote_vals[i],
                                                     i,
                                                     data_len * value_bytes,
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(&res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              res.d_trans_vals,
                              trans_id,
                              data_len * value_bytes,
                              stream));
    }
    */
  }
  // 从其他卡把VALUE拷贝回来了
  // restore vals
  scatter_vals(
      reinterpret_cast<const float *>(res.d_vals_parted),  // in
      reinterpret_cast<float *>(d_out_vals),               // out
      res.d_idx,
      total_fea_num,
      value_bytes,
      stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    scatter_inter_vals_by_copy(const int &gpu_id,
                               const size_t &fea_size, // gpuid卡上去重前的fea size
                               const char *d_in_vals, // 去重后的val
                               void *d_out_vals, // d_vals 
                               const size_t &value_bytes, // pull_type_size 
                               const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];

  // restore vals
  // heter_comm_kernel_->scatter_vals(
  //     (const float *)(d_in_vals),                              // in
  //     reinterpret_cast<float *>(my_cache.d_merged_push_vals),  // out
  //     my_cache.pull_res.d_restore_keys_idx,
  //     my_cache.pull_res.h_recv_fea_num,
  //     value_bytes,
  //    stream);

  // restore vals
  // 恢复到排序去重前的顺序,保存在d_merged_push_vals里
  gather_vals(
      reinterpret_cast<float *>(my_cache.d_merged_push_vals),  // out
      reinterpret_cast<const float *>(d_in_vals),              // in
      my_cache.pull_res.d_restore_keys_idx,
      my_cache.pull_res.h_recv_fea_num,
      value_bytes,
      stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  auto &res = my_cache.inner_res;
  int trans_id = -1;
  // if (topo_aware_ && device_num_ > 4) {
  //   trans_id = get_transfer_devid(gpu_id);
  // }
  // my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();

  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id]; // gpu_id卡发送到第i张卡的key的开始位置
    res.d_remote_vals[i] = &cache.d_merged_push_vals[recv_offset * value_bytes];
    // if (trans_id >= 0) {
    //   // set transfer buffer
    //   auto &trans_cache = storage_[trans_id];
    //   res.d_trans_vals =
    //       &trans_cache
    //            .d_merged_push_vals[trans_cache.h_trans_offset * value_bytes];
    // }
  }


  //对于gpuid卡,我们要做的就是把d_remote_vals指向的数据都拷贝回来
  res.d_vals_parted = my_cache.d_merged_vals;
  // my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();
  // recv all pull sparse vals
  scatter_inner_vals_p2p(fea_size,
                         d_out_vals,  // out
                         res,
                         gpu_id,
                         device_num_,
                         trans_id,
                         value_bytes,
                         stream);
}

// ===== all2all ======
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse_all2all(
    const int& gpu_num, KeyType* d_keys, ValType* d_vals, const size_t& fea_num) {

  platform::Timer time_line;
  time_line.Start();
 
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);

  auto& pull_storage = storage_[gpu_num];

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  // 这是什么stream,通信还是计算??应该是通信stream
  auto stream = resource_->local_stream(gpu_num, 0);

  /*
  // auto accessor_wrapper_ptr =
  //   GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  // size_t pull_value_size =
  //   accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);

  auto d_idx = memory::Alloc(place, fea_num * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, fea_num * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  auto d_shard_vals = memory::Alloc(place, fea_num * pull_value_size);
  ValType* d_shard_vals_ptr = reinterpret_cast<ValType*>(d_shard_vals->ptr());


  size_t grid_size = (len - 1) / block_size_ + 1;
  split_input_to_shard(d_keys, d_idx_ptr, fea_num, d_left_ptr, d_right_ptr,
                       gpu_num);
   
  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);

  // pull_sparse不需要copy dvals
  // check
  // dim3 block_dims(32, 32);
  // grid_size = (len - 1) / 32 + 1; 
  // dim3 grid_dims(grid_size);

  // fill_shard_vals和fill_shard_grads基本逻辑相同，直接复用
  // dy_mf_fill_shard_vals<<<grid_dims, block_dims, 0, stream>>>(
  //     d_shard_keys_ptr, d_keys, d_shard_vals_ptr, d_vals, d_idx_ptr,
  //     len, pull_value_size, gpu_accessor_);

  // d_shard_keys_ptr 是d_keys按照shard id排序后的结果，同理，d_shard_grads_ptr也是
  // VLOG(0) << "after dymf fill shard grads";

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


  pull_storage.h_local_left = h_left;
  pull_storage.h_local_right = h_right;
  

  // 等从其他卡上把feasign拉取回来，拷贝回tmp_local_grads
  pull_storage.tmp_local_keys = d_shard_keys_ptr;
  pull_storage.tmp_local_grads = (char*)d_shard_vals_ptr;


  int h_merge_offset[total_gpu];
  int h_merge_len[total_gpu];

  int cur_offset = 0;
  size_t inter_len = 0;

  time_line.Pause();
  time_line.Start();

  h_barrier.Wait();
  time_line.Pause();

  time_line.Start();

  // 8卡barrier以后, pull_storage_都填充满了
  // 每张卡的pull_storage的h_local_left, h_local_right都有
  for (int i = 0; i < total_gpu; ++i) {

    auto& cur_storage = pull_storage_[i];

    h_merge_offset[i] = cur_offset; // 0
    h_merge_len[i] = cur_storage.h_local_right[gpu_num] - cur_storage.h_local_left[gpu_num] + 1; // 第i张卡上分shard到第gpu_num号卡上的数据量

    cur_offset += h_merge_len[i];
    inter_len += h_merge_len[i];

  }

  pull_storage.alloc_for_inter_copy(inter_len);

  // 把所有应该放到gpu_num卡上的数据都拷贝过来
  // 对于pull_sparse来说，不需要拷贝vals,只需要拷贝keys
  // 
  for (int i = 0; i < total_gpu; i++) {

    auto& cur_storage = pull_storage_[i];

    cudaMemcpyAsync(pull_storage.local_keys + h_merge_offset[i],
                    cur_storage.tmp_local_keys + cur_storage.h_local_left[gpu_num],
                    h_merge_len[i] * sizeof(uint64_t),
                    cudaMemcpyDefault,
                    resource_->remote_stream(gpu_num, i));

    // cudaMemcpyAsync((char*)(pull_storage.local_grads) + h_merge_offset[i] * pull_value_size,
    //                  (char*)(cur_storage.tmp_local_grads)+ cur_storage.h_local_left[gpu_num] * pull_value_size,
    //                  h_merge_len[i] * pull_value_size,
    //                  cudaMemcpyDefault,
    //                  resource_->remote_stream(gpu_num, i));
  }

  for (int i = 0; i < total_gpu; i++) {
    cudaStreamSynchronize(resource_->remote_stream(gpu_num, i));
  }

  time_line.Pause();

  time_line.Start();

  int ret = inter_len;

  // ret返回去重后的长度
  auto d_fea_num_info =
      memory::Alloc(place, sizeof(uint32_t) * (inter_len * 3 + 1));

  d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());

  // merge以后的key都放在pull_storage.local_keys
  // gather_inner_size是去重以后的key数量
  size_t gather_inner_size = merge_key(gpu_num, pull_storage.local_keys, d_fea_num_info_ptr, inter_len);
  
  time_line.Pause();

  h_barrier.wait();

  */

  size_t gather_inner_size = gather_inter_keys_by_copy(gpu_num, fea_num, d_keys, stream);

  // int gather_inner_size = ret;
  // pull_size就是其他机器发送到当前机器当前卡的的key数量,机器之间没有去重
  size_t pull_size = gather_sparse_keys_by_all2all(gpu_num,
                                                   gather_inner_size,
                                                   pull_storage.d_merged_keys,
                                                   pull_storage.d_merged_keys,
                                                   pull_storage.d_merged_push_keys,
                                                   stream);

  // pull one table
  // 拉取d_merged_keys对应的d_merged_vals
  pull_one_table(gpu_num,
                 pull_storage.d_merged_keys,
                 (float *)pull_storage.d_merged_vals,
                 pull_size,
                 stream);


  scatter_sparse_vals_by_all2all(gpu_num,
                                 gather_inner_size,
                                 pull_storage.d_merged_vals,
                                 pull_storage.d_merged_vals,
                                 pull_type_size_,
                                 pull_storage.d_merged_push_vals,
                                 stream);

  // pull_storage.d_merged_vals里保存的是根据device_id排序前的key
  // 不过是去重后的,得把key再拷贝回原来的device
  scatter_inter_vals_by_copy(gpu_num,
                             fea_num,
                             pull_storage.d_merged_vals,
                             d_vals,
                             pull_type_size_,
                             stream);

}
/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse_multi_node(int gpu_num,
                                                                    KeyType* d_keys,
                                                                    ValType* d_vals,
                                                                    size_t len) {
  if (len == 0) {
    return;
  }

  platform::Timer time_line;
  int uniq_len = len;
  time_line.Start();
  if (multi_node_) {
    pull_sparse_all2all(gpu_num, d_keys, d_vals, uniq_len);
  }
  time_line.Pause();

}
*/

// ==== add for all2all ===

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::pull_sparse(int num,
                                                                    KeyType* d_keys,
                                                                    ValType* d_vals,
                                                                    size_t len) {
  if (len == 0) {
    return;
  }

  if (multi_node_) {
    pull_sparse_all2all(num, d_keys, d_vals, len);
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(num, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  // size_t val_type_size = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
  size_t val_type_size = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  auto d_shard_vals = memory::Alloc(place, len * val_type_size);
  float* d_shard_vals_ptr = reinterpret_cast<float*>(d_shard_vals->ptr());

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr, num);
  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr,
                                                        d_keys, d_idx_ptr, len);

  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  // for (auto k = 0; k < total_gpu; k++) {
  //   VLOG(0) << "yxfffpull gpunum: " << num << " k: " << k << " left: " << h_left[k] << " right: " << h_right[k]; 
  // }

  if (!direct_access_) {
    for (int i = 0; i < total_gpu; ++i) {
      int shard_len = h_right[i] - h_left[i] + 1;
      if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
      create_storage(num, i, shard_len * sizeof(KeyType),
                    shard_len * val_type_size);
    }

    walk_to_dest(num, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);
  }

  // std::vector<platform::Timer> time_lines;
  // time_lines.resize(total_gpu);
  // 对每张卡for循环从gpu table拿key对应的embedding
  // all2all需要分该key是否属于当前机器 
  for (int i = 0; i < total_gpu; ++i) {
    // time_lines[i].Start();
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[num][i].nodes_.back();
    if (!direct_access_) {
      cudaStreamSynchronize(node.in_stream);
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
/* 
    // VLOG(0) << "yxf:: start table get in device: " << num << " remote device: " << i;
    ptr_tables_[i]->rwlock_->RDLock();
    // VLOG(0) << "after lock yxf:: start table get in device: " << num << " remote device: " << i;
    if (!direct_access_) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                          node.val_storage, h_right[i] - h_left[i] + 1,
                          resource_->remote_stream(i, num));
    } else {
      ptr_tables_[i]->get(
          d_shard_keys_ptr + h_left[i],
          reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
          h_right[i] - h_left[i] + 1, resource_->remote_stream(i, num));
    }
*/

    ptr_tables_[i]->rwlock_->RDLock();
    if (!direct_access_) {
      ptr_tables_[i]->get(reinterpret_cast<KeyType*>(node.key_storage),
                        node.val_storage, h_right[i] - h_left[i] + 1,
                        resource_->remote_stream(i, num),
                        gpu_accessor_);
    } else {

      // ptr_tables_[i]->get(
      //    d_shard_keys_ptr + h_left[i],
      //    reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
      //    h_right[i] - h_left[i] + 1, resource_->remote_stream(i, num));
      // check
      ptr_tables_[i]->get(
          d_shard_keys_ptr + h_left[i],
          reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
          h_right[i] - h_left[i] + 1, resource_->remote_stream(i, num), gpu_accessor_);

      // 现在get的时候不为0的feasign也可能不存在，因为是其他机器的feasign,在当前机器的HBM里没有
      // 这个名字取得还挺好
      // 每台机器都需要gather key向量, 因为key向量分布在多台机器上 
      // gather_multi_node_emb(
      //    num,
      //    d_shard_keys_ptr + h_left[i],
      //    reinterpret_cast<char*>(d_shard_vals_ptr) + h_left[i] * val_type_size,
      //    h_right[i] - h_left[i] + 1, resource_->remote_stream(i, num), gpu_accessor_);
    }

  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, num));
    if (h_left[i] == -1) {
      continue;
    }
    ptr_tables_[i]->rwlock_->UNLock();
    // time_lines[i].Pause();
    // mg_time_2[i] += time_lines[i].ElapsedSec();
  }

  if (!direct_access_) {
    walk_to_src(num, total_gpu, h_left, h_right, reinterpret_cast<char*>(d_shard_vals_ptr), val_type_size);
    
  // walk_to_src(num, total_gpu, h_left, h_right, reinterpret_cast<char*>(d_shard_vals_ptr), val_type_size);

    for (int i = 0; i < total_gpu; ++i) {
      auto& node = path_[num][i].nodes_.front();
      cudaStreamSynchronize(node.out_stream);
    }
  }

  dim3 block_dims(32, 32);
  const size_t grid_size_ = (len - 1) / 32 + 1;
  dim3 grid_dims(grid_size_);
    
  dy_mf_fill_dvals<<<grid_dims, block_dims, 0, stream>>>(
        d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size, gpu_accessor_);

  // VLOG(0) << "yxf::finish walk to src: " << num;

/*
  dim3 block_dims(32,32);
  const size_t grid_size_ = (len - 1) / 32 + 1;
  dim3 grid_dims(grid_size_);


  dy_mf_fill_dvals<<<grid_dims, block_dims, 0, stream>>>(
      d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size);
  // dy_mf_fill_dvals<<<grid_size, block_size_, 0, stream>>>(
  //     d_shard_vals_ptr, d_vals, d_idx_ptr, len, val_type_size);
*/

  cudaStreamSynchronize(stream);
  // VLOG(0) << "yxf::finish walk to fill: " << num;
  if (!direct_access_) {
    for (int i = 0; i < total_gpu; ++i) {
      if (h_left[i] == -1 || h_right[i] == -1) {
        continue;
      }
      destroy_storage(num, i);
      // VLOG(0) << "yxf::end get device: " << num << "from device: " << i;
    }
  }
}

//  === push_sparse all2all ==== 

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::update_one_table(
    int gpu_id,
    KeyType *d_keys,
    GradType *d_grads,
    size_t len,
    Sgd &sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_id, 0);
  // no mf dim
  // if (!multi_mf_dim_) {
  //   auto &table = tables_[gpu_id];
  //   table->rwlock_->WRLock();
  //   table->update(
  //       d_keys, reinterpret_cast<const char *>(d_grads), len, sgd, stream);
  //   table->rwlock_->UNLock();
  // } else {
    auto &table = ptr_tables_[gpu_id];
    table->rwlock_->WRLock();
    table->update(
        d_keys, reinterpret_cast<const char *>(d_grads), len, sgd, stream, dev_id);
    table->rwlock_->UNLock();
  // }
  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
__global__ void scale_grad_kernel(const size_t N,
                                  char* grads,
                                  const size_t value_bytes,
                                  const size_t grad_dim,
                                  GPUAccessor accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    size_t idx = i / grad_dim;
    size_t field_id = i % grad_dim;

    float* vals = reinterpret_cast<float*>(&grads[idx * value_bytes]);
    float& show = vals[accessor.common_push_value.ShowIndex()];
    if (show > 0.0) {
      vals[accessor.common_push_value.EmbedGIndex() + field_id] /= show;
    }
  }
}

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::scale_grad(const size_t& len,
                                                                    char* grads,
                                                                    const size_t& value_bytes,
                                                                    const size_t& max_mf_dim,
                                                                    const cudaStream_t& stream,
                                                                    const GPUAccessor& gpu_accessor) {
  const size_t grad_dim = (max_mf_dim + 1); // 为啥要加1
  const size_t N = len * grad_dim;
  const int grid_size = (N - 1) / block_size_ + 1;
  scale_grad_kernel<<<grid_size, block_size_, 0, stream>>>(
      N, grads, value_bytes, grad_dim, gpu_accessor);
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_inner_data_p2p(
    const size_t &total_fea_num,
    const KeyType *d_keys,
    const void *d_vals,
    HeterCommType::InnerResource &res,  // NOLINT
    const int &gpu_id,
    const int &gpu_num,
    const int &trans_id,
    const size_t &value_bytes,
    const cudaStream_t &stream) {
  // gather all datas
  gather_keys(
      res.d_keys_parted, d_keys, res.d_idx, total_fea_num, stream);
  gather_vals(reinterpret_cast<float *>(res.d_vals_parted),
                                  reinterpret_cast<const float *>(d_vals),
                                  res.d_idx,
                                  total_fea_num,
                                  value_bytes,
                                  stream);

  // d_keys和d_vals都按照shard id排序放入res.d_keys_parted和res.d_vals_parted
  
  // p2p copy key and values
  if (trans_id < 0) {
    // not need transfer
    for (int i = 0; i < gpu_num; ++i) {
      size_t &data_len = res.h_part_sizes[i];
      if (data_len == 0) {
        continue;
      }
      size_t &offset = res.h_offsets[i];
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(res.d_remote_vals[i],
                              i,
                              &res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              data_len * value_bytes,
                              stream));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    return;
  }
  /*
  // need transfer
  for (int i = 0; i < gpu_num; ++i) {
    size_t data_len = res.h_part_sizes[i];
    if (data_len == 0) {
      continue;
    }
    size_t &offset = res.h_offsets[i];
    if (!need_transfer(gpu_id, i)) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                     i,
                                                     &res.d_keys_parted[offset],
                                                     gpu_id,
                                                     data_len * sizeof(KeyType),
                                                     stream));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpyPeerAsync(res.d_remote_vals[i],
                              i,
                              &res.d_vals_parted[offset * value_bytes],
                              gpu_id,
                              data_len * value_bytes,
                              stream));
      continue;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_trans_keys,
                                                   trans_id,
                                                   &res.d_keys_parted[offset],
                                                   gpu_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_keys[i],
                                                   i,
                                                   res.d_trans_keys,
                                                   trans_id,
                                                   data_len * sizeof(KeyType),
                                                   stream));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyPeerAsync(res.d_trans_vals,
                            trans_id,
                            &res.d_vals_parted[offset * value_bytes],
                            gpu_id,
                            data_len * value_bytes,
                            stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyPeerAsync(res.d_remote_vals[i],
                                                   i,
                                                   res.d_trans_vals,
                                                   trans_id,
                                                   data_len * value_bytes,
                                                   stream));
  
  }
  */
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename GPUAccessor>
__global__ void merge_gradients_basic_kernel(const KeyType* d_keys,
                                             const uint32_t* offset,
                                             const uint32_t* fea_num,
                                             const uint32_t* index,
                                             const char* input,
                                             char* output,
                                             int n,
                                             size_t grad_value_size,
                                            //  const DynamicGradMerger& merger,
                                            CustomGradMerger& merger,
                                            GPUAccessor& gpu_accessor) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t start = offset[i];
    uint32_t num = fea_num[i];
    int ori_index = index[start];
    float* out = reinterpret_cast<float*>(output + i * grad_value_size);
    const float* in = reinterpret_cast<const float*>(
        input + size_t(ori_index) * grad_value_size);
    // merger.update_basic(out, in, gpu_accessor);
    merger.copy_basic_field(out, in, gpu_accessor);
    KeyType key = d_keys[i];
    if (key != 0) {
      for (int j = 1; j < num; ++j) {
        ori_index = index[start + j];
        in = (float*)(input + size_t(ori_index) * grad_value_size);  // NOLINT
        // merger.merge_basic(out, in, gpu_accessor);
        merger.add_basic_field(out, in, gpu_accessor);
      }
    }
  }
}

template <typename KeyType, typename GPUAccessor>
__global__ void merge_gradients_embedx_kernel(const KeyType* d_keys,
                                              const uint32_t* offset,
                                              const uint32_t* fea_num,
                                              const uint32_t* index,
                                              const char* input,
                                              char* output,
                                              int n,
                                              size_t grad_dim,
                                              size_t grad_value_size,
                                              // const DynamicGradMerger& merger,
                                              CustomGradMerger& merger,
                                              GPUAccessor& gpu_accessor) {
/*
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    size_t value_idx = i / grad_dim;
    const uint32_t& start = offset[value_idx];
    const uint32_t& num = fea_num[value_idx];

    double val = 0;
    uint32_t off =
        gpu_accessor.common_push_value.EmbedxGIndex() + (i % grad_dim);
    for (uint32_t j = 0; j < num; ++j) {
      val += ((float*)(&input[size_t(index[start + j]) *  // NOLINT
                              grad_value_size]))[off];
    }
    (reinterpret_cast<float*>(&output[value_idx * grad_value_size]))[off] = val;
  }
*/
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t value_idx = i / grad_dim;
    if (d_keys[value_idx] > 0) {
/*
      size_t field_idx = i % grad_dim;

      uint32_t start = offset[value_idx];
      uint32_t num = fea_num[value_idx];
      int ori_index = index[start];

      FeaturePushValue& in = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
      FeaturePushValue& lhs = *(FeaturePushValue*)(output + value_idx * grad_value_size);

      merger.copy_embedx_field(lhs, in, field_idx);

      for (int j = 1; j < num; ++j) {
        int ori_index = index[start + j];
        FeaturePushValue& rhs = *(FeaturePushValue*)(input + size_t(ori_index) * grad_value_size);
        merger.add_embedx_field(lhs, rhs, field_idx);
      }
*/
      size_t field_idx = i % grad_dim;
      uint32_t start = offset[value_idx];
      uint32_t num = fea_num[value_idx];
      int ori_index = index[start];
      float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
      float* lhs = (float*)(output + value_idx * grad_value_size);
      merger.copy_embedx_field(lhs, rhs, field_idx, gpu_accessor);
      for (int j = 1; j < num; ++j) {
        int ori_index = index[start + j];
        float* rhs = (float*)(input + size_t(ori_index) * grad_value_size);
        merger.add_embedx_field(lhs, rhs, field_idx, gpu_accessor);
      }
    }
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_gradient(const KeyType* d_keys,
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
                                                                        GPUAccessor& gpu_accessor) {
  int grid_size1 = (n - 1) / block_size_ + 1;
  merge_gradients_basic_kernel<<<grid_size1, block_size_, 0, stream>>>(
      d_keys,
      offset,
      fea_num,
      index,
      input,
      output,
      n,
      grad_value_size,
      merger,
      gpu_accessor);
  if (grad_dim > 0) {
    int grid_size2 = (n * grad_dim - 1) / block_size_ + 1;
    merge_gradients_embedx_kernel<<<grid_size2, block_size_, 0, stream>>>(
        d_keys,
        offset,
        fea_num,
        index,
        input,
        output,
        n * grad_dim,
        grad_dim,
        grad_value_size,
        merger,
        gpu_accessor);
  }
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::merge_grads(
    const int &gpu_id,
    const size_t &len,
    const KeyType *d_in_keys,
    KeyType *d_out_keys,
    const void *d_in_grads,
    void *d_out_grads,
    const cudaStream_t &stream) {
  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(gpu_id);
  auto place = platform::CUDAPlace(gpu_id);
  auto d_fea_num_info =
      memory::Alloc(place,
                    sizeof(uint32_t) * len * 4);

  uint32_t *d_offset = reinterpret_cast<uint32_t *>(d_fea_num_info->ptr());
  uint32_t *d_sorted_idx = &d_offset[len];
  uint32_t *d_restore_idx = &d_sorted_idx[len];
  uint32_t *d_merged_cnts = &d_restore_idx[len];

  auto d_sort_keys_ptr =
      memory::Alloc(place,
                    sizeof(KeyType) * len);
  KeyType *d_sorted_keys = reinterpret_cast<KeyType *>(d_sort_keys_ptr->ptr());

  size_t merge_size = dedup_keys_and_fillidx(gpu_id,
                                             len,
                                             d_in_keys,   // input
                                             d_out_keys,  // output
                                             d_sorted_keys,
                                             d_restore_idx,
                                             d_sorted_idx,
                                             d_offset,
                                             d_merged_cnts,
                                             false,
                                             stream);

  merge_gradient(d_out_keys,
                 d_offset,
                 d_merged_cnts,
                 d_sorted_idx,
                 (char *)d_in_grads,
                 (char *)d_out_grads,
                 (int)merge_size,
                 max_mf_dim_,
                 grad_type_size_,
                 merger_,
                 stream,
                 gpu_accessor_);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return merge_size;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    gather_inter_gradient_by_copy(const int &gpu_id,
                                  const size_t &push_size,
                                  KeyType *d_keys,
                                  void *d_push_vals,
                                  const size_t &value_bytes,
                                  const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];

  my_cache.init_inner(push_size, device_num_);
  auto &res = my_cache.inner_res;
  res.h_part_sizes = &my_cache.h_fea_sizes[0];

  // shard data
  shard_inner_keys(push_size, d_keys, gpu_id, device_num_, &res, stream);
  // my_cache.inner_barrier_.Resume();
  // barrier wait all gpu done
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();

  size_t max_part_size = 0;
  size_t shard_recv_offset = 0;
  size_t shard_send_offset = 0;
  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    my_cache.h_recv_offsets[i] = shard_recv_offset;
    shard_recv_offset += cache.h_fea_sizes[gpu_id];
    res.h_offsets[i] = shard_send_offset;
    shard_send_offset += res.h_part_sizes[i];
    if (res.h_part_sizes[i] > max_part_size) {
      max_part_size = res.h_part_sizes[i];
    }
  }

  size_t trans_need_size = std::max(shard_recv_offset, push_size);
  int trans_id = -1;

  // if (topo_aware_ && device_num_ > 4) {
  //   trans_id = get_transfer_devid(gpu_id);
  //   storage_[trans_id].h_trans_size = max_part_size;
  //   // barrier wait all set trans length [0-4, 1-5, 3-7, 2-6]
  //   barrier_.wait();
  //   my_cache.h_trans_offset = trans_need_size;
  //   trans_need_size += my_cache.h_trans_size;
  // }

  my_cache.alloc(trans_need_size, max_type_size_);
  // my_cache.inner_barrier_.Resume();
  // barrier wait all hbm malloc size
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();

  for (int i = 0; i < device_num_; ++i) {
    auto &cache = storage_[i];
    size_t &recv_offset = cache.h_recv_offsets[gpu_id];
    res.d_remote_keys[i] = &cache.d_merged_keys[recv_offset];
    res.d_remote_vals[i] = &cache.d_merged_vals[recv_offset * value_bytes];
    // if (trans_id >= 0) {
    //   // set transfer buffer
    //   auto &trans_cache = storage_[trans_id];
    //   res.d_trans_keys = &trans_cache.d_merged_keys[trans_cache.h_trans_offset];
    //   res.d_trans_vals =
    //       &trans_cache.d_merged_vals[trans_cache.h_trans_offset * value_bytes];
    // }
  }
  res.d_keys_parted = my_cache.d_merged_push_keys;
  res.d_vals_parted = my_cache.d_merged_push_vals;
  // my_cache.inner_barrier_.Resume();
  // barrier wait set buffer ptr
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();
  // 把数据拷贝到对应的卡上
  gather_inner_data_p2p(push_size,
                        d_keys,   // input
                        d_push_vals, // input
                        res,
                        gpu_id,
                        device_num_,
                        trans_id,
                        value_bytes,
                        stream);
  // barrier wait all gpu aync memcpy data
  // my_cache.inner_barrier_.Resume();
  h_barrier.wait();
  // my_cache.inner_barrier_.Pause();
  // all embedx merge
  // 把梯度分shard以后需要Merge
  // 拷贝完以后,my_cache.d_merged_keys和d_merged_vals已经有从其他卡拷贝过来的key和梯度了
  // total_push_size是merge以后的长度
  size_t total_push_size = merge_grads(gpu_id,
                                      shard_recv_offset,
                                      my_cache.d_merged_keys,
                                      my_cache.d_merged_push_keys,
                                      my_cache.d_merged_vals,
                                      my_cache.d_merged_push_vals,
                                      stream);

  return total_push_size;
}


// 把gradient all2all发送到其他机器
template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
size_t HeterComm<KeyType, ValType, GradType, GPUAccessor>::
    gather_sparse_gradient_by_all2all(const int &gpu_id,
                                      const size_t &fea_size,
                                      const KeyType *d_keys, // d_merged_push_keys
                                      const char *d_push_vals, // d_merged_push_vals
                                      const size_t &value_bytes,
                                      KeyType *d_out_keys, // d_merged_push_keys
                                      KeyType *d_tmp_keys, // d_merged_keys
                                      char *d_out_vals,  // d_merged_push_vals
                                      char *d_tmp_vals,  // d_merged_vals
                                      const cudaStream_t &stream) {
  auto &my_cache = storage_[gpu_id];
  my_cache.init_shard(fea_size, node_size_);
  auto &res = my_cache.shard_res;

  size_t *h_local_part_sizes = res.h_local_part_sizes.data();
  size_t *h_local_part_offsets = res.h_local_part_offsets.data();
  uint32_t *h_push_fea_sizes = res.h_push_fea_sizes.data();

  partition_shard_keys(gpu_id,
                       fea_size,
                       d_keys,
                       res.d_local_idx_parted,
                       d_tmp_keys,
                       h_local_part_sizes,
                       node_size_,
                       stream);

  int all_shard_part_size = node_size_ * node_size_;
  h_local_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = rank_id_ * node_size_ + i;
    h_push_fea_sizes[offset] = h_local_part_sizes[i];
    h_local_part_offsets[i + 1] =
        h_local_part_offsets[i] + h_local_part_sizes[i];
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(res.d_node_size_ptr,
                                             &h_push_fea_sizes[0],
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             stream));

  // my_cache.node_barrier_.Resume();
  auto &comm = nccl_inter_comms_[gpu_id];
  auto nccl_stream = resource_->comm_stream(gpu_id, 0);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (const void *)(&res.d_node_size_ptr[rank_id_ * node_size_]),
      reinterpret_cast<void *>(res.d_node_size_ptr),
      node_size_,
      ncclInt,
      comm,
      stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&h_push_fea_sizes[0],
                                             res.d_node_size_ptr,
                                             all_shard_part_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // my_cache.node_barrier_.Pause();

  size_t *h_remote_part_sizes = res.h_remote_part_sizes.data();
  size_t *h_remote_part_offsets = res.h_remote_part_offsets.data();

  h_remote_part_offsets[0] = 0;
  for (int i = 0; i < node_size_; i++) {
    int offset = node_size_ * i + rank_id_;
    int recv_num = h_push_fea_sizes[offset];
    h_remote_part_sizes[i] = recv_num;
    h_remote_part_offsets[i + 1] = h_remote_part_offsets[i] + recv_num;
  }
  size_t total_recv_fea_num = h_remote_part_offsets[node_size_];
  my_cache.alloc(total_recv_fea_num, max_type_size_, HeterCommType::COPY_ALL);
  // fill shard vals
  // d_merged_push_vals根据node id排序,放到Merged_vals
  // 
  gather_vals(
      reinterpret_cast<float *>(d_tmp_vals),         // out
      reinterpret_cast<const float *>(d_push_vals),  // in
      res.d_local_idx_parted,
      fea_size,
      value_bytes,
      stream);

  size_t total_send_recv = 0;
  // if (rdma_checker_->need_rdma_trans()) {
  //   total_send_recv = send_gradient_by_all2all_trans(gpu_id,
  //                                                   rank_id_,
  //                                                    node_size_,
  //                                                   fea_size,
  //                                                   d_tmp_keys,
  //                                                   d_tmp_vals,
  //                                                   value_bytes,
  //                                                   d_out_keys,
  //                                                   d_out_vals,
  //                                                   stream);
  // } else {
    // send local device
    total_send_recv = send_data_by_all2all(gpu_id,
                                           node_size_,
                                           rank_id_,
                                           sizeof(KeyType),
                                           h_local_part_sizes,
                                           h_local_part_offsets,
                                           h_remote_part_sizes,
                                           h_remote_part_offsets,
                                           (const char *)(d_tmp_keys),
                                           reinterpret_cast<char *>(d_out_keys), // d_merged_push_keys
                                           stream);
    send_data_by_all2all(gpu_id,
                         node_size_,
                         rank_id_,
                         value_bytes,
                         h_local_part_sizes,
                         h_local_part_offsets,
                         h_remote_part_sizes,
                         h_remote_part_offsets,
                         (const char *)(d_tmp_vals),
                         reinterpret_cast<char *>(d_out_vals), // d_merged_push_vals
                         stream);
  // }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return total_recv_fea_num;
}

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse_all2all(
    const int &gpu_id,
    KeyType *d_keys,
    GradType *d_grads,
    const size_t &len,
    Sgd &sgd) {  // NOLINT
  if (len == 0) {
    return;
  }
  auto &my_cache = storage_[gpu_id];
  // my_cache.all2all_span_.Resume();

  auto stream = resource_->local_stream(gpu_id, 0);
  // tracker
  // if (FLAGS_enable_tracker_all2all) {
  //   // check push grads
  //   heter_comm_kernel_->check_valid_values(
  //       10,
  //       len,
  //       d_keys,
  //       reinterpret_cast<const char *>(d_grads),
  //       grad_type_size_,
  //       stream,
  //       (gpu_id == 0));
  // }

  // scale grad
  // 这里为啥要scale下??
  scale_grad(len,
             reinterpret_cast<char *>(d_grads),
             grad_type_size_,
             max_mf_dim_,
             stream,
             gpu_accessor_);

  // size_t inter_push_len = 0;
  // size_t node_push_len = 0;
  size_t value_bytes = grad_type_size_;

  // enable inner gather
  // if (FLAGS_enable_sparse_inner_gather) {
    // my_cache.inner_span_.Resume();
   // 分卡去重以后的长度是inter_push_len
   // 去重以后的key和val保存在d_merge_push_keys和d_merged_push_vals
  size_t inter_push_len =
        gather_inter_gradient_by_copy(gpu_id,
                                      len,
                                      d_keys,
                                      reinterpret_cast<void *>(d_grads),
                                      grad_type_size_,
                                      stream);
    // my_cache.inner_span_.Pause();

    // my_cache.node_span_.Resume();
/*
    if (FLAGS_enable_all2all_use_fp16) {  // use fp16
      value_bytes = heter_comm_kernel_->compress_values(
          inter_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_push_vals),
          reinterpret_cast<char *>(my_cache.d_merged_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            inter_push_len,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_vals,
                                            my_cache.d_merged_push_vals,
                                            stream);
      heter_comm_kernel_->uncompress_values(
          node_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_vals),
          reinterpret_cast<char *>(my_cache.d_merged_push_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {

*/
  //把key发送到其他机器 
  size_t node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            inter_push_len,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_push_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_push_vals,
                                            my_cache.d_merged_vals,
                                            stream);
    // }
    // my_cache.node_span_.Pause();
/*
  } else {  // only node all2all
    my_cache.node_span_.Resume();
    if (FLAGS_enable_all2all_use_fp16) {  // use fp16
      value_bytes = heter_comm_kernel_->compress_values(
          len,
          reinterpret_cast<const char *>(d_grads),
          reinterpret_cast<char *>(my_cache.d_merged_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      node_push_len =
          gather_sparse_gradient_by_all2all(gpu_id,
                                            len,
                                            d_keys,
                                            my_cache.d_merged_vals,
                                            value_bytes,
                                            my_cache.d_merged_push_keys,
                                            my_cache.d_merged_keys,
                                            my_cache.d_merged_vals,
                                            my_cache.d_merged_push_vals,
                                            stream);
      heter_comm_kernel_->uncompress_values(
          node_push_len,
          reinterpret_cast<const char *>(my_cache.d_merged_vals),
          reinterpret_cast<char *>(my_cache.d_merged_push_vals),
          grad_type_size_,
          max_mf_dim_,
          max_grad_bound_,
          stream);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    } else {
      node_push_len = gather_sparse_gradient_by_all2all(
          gpu_id,
          len,
          d_keys,                                   // in
          reinterpret_cast<const char *>(d_grads),  // in
          value_bytes,
          my_cache.d_merged_push_keys,  // out
          my_cache.d_merged_keys,       // tmp
          my_cache.d_merged_push_vals,  // out
          my_cache.d_merged_vals,       // tmp
          stream);
    }
    my_cache.node_span_.Pause();
  }
*/
  // if (FLAGS_enable_tracker_all2all) {
  //   VLOG(0) << "push gpu id=" << gpu_id
  //           << ", gather_sparse_gradient_by_all2all len=" << node_push_len;
  // }
  // all embedx merge
  size_t uniq_len = merge_grads(gpu_id,
                               node_push_len,
                               my_cache.d_merged_push_keys,  // in
                               my_cache.d_merged_keys,       // out
                               my_cache.d_merged_push_vals,  // in
                               my_cache.d_merged_vals,       // out 
                               stream);
  // if (FLAGS_enable_tracker_all2all) {
  //   // check all2ll merge grads
  //   heter_comm_kernel_->check_valid_values(
  //       11,
  //       uniq_len,
  //       my_cache.d_merged_keys,
  //       reinterpret_cast<const char *>(my_cache.d_merged_vals),
  //       grad_type_size_,
  //       stream,
  //       (gpu_id == 0));
  // }
  // if (FLAGS_enable_sparse_inner_gather) {
    // update all grad
    update_one_table(gpu_id,
                     my_cache.d_merged_keys,
                     reinterpret_cast<GradType *>(my_cache.d_merged_vals),
                     uniq_len,
                     sgd);
  // } else {
  //   // update all tables
  //   push_normal_sparse(gpu_id,
  //                      my_cache.d_merged_keys,
  //                      reinterpret_cast<float *>(my_cache.d_merged_vals),
  //                      uniq_len,
  //                      sgd);
  // }
  // my_cache.all2all_span_.Pause();
  // push
  // if (FLAGS_enable_tracker_all2all) {
  //   VLOG(0) << "push gpu id=" << gpu_id << ", push len=" << len
  //           << ", inner=" << inter_push_len << ", node=" << node_push_len
  //           << ", update=" << uniq_len << ", compress bytes=" << value_bytes
  //           << ", grad_type_size=" << grad_type_size_;
  // }
  // print_debug_time(gpu_id);
}

//  === push_sparse all2all ==== 

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse(int gpu_num,
                                                                    KeyType* d_keys,
                                                                    GradType* d_grads,
                                                                    size_t len,
                                                                    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  if (multi_node_) {
    push_sparse_all2all(gpu_num, d_keys, d_grads, len, sgd);
    return;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  // size_t grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  // int h_left[total_gpu];   // NOLINT
  // int h_right[total_gpu];  // NOLINT
  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
  float* d_shard_grads_ptr = reinterpret_cast<float*>(d_shard_grads->ptr());

  int uniq_len = len;
  merge_grad(gpu_num, d_keys, d_grads, NULL, len, uniq_len);

  int grid_size = (uniq_len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, uniq_len, d_left_ptr, d_right_ptr,
                       gpu_num);

  dim3 block_dims(32, 32);
  const size_t grid_size_ = (uniq_len - 1) / 32 + 1; 
  dim3 grid_dims(grid_size_);

  dy_mf_fill_shard_grads<<<grid_dims, block_dims, 0, stream>>>(
        d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
        uniq_len, grad_value_size, gpu_accessor_);

  // d_shard_keys_ptr 是d_keys按照shard id排序后的结果，同理，d_shard_grads_ptr也是

  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_right[i] - h_left[i] + 1;
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    create_storage(gpu_num, i, shard_len * sizeof(KeyType),
                   shard_len * grad_value_size);
  }

  walk_to_dest(gpu_num, total_gpu, h_left, h_right, d_shard_keys_ptr,
               reinterpret_cast<char*>(d_shard_grads_ptr), grad_value_size);

  // std::vector<platform::Timer> time_lines;
  // time_lines.resize(total_gpu);

  for (int i = 0; i < total_gpu; ++i) {
    // time_lines[i].Start();
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_num][i].nodes_.back();
    cudaStreamSynchronize(node.in_stream);

    platform::CUDADeviceGuard guard(resource_->dev_id(i));

    ptr_tables_[i]->rwlock_->WRLock();
    ptr_tables_[i]->update(reinterpret_cast<KeyType*>(node.key_storage),
                           node.val_storage, h_right[i] - h_left[i] + 1, sgd,
                           resource_->remote_stream(i, gpu_num), i);
  }
  for (int i = 0; i < total_gpu; ++i) {
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_num));
    if (h_left[i] != -1) {
      // if (!multi_mf_dim_) {
      //  tables_[i]->rwlock_->UNLock();
      //} else {
      ptr_tables_[i]->rwlock_->UNLock();
      // }
      // time_lines[i].Pause();
      // mg_time_3[i] += time_lines[i].ElapsedSec();
    }
  }
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    destroy_storage(gpu_num, i);
  }
}
/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::update_one_table(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT
  if (len == 0) {
    return;
  }

  int dev_id = resource_->dev_id(gpu_num);
  platform::CUDADeviceGuard guard(dev_id);
  ptr_tables_[gpu_num]->rwlock_->WRLock();
  // ptr_tables_[gpu_num]->update(d_keys, reinterpret_cast<char*>(d_grads), len, sgd,
  //                         resource_->remote_stream(gpu_num, gpu_num));
  ptr_tables_[gpu_num]->update(d_keys, reinterpret_cast<char*>(d_grads), len, sgd,
                           resource_->remote_stream(gpu_num, gpu_num), dev_id);
  ptr_tables_[gpu_num]->rwlock_->UNLock();
  cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
}
*/

/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
template <typename Sgd>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::push_sparse_multi_node(
    int gpu_num, KeyType* d_keys, GradType* d_grads, size_t len,
    Sgd& sgd) {  // NOLINT

  if (len == 0) {
    return;
  }

  platform::Timer time_line;
  time_line.Start();
  int uniq_len = len;

  // VLOG(0) << "yxf::multinode merge grad start: len: " << len << "uniq len: " << uniq_len << " gpunum: " << gpu_num;
  if (!FLAGS_gpups_dedup_pull_push_mode) {
    merge_grad(gpu_num, d_keys, d_grads, NULL, len, uniq_len);
  }
  // d_keys, d_grads 都是排序去重以后的了,uniq_len也是去重以后的长度

  time_line.Pause();
  // mg_time_13[gpu_num] += time_line.ElapsedSec();

  // VLOG(0) << "yxf:: mergelen: " << len << " uniq_len: " << uniq_len;
  // char* test_grad_values =
  //     (char*)malloc(grad_type_size_ * uniq_len);
  // char* test_grad_keys =
  //     (char*)malloc(sizeof(KeyType) * uniq_len);
  // cudaMemcpy(test_grad_values, d_grads,
  //           grad_type_size_ * uniq_len, cudaMemcpyDeviceToHost);
  // cudaMemcpy(test_grad_keys, d_keys,
  //           sizeof(KeyType) * uniq_len, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < uniq_len; i++) {
  //   FeaturePushValue* cur =
  //       (FeaturePushValue*)(test_grad_values + i * grad_type_size_);
  //   if (cur->mf_dim <= 0 || cur->mf_dim > 64) {
      
  //   // VLOG(0) << "yxfpush merge:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
  //   //         << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
  //   VLOG(0) << "yxf push merge000:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur feature: " << *cur;
  //   }
    
  // }
  // free(test_grad_values);
  // free(test_grad_keys);
  // VLOG(0) << "yxf::merge len: enddd";
  // VLOG(0) << "yxf222";

  // VLOG(0) << "yxf::multinode merge grad done: len: " << len << "uniq len: " << uniq_len << " gpunum: " << gpu_num;
  time_line.Start();

  // 将所有卡上shard id为gpu_num的key，排序，去重, uniq_len就是去重以后的长度，然后去重以后的key和grad保存在
  // storage_[gpu_num].local_keys和local_grads
  uniq_len = gather_one_node_grad_v2(gpu_num, d_keys, d_grads, uniq_len);

  time_line.Pause();
  // mg_time_10[gpu_num] += time_line.ElapsedSec();
  // VLOG(0) << "yxf::multinode gather_one_node_grad done: uniq len: " << uniq_len << " gpunum: " << gpu_num;
  
  // test_grad_values =
  //     (char*)malloc(grad_type_size_ * uniq_len);
  // test_grad_keys =
  //     (char*)malloc(sizeof(KeyType) * uniq_len);
  // cudaMemcpy(test_grad_values, storage_[gpu_num].local_grads,
  //           grad_type_size_ * uniq_len, cudaMemcpyDeviceToHost);
  // cudaMemcpy(test_grad_keys, storage_[gpu_num].local_keys,
  //           sizeof(KeyType) * uniq_len, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < uniq_len; i++) {
  //   FeaturePushValue* cur =
  //       (FeaturePushValue*)(test_grad_values + i * grad_type_size_);
  //   if (cur->mf_dim <= 0 || cur->mf_dim > 100) {
      
  //   // VLOG(0) << "yxfpush merge:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
  //   //         << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
  //   VLOG(0) << "yxf push merge111:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur feature: " << *cur;
  //   }
    
  // }
  // free(test_grad_values);
  // free(test_grad_keys);

  time_line.Start();

  // 对2,3,4,5卡和其余卡采取的逻辑不同
  // 
  if (gpu_num >= 2 && gpu_num <= 5) {
    uniq_len = gather_multi_node_grad(gpu_num, storage_[gpu_num].local_keys,
                                  storage_[gpu_num].local_grads, uniq_len);
  } else {
    uniq_len = gather_multi_node_grad_v4(gpu_num, storage_[gpu_num].local_keys,
                                  storage_[gpu_num].local_grads, uniq_len);
  }
  
  time_line.Pause();
  //mg_time_11[gpu_num] += time_line.ElapsedSec();
  VLOG(3) << "yxf::multinode gather_multi_node_grad done: uniq len: " << uniq_len << " gpunum: " << gpu_num;
  
  // test_grad_values =
  //     (char*)malloc(grad_type_size_ * uniq_len);
  // test_grad_keys =
  //     (char*)malloc(sizeof(KeyType) * uniq_len);
  // cudaMemcpy(test_grad_values, storage_[gpu_num].local_grads,
  //           grad_type_size_ * uniq_len, cudaMemcpyDeviceToHost);
  // cudaMemcpy(test_grad_keys, storage_[gpu_num].local_keys,
  //           sizeof(KeyType) * uniq_len, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < uniq_len; i++) {
  //   FeaturePushValue* cur =
  //       (FeaturePushValue*)(test_grad_values + i * grad_type_size_);
  //   if (cur->mf_dim < 0 || cur->mf_dim > 100) {
  //     FeaturePushValue* cur =
  //       (FeaturePushValue*)(test_grad_values + i * grad_type_size_);
  //   // VLOG(0) << "yxfpush merge:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur->slot: " << cur->slot << " mf_dim: " << cur->mf_dim
  //   //         << " show: " << cur->show << " clk: " << cur->clk << " : " << cur->mf_g[0] << " : " << cur->mf_g[1] << " : " << cur->mf_g[2] << " : " << cur->mf_g[3] << " : " << cur->mf_g[4] << " : " << cur->mf_g[5] << " : " << cur->mf_g[6] << " : " << cur->mf_g[7];
  //   VLOG(0) << "yxf push merge222:: i: " << i << " key: " << ((uint64_t*)test_grad_keys)[i] << " cur feature: " << *cur;
  //   }
    
  // }
  // free(test_grad_values);
  // free(test_grad_keys);

  // uniq_len是gpu_num卡所有机器上的grads排序去重以后的数据长度
  // 
  time_line.Start();
  update_one_table(gpu_num, storage_[gpu_num].local_keys,
                   storage_[gpu_num].local_grads, uniq_len, sgd);
  time_line.Pause();
  // mg_time_12[gpu_num] += time_line.ElapsedSec();
  VLOG(3) << "yxf::multinode update_one_table done: uniq len: " << uniq_len << " gpunum: " << gpu_num;
}

// uniq_len = gather_one_node_grad_v2(gpu_num, d_keys, d_grads, uniq_len);
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_one_node_grad_v2(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {

  platform::Timer time_line;

  time_line.Start();
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);



  auto& storage = storage_[gpu_num];

  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);

  // VLOG(0) << "in gather one node grad v2";

  // size_t grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (max_mf_dim_ * sizeof(float)));
  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t grad_value_size =
      accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);

  // VLOG(0) << "before alloc memory";
  // split keys grad in shard in current gpu 

  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);

  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());

  auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
  GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());

  // VLOG(0) << "after alloc memory";
  size_t grid_size = (len - 1) / block_size_ + 1;

  split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr,
                       gpu_num);

  // left, right 保存每张卡上的左右边界，比如10个key,如果前三个key,shard id分别是0 0 1, 那么left[0] = 0, right[0] = 1
  // d_idx_ptr保存每个key原来的在d_keys上的顺序

  // VLOG(0) << "after split input to shard";

  cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost, stream);

  // check
  dim3 block_dims(32, 32);
  grid_size = (len - 1) / 32 + 1; 
  dim3 grid_dims(grid_size);

  dy_mf_fill_shard_grads<<<grid_dims, block_dims, 0, stream>>>(
      d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
      len, grad_value_size, gpu_accessor_);

  // d_shard_keys_ptr 是d_keys按照shard id排序后的结果，同理，d_shard_grads_ptr也是
  // VLOG(0) << "after dymf fill shard grads";

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  // split keys grad in shard in current gpu end 
  storage.h_local_left = h_left;
  storage.h_local_right = h_right;

  storage.tmp_local_keys = d_shard_keys_ptr;
  storage.tmp_local_grads = (char*)d_shard_grads_ptr;


  // 每张卡都会有一份
  //
  int h_merge_offset[total_gpu];
  int h_merge_len[total_gpu];

  int cur_offset = 0;
  size_t inter_len = 0;

  time_line.Pause();
  // mg_time_14[gpu_num] += time_line.ElapsedSec();
  time_line.Start();

  h_barrier.wait(); // 8卡全部走到这里
  
  time_line.Pause();
  // mg_time_1[gpu_num] += time_line.ElapsedSec();
  time_line.Start();

  // 8卡barrier以后, storage_都填充满了
  // 没张卡的storage的h_local_left, h_local_right都有
  for (int i = 0; i < total_gpu; ++i) {

    auto& cur_storage = storage_[i];

    h_merge_offset[i] = cur_offset; // 0
    h_merge_len[i] = cur_storage.h_local_right[gpu_num] - cur_storage.h_local_left[gpu_num] + 1; // 第i张卡上分shard到第gpu_num号卡上的数据量

    cur_offset += h_merge_len[i];
    inter_len += h_merge_len[i];

  }

  // inter_len是分配到gpu_num号卡上的所有数据总长度
  // VLOG(0) << "after merge" << inter_len;

  storage.alloc_for_inter_copy(inter_len);

  // 更新了storage里的local keys和local grads

  // VLOG(0) << "gpu_num:" << gpu_num << " after alloc_for_inter_copy" << inter_len << " grad value size:" << grad_value_size;
  // VLOG(0) << "gpu_num: " << gpu_num << " h_merge_offset:" << h_merge_offset[0] << " " << h_merge_offset[1] <<  " " << h_merge_offset[2] << " " << h_merge_offset[3] << " " << h_merge_offset[4] << " " << h_merge_offset[5] << " " << h_merge_offset[6] << " " << h_merge_offset[7];
  // VLOG(0) << "gpu_num: " << gpu_num << " h_merge_len:" << h_merge_len[0] << " " << h_merge_len[1] << " " << h_merge_len[2] << " " << h_merge_len[3] << " " << h_merge_len[4] << " " << h_merge_len[5] << " " << h_merge_len[6] << " " << h_merge_len[7];
  // VLOG(0) << "gpu_num: " << gpu_num << " h_local_left: " << storage_[0].h_local_left[gpu_num] << " " << storage_[1].h_local_left[gpu_num] << " " << storage_[2].h_local_left[gpu_num] << " " << storage_[3].h_local_left[gpu_num] << " " << storage_[4].h_local_left[gpu_num]
  //        << " " << storage_[5].h_local_left[gpu_num] << " " << storage_[6].h_local_left[gpu_num] << " " << storage_[7].h_local_left[gpu_num];

  // 把所有应该放到gpu_num卡上的数据都拷贝过来
  for (int i = 0; i < total_gpu; i++) {

    auto& cur_storage = storage_[i];

    cudaMemcpyAsync(storage.local_keys + h_merge_offset[i],
                      cur_storage.tmp_local_keys + cur_storage.h_local_left[gpu_num],
                      h_merge_len[i] * sizeof(uint64_t),
                      cudaMemcpyDefault,
                      resource_->remote_stream(gpu_num, i));

    cudaMemcpyAsync((char*)(storage.local_grads) + h_merge_offset[i] * grad_value_size,
                      (char*)(cur_storage.tmp_local_grads)+ cur_storage.h_local_left[gpu_num] * grad_value_size,
                      h_merge_len[i] * grad_value_size,
                      cudaMemcpyDefault,
                      resource_->remote_stream(gpu_num, i));
  }

  for (int i = 0; i < total_gpu; i++) {
    cudaStreamSynchronize(resource_->remote_stream(gpu_num, i));
  }

  // VLOG(0) << "after memory copy async";

  time_line.Pause();
  // mg_time_4[gpu_num] += time_line.ElapsedSec();

  time_line.Start();
  int ret = inter_len;

  // 拷贝过来以后，再做一次merge_grad, ret就是最后去重以后的key数量，最后去重以后的key, grads,保存在storage.local_keys和storage.local_grads
  // 
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, NULL, inter_len, ret);
  time_line.Pause();
  // mg_time_6[gpu_num] += time_line.ElapsedSec();
  h_barrier.wait();
  // VLOG(0) << "before wait";
  return ret;
  // storage.gather_one_node_len = ret;

  // h_barrier.wait();
  // int trans_id = -1;
  // size_t trans_total_len = 0;
  // if (gpu_num >= 2 && gpu_num <= 5) {
  //   trans_id = (gpu_num + 4) % 8;
  //   trans_total_len = storage_[trans_id].gather_one_node_len + ret;
  //   storage.alloc_for_data_transfer(trans_total_len);
  // }
  // if (trans_id >= 0) {
  //   cudaMemcpyAsync(storage.all_keys,
  //                     storage.local_keys,
  //                     ret * sizeof(uint64_t),
  //                     cudaMemcpyDefault,
  //                     resource_->remote_stream(gpu_num, gpu_num));
  //   cudaMemcpyAsync((char*)(storage.all_grads),
  //                     (char*)(storage.local_grads),
  //                     ret * grad_value_size,
  //                     cudaMemcpyDefault,
  //                     resource_->remote_stream(gpu_num, gpu_num));
  //   cudaMemcpyAsync(storage.all_keys + ret,
  //                     storage_[trans_id].local_keys,
  //                     storage_[trans_id].gather_one_node_len * sizeof(uint64_t),
  //                     cudaMemcpyDefault,
  //                     resource_->remote_stream(gpu_num, trans_id));
  //   cudaMemcpyAsync((char*)(storage.all_grads) + ret * grad_value_size,
  //                     (char*)(storage_[trans_id].local_grads),
  //                     storage_[trans_id].gather_one_node_len * grad_value_size,
  //                     cudaMemcpyDefault,
  //                     resource_->remote_stream(gpu_num, trans_id));
  //   cudaStreamSynchronize(resource_->remote_stream(gpu_num, gpu_num));
  //   cudaStreamSynchronize(resource_->remote_stream(gpu_num, trans_id));
  // } 
  
  // return trans_total_len > 0 ? trans_total_len : ret;
}
*/


/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad_v3(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  platform::Timer time_line;
  time_line.Start();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  int h_node_len[node_size_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);
  time_line.Pause();
  // mg_time_15[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // auto d_nccl_rank_mem = memory::Alloc(place, sizeof(int));
  // int* d_nccl_rank = reinterpret_cast<int*>(d_nccl_rank_mem->ptr());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommUserRank(nccl_inter_comm, d_nccl_rank));
  // int h_nccl_rank[1];  // NOLINT
  // cudaMemcpy(h_nccl_rank, d_nccl_rank, sizeof(int), cudaMemcpyDeviceToHost);

  // allgather grad len
  if (gpu_num >= 2 && gpu_num <= 5) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
              cudaMemcpyDeviceToHost);

    for (int i = 0; i < node_size_; ++i) {
      if (h_node_len[i] > max_size) {
        max_size = h_node_len[i];
      }
    }
    
    time_line.Pause();
    // mg_time_7[gpu_num] += time_line.ElapsedSec();
    time_line.Start();
    
    // test all gather with different send len
    storage.alloc_for_data_transfer_nccl(max_size * node_size_);

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        d_keys, storage.local_keys, max_size, ncclUint64, nccl_inter_comm, stream));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        (void*)(d_grads), (void*)storage.local_grads, max_size * grad_type_size_, ncclUint8,
        nccl_inter_comm, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  }
  
  time_line.Pause();
  // mg_time_8[gpu_num] += time_line.ElapsedSec();
  
  return 0;
}
*/
/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_one_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  platform::Timer time_line;
  time_line.Start();
  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;

  ncclComm_t nccl_inner_comm = nccl_inner_comms_[gpu_num];
  
  // // compute each shard len 
  // auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  // auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  // int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  // int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  // cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  // cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  // //
  // auto d_idx = memory::Alloc(place, len * sizeof(int));
  // int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  // split_input_to_shard(d_keys, d_idx_ptr, len, d_left_ptr, d_right_ptr,
  //                      gpu_num);
  // auto d_shard_keys = memory::Alloc(place, len * sizeof(KeyType));
  // KeyType* d_shard_keys_ptr = reinterpret_cast<KeyType*>(d_shard_keys->ptr());
  // auto d_shard_grads = memory::Alloc(place, len * grad_value_size);
  // GradType* d_shard_grads_ptr = reinterpret_cast<GradType*>(d_shard_grads->ptr());
  // int grid_size = (len - 1) / block_size_ + 1;
  // dy_mf_fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
  //       d_shard_keys_ptr, d_keys, d_shard_grads_ptr, d_grads, d_idx_ptr,
  //       len, grad_value_size);
  // auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  // auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * total_gpu);
  // int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  // int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());
  // cudaMemcpyAsync(h_left, d_left_ptr, total_gpu * sizeof(int),
  //            cudaMemcpyDeviceToHost, stream);
  // cudaMemcpyAsync(h_right, d_right_ptr, total_gpu * sizeof(int),
  //            cudaMemcpyDeviceToHost, stream);
  // cudaStreamSynchronize(stream);
  // int total_shard_len = total_gpu * total_gpu;
  // int cur_shard_offset = total_gpu * gpu_num;
  // int h_node_shard_len[total_shard_len];
  // for (int local_shard_id = 0; local_shard_id < gpu_num; local_shard_id++) {
  //   h_node_shard_len[cur_shard_offset + local_shard_id] = h_right[local_shard_id] - h_left[local_shard_id] + 1;
  // }
  // auto d_node_shard_len_mem = memory::Alloc(place, total_gpu * total_gpu * sizeof(int));
  // int* d_node_shard_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  // cudaMemcpy(d_node_shard_len + cur_shard_offset, h_node_shard_len, sizeof(int) * total_gpu,
  //            cudaMemcpyHostToDevice);

  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  // PADDLE_ENFORCE_GPU_SUCCESS(
  //     platform::dynload::ncclAllGather((const void*)(d_node_shard_len + cur_shard_offset),
  //                                      (void*)d_node_shard_len, total_gpu, ncclInt,  // NOLINT
  //                                      nccl_inner_comm, stream));
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // cudaMemcpy(h_node_shard_len, d_node_shard_len, sizeof(int) * total_shard_len,
  //            cudaMemcpyDeviceToHost);
  // for (int i = 0; i < total_gpu; ++i) {
  //   if (h_node_len[i*total_gpu + gpu_num] > max_size) {
  //     max_size = h_node_len[i*total_gpu + gpu_num];
  //   }
  // }
  // storage.alloc(max_size * total_gpu);
  // // compute each shard len enddddddddd
  
  // alloc for size
  int h_node_len[total_gpu];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[gpu_num] = len;
  cudaMemcpy(d_node_len + gpu_num, h_node_len + gpu_num, sizeof(int),
             cudaMemcpyHostToDevice);

  time_line.Pause();
  // mg_time_14[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::ncclAllGather((const void*)(d_node_len + gpu_num),
                                       (void*)d_node_len, 1, ncclInt,  // NOLINT
                                       nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  time_line.Pause();
  // mg_time_4[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * total_gpu,
             cudaMemcpyDeviceToHost);
  // VLOG(0) << "yxffff: end one nccl: gpu: " << gpu_num;
  for (int i = 0; i < total_gpu; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  
  storage.alloc(max_size * total_gpu);
  char* gather_local_keys = (char*)(storage.all_keys + max_size * gpu_num);
  char* gather_local_grads = (char*)((char*)(storage.all_grads) + max_size * gpu_num * grad_type_size_);
  cudaMemcpyAsync(gather_local_keys, (char*)d_keys,
                    h_node_len[gpu_num] * sizeof(KeyType), cudaMemcpyDefault, stream);
  
  cudaMemcpyAsync(gather_local_grads, (char*)d_grads,
                    h_node_len[gpu_num] * grad_type_size_, cudaMemcpyDefault, stream);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      gather_local_keys, storage.all_keys, max_size, ncclUint64, nccl_inner_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
     gather_local_grads, (void*)(storage.all_grads), max_size * grad_type_size_, ncclUint8,
      nccl_inner_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // VLOG(0) << "yxffff: end two nccl: gpu: " << gpu_num << " max_size: " << max_size << " size: " << max_size * grad_type_size_;
  time_line.Pause();
  // mg_time_5[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT
  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  int merge_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    int index = i * max_size;
    auto d_idx = memory::Alloc(place, h_node_len[i] * sizeof(int));
    int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

    // cudaMemset(d_left_ptr, -1, total_gpu * sizeof(int));
    // cudaMemset(d_right_ptr, -1, total_gpu * sizeof(int));
    // cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
    // cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));   

    split_input_to_shard(storage.all_keys + index, d_idx_ptr, h_node_len[i],
                         d_left_ptr, d_right_ptr, gpu_num);
                      
    cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // VLOG(0) << "yxf111 i: " << i << " index: " << index << " h_node_len: " << h_node_len[i] << " right: " << h_right[gpu_num] << " left: " << h_left[gpu_num] << " index: " << index * grad_type_size_ << " merge: " << merge_num * grad_type_size_;
    size_t cur_len = size_t(h_right[gpu_num]) - size_t(h_left[gpu_num]) + 1;
    // int grid_size = (cur_len - 1) / block_size_ + 1;
    GradType* current_all_grads = (GradType*)((char*)(storage.all_grads) + index * grad_type_size_);
    GradType* current_local_grads = (GradType*)((char*)(storage.local_grads) + merge_num * grad_type_size_);
    
    dim3 block_dims(32, 32);
    const size_t grid_size = (cur_len - 1) / 32 + 1; 
    dim3 grid_dims(grid_size);
    dy_mf_fill_shard_grads<<<grid_dims, block_dims, 0, stream>>>(
        storage.local_keys + merge_num, storage.all_keys + index,
        current_local_grads, current_all_grads,
        d_idx_ptr + h_left[gpu_num], cur_len, grad_type_size_);
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    merge_num = merge_num + h_right[gpu_num] - h_left[gpu_num] + 1;
    // VLOG(0) << "yxf222 i: " << i << " gpu_num: " << gpu_num << " index: " << index << " h_node_len: " << h_node_len[i] << " right: " << h_right[gpu_num] << " left: " << h_left[gpu_num] << " index: " << index * grad_type_size_ << " merge: " << merge_num * grad_type_size_;
    // for (auto k = 0; k < total_gpu; k++) {
    //   VLOG(0) << "yxf333 i: " << i << " k: " << k << " gpu_num: " << gpu_num << " index: " << index << " h_node_len: " << h_node_len[k] << " right: " << h_right[k] << " left: " << h_left[k];
    // }
  }
  // VLOG(0) << "yxf gather one node keys done grad_type_size_: " << grad_type_size_;
  int ret = merge_num;
  time_line.Pause();
  // mg_time_1[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, NULL, merge_num, ret);
  time_line.Pause();
  // mg_time_6[gpu_num] += time_line.ElapsedSec();
  return ret;
}
*/


/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad_v4(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {

  platform::Timer time_line;
  time_line.Start();
  int dev_id = resource_->dev_id(gpu_num);
  int trans_id = (dev_id + 4) % 8; // 0卡先拷贝到4卡，1卡先拷贝到5卡, 6卡先拷贝到2卡，7卡先拷贝到3卡

  auto& storage = storage_[gpu_num];
  auto& trans_storage = storage_[trans_id]; // 拿到需要拷贝到该卡的storage

  platform::CUDAPlace place = platform::CUDAPlace(trans_id);
  platform::CUDADeviceGuard guard(trans_id);
  // VLOG(0) << "yxfff: trans_id: " << trans_id << " gpu num: " << gpu_num << " devid: " << dev_id;
  auto stream = resource_->local_stream(gpu_num, 0);
  auto trans_stream = resource_->remote_stream(trans_id, gpu_num);

  int max_size = 0;

  // 用的还是transid的inter comm
  ncclComm_t nccl_inter_comm = nccl_trans_inter_comms_[trans_id];

  // alloc for size
  int h_node_len[node_size_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);
  time_line.Pause();
  // mg_time_15[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // auto d_nccl_rank_mem = memory::Alloc(place, sizeof(int));
  // int* d_nccl_rank = reinterpret_cast<int*>(d_nccl_rank_mem->ptr());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommUserRank(nccl_inter_comm, d_nccl_rank));
  // int h_nccl_rank[1];  // NOLINT
  // cudaMemcpy(h_nccl_rank, d_nccl_rank, sizeof(int), cudaMemcpyDeviceToHost);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, trans_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(trans_stream));

  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  // VLOG(0) << "yxfff00 " << h_node_len[0] << " | " << h_node_len[1];
  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  
  time_line.Pause();
  // mg_time_7[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  
  // test all gather with different send len
  // VLOG(0) << "yxfff44";
  // storage_[trans_id].alloc_in_transfer(max_size * node_size_);
  // hetercomm初始化的时候，预留了一定空间fea_num_
  if (trans_keys[trans_id] == nullptr && trans_grads[trans_id] == nullptr) {
    trans_keys[trans_id] = memory::Alloc(place, max_size * node_size_ * sizeof(KeyType));
    trans_grads[trans_id] = memory::Alloc(place, max_size * node_size_ * grad_type_size_);
  } else if (max_size * node_size_ > trans_keys[trans_id]->size()) {
    VLOG(0) << "yxfff55" << "cur size: " << max_size * node_size_ << " ori size: " << trans_keys[trans_id]->size();
    trans_keys[trans_id].reset();
    trans_grads[trans_id].reset();
    trans_keys[trans_id] = memory::Alloc(place, max_size * node_size_ * sizeof(KeyType));
    trans_grads[trans_id] = memory::Alloc(place, max_size * node_size_ * grad_type_size_);
  }

  // auto trans_all_keys_mem = memory::Alloc(place, max_size * node_size_ * sizeof(KeyType));
  // // VLOG(0) << "yxf LocalStorage111 alloc grad size: " << grad_type_size_;
  // auto trans_all_grads_mem = memory::Alloc(place, max_size * node_size_ * grad_type_size_);
  auto* trans_all_keys = reinterpret_cast<KeyType*>(trans_keys[trans_id]->ptr());
  auto* trans_all_grads = reinterpret_cast<GradType*>(trans_grads[trans_id]->ptr());

  // VLOG(0) << "yxfff33";
  // storage.alloc(max_size * node_size_);


  // 先把当前卡的数据拷贝到trans_id对应的卡上
  cudaMemcpyAsync((char*)(trans_all_keys), (char*)d_keys,
                     len * sizeof(KeyType), cudaMemcpyDefault, trans_stream);
  cudaMemcpyAsync((char*)(trans_all_grads), (char*)d_grads,
                    len * grad_type_size_, cudaMemcpyDefault, trans_stream);
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(trans_stream));
  // VLOG(0) << "yxfff22";
  // storage.alloc(max_size * node_size_);

    //test
  // int test_rank = 0;
  // for (auto test_idx = 0; test_idx < node_size_; test_idx++) {
  //   if (h_node_len[test_idx] == len) {
  //     test_rank = test_idx;
  //     break;
  //   }
  // }
  // assert(test_rank==node_rank_);

  // char* gather_local_keys = (char*)(storage.all_keys + max_size * node_rank_);
  // char* gather_local_grads = (char*)(storage.all_grads) + max_size * node_rank_ * grad_type_size_;
  // cudaMemcpyAsync(gather_local_keys, (char*)d_keys,
  //                   len * sizeof(KeyType), cudaMemcpyDefault, stream);
  // cudaMemcpyAsync(gather_local_grads, (char*)d_grads,
  //                   len * grad_type_size_, cudaMemcpyDefault, stream);

  // // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      trans_all_keys, trans_all_keys, max_size, ncclUint64, nccl_inter_comm, trans_stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (void*)(trans_all_grads), (void*)(trans_all_grads), max_size * grad_type_size_, ncclUint8,
      nccl_inter_comm, trans_stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(trans_stream));

  // VLOG(0) << "yxfff11";
  time_line.Pause();
  // mg_time_8[gpu_num] += time_line.ElapsedSec();
  time_line.Start();

  // copy from trans gpu to current gpu
  guard.SetDeviceIndex(dev_id);
  storage.alloc_for_data_transfer_nccl(max_size * node_size_);
  
  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {
    int index = i * max_size;
    char* current_local_grads = (char*)(storage.local_grads) + merge_num * grad_type_size_;
    char* current_all_grads = (char*)(trans_all_grads) + index * grad_type_size_;
    cudaMemcpyAsync(storage.local_keys + merge_num, trans_all_keys + index,
                    h_node_len[i] * sizeof(KeyType), cudaMemcpyDefault, stream);
    cudaMemcpyAsync(current_local_grads, current_all_grads,
                    h_node_len[i] * grad_type_size_, cudaMemcpyDefault, stream);
    merge_num += h_node_len[i];
    // VLOG(0) << "yxf end4444: i: " << i << " index: " << index << " merge_num: " << merge_num;
  }
  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, NULL, merge_num, ret);

  // for hbm oom debug 
  trans_keys[trans_id].reset();
  trans_grads[trans_id].reset();

  time_line.Pause();
  // mg_time_9[gpu_num] += time_line.ElapsedSec();
  return ret;
}
*/

/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad_v5(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  platform::Timer time_line;
  time_line.Start();
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  int h_node_len[node_size_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);

  auto d_test_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_test_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  int test_node_len[2];
  size_t reoeder_grad_size = 0;
  // VLOG(0) << "yxf00";
  reorder_input_by_mfdim(d_keys, d_grads, len, test_node_len, gpu_num, reoeder_grad_size);
  // VLOG(0) << "yxf11";

  time_line.Pause();
  // mg_time_15[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // auto d_nccl_rank_mem = memory::Alloc(place, sizeof(int));
  // int* d_nccl_rank = reinterpret_cast<int*>(d_nccl_rank_mem->ptr());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommUserRank(nccl_inter_comm, d_nccl_rank));
  // int h_nccl_rank[1];  // NOLINT
  // cudaMemcpy(h_nccl_rank, d_nccl_rank, sizeof(int), cudaMemcpyDeviceToHost);

  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  
  time_line.Pause();
  // mg_time_7[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // VLOG(0) << "yxf22";
  // test all gather with different send len
  storage.alloc_for_multi_node_nccl(max_size * node_size_);
  // storage.alloc(max_size * node_size_);

    //test
  // int test_rank = 0;
  // for (auto test_idx = 0; test_idx < node_size_; test_idx++) {
  //   if (h_node_len[test_idx] == len) {
  //     test_rank = test_idx;
  //     break;
  //   }
  // }
  // assert(test_rank==node_rank_);

  // char* gather_local_keys = (char*)(storage.all_keys + max_size * node_rank_);
  // char* gather_local_grads = (char*)(storage.all_grads) + max_size * node_rank_ * grad_type_size_;
  // cudaMemcpyAsync(gather_local_keys, (char*)d_keys,
  //                   len * sizeof(KeyType), cudaMemcpyDefault, stream);
  // cudaMemcpyAsync(gather_local_grads, (char*)d_grads,
  //                   len * grad_type_size_, cudaMemcpyDefault, stream);

  // // allgather keys and grads
  // VLOG(0) << "yxf33";
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (void*)(d_grads), (void*)storage.all_grads, reoeder_grad_size, ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  // VLOG(0) << "yxf44";
  time_line.Pause();
  // mg_time_8[gpu_num] += time_line.ElapsedSec();
  return 0;
}
*/
/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {

  platform::Timer time_line;
  time_line.Start();
  int dev_id = resource_->dev_id(gpu_num);

  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);


  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];

  // alloc for size
  int h_node_len[node_size_];  // NOLINT

  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());

  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);
  time_line.Pause();

  // mg_time_15[gpu_num] += time_line.ElapsedSec();
  time_line.Start();

  // auto d_nccl_rank_mem = memory::Alloc(place, sizeof(int));
  // int* d_nccl_rank = reinterpret_cast<int*>(d_nccl_rank_mem->ptr());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommUserRank(nccl_inter_comm, d_nccl_rank));
  // int h_nccl_rank[1];  // NOLINT
  // cudaMemcpy(h_nccl_rank, d_nccl_rank, sizeof(int), cudaMemcpyDeviceToHost);


  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  // 两机都拿到d_node_len,然后拷贝到h_node_len,两机的h_node_len是一样的
  // 然后拿一个最大值
  //
  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  
  time_line.Pause();
  // mg_time_7[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  
  // 分配空间用来存储多机allgather以后的所有数据
  // test all gather with different send len
  storage.alloc_for_multi_node_nccl(max_size * node_size_);
  // storage.alloc(max_size * node_size_);

    //test
  // int test_rank = 0;
  // for (auto test_idx = 0; test_idx < node_size_; test_idx++) {
  //   if (h_node_len[test_idx] == len) {
  //     test_rank = test_idx;
  //     break;
  //   }
  // }
  // assert(test_rank==node_rank_);

  // char* gather_local_keys = (char*)(storage.all_keys + max_size * node_rank_);
  // char* gather_local_grads = (char*)(storage.all_grads) + max_size * node_rank_ * grad_type_size_;
  // cudaMemcpyAsync(gather_local_keys, (char*)d_keys,
  //                   len * sizeof(KeyType), cudaMemcpyDefault, stream);
  // cudaMemcpyAsync(gather_local_grads, (char*)d_grads,
  //                   len * grad_type_size_, cudaMemcpyDefault, stream);

  // // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (void*)(d_grads), (void*)storage.all_grads, max_size * grad_type_size_, ncclUint8,
      nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  time_line.Pause();
  // mg_time_8[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // //test all gather time

  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     (void*)(d_grads), (void*)storage.all_grads, max_size * grad_type_size_, ncclUint8,
  //     nccl_inter_comm, stream));
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


  // 把所有机器数据又拷贝回storage.local_keys和storage.local_grads
  //
  // //test all gather time end
  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {

    int index = i * max_size;

    char* current_local_grads = (char*)(storage.local_grads) + merge_num * grad_type_size_;
    char* current_all_grads = (char*)(storage.all_grads) + index * grad_type_size_;

    cudaMemcpyAsync(storage.local_keys + merge_num, storage.all_keys + index,
                    h_node_len[i] * sizeof(KeyType), cudaMemcpyDefault, stream);

    cudaMemcpyAsync(current_local_grads, current_all_grads,
                    h_node_len[i] * grad_type_size_, cudaMemcpyDefault, stream);

    merge_num += h_node_len[i];
    // VLOG(0) << "yxf end4444: i: " << i << " index: " << index << " merge_num: " << merge_num;
  }

  // 然后又排序，去重,然后storage.local_keys,就是所有机器，gpu_num卡上的grads数据了，然后就是去更新hashtable
  // 
  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, NULL, merge_num, ret);

  time_line.Pause();
  // mg_time_9[gpu_num] += time_line.ElapsedSec();
  return ret;
}
*/

/*
// allgather key, pull embedding to d_vals
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_val(
    int gpu_num, KeyType* d_keys, ValType* d_vals, int len) {

  platform::Timer time_line;
  time_line.Start();
  int dev_id = resource_->dev_id(gpu_num);

  auto& pull_storage = pull_storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);


  int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];

  // alloc for size
  int h_node_len[node_size_];  // NOLINT

  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());

  h_node_len[0] = len;

  cudaMemcpy(d_node_len, h_node_len, sizeof(int), cudaMemcpyHostToDevice);
  time_line.Pause();

  // mg_time_15[gpu_num] += time_line.ElapsedSec();
  time_line.Start();

  // auto d_nccl_rank_mem = memory::Alloc(place, sizeof(int));
  // int* d_nccl_rank = reinterpret_cast<int*>(d_nccl_rank_mem->ptr());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclCommUserRank(nccl_inter_comm, d_nccl_rank));
  // int h_nccl_rank[1];  // NOLINT
  // cudaMemcpy(h_nccl_rank, d_nccl_rank, sizeof(int), cudaMemcpyDeviceToHost);


  // allgather grad len
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 1, ncclInt, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_,
             cudaMemcpyDeviceToHost);

  // 两机都拿到d_node_len,然后拷贝到h_node_len,两机的h_node_len是一样的
  // 然后拿一个最大值
  //
  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i] > max_size) {
      max_size = h_node_len[i];
    }
  }
  
  time_line.Pause();
  // mg_time_7[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  
  // 分配空间用来存储多机allgather以后的所有数据
  // test all gather with different send len
  pull_storage.alloc_for_multi_node_nccl(max_size * node_size_);
  // storage.alloc(max_size * node_size_);

    //test
  // int test_rank = 0;
  // for (auto test_idx = 0; test_idx < node_size_; test_idx++) {
  //   if (h_node_len[test_idx] == len) {
  //     test_rank = test_idx;
  //     break;
  //   }
  // }
  // assert(test_rank==node_rank_);

  // char* gather_local_keys = (char*)(storage.all_keys + max_size * node_rank_);
  // char* gather_local_grads = (char*)(storage.all_grads) + max_size * node_rank_ * grad_type_size_;
  // cudaMemcpyAsync(gather_local_keys, (char*)d_keys,
  //                   len * sizeof(KeyType), cudaMemcpyDefault, stream);
  // cudaMemcpyAsync(gather_local_grads, (char*)d_grads,
  //                   len * grad_type_size_, cudaMemcpyDefault, stream);

  // // allgather keys and grads
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_keys, pull_storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //    (void*)(d_grads), (void*)storage.all_grads, max_size * grad_type_size_, ncclUint8,
  //     nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  time_line.Pause();
  // mg_time_8[gpu_num] += time_line.ElapsedSec();
  time_line.Start();
  // //test all gather time


  // 每台机器都拿到所有的key，应该去重
  // 其实这里不去重会方便些，但是会导致去hashtable的重复拉取
  // ret返回去重后的长度
  auto d_fea_num_info =
      memory::Alloc(place, sizeof(uint32_t) * (len * 3 + 1));

  d_fea_num_info_ptr =
      reinterpret_cast<uint32_t*>(d_fea_num_info->ptr());
  
  merge_key(gpu_num, pull_storage.all_keys, d_fea_num_info_ptr, inter_len, ret);


  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     (void*)(d_grads), (void*)storage.all_grads, max_size * grad_type_size_, ncclUint8,
  //     nccl_inter_comm, stream));
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  // PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));


  // 把所有机器数据又拷贝回storage.local_keys和storage.local_grads
  //
  // //test all gather time end
  int merge_num = 0;
  for (int i = 0; i < node_size_; ++i) {

    int index = i * max_size;

    char* current_local_grads = (char*)(storage.local_grads) + merge_num * grad_type_size_;
    char* current_all_grads = (char*)(storage.all_grads) + index * grad_type_size_;

    cudaMemcpyAsync(storage.local_keys + merge_num, storage.all_keys + index,
                    h_node_len[i] * sizeof(KeyType), cudaMemcpyDefault, stream);

    cudaMemcpyAsync(current_local_grads, current_all_grads,
                    h_node_len[i] * grad_type_size_, cudaMemcpyDefault, stream);

    merge_num += h_node_len[i];
    // VLOG(0) << "yxf end4444: i: " << i << " index: " << index << " merge_num: " << merge_num;
  }

  // 然后又排序，去重,然后storage.local_keys,就是所有机器，gpu_num卡上的grads数据了，然后就是去更新hashtable
  // 
  int ret = merge_num;
  merge_grad(gpu_num, storage.local_keys, storage.local_grads, NULL, merge_num, ret);

  time_line.Pause();
  // mg_time_9[gpu_num] += time_line.ElapsedSec();
  return ret;






}
*/






/*
template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::gather_multi_node_grad_v2(
    int gpu_num, KeyType* d_keys, GradType* d_grads, int len) {
  int dev_id = resource_->dev_id(gpu_num);
  auto& storage = storage_[gpu_num];
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_num, 0);
  // int max_size = 0;
  ncclComm_t nccl_inter_comm = nccl_inter_comms_[gpu_num];
  // alloc for size
  // int h_node_len[node_size_];  // NOLINT
  // auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int));
  // int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  // h_node_len[0] = len;

  // calculate len with different mf dim
  auto h_left_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * multi_mf_dim_);
  auto h_right_alloc = memory::Alloc(phi::GPUPinnedPlace(), sizeof(int) * multi_mf_dim_);
  int* h_left = reinterpret_cast<int*>(h_left_alloc->ptr());
  int* h_right = reinterpret_cast<int*>(h_right_alloc->ptr());
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  auto d_left = memory::Alloc(place, multi_mf_dim_ * sizeof(int));
  auto d_right = memory::Alloc(place, multi_mf_dim_ * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  split_input_by_mfdim(d_grads, d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_num);
  cudaMemcpyAsync(h_left, d_left_ptr, multi_mf_dim_ * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_right, d_right_ptr, multi_mf_dim_ * sizeof(int),
             cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);


  auto accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
  size_t dim8_grad_value_size =
      accessor_wrapper_ptr->GetPushValueSize(8);
  size_t dim64_grad_value_size =
      accessor_wrapper_ptr->GetPushValueSize(64);

  // size_t dim8_grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (8 * sizeof(float)));
  // size_t dim64_grad_value_size =
  //    TYPEALIGN(8, sizeof(FeaturePushValue) + (64 * sizeof(float)));

  size_t dim8_len = h_right[0] - h_left[0] + 1;
  auto d_dim8_grads = memory::Alloc(place, dim8_len * dim8_grad_value_size);
  GradType* d_dim8_grads_ptr = reinterpret_cast<GradType*>(d_dim8_grads->ptr());
  int grid_size = (dim8_len - 1) / block_size_ + 1;
  dy_mf_fill_mfdim_grads<<<grid_size, block_size_, 0, stream>>>(d_dim8_grads_ptr, d_grads, d_idx_ptr, dim8_len, dim8_grad_value_size, size_t(0), dim64_grad_value_size);

  size_t dim64_len = h_right[1] - h_left[1] + 1;
  auto d_dim64_grads = memory::Alloc(place, dim64_len * dim64_grad_value_size);
  GradType* d_dim64_grads_ptr = reinterpret_cast<GradType*>(d_dim64_grads->ptr());
  grid_size = (dim64_len - 1) / block_size_ + 1;
  dy_mf_fill_mfdim_grads<<<grid_size, block_size_, 0, stream>>>(d_dim64_grads_ptr, d_grads, d_idx_ptr, dim64_len, dim64_grad_value_size, size_t(h_left[1]), dim64_grad_value_size);
  int h_node_len[node_size_ * multi_mf_dim_];  // NOLINT
  auto d_node_len_mem = memory::Alloc(place, node_size_ * sizeof(int) * multi_mf_dim_);
  int* d_node_len = reinterpret_cast<int*>(d_node_len_mem->ptr());
  h_node_len[0] = dim8_len;
  h_node_len[1] = dim64_len;
  cudaMemcpy(d_node_len, h_node_len, sizeof(int) * multi_mf_dim_, cudaMemcpyHostToDevice);
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      d_node_len, d_node_len, 2, ncclInt, nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  cudaMemcpy(h_node_len, d_node_len, sizeof(int) * node_size_ * multi_mf_dim_,
             cudaMemcpyDeviceToHost);
  int dim8_max_size = 0;
  int dim64_max_size = 0;
  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i*2] > dim8_max_size) {
      dim8_max_size = h_node_len[i*2];
    }
  }
  for (int i = 0; i < node_size_; ++i) {
    if (h_node_len[i*2+1] > dim64_max_size) {
      dim64_max_size = h_node_len[i*2+1];
    }
  }
  VLOG(0) << "yxfdim8len: " << dim8_max_size << " dim64len: " << dim64_max_size;

  auto dim8_all_grads_mem = memory::Alloc(place, dim8_max_size * dim8_grad_value_size * 2);
  GradType* dim8_all_grads = reinterpret_cast<GradType*>(dim8_all_grads_mem->ptr());
  auto dim64_all_grads_mem = memory::Alloc(place, dim64_max_size * dim64_grad_value_size * 2);
  GradType* dim64_all_grads = reinterpret_cast<GradType*>(dim64_all_grads_mem->ptr());

  cudaMemcpyAsync(dim8_all_grads, (char*)d_dim8_grads_ptr,
                     dim8_len * dim8_grad_value_size, cudaMemcpyDefault, stream);
  cudaMemcpyAsync(dim64_all_grads, (char*)d_dim64_grads_ptr,
                     dim64_len * dim64_grad_value_size, cudaMemcpyDefault, stream);

  platform::Timer time_line;
  time_line.Start();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
  // PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
  //     d_keys, storage.all_keys, max_size, ncclUint64, nccl_inter_comm, stream));

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (void*)(dim8_all_grads), (void*)dim8_all_grads, dim8_max_size * dim8_grad_value_size, ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
      (void*)(dim64_all_grads), (void*)dim64_all_grads, dim64_max_size * dim64_grad_value_size, ncclUint8,
      nccl_inter_comm, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  time_line.Pause();
  // mg_time_7[gpu_num] += time_line.ElapsedSec();


  // calculaye len with different mf dim end
  return 0;
}
*/

template <typename KeyType, typename ValType, typename GradType, typename GPUAccessor>
void HeterComm<KeyType, ValType, GradType, GPUAccessor>::end_pass() {
  int total_gpu = resource_->total_gpu();
  std::vector<std::thread> threads;

  auto dump_to_cpu_func = [this](int index) {
    auto stream = resource_->local_stream(index, 0);
    int dev_id = resource_->dev_id(index);
    platform::CUDADeviceGuard guard(dev_id);
    tables_[index]->dump_to_cpu(dev_id, stream);
  };

  if (!multi_mf_dim_) {
    for (int i = 0; i < total_gpu; ++i) {
      threads.push_back(std::thread(dump_to_cpu_func, i));
    }
    for (auto& t : threads) {
      t.join();
    }
  }
}

/*
      uint32_t* d_restore_idx =
          reinterpret_cast<uint32_t*>(&key2slot[total_length]);
      uint32_t* d_sorted_idx =
          reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
      uint32_t* d_offset =
          reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
      uint32_t* d_merged_cnts =
          reinterpret_cast<uint32_t*>(&d_offset[total_length]);

      uint64_t* d_merged_keys =
          reinterpret_cast<uint64_t*>(&total_keys[total_length]);
      uint64_t* d_sorted_keys =
          reinterpret_cast<uint64_t*>(&d_merged_keys[total_length]);

      int dedup_size = HeterPs_->dedup_keys_and_fillidx(
          devid_2_index,
          static_cast<int>(total_length),
          total_keys,     // input
          d_merged_keys,  // output
          d_sorted_keys,  // sort keys
          d_restore_idx,  // pull fill idx
          d_sorted_idx,   // sort old idx
          d_offset,       // offset
          d_merged_cnts,
          FLAGS_gpups_dedup_pull_push_mode & 0x02);
*/

template <typename KeyType,
          typename ValType,
          typename GradType,
          typename GPUAccessor>
int HeterComm<KeyType, ValType, GradType, GPUAccessor>::dedup_keys_and_fillidx(
    const int gpu_id,
    const int total_fea_num, // 所有的feasign 
    const KeyType* d_keys,   // input
    KeyType* d_merged_keys,  // output
    KeyType* d_sorted_keys,
    uint32_t* d_restore_idx,
    uint32_t* d_sorted_idx,
    uint32_t* d_offset,
    uint32_t* d_merged_cnts,
    bool filter_zero, // true
    cudaStream_t stream) {

  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);

  if (stream == 0) {
    stream = resource_->local_stream(gpu_id, 0);
  }


  // auto stream = resource_->local_stream(gpu_id, 0);

  assert(total_fea_num > 0);
  int merged_size = 0;

  // 最后一个uint32_t保存d_merged_size
  size_t byte_size = sizeof(uint32_t) * (total_fea_num + 1);

  auto d_index_ptr = memory::Alloc(place,
                                   byte_size);

  uint32_t* d_index_in = reinterpret_cast<uint32_t*>(d_index_ptr->ptr());
  int* d_merged_size = reinterpret_cast<int*>(&d_index_in[total_fea_num]);

  // heter_comm_kernel_->fill_idx(d_index_in, total_fea_num, stream);
  int grid_size = (total_fea_num - 1) / block_size_ + 1;

  // 从0开始 
  fill_idx<<<grid_size, block_size_, 0, stream>>>(d_index_in, total_fea_num);

  void* d_buf = NULL;
  size_t temp_storage_bytes = 0;

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(NULL,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));
  auto d_cache_ptr = memory::Alloc(place,
                                   temp_storage_bytes);

  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRadixSort::SortPairs(d_buf,
                                      temp_storage_bytes,
                                      d_keys,
                                      d_sorted_keys,
                                      d_index_in,
                                      d_sorted_idx,
                                      total_fea_num,
                                      0,
                                      8 * sizeof(KeyType),
                                      stream,
                                      false));

  // 把d_keys排序后放在d_sorted_keys
  // 对应的index放在d_sorted_idx

  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(NULL,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  if (d_cache_ptr->size() < temp_storage_bytes) {
    d_cache_ptr.reset();
    d_cache_ptr = memory::Alloc(place,
                                temp_storage_bytes);
  }
  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceRunLengthEncode::Encode(d_buf,
                                         temp_storage_bytes,
                                         d_sorted_keys,
                                         d_merged_keys,
                                         d_merged_cnts,
                                         d_merged_size,
                                         total_fea_num,
                                         stream));
  // 去重以后的key放在d_merged_keys
  // 每个key出现的次数放在d_merged_cnts
  // 去重以后的key个数放在d_merged_size
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync((void*)&merged_size,
                                             (void*)d_merged_size,
                                             sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));

  if (d_cache_ptr->size() < temp_storage_bytes) {
    d_cache_ptr.reset();
    d_cache_ptr = memory::Alloc(place,
                                temp_storage_bytes);
  }
  d_buf = reinterpret_cast<int*>(d_cache_ptr->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::ExclusiveSum(
      d_buf, temp_storage_bytes, d_merged_cnts, d_offset, merged_size, stream));

  // d_offset是d_merged_cnts的前缀和
  // [8,6,7,5,3,0,9] -> [0, 8, 14, 21, 26, 29, 29]

  // 初始化为0
  //
  if (filter_zero) {
    cudaMemsetAsync(d_restore_idx, 0, total_fea_num * sizeof(uint32_t), stream);
  }
  // fill restore idx [1,3,5,2,4,6] = [1,2,1,3,2,1]
  fill_restore_idx(filter_zero,
                   total_fea_num,
                   merged_size,   // 去重以后的feasign数
                   d_merged_keys, // 去重以后的key
                   d_sorted_idx,  // 排序以后，保存每个key原来的idx 
                   d_offset,      // 保存每个feasign的数量的数组的前缀和数组
                   d_merged_cnts, // 每个feasign的数量
                   d_restore_idx, // 输出
                   stream);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));

  return merged_size;
  
}

// template <typename KeyType, typename ValType, typename GradType>
// void HeterComm<KeyType, ValType, GradType>::dump_to_cpu(int index) {
//  auto stream = resource_->local_stream(index, 0);
//  int dev_id = resource_->dev_id(index);
//  platform::CUDADeviceGuard guard(dev_id);
//  tables_[index]->dump_to_cpu(dev_id, stream);
//}

}  // end namespace framework
}  // end namespace paddle

#endif
