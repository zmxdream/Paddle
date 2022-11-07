/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include "paddle/fluid/framework/lod_tensor.h"
// #include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace framework {

const int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0


template <typename GPUAccessor>
__global__ void PullCopy(float** dest,
                         const float* src,
                         const int64_t* len,
                         int slot_num,
                         int total_len,
                         uint64_t** keys,
                         uint64_t max_val_size,
                         int* gpu_dim,
                         GPUAccessor gpu_accessor) {
  
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);

    float* feature_value_ptr =
        (float*)((char*)src + uint64_t(i) * uint64_t(max_val_size));
    int mf_dim = gpu_dim[x] - 3;

    gpu_accessor.Select(
        dest[x] + y * (mf_dim + 3), feature_value_ptr, keys[x] + y, mf_dim);
  }
}

// ==== hbm optimized ====
template <typename GPUAccessor>
__global__ void PullDedupCopy(const size_t N,    // total_length * (max_mf_dim_ + 3) == total_length * hidden
                              const uint64_t* total_keys,
                              float** dest,
                              const float* src,  // 去重以后的feature value，和去重以后的key一一对应
                              const int64_t* slot_lens,
                              uint64_t max_val_size,
                              const int* slot_dims,
                              const int hidden,
                              const int* key2slot,
                              const uint32_t* restore_idx,
                              int* gpu_dim,
                              GPUAccessor gpu_accessor) {
  CUDA_KERNEL_LOOP(idx, N) {
    int i = idx / hidden; // 第i个feasign
    int off = idx % hidden; // 第i个feasign的第off维度

    int x = key2slot[i]; // 第i个slot属于第x个slot
    int y = i - slot_lens[x]; // y表示这是第x个slot里的第y个feasign

    // assert(slot_dims[x] == hidden);
    int mf_dim = gpu_dim[x]; // 第x个slot对应的dim + 3
    float* dest_ptr = dest[x] + y * mf_dim;

    if (off >= mf_dim) {
      return;
    }
    // 0 key fill zero
    if (total_keys[i] == 0) {
      *(dest_ptr + off) = 0;
      return;
    }

    // FeatureValue* feature_value_ptr =
    //    (FeatureValue*)((char*)src + uint64_t(restore_idx[i]) * uint64_t(max_val_size));
    // 拿到对应向量 
    // float* feature_value_ptr =
    //    (float*)((char*)src + uint64_t(restore_idx[i]) * uint64_t(max_val_size));

    // gpu_accessor.SelectV2(
    //    dest_ptr, feature_value_ptr, total_keys + i, off);
    float* src_ptr = (float*)((char*)src + uint64_t(restore_idx[i]) *
                                                uint64_t(max_val_size));
    switch (off) {
      case 0:
        *(dest_ptr + off) = src_ptr[gpu_accessor.common_pull_value.ShowIndex()];
        break;
      case 1:
        *(dest_ptr + off) = src_ptr[gpu_accessor.common_pull_value.ClickIndex()];
        break;
      case 2:
        *(dest_ptr + off) = src_ptr[gpu_accessor.common_pull_value.EmbedWIndex()];
        break;
      default:
        // check
        if (src_ptr[gpu_accessor.common_pull_value.MfSizeIndex()] == 0) {
          *(dest_ptr + off) = 0;
        } else {
          *(dest_ptr + off) = src_ptr[gpu_accessor.common_pull_value.EmbedxWIndex() + off - 3];
        }
        break;
    }
    /*
    switch (off) {
      case 0:
        *(dest_ptr + off) = feature_value_ptr->show;
        break;
      case 1:
        *(dest_ptr + off) = feature_value_ptr->clk;
        break;
      case 2:
        *(dest_ptr + off) = feature_value_ptr->lr;
        break;
      default:
        if ((feature_value_ptr)->mf_size == 0) {
          *(dest_ptr + off) = 0;
        } else {
          *(dest_ptr + off) = feature_value_ptr->mf[off - 2];
        }
        break;
    }
    */
 
  }
}
// ==== hbm optimized ====

template <typename GPUAccessor>
__global__ void PushCopyWithPool(float* dest,
                                 float** src,
                                 int64_t* len,
                                 int slot_num,
                                 uint64_t total_len,
                                 int bs,
                                 int* slot_vector,
                                 int* mf_dim_vector,
                                 size_t grad_value_size,
                                 GPUAccessor gpu_accessor) {

  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    float* cur =
        (float*)((char*)dest + i * grad_value_size);
   
    int mf_dim = mf_dim_vector[x];  // slot_vector holds both slot and
                                    // slot:mf_dim information
    gpu_accessor.GradientSelect(cur, src[x] + y * (mf_dim + 3), slot_vector[x], mf_dim, bs);
  }
}

// === hbm optimized =====
// 目的就是把total_values_gpu中的emb放到gpu_values中去
template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPull(const paddle::platform::Place& place,
                                               const uint64_t* total_keys,
                                               float** gpu_values,
                                               const float* total_values_gpu,
                                               const int64_t* slot_lens,
                                               const int* key2slot,
                                               const int hidden_size,
                                               const int64_t total_length,
                                               const int* slot_dims,
                                               const uint32_t* gpu_restore_idx,
                                               int* gpu_dim,
                                               size_t val_type_size) {

  auto stream = dynamic_cast<phi::GPUContext*>(
                    paddle::platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  size_t N = total_length * hidden_size;

  PullDedupCopy<<<CUDA_BLOCK(N), stream>>>(N,
                                           total_keys,
                                           gpu_values,
                                           total_values_gpu,
                                           slot_lens,
                                           val_type_size,
                                           slot_dims,
                                           hidden_size,
                                           key2slot,
                                           gpu_restore_idx,
                                           gpu_dim,
                                           gpu_accessor_);
  cudaStreamSynchronize(stream);                       
}
// ==== hbm optimized ====

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPull(const paddle::platform::Place& place,
                                               uint64_t** gpu_keys,
                                               const std::vector<float*>& values,
                                               const float* total_values_gpu,
                                               const int64_t* gpu_len,
                                               const int slot_num,
                                               const int hidden_size,
                                               const int64_t total_length,
                                               int* gpu_dim,
                                               size_t val_type_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto buf_value = memory::Alloc(place, values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(buf_value->ptr());
  cudaMemcpy(gpu_values, values.data(), values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
  PullCopy<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
        gpu_values, total_values_gpu, gpu_len, slot_num, total_length, gpu_keys,
        val_type_size, gpu_dim, gpu_accessor_);
  cudaStreamSynchronize(stream);
}

// ========= hbm optimized ====
template <typename GPUAccessor>
__global__ void PushMergeCopyAtomic(const size_t N,
                                    const uint64_t* total_keys,
                                    float* dest,
                                    float** src,
                                    const int hidden,
                                    const int bs,
                                    const int* slot_vector,
                                    const int* slot_dims,
                                    const int64_t* slot_lens,
                                    const int* key2slot,
                                    const uint32_t* d_restore_idx,
                                    size_t grad_value_size,
                                    GPUAccessor gpu_accessor) {
  CUDA_KERNEL_LOOP(idx, N) {
    int i = idx / hidden;
    int off = idx % hidden;
    // filter 0 keys
    if (total_keys[i] == 0) {
      return;
    }

    int x = key2slot[i];
    int y = i - slot_lens[x];

    int mf_dim = slot_dims[x] - 3;
    const float* ptr = src[x] + y * (mf_dim + 3); // 梯度,需要merge

    // FeaturePushValue* cur = (FeaturePushValue*)((char*)dest + d_restore_idx[i] * grad_value_size);
    // d_restore_idx[i]表示第i个key去重以后的idx
    float* cur = (float*)((char*)dest + d_restore_idx[i] * grad_value_size);
 
    if (off - 3 >= mf_dim) {
      return;
    }

    switch (off) {
      case 0:
        cur[gpu_accessor.common_push_value.SlotIndex()] = (float)slot_vector[x];
        cur[gpu_accessor.common_push_value.MfDimIndex()] = mf_dim;
        paddle::platform::CudaAtomicAdd(&cur[gpu_accessor.common_push_value.ShowIndex()],
                                        *(ptr + off));
        break;
      case 1:
        paddle::platform::CudaAtomicAdd(&cur[gpu_accessor.common_push_value.ClickIndex()],
                                        *(ptr + off));
        break;
      case 2:
        paddle::platform::CudaAtomicAdd(&cur[gpu_accessor.common_push_value.EmbedGIndex()],
                                        *(ptr + off) * -1. * bs);
        break;
      default:
        int embedx_idx = off - 3;
        if (mf_dim < embedx_idx) {
          return;
        }
        paddle::platform::CudaAtomicAdd(
            &cur[gpu_accessor.common_push_value.EmbedxGIndex() + embedx_idx],
            *(ptr + off) * -1. * bs);
        break;
    }

    // gpu_accessor.GradientSelectV2(cur, src[x] + y * (mf_dim + 3), slot_vector[x], mf_dim, bs, off);

/*
    switch (off) {
      case 0:
        cur->slot = (float)slot_vector[x];
        cur->mf_dim = mf_dim;
        paddle::platform::CudaAtomicAdd(&(cur->show),
                                        *(ptr + off));
        break;
      case 1:
        paddle::platform::CudaAtomicAdd(&(cur->clk),
                                        *(ptr + off));
        break;
      case 2:
        paddle::platform::CudaAtomicAdd(&(cur->lr_g),
                                        *(ptr + off) * -1. * bs);
        break;
      default:
        int embedx_idx = off - 3;

        paddle::platform::CudaAtomicAdd(
            &(cur->mf_g[embedx_idx]),
            *(ptr + off) * -1. * bs);
        break;
    }
*/

  }
}

#define SUM_GRAD_VALUE                                             \
  for (uint32_t j = 0; j < count; ++j) {                           \
    const uint32_t& pos = d_sort_idx[start + j];                   \
    const int& x = key2slot[pos];                                  \
    y = pos - slot_lens[x];                                        \
    val += *(reinterpret_cast<float*>(src[x] + y * (mf_dim + 3) + off)); \
  }

template <typename GPUAccessor>
__global__ void PushMergeCopy(const size_t N,
                              const uint64_t* total_keys,
                              float* dest,
                              float** src,
                              const int hidden,
                              const int bs,
                              const int* slot_vector,
                              const int* slot_dims,
                              const int64_t* slot_lens,
                              const int* key2slot,
                              const uint32_t* d_sort_idx,
                              const uint32_t* d_sort_offset,
                              const uint32_t* d_sort_cnt,
                              size_t grad_value_size,
                              GPUAccessor gpu_accessor) {
  CUDA_KERNEL_LOOP(idx, N) {
    int i = idx / hidden;
    int off = idx % hidden;
    // filter 0 keys
    // FeaturePushValue* cur = (FeaturePushValue*)((char*)dest + i * grad_value_size);
    float* cur = (float*)((char*)dest + i * grad_value_size);
    
    const uint32_t& start = d_sort_offset[i]; // 前面有多少个feasign，包括重复的数量
    const uint32_t& count = d_sort_cnt[i];    // 当前feasign的数量 
    const uint32_t& pos = d_sort_idx[start];  // 排序后第一个这个feasign原来的index,

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];
    int mf_dim = slot_dims[x] - 3;
    if (off - 3 >= mf_dim) {
      return;
    }

    if (total_keys[i] == 0) {
      switch (off) {
        case 0:
          cur[gpu_accessor.common_push_value.SlotIndex()] = 0;
          cur[gpu_accessor.common_push_value.MfDimIndex()] = 0;
          cur[gpu_accessor.common_push_value.ShowIndex()] = 0.0;
          break;
        case 1:
          cur[gpu_accessor.common_push_value.ClickIndex()] = 0.0;
          break;
        case 2:
          cur[gpu_accessor.common_push_value.EmbedGIndex()] = 0.0;
          break;
        default:
          cur[gpu_accessor.common_push_value.EmbedxGIndex() + off - 3] = 0.0;
          break;
      }
      return;
    }

    double val = 0.0;

    switch (off) {
      case 0:
        cur[gpu_accessor.common_push_value.SlotIndex()] = (float)slot_vector[x];
        cur[gpu_accessor.common_push_value.MfDimIndex()] = mf_dim;
        SUM_GRAD_VALUE
        cur[gpu_accessor.common_push_value.ShowIndex()] = val;
        break;
      case 1:
        SUM_GRAD_VALUE
        cur[gpu_accessor.common_push_value.ClickIndex()] = val;
        break;
      case 2:
        SUM_GRAD_VALUE
        cur[gpu_accessor.common_push_value.EmbedGIndex()] = val * -1. * bs;
        break;
      default:
        int embedx_idx = off - 3;
        if (mf_dim <= embedx_idx) {
          cur[gpu_accessor.common_push_value.EmbedxGIndex() + embedx_idx] = 0.0;
          return;
        }
        SUM_GRAD_VALUE
        cur[gpu_accessor.common_push_value.EmbedxGIndex() + embedx_idx] = val * -1. * bs;
        break;
    }

    // gpu_accessor.GradientSelectV3(cur, src, slot_vector[x], mf_dim, bs,
    //                              total_keys[i], off, start, count, d_sort_idx, key2slot, slot_lens);
    /*
    if (total_keys[i] == 0) {
      switch (off) {
        case 0:
          cur->slot = 0;
          cur->mf_dim = 0;
          cur->show = 0.0;
          break;
        case 1:
          cur->clk = 0.0;
          break;
        case 2:
          cur->lr_g = 0.0;
          break;
        default:
          
          cur->mf_g[off - 3] = 0.0;
          break;
      }
      return;
    }

    double val = 0.0;

    switch (off) {
      case 0:
        cur->slot = (float)slot_vector[x];
        cur->mf_dim = mf_dim;
        SUM_GRAD_VALUE
        cur->show = val;
        break;
      case 1:
        SUM_GRAD_VALUE
        cur->clk = val;
        break;
      case 2:
        SUM_GRAD_VALUE
        cur->lr_g = val * -1. * bs;
        break;
      default:
        int embedx_idx = off - 3;
        if (mf_dim < embedx_idx) {
          cur->mf_g[embedx_idx] = 0.0;
          return;
        }
        SUM_GRAD_VALUE
        cur->mf_g[embedx_idx] = val * -1. * bs;
        break;
    }
   */

  }
}
/*
template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPush(const paddle::platform::Place& place,
                                               const uint64_t* total_keys,
                                               float** grad_values,
                                               float* total_grad_values_gpu,
                                               const int* slots,
                                               const int64_t* slot_lens,
                                               const int hidden_size,
                                               const int64_t total_length,
                                               const int64_t dedup_length,
                                               const int batch_size,
                                               const int* slot_dims,
                                               const int* key2slot,
                                               const uint32_t* d_restore_idx,
                                               const size_t grad_value_size) {
    CopyForPushDedupImpl(place,
                         total_keys,
                         grad_values,
                         total_grad_values_gpu,
                         slots,
                         slot_lens,
                         hidden_size,
                         total_length,
                         dedup_length,
                         batch_size,
                         slot_dims,
                         key2slot,
                         d_restore_idx,
                         grad_value_size);
  
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPush(const paddle::platform::Place& place,
                                               const uint64_t* total_keys,
                                               float** grad_values,
                                               float* total_grad_values_gpu,
                                               const int* slots,
                                               const int64_t* slot_lens,
                                               const int hidden_size,
                                               const int64_t total_length,
                                               const int64_t dedup_length,
                                               const int batch_size,
                                               const int* slot_dims,
                                               const int* key2slot,
                                               const uint32_t* gpu_sort_idx,
                                               const uint32_t* gpu_sort_offset,
                                               const uint32_t* gpu_sort_lens,
                                               const size_t grad_value_size) {
  CopyForPushDedupImpl(place,
                       total_keys,
                       grad_values,
                       total_grad_values_gpu,
                       slots,
                       slot_lens,
                       hidden_size,
                       total_length,
                       dedup_length,
                       batch_size,
                       slot_dims,
                       key2slot,
                       gpu_sort_idx,
                       gpu_sort_offset,
                       gpu_sort_lens,
                       grad_value_size);
}
*/


template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPush(const paddle::platform::Place& place,
                                                        const uint64_t* total_keys,
                                                        float** grad_values,
                                                        float* total_grad_values_gpu,
                                                        const int* slots,
                                                        const int64_t* slot_lens,
                                                        const int hidden_size,
                                                        const int64_t total_length,
                                                        const int64_t dedup_length,
                                                        const int batch_size,
                                                        const int* slot_dims,
                                                        const int* key2slot,
                                                        const uint32_t* d_restore_idx,
                                                        const size_t grad_value_size) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                    paddle::platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  // 初始化为0
  cudaMemsetAsync(
      total_grad_values_gpu, 0, dedup_length * grad_value_size, stream);
  
  size_t N = total_length * hidden_size;

  PushMergeCopyAtomic<<<CUDA_BLOCK(N), stream>>>(
      N,
      total_keys,
      total_grad_values_gpu,
      grad_values,
      hidden_size,
      batch_size,
      slots,
      slot_dims,
      slot_lens,
      key2slot,
      d_restore_idx,
      grad_value_size,
      gpu_accessor_);

  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPush(const paddle::platform::Place& place,
                                               const uint64_t* total_keys,
                                               float** grad_values,
                                               float* total_grad_values_gpu,
                                               const int* slots,
                                               const int64_t* slot_lens,
                                               const int hidden_size,
                                               const int64_t total_length,
                                               const int64_t dedup_length,
                                               const int batch_size,
                                               const int* slot_dims,
                                               const int* key2slot,
                                               const uint32_t* gpu_sort_idx,
                                               const uint32_t* gpu_sort_offset,
                                               const uint32_t* gpu_sort_lens,
                                               const size_t grad_value_size) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                    paddle::platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  // merge all grad to one
  size_t N = dedup_length * hidden_size;
  PushMergeCopy<<<CUDA_BLOCK(N), stream>>>(N,
                                           total_keys,
                                           total_grad_values_gpu,
                                           grad_values,
                                           hidden_size,
                                           batch_size,
                                           slots,
                                           slot_dims,
                                           slot_lens,
                                           key2slot,
                                           gpu_sort_idx,
                                           gpu_sort_offset,
                                           gpu_sort_lens,
                                           grad_value_size,
                                           gpu_accessor_);
  cudaStreamSynchronize(stream);
}
// ======= hbm optimized ===========================

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPush(const paddle::platform::Place& place,
                                               const std::vector<const float*>& grad_values,
                                               float* total_grad_values_gpu,
                                               const std::vector<int64_t>& slot_lengths,
                                               const uint64_t total_length,
                                               const int batch_size,
                                               size_t grad_value_size,
                                               std::vector<int>& slot_vector,
                                               std::vector<int>& slot_mf_dim_vector) {

  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  auto buf_grad_value =
      memory::Alloc(place, grad_values.size() * sizeof(float*));
  auto buf_length =
      memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
  auto buf_slot_vector =
      memory::Alloc(place, slot_lengths_lod.size() * sizeof(int));
  auto buf_mf_dim_vector =
      memory::Alloc(place, slot_lengths_lod.size() * sizeof(int));

  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());
  int* d_mf_dim_vector = reinterpret_cast<int*>(buf_mf_dim_vector->ptr());

  cudaMemcpy(gpu_values, grad_values.data(),
             grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len, slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector, slot_vector.data(),
             slot_lengths_lod.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mf_dim_vector, slot_mf_dim_vector.data(),
             slot_lengths_lod.size() * sizeof(int), cudaMemcpyHostToDevice);

  PushCopyWithPool<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
      total_grad_values_gpu, gpu_values, gpu_len, slot_lengths.size(),
      total_length, batch_size, d_slot_vector, d_mf_dim_vector,
      grad_value_size, gpu_accessor_);

  cudaStreamSynchronize(stream);
}

#ifdef PADDLE_WITH_PSLIB
template class AccessorWrapper<CommonFeatureValueAccessor>;
#endif



}  // end namespace framework
}  // end namespace paddle
#endif
