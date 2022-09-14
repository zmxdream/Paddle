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
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

/*
__global__ void PullCopy(float** dest, const FeatureValue* src,
                         const int64_t* len, int hidden, int slot_num,
                         int total_len, uint64_t** keys) {
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
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * hidden) = 0;
      *(dest[x] + y * hidden + 1) = 0;
      *(dest[x] + y * hidden + 2) = 0;
    } else {
      *(dest[x] + y * hidden) = (src + i)->show;
      *(dest[x] + y * hidden + 1) = (src + i)->clk;
      *(dest[x] + y * hidden + 2) = (src + i)->lr;
    }
    if ((src + i)->mf_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = (src + i)->mf[1 + j];
      }
    }
  }
}

__global__ void PullCopy(float** dest, const FeatureValue* src,
                         const int64_t* len, int slot_num, int total_len,
                         uint64_t** keys, uint64_t max_val_size, int* gpu_dim) {
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
    int cur_dim = gpu_dim[x] - 3;
    FeatureValue* feature_value_ptr =
        (FeatureValue*)((char*)src + uint64_t(i) * uint64_t(max_val_size));
    
    int mf_dim = feature_value_ptr->mf_dim;
    mf_dim = gpu_dim[x] - 3;
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * (cur_dim + 3)) = 0;
      *(dest[x] + y * (cur_dim + 3) + 1) = 0;
      *(dest[x] + y * (cur_dim + 3) + 2) = 0;
    } else {
      *(dest[x] + y * (mf_dim + 3)) = feature_value_ptr->show;
      *(dest[x] + y * (mf_dim + 3) + 1) = feature_value_ptr->clk;
      *(dest[x] + y * (mf_dim + 3) + 2) = feature_value_ptr->lr;
    }
    if ((feature_value_ptr)->mf_size == 0 || *(keys[x] + y) == 0) {
      if (*(keys[x] + y) == 0) {
        for (int j = 0; j < cur_dim; j++) {
          *(dest[x] + y * (cur_dim + 3) + 3 + j) = 0;
        }
      } else {
        for (int j = 0; j < mf_dim; j++) {
          *(dest[x] + y * (mf_dim + 3) + 3 + j) = 0;
        }
      }
      
    } else {
      for (int j = 0; j < mf_dim; j++) {
        *(dest[x] + y * (mf_dim + 3) + 3 + j) = feature_value_ptr->mf[1 + j];
      }
    }
  }
}
*/

__global__ void CopyKeysKernel(uint64_t** src_keys, uint64_t* dest_total_keys,
                               const int64_t* len, int slot_num,
                               int total_len) {
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
    dest_total_keys[i] = src_keys[x][y];
  }
}

void PSGPUWrapper::CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys, uint64_t* total_keys,
                            const int64_t* gpu_len, int slot_num,
                            int total_len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  CopyKeysKernel<<<(total_len + 1024 - 1) / 1024, 1024, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::SetSparseSGD(float nonclk_coeff, float clk_coeff,
                                float min_bound, float max_bound,
                                float learning_rate, float initial_g2sum,
                                float initial_range) {
  optimizer_config_.set_sparse_sgd(nonclk_coeff,
                                  clk_coeff,
                                  min_bound,
                                  max_bound,
                                  learning_rate,
                                  initial_g2sum,
                                  initial_range);
}

void PSGPUWrapper::SetEmbedxSGD(float mf_create_thresholds,
                                float mf_learning_rate, float mf_initial_g2sum,
                                float mf_initial_range, float mf_min_bound,
                                float mf_max_bound) {
  optimizer_config_.set_embedx_sgd(mf_create_thresholds,
                                  mf_learning_rate,
                                  mf_initial_g2sum,
                                  mf_initial_range,
                                  mf_min_bound,
                                  mf_max_bound);
}

}  // end namespace framework
}  // end namespace paddle
#endif
