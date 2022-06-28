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
  cudaMemcpyToSymbol(optimizer_config::nonclk_coeff, &nonclk_coeff,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::clk_coeff, &clk_coeff, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::min_bound, &min_bound, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::max_bound, &max_bound, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::learning_rate, &learning_rate,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::initial_g2sum, &initial_g2sum,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::initial_range, &initial_range,
                     sizeof(float));
}

void PSGPUWrapper::SetEmbedxSGD(float mf_create_thresholds,
                                float mf_learning_rate, float mf_initial_g2sum,
                                float mf_initial_range, float mf_min_bound,
                                float mf_max_bound) {
  cudaMemcpyToSymbol(optimizer_config::mf_create_thresholds,
                     &mf_create_thresholds, sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_learning_rate, &mf_learning_rate,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_initial_g2sum, &mf_initial_g2sum,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_initial_range, &mf_initial_range,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_min_bound, &mf_min_bound,
                     sizeof(float));
  cudaMemcpyToSymbol(optimizer_config::mf_max_bound, &mf_max_bound,
                     sizeof(float));
}

}  // end namespace framework
}  // end namespace paddle
#endif
