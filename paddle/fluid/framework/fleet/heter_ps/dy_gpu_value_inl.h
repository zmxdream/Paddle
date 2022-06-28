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

#ifdef PADDLE_WITH_HETERPS

#include "paddle/fluid/framework/fleet/heter_ps/feature_value_inl.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"

namespace paddle {
namespace framework {

struct DyGpuValue {
  float delta_score;
  float show;
  float clk;
  int slot;
  float lr;
  float lr_g2sum;
  int mf_size;
  int mf_dim;
  uint64_t cpu_ptr;
  float mf[0];
  __host__ __device__ __forceinline__ DyGpuValue() {
    delta_score = 0;
    show = 0;
    clk = 0;
    slot = -1;
    lr = 0;
    lr_g2sum = 0;
    mf_size = 0;
    mf_dim = 0;
    cpu_ptr = 0;
  }
  __device__ __forceinline__ void operator=(const DyGpuValue& in) {
    delta_score = in.delta_score;
    show = in.show;
    clk = in.clk;
    slot = in.slot;
    lr = in.lr;
    lr_g2sum = in.lr_g2sum;
    mf_size = in.mf_size;
    mf_dim = in.mf_dim;
    cpu_ptr = in.cpu_ptr;
    for (int i = 0; i < mf_dim + 1; i++) {
      mf[i] = in.mf[i];
    }
  }
  __device__ __forceinline__ void to_cvm(float* des, int dim) {
    des[0] = show;
    des[1] = clk;
    des[2] = lr;
    if (mf_size == 0) {
      for (int i = 0; i < dim; i++) {
        des[3 + i] = 0;
      }
    } else {
      for (int i = 0; i < dim; i++) {
        des[3 + i] = mf[1 + i];
      }
    }
  }
};

struct DyGpuPushValue {
  float show;
  float clk;
  int slot;
  float lr_g;
  int mf_dim;
  float mf_g[0];
  __device__ __forceinline__ void from_grad(const float* grad, int dim, int slot_id, int batch_size) {
    this->slot = slot_id;
    this->mf_dim = dim;
    this->show = grad[0];
    this->clk = grad[1];
    this->lr_g = grad[2] * -1. * batch_size;
    for (int j = 0; j < dim; j++) {
      this->mf_g[j] = grad[3 + j] * -1. * batch_size;
    }
  }
  __device__ __forceinline__ DyGpuPushValue& operator+=(const DyGpuPushValue& input) {
    show += input.show;
    clk += input.clk;
    lr_g += input.lr_g;
    for (int i = 0; i < input.mf_dim; i++) {
      mf_g[i] += input.mf_g[i];
    }
    return *this;
  }
  __device__ __forceinline__ void operator=(const DyGpuPushValue& input) {
    show = input.show;
    clk = input.clk;
    slot = input.slot;
    lr_g = input.lr_g;
    mf_dim = input.mf_dim;
    for (int i = 0; i < mf_dim; i++) {
     mf_g[i] = input.mf_g[i];
    }
  }
};

template <>
class Optimizer<DyGpuValue, DyGpuPushValue> {
 public:
  Optimizer() {}

  ~Optimizer() {}

  void initialize() {}

  __device__ void update_lr(float& w, float& g2sum, float g, float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config::learning_rate *
                   sqrt(optimizer_config::initial_g2sum /
                        (optimizer_config::initial_g2sum + g2sum));
    double scaled_grad = g / scale;
    w += scaled_grad * ratio;
    if (w < optimizer_config::min_bound) w = optimizer_config::min_bound;
    if (w > optimizer_config::max_bound) w = optimizer_config::max_bound;
    add_g2sum += scaled_grad * scaled_grad;
    g2sum += add_g2sum;
  }

  __device__ void update_mf(int n, float* w, float& g2sum, const float* g,
                            float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config::mf_learning_rate *
                   sqrt(optimizer_config::mf_initial_g2sum /
                        (optimizer_config::mf_initial_g2sum + g2sum));
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;
      w[i] += scaled_grad * ratio;
      if (w[i] < optimizer_config::mf_min_bound)
        w[i] = optimizer_config::mf_min_bound;
      if (w[i] > optimizer_config::mf_max_bound)
        w[i] = optimizer_config::mf_max_bound;
      add_g2sum += scaled_grad * scaled_grad;
    }
    g2sum += add_g2sum / n;
  }
  
  __device__ void update_value(DyGpuValue* ptr, const DyGpuPushValue& grad, curandState& state) {
    ptr->slot = grad.slot;
    ptr->show += grad.show;
    ptr->clk += grad.clk;
    ptr->delta_score += optimizer_config::nonclk_coeff * (grad.show - grad.clk) +
                       optimizer_config::clk_coeff * grad.clk;

    update_lr(ptr->lr, ptr->lr_g2sum, grad.lr_g, grad.show);

    if (ptr->mf_size == 0) {
      if (optimizer_config::mf_create_thresholds <=
          optimizer_config::nonclk_coeff * (ptr->show - ptr->clk) +
              optimizer_config::clk_coeff * ptr->clk) {
        ptr->mf_size = ptr->mf_dim + 1;
        ptr->mf[0] = 0;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = 0; i < ptr->mf_dim; ++i) {
          ptr->mf[i + 1] =
              (curand_uniform(&state)) * optimizer_config::mf_initial_range;
        }
      }
    } else {
      update_mf(ptr->mf_dim, &(ptr->mf[1]), ptr->mf[0], grad.mf_g,
                grad.show);  // for local test
    }
  }
};


class T_DyGpuValue_DownpourCtrDymfAccessor : public ValueTransforImp {
public:
  virtual int get_gpu_value_size(int dim_size) {
    int ret = sizeof(DyGpuValue) + (dim_size + 1) * sizeof(float);
    return TYPE_ALIGN(8, ret);
  }
  virtual int get_gpu_push_value_size(int dim_size) {
    int ret = sizeof(DyGpuPushValue) + (dim_size) * sizeof(float);
    return TYPE_ALIGN(8, ret);
  }
  virtual void value_cpu_to_gpu(void* cpu, void* gpu, int dim_size) {
#ifdef PADDLE_WITH_PSLIB
    paddle::ps::DownpourFixedFeatureValue* cpu_value = (paddle::ps::DownpourFixedFeatureValue*)cpu;
    DyGpuValue* gpu_value = (DyGpuValue*)gpu;
    const float* ptr_cpu_data = cpu_value->data();
    size_t dim = cpu_value->size();
    uint64_t tmp_aa = (uint64_t)(cpu);
    gpu_value->delta_score = ptr_cpu_data[1];
    gpu_value->show = ptr_cpu_data[2];
    gpu_value->clk = ptr_cpu_data[3];
    gpu_value->slot = int(ptr_cpu_data[6]);
    gpu_value->lr = ptr_cpu_data[4];
    gpu_value->lr_g2sum = ptr_cpu_data[5];
    gpu_value->cpu_ptr = (uint64_t)(cpu);
    gpu_value->mf_dim = dim_size;
    if (dim > 8) {
        gpu_value->mf_size = dim_size + 1;
        for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
          gpu_value->mf[x] = ptr_cpu_data[x + 8];
        }
      } else {
        gpu_value->mf_size = 0;
        for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
          gpu_value->mf[x] = 0 ;
        }
      }
#endif
  }
  virtual void value_gpu_to_cpu(void* gpu) {
#ifdef PADDLE_WITH_PSLIB
    DyGpuValue* gpu_value = (DyGpuValue*)gpu;
    paddle::ps::DownpourFixedFeatureValue& cpu_fix = *((paddle::ps::DownpourFixedFeatureValue*)(gpu_value->cpu_ptr));
    if (gpu_value->mf_size > 0) {
      cpu_fix.resize(8 + 1 + gpu_value->mf_dim);
    }
    float* cpu_value = cpu_fix.data();
    cpu_value[1] = gpu_value->delta_score;
    cpu_value[2] = gpu_value->show;
    cpu_value[3] = gpu_value->clk;
    cpu_value[4] = gpu_value->lr;
    cpu_value[5] = gpu_value->lr_g2sum;
    cpu_value[6] = gpu_value->slot;
    if (gpu_value->mf_size > 0) {
       for (int x = 0; x < gpu_value->mf_dim + 1; x++) {
         cpu_value[x + 8] = gpu_value->mf[x];
       }
    }
#endif
  }
  virtual void value_to_cvm(float** gpu_cvm,
                            const void* gpu_value,
                            FeatureKey** gpu_keys,
                            const int slot_num,
                            const int64_t* key_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            cudaStream_t stream
                            ) {
    value_to_cvm_impl(gpu_cvm, (DyGpuValue*)gpu_value, gpu_keys, slot_num, key_len,
                        slot_dim, total_length, hidden_size, value_size, stream);
  }
  virtual void grad_to_push(void* push_value,
                            float** grad_value,
                            const int slot_num,
                            const int64_t* grad_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            int batch_size,
                            const int* slot_vector,
                            cudaStream_t stream
                            ) {
    grad_to_push_impl((DyGpuPushValue*)push_value, grad_value, slot_num, grad_len, slot_dim,
                       total_length, hidden_size, value_size, batch_size, slot_vector, stream);
  }
};

}
}

#endif