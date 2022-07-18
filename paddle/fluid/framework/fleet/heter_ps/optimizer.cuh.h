/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <curand_kernel.h>
#include <vector>
#include "optimizer_conf.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

template <typename FVAccessor>
class Optimizer {
 public:
  Optimizer() {}
  ~Optimizer() {}

  virtual __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr,
                                     const float* grad,
                                     curandState& state) = 0;
};

template <typename FVAccessor>
class SparseAdagradOptimizer : public Optimizer<FVAccessor> {

public:
  SparseAdagradOptimizer() {}
  SparseAdagradOptimizer(FVAccessor& feature_value_accessor): Optimizer<FVAccessor>() {
    feature_value_accessor_ = feature_value_accessor;

/* use VLOG instead of std::cout 
    std::cout << "=============Hashtable GPUAccesor FeatureValue INFO=========" << std::endl;
    std::cout << "optimizer type:" << feature_value_accessor_.common_feature_value.optimizer_type_ << std::endl;
    std::cout << "Dim:" << feature_value_accessor_.common_feature_value.Dim() << std::endl;
    std::cout << "EmbedDim:" << feature_value_accessor_.common_feature_value.EmbedDim() << std::endl;
    std::cout << "EmbedXDim:" << feature_value_accessor_.common_feature_value.EmbedXDim() << std::endl;
    std::cout << "EmbedWDim:" << feature_value_accessor_.common_feature_value.EmbedWDim() << std::endl;
    std::cout << "CpuPtrIndex:" << feature_value_accessor_.common_feature_value.CpuPtrIndex() << std::endl;
    std::cout << "DeltaScoreIndex:" << feature_value_accessor_.common_feature_value.DeltaScoreIndex() << std::endl;
    std::cout << "ShowIndex:" << feature_value_accessor_.common_feature_value.ShowIndex() << std::endl;
    std::cout << "ClickIndex:" << feature_value_accessor_.common_feature_value.ClickIndex() << std::endl;
    std::cout << "EmbedWIndex:" << feature_value_accessor_.common_feature_value.EmbedWIndex() << std::endl;
    std::cout << "EmbedG2SumIndex:" << feature_value_accessor_.common_feature_value.EmbedG2SumIndex() << std::endl;
    std::cout << "SlotIndex:" << feature_value_accessor_.common_feature_value.SlotIndex() << std::endl;
    std::cout << "MfDimIndex:" << feature_value_accessor_.common_feature_value.MfDimIndex() << std::endl;
    std::cout << "MfSizeIndex:" << feature_value_accessor_.common_feature_value.MfSizeIndex() << std::endl;
    std::cout << "EmbedxG2SumIndex:" << feature_value_accessor_.common_feature_value.EmbedxG2SumIndex() << std::endl;
    std::cout << "EmbedxWIndex:" << feature_value_accessor_.common_feature_value.EmbedxWIndex() << std::endl;
    std::cout << "=============Hashtable GPUAccesor FeatureValue INFO=========" << std::endl;
    std::cout << "=============Hashtable GPUAccesor PushValue INFO=========" << std::endl;
    std::cout << "push slotIndex:" << feature_value_accessor_.common_push_value.SlotIndex() << std::endl;
    std::cout << "push showIndex:" << feature_value_accessor_.common_push_value.ShowIndex() << std::endl;
    std::cout << "push ClickIndex:" << feature_value_accessor_.common_push_value.ClickIndex() << std::endl;
    std::cout << "push MfDimIndex:" << feature_value_accessor_.common_push_value.MfDimIndex() << std::endl;
    std::cout << "push EmbedGIndex:" << feature_value_accessor_.common_push_value.EmbedGIndex() << std::endl;
    std::cout << "push EmbedxGIndex:" << feature_value_accessor_.common_push_value.EmbedxGIndex() << std::endl;
    std::cout << "=============Hashtable GPUAccesor PushValue INFO=========" << std::endl;
*/

  }

  ~SparseAdagradOptimizer() {}

  __device__ void update_lr(const OptimizerConfig& optimizer_config, float& w, float& g2sum, float g, float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config.learning_rate *
                   sqrt(optimizer_config.initial_g2sum /
                        (optimizer_config.initial_g2sum + g2sum));
    double scaled_grad = g / scale;

    w += scaled_grad * ratio;

    if (w < optimizer_config.min_bound) w = optimizer_config.min_bound;
    if (w > optimizer_config.max_bound) w = optimizer_config.max_bound;

    add_g2sum += scaled_grad * scaled_grad;

    g2sum += add_g2sum;
  }

  __device__ void update_mf(const OptimizerConfig& optimizer_config, int n, float* w, float& g2sum, const float* g,
                            float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config.mf_learning_rate *
                   sqrt(optimizer_config.mf_initial_g2sum /
                        (optimizer_config.mf_initial_g2sum + g2sum));
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      w[i] += scaled_grad * ratio;

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;
      add_g2sum += scaled_grad * scaled_grad;
    }

    g2sum += add_g2sum / n;
  }

 /*
  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config, ValType* ptr, const GradType* grad) {
    ptr->slot = grad.slot;
    ptr->show += grad.show;
    ptr->clk += grad.clk;
    ptr->delta_score += optimizer_config.nonclk_coeff * (grad.show - grad.clk) +
                       optimizer_config.clk_coeff * grad.clk;

    update_lr(optimizer_config, ptr->lr, ptr->lr_g2sum, grad.lr_g, grad.show);
    // ptr->mf_dim = grad.mf_dim;

    if (ptr->mf_size == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * (ptr->show - ptr->clk) +
              optimizer_config.clk_coeff * ptr->clk) {
        ptr->mf_size = ptr->mf_dim + 1;
        ptr->mf[0] = 0;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < ptr->mf_dim; ++i) {
          ptr->mf[i + 1] =
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_mf(optimizer_config, ptr->mf_dim, &(ptr->mf[1]), ptr->mf[0], grad.mf_g,
                grad.show);  // for local test
    }
  }
  */

  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config, float* ptr, const float* grad, curandState& state) {

    // ptr->slot = grad.slot;
    // ptr->show += grad.show;
    // ptr->clk += grad.clk;
   
    // int grad_show_index = feature_value_accessor_.common_push_value.ShowIndex();
    // int grad_clk_index = feature_value_accessor_.common_push_value.ClickIndex();

    // float grad_show = grad[feature_value_accessor_.common_push_value.ShowIndex()];
    // float grad_clk = grad[feature_value_accessor_.common_push_value.ClickIndex()];

    float grad_show = grad[1];
    float grad_clk = grad[2];
      
    // int ptr_slot_index = feature_value_accessor_.common_feature_value.SlotIndex();
    // int ptr_slot = (int)ptr[feature_value_accessor_.common_feature_value.SlotIndex()];

    ptr[7] = grad[0];
    // ptr[feature_value_accessor_.common_feature_value.SlotIndex()] =
    //    grad[feature_value_accessor_.common_push_value.SlotIndex()];

    ptr[3] += grad_show;
    ptr[4] += grad_clk;
    //ptr[feature_value_accessor_.common_feature_value.ShowIndex()] += grad_show;
    //ptr[feature_value_accessor_.common_feature_value.ClickIndex()] += grad_clk;

    // ptr[feature_value_accessor_.common_feature_value.DeltaScoreIndex()] +=
    ptr[2] += optimizer_config.nonclk_coeff * (grad_show - grad_clk) +
                       optimizer_config.clk_coeff * grad_clk;
   
    // float ptr_lr = ptr[feature_value_accessor_.common_feature_value.EmbedWIndex()];
    // float ptr_lr_g2sum = ptr[feature_value_accessor_.common_feature_value.EmbedG2SumIndex()];

    float ptr_show = ptr[3];
    float ptr_clk = ptr[4];

    // float ptr_show = ptr[feature_value_accessor_.common_feature_value.ShowIndex()];
    // float ptr_clk = ptr[feature_value_accessor_.common_feature_value.ClickIndex()];
    // float grad_lr_g = grad[feature_value_accessor_.common_push_value.EmbedGIndex()];
    float grad_lr_g = grad[4];

    // float& ptr_mf_size = ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()];
    float ptr_mf_size = ptr[9];

    // int mf_dim_index = feature_value_accessor_.common_feature_value.MfDimIndex();
    // int ptr_mf_dim = (int)(ptr[feature_value_accessor_.common_feature_value.MfDimIndex()]);
    float ptr_mf_dim = ptr[8];
    
    // int ptr_slot = (int)ptr[feature_value_accessor_.common_feature_value.SlotIndex()];

    update_lr(
        optimizer_config,
        ptr[5],
        ptr[6],
        grad_lr_g,
        grad_show);

    // ptr->mf_dim = grad.mf_dim;

    if (ptr_mf_size == (float)0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * (ptr_show - ptr_clk) +
              optimizer_config.clk_coeff * ptr_clk) {

        ptr[9] = ptr_mf_dim + 1;
        // ptr_mf_size =  feature_value_accessor_.common_feature_value.MFSize(ptr_mf_dim) / sizeof(float);
        // ptr->mf_size = ptr->mf_dim + 1;
        // ptr->mf[0] = 0;
        ptr[10] = 0;
        // ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex()] = 0;
        // int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        //curandState state;
        //curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < ptr_mf_dim; ++i) {
          // ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + i + 1] =
          ptr[10 + i + 1] = (curand_uniform(&state)) * optimizer_config.mf_initial_range;
          // ptr->mf[i + 1] =
          //    (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_mf(
          optimizer_config,
          ptr_mf_dim,
          &ptr[11],
          ptr[10],
          &grad[5],
          grad_show);
          // &(ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + 1]),
          // ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex()],
          // &(grad[feature_value_accessor_.common_push_value.EmbedxGIndex()]),
          // grad_show);  // for local test
    }
  }

private:
  FVAccessor feature_value_accessor_;
};


}  // end namespace framework
}  // end namespace paddle
#endif
