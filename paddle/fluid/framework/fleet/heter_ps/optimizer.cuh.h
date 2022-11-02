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

template <typename GPUAccessor>
class SparseAdagradOptimizer {

public:
  SparseAdagradOptimizer() {}
  SparseAdagradOptimizer(GPUAccessor& gpu_accessor) {
    gpu_accessor_ = gpu_accessor;
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

    float grad_show = grad[gpu_accessor_.common_push_value.ShowIndex()];
    float grad_clk = grad[gpu_accessor_.common_push_value.ClickIndex()];

    ptr[gpu_accessor_.common_feature_value.SlotIndex()] =
        grad[gpu_accessor_.common_push_value.SlotIndex()];

    ptr[gpu_accessor_.common_feature_value.ShowIndex()] += grad_show;
    ptr[gpu_accessor_.common_feature_value.ClickIndex()] += grad_clk;

    ptr[gpu_accessor_.common_feature_value.DeltaScoreIndex()] +=
      optimizer_config.nonclk_coeff * (grad_show - grad_clk) +
                       optimizer_config.clk_coeff * grad_clk;
   
    float ptr_show = ptr[gpu_accessor_.common_feature_value.ShowIndex()];
    float ptr_clk = ptr[gpu_accessor_.common_feature_value.ClickIndex()];
    float grad_lr_g = grad[gpu_accessor_.common_push_value.EmbedGIndex()];

    float ptr_mf_size = ptr[gpu_accessor_.common_feature_value.MfSizeIndex()];

    int ptr_mf_dim = (int)(ptr[gpu_accessor_.common_feature_value.MfDimIndex()]);
    
    update_lr(
        optimizer_config,
        ptr[gpu_accessor_.common_feature_value.EmbedWIndex()],
        ptr[gpu_accessor_.common_feature_value.EmbedG2SumIndex()],
        grad_lr_g,
        grad_show);

    if (ptr_mf_size == (float)0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * (ptr_show - ptr_clk) +
              optimizer_config.clk_coeff * ptr_clk) {

        ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] =
          gpu_accessor_.common_feature_value.MFSize(ptr_mf_dim) / sizeof(float);
        ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex()] = 0;
        for (int i = 0; i < ptr_mf_dim; ++i) {
          ptr[gpu_accessor_.common_feature_value.EmbedxWIndex() + i] =
            (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_mf(
          optimizer_config,
          ptr_mf_dim,
          &ptr[gpu_accessor_.common_feature_value.EmbedxWIndex()],
          ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex()],
          &grad[gpu_accessor_.common_push_value.EmbedxGIndex()],
          grad_show);
    }
  }

private:
  GPUAccessor gpu_accessor_;
};

template <typename GPUAccessor>
class StdAdagradOptimizer {

public:
  StdAdagradOptimizer() {}
  StdAdagradOptimizer(GPUAccessor& gpu_accessor) {
    gpu_accessor_ = gpu_accessor;
  }

  ~StdAdagradOptimizer() {}

  __device__ void update_lr(const OptimizerConfig& optimizer_config, float& w, float& g2sum, float g, float scale) {
    // double add_g2sum = 0;
    double ratio = optimizer_config.learning_rate *
                   sqrt(optimizer_config.initial_g2sum /
                        (optimizer_config.initial_g2sum + g2sum));
    double scaled_grad = g / scale;

    w += scaled_grad * ratio;

    if (w < optimizer_config.min_bound) w = optimizer_config.min_bound;
    if (w > optimizer_config.max_bound) w = optimizer_config.max_bound;

    g2sum += scaled_grad * scaled_grad;

    // g2sum += add_g2sum;
  }

  __device__ int g2sum_index() {
    return 0; 
  }

  __device__ void update_mf(const OptimizerConfig& optimizer_config, int n, float* w, float* sgd, const float* g,
                            float scale) {
    // double add_g2sum = 0;
    // double ratio = optimizer_config.mf_learning_rate *
    //               sqrt(optimizer_config.mf_initial_g2sum /
    //                    (optimizer_config.mf_initial_g2sum + g2sum));
    for (int i = 0; i < n; ++i) {
      float& g2sum = sgd[g2sum_index() + i];
      double scaled_grad = g[i] / scale;

      double ratio = optimizer_config.mf_learning_rate *
                   sqrt(optimizer_config.mf_initial_g2sum /
                        (optimizer_config.mf_initial_g2sum + g2sum));
      
      w[i] += scaled_grad * ratio;

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;

      g2sum += scaled_grad * scaled_grad;
    }

    // g2sum += add_g2sum / n;
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
    float grad_show = grad[gpu_accessor_.common_push_value.ShowIndex()];
    float grad_clk = grad[gpu_accessor_.common_push_value.ClickIndex()];

    ptr[gpu_accessor_.common_feature_value.SlotIndex()] =
        grad[gpu_accessor_.common_push_value.SlotIndex()];

    ptr[gpu_accessor_.common_feature_value.ShowIndex()] += grad_show;
    ptr[gpu_accessor_.common_feature_value.ClickIndex()] += grad_clk;

    ptr[gpu_accessor_.common_feature_value.DeltaScoreIndex()] +=
      optimizer_config.nonclk_coeff * (grad_show - grad_clk) +
                       optimizer_config.clk_coeff * grad_clk;
   
    float ptr_show = ptr[gpu_accessor_.common_feature_value.ShowIndex()];
    float ptr_clk = ptr[gpu_accessor_.common_feature_value.ClickIndex()];
    float grad_lr_g = grad[gpu_accessor_.common_push_value.EmbedGIndex()];

    float ptr_mf_size = ptr[gpu_accessor_.common_feature_value.MfSizeIndex()];
    int ptr_mf_dim = (int)(ptr[gpu_accessor_.common_feature_value.MfDimIndex()]);
    
    update_lr(
        optimizer_config,
        ptr[gpu_accessor_.common_feature_value.EmbedWIndex()],
        ptr[gpu_accessor_.common_feature_value.EmbedG2SumIndex()],
        grad_lr_g,
        grad_show);

    if (ptr_mf_size == (float)0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * (ptr_show - ptr_clk) +
              optimizer_config.clk_coeff * ptr_clk) {

        ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] =
          gpu_accessor_.common_feature_value.MFSize(ptr_mf_dim) / sizeof(float);
 
        // get embedxw index
        int embedx_w_index = gpu_accessor_.common_feature_value.EmbedxWOffsetIndex(ptr);
        
        for (int i = 0; i < ptr_mf_dim; ++i) {
          ptr[embedx_w_index + i] =
            (curand_uniform(&state)) * optimizer_config.mf_initial_range;
          ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex() + i] = 0;
        }
      }
    } else {
      int embedx_w_index = gpu_accessor_.common_feature_value.EmbedxWOffsetIndex(ptr);
      update_mf(
          optimizer_config,
          ptr_mf_dim,
          &ptr[embedx_w_index],
          &ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex()],
          &grad[gpu_accessor_.common_push_value.EmbedxGIndex()],
          grad_show);
    }
  }

private:
  GPUAccessor gpu_accessor_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
