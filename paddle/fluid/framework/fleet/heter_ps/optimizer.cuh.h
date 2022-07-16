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
  // Optimizer(FVAccessor& feature_value_accessor) {
  //  feature_value_accessor_ = feature_value_accessor;
  // }
  ~Optimizer() {}

  virtual __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr,
                                     const float* grad,
                                     curandState& state) = 0;

  // FVAccessor feature_value_accessor_;

};

template <typename FVAccessor>
class SparseAdagradOptimizer : public Optimizer<FVAccessor> {

public:
  SparseAdagradOptimizer() {}
  SparseAdagradOptimizer(FVAccessor& feature_value_accessor): Optimizer<FVAccessor>() {
    feature_value_accessor_ = feature_value_accessor;
/*
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


  __device__ void print_accessor() {
  
    printf("=============Print Accessor=================================\n");
    printf("=============Hashtable GPUAccesor FeatureValue INFO=========\n");
    printf("optimizer type:%d\n", feature_value_accessor_.common_feature_value.optimizer_type_);
    printf("Dim:%d\n",feature_value_accessor_.common_feature_value.Dim());
    printf("EmbedDim:%d\n", feature_value_accessor_.common_feature_value.EmbedDim());
    printf("EmbedXDim:%d\n", feature_value_accessor_.common_feature_value.EmbedXDim());
    printf("EmbedWDim:%d\n", feature_value_accessor_.common_feature_value.EmbedWDim());
    printf("CpuPtrIndex:%d\n", feature_value_accessor_.common_feature_value.CpuPtrIndex());
    printf("DeltaScoreIndex:%d\n", feature_value_accessor_.common_feature_value.DeltaScoreIndex());
    printf("ShowIndex:%d\n", feature_value_accessor_.common_feature_value.ShowIndex());
    printf("ClickIndex:%d\n", feature_value_accessor_.common_feature_value.ClickIndex());
    printf("EmbedWIndex:%d\n", feature_value_accessor_.common_feature_value.EmbedWIndex());
    printf("EmbedG2SumIndex:%d\n", feature_value_accessor_.common_feature_value.EmbedG2SumIndex());
    printf("SlotIndex:%d\n", feature_value_accessor_.common_feature_value.SlotIndex());
    printf("MfDimIndex:%d\n", feature_value_accessor_.common_feature_value.MfDimIndex());
    printf("MfSizeIndex:%d\n", feature_value_accessor_.common_feature_value.MfSizeIndex());
    printf("EmbedxG2SumIndex:%d\n", feature_value_accessor_.common_feature_value.EmbedxG2SumIndex());
    printf("EmbedxWIndex:%d\n", feature_value_accessor_.common_feature_value.EmbedxWIndex());
    printf("=============Hashtable GPUAccesor FeatureValue INFO=========\n");
    printf("=============Hashtable GPUAccesor PushValue INFO=========\n");
    printf("push slotIndex:%d\n", feature_value_accessor_.common_push_value.SlotIndex());
    printf("push showIndex:%d\n",feature_value_accessor_.common_push_value.ShowIndex());
    printf("push ClickIndex:%d\n", feature_value_accessor_.common_push_value.ClickIndex());
    printf("push MfDimIndex:%d\n", feature_value_accessor_.common_push_value.MfDimIndex());
    printf("push EmbedGIndex:%d\n",feature_value_accessor_.common_push_value.EmbedGIndex());
    printf("push EmbedxGIndex:%d\n", feature_value_accessor_.common_push_value.EmbedxGIndex());
    printf("=============Hashtable GPUAccesor PushValue INFO=========\n");


  }

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

  // __device__ void update_value(const OptimizerConfig& optimizer_config, ValType& val, const GradType& grad) {
  //  val.slot = grad.slot;
  //  val.show += grad.show;
  //  val.clk += grad.clk;
  //  val.delta_score += optimizer_config.nonclk_coeff * (grad.show - grad.clk) +
  //                     optimizer_config.clk_coeff * grad.clk;
  //
  //  update_lr(optimizer_config, val.lr, val.lr_g2sum, grad.lr_g, grad.show);
  //
  //  if (val.mf_size == 0) {
  //    if (optimizer_config.mf_create_thresholds <=
  //        optimizer_config.nonclk_coeff * (val.show - val.clk) +
  //            optimizer_config.clk_coeff * val.clk) {
  //      val.mf_size = MF_DIM + 1;
  //      val.mf[0] = 0;
  //      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  //      curandState state;
  //      curand_init(clock64(), tid_x, 0, &state);
  //      for (int i = 0; i < MF_DIM; ++i) {
  //        val.mf[i + 1] =
  //            (curand_uniform(&state)) * optimizer_config.mf_initial_range;
  //      }
  //    }
  //  } else {
  //    update_mf(optimizer_config, MF_DIM, &val.mf[1], val.mf[0], grad.mf_g, grad.show);
  //  }
  // }

  // __device__ void update_value(const OptimizerConfig& optimizer_config, ValType& val, const GradType& grad, curandState& state) {
  //  val.slot = grad.slot;
  //  val.show += grad.show;
  //  val.clk += grad.clk;
  //  val.delta_score += optimizer_config.nonclk_coeff * (grad.show - grad.clk) +
  //                     optimizer_config.clk_coeff * grad.clk;
  //
  //  update_lr(optimizer_config, val.lr, val.lr_g2sum, grad.lr_g, grad.show);
  //
  //  if (val.mf_size == 0) {
  //    if (optimizer_config.mf_create_thresholds <=
  //        optimizer_config.nonclk_coeff * (val.show - val.clk) +
  //            optimizer_config.clk_coeff * val.clk) {
  //      val.mf_size = MF_DIM + 1;
  //      val.mf[0] = 0;
  //      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  //      for (int i = 0; i < MF_DIM; ++i) {
  //        val.mf[i + 1] =
  //            (curand_uniform(&state)) * optimizer_config.mf_initial_range;
  //      }
  //    }
  //  } else {
  //    update_mf(optimizer_config, MF_DIM, &val.mf[1], val.mf[0], grad.mf_g, grad.show);
  //  }
  // }

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
   
    // debug 
    // printf("zmx::optimizer config, mf_initial_range:%f, learning_rate:%f, initial_g2sum:%f, min_bound:%f, max_bound:%f, mf_lr:%f, mf_initial_g2sum:%f, mf_min_bound:%f, mf_max_bound:%f", 
    //       optimizer_config.mf_initial_range, optimizer_config.learning_rate, optimizer_config.initial_g2sum, optimizer_config.min_bound, optimizer_config.max_bound,
    //       optimizer_config.mf_learning_rate, optimizer_config.mf_initial_g2sum, optimizer_config.mf_min_bound, optimizer_config.mf_max_bound);
   
    // printf("zmx::debug: grad_show:%f, grad_clk:%f, ptr_slot:%d, grad_slot:%d, ptr_show:%f, ptr_clk:%f, ptr_mf_size:%d, ptr_mf_dim:%d\n",
    //          grad_show, grad_clk, ptr[feature_value_accessor_.common_feature_value.SlotIndex()], grad[feature_value_accessor_.common_push_value.SlotIndex()],
    //       ptr_show, ptr_clk, ptr_mf_size, ptr_mf_dim);
 

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
    // printf("zmx::debug: mf_dim_index:%d, grad_show_index:%d, grad_clk_index:%d, grad_show:%f, grad_clk:%f, ptr_slot_index:%d,ptr_slot:%d, grad_slot:%d, ptr_show:%f, ptr_clk:%f, ptr_mf_size:%d,mf_dim_index:%d, ptr_mf_dim:%d\n",
    //          mf_dim_index, grad_show_index, grad_clk_index, grad_show, grad_clk, ptr_slot_index, ptr_slot, (int)grad[feature_value_accessor_.common_push_value.SlotIndex()],
    //       ptr_show, ptr_clk, ptr_mf_size, mf_dim_index, ptr_mf_dim);

    // printf("zmx::debug, mf_dim_index:%d, mf_dim_index:%d\n", mf_dim_index, mf_dim_index);

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
        ptr[11] = 0;
        // ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex()] = 0;
        // int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        //curandState state;
        //curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < ptr_mf_dim; ++i) {
          // ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + i + 1] =
          ptr[11 + i + 1] = (curand_uniform(&state)) * optimizer_config.mf_initial_range;
          // ptr->mf[i + 1] =
          //    (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_mf(
          optimizer_config,
          ptr_mf_dim,
          &ptr[12],
          ptr[11],
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
