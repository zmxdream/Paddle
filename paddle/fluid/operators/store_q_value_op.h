/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/tensor.h"



namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoDTensorArray = framework::LoDTensorArray;


template <typename T>
static void StoreQValueFunctor(const framework::ExecutionContext &ctx) {

  auto inputs = ctx.MultiInput<LoDTensor>("Ids"); //std::vector<const Tensor*>
  auto ctx_place = ctx.GetPlace();
  int device_id = boost::get<platform::CUDAPlace>(ctx_place).GetDeviceId();

  const auto qvalues_size = inputs.size();
  std::vector<framework::Tensor> cpu_qvalues(qvalues_size);

  for (size_t i = 0; i < qvalues_size; ++i) {
    framework::TensorCopy(*inputs[i], platform::CPUPlace(), &cpu_qvalues[i]);
  }
  framework::BatchGpuPackMgr().store_qvalue(device_id, cpu_qvalues);
}

template <typename T>
class StoreQValueCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    StoreQValueFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
