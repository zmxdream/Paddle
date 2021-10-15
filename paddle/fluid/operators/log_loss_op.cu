/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/log_loss_op.h"

namespace paddle {
namespace operators {
template <typename T>
__global__ void kernel_logloss(const size_t N, const T* label,
                               const T* prediction, const float epsilon,
                               T* out_data) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = (-(label[idx] * log(prediction[idx] + epsilon)) -
                     ((static_cast<T>(1.0) - label[idx]) *
                      log(static_cast<T>(1.0) - prediction[idx] + epsilon)));
  }
}
template <typename T>
__global__ void kernel_loglossgrad(const size_t N, const T* label,
                                   const T* prediction, const T* dloss,
                                   const float epsilon, T* out_grad) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_grad[idx] =
        dloss[idx] * (-(label[idx] / (prediction[idx] + epsilon)) +
                      ((static_cast<T>(1) - label[idx]) /
                       (static_cast<T>(1) - prediction[idx] + epsilon)));
  }
}
template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto loss_out = ctx.Output<Tensor>("Loss");
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto prediction = ctx.Input<Tensor>("Predicted");
    auto label = ctx.Input<Tensor>("Labels");

    auto stream = ctx.cuda_device_context().stream();
    const T* prediction_data = prediction->data<T>();
    const T* label_data = label->data<T>();
    T* out_data = loss_out->mutable_data<T>(ctx.GetPlace());

    size_t N = static_cast<size_t>(label->numel());
    int threads = 512;
    int grid = (N + threads - 1) / threads;
    kernel_logloss<T><<<grid, threads, 0, stream>>>(
        N, label_data, prediction_data, static_cast<float>(epsilon), out_data);
  }
};

template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));

    auto prediction = ctx.Input<Tensor>("Predicted");
    auto label = ctx.Input<Tensor>("Labels");

    auto* dloss = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* dpred = ctx.Output<Tensor>(framework::GradVarName("Predicted"));

    if (dpred) {
      const T* prediction_data = prediction->data<T>();
      const T* label_data = label->data<T>();
      const T* dloss_data = dloss->data<T>();
      T* out_grad = dpred->mutable_data<T>(ctx.GetPlace());

      auto stream = ctx.cuda_device_context().stream();
      size_t N = static_cast<size_t>(label->numel());
      int threads = 512;
      int grid = (N + threads - 1) / threads;
      kernel_loglossgrad<T><<<grid, threads, 0, stream>>>(
          N, label_data, prediction_data, dloss_data,
          static_cast<float>(epsilon), out_grad);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    log_loss,
    ops::LogLossCUDAKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    log_loss_grad,
    ops::LogLossGradCUDAKernel<paddle::platform::CUDADeviceContext, float>);
