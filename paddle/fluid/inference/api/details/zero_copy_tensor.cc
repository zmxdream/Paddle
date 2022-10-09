// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {

void ZeroCopyTensor::Reshape(const std::vector<int> &shape) {
  PADDLE_ENFORCE_EQ(
      name_.empty(), false,
      platform::errors::PreconditionNotMet(
          "Need to SetName first, so that the corresponding tensor can "
          "be retrieved."));
  PADDLE_ENFORCE_EQ(input_or_output_, true,
                    platform::errors::PermissionDenied(
                        "Can't reshape the output tensor, it is readonly"));
  PADDLE_ENFORCE_NOT_NULL(scope_, platform::errors::PreconditionNotMet(
                                      "The scope should not be nullptr."));
  auto *scope = static_cast<framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::PreconditionNotMet(
               "No tensor called [%s] in the runtime scope", name_));
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim(shape));
}

#define EAGER_GET_TENSOR    \
  if (!tensor_) {           \
    tensor_ = FindTensor(); \
  }                         \
  auto *tensor = static_cast<framework::LoDTensor *>(tensor_);

template <typename T>
T *ZeroCopyTensor::mutable_data(PaddlePlace place) {
  EAGER_GET_TENSOR;
  PADDLE_ENFORCE_GT(
      tensor->numel(), 0,
      platform::errors::PreconditionNotMet(
          "You should call ZeroCopyTensor::Reshape(const std::vector<int> "
          "&shape)"
          "function before retrieving mutable_data from input tensor."));
  switch (static_cast<int>(place)) {
    case static_cast<int>(PaddlePlace::kCPU): {
      return tensor->mutable_data<T>(platform::CPUPlace());
    }
    case static_cast<int>(PaddlePlace::kGPU): {
      return tensor->mutable_data<T>(platform::CUDAPlace(device_));
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupported place: %d",
                                                 static_cast<int>(place)));
      break;
  }
  return nullptr;
}

template <typename T>
T *ZeroCopyTensor::data(PaddlePlace *place, int *size) const {
  EAGER_GET_TENSOR;
  auto *res = tensor->data<T>();

  if (platform::is_cpu_place(tensor->place())) {
    *place = PaddlePlace::kCPU;
  } else if (platform::is_gpu_place(tensor->place())) {
    *place = PaddlePlace::kGPU;
  } else {
    *place = PaddlePlace::kUNK;
  }

  *size = tensor->numel();
  return res;
}

PaddleDType ZeroCopyTensor::type() const {
  EAGER_GET_TENSOR;
  auto type = tensor->type();
  if (type == framework::proto::VarType::FP32) {
    return PaddleDType::FLOAT32;
  } else if (type == framework::proto::VarType::INT64) {
    return PaddleDType::INT64;
  } else if (type == framework::proto::VarType::INT32) {
    return PaddleDType::INT32;
  } else if (type == framework::proto::VarType::UINT8) {
    return PaddleDType::UINT8;
  }
  return PaddleDType::FLOAT32;
}

template <typename T>
void ZeroCopyTensor::copy_from_cpu(const T *data) {
  EAGER_GET_TENSOR;
  PADDLE_ENFORCE_GE(tensor->numel(), 0,
                    platform::errors::PreconditionNotMet(
                        "You should call ZeroCopyTensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);

  if (place_ == PaddlePlace::kCPU) {
    auto *t_data = tensor->mutable_data<T>(platform::CPUPlace());
    std::memcpy(static_cast<void *>(t_data), data, ele_size);
  } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    platform::CUDAPlace gpu_place(device_);
    auto *t_data = tensor->mutable_data<T>(gpu_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));

    memory::Copy(gpu_place, static_cast<void *>(t_data), platform::CPUPlace(),
                 data, ele_size, dev_ctx->stream());
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Not compiled with CUDA, should not reach here."));
#endif
  }
}

template <typename T>
void ZeroCopyTensor::copy_to_cpu(T *data) {
  EAGER_GET_TENSOR;
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();

  if (platform::is_cpu_place(t_place)) {
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
  } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, t_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
    memory::Copy(platform::CPUPlace(), static_cast<void *>(data), gpu_place,
                 t_data, ele_num * sizeof(T), dev_ctx->stream());

    cudaStreamSynchronize(dev_ctx->stream());
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Not compile with CUDA, should not reach here."));
#endif
  }
}
template PD_INFER_DECL void ZeroCopyTensor::copy_from_cpu<float>(
    const float *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_from_cpu<int64_t>(
    const int64_t *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_from_cpu<int32_t>(
    const int32_t *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_from_cpu<uint8_t>(
    const uint8_t *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_to_cpu<float>(float *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_to_cpu<int64_t>(int64_t *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_to_cpu<int32_t>(int32_t *data);
template PD_INFER_DECL void ZeroCopyTensor::copy_to_cpu<uint8_t>(uint8_t *data);

template PD_INFER_DECL float *ZeroCopyTensor::data<float>(PaddlePlace *place,
                                                          int *size) const;
template PD_INFER_DECL int64_t *ZeroCopyTensor::data<int64_t>(
    PaddlePlace *place, int *size) const;
template PD_INFER_DECL int32_t *ZeroCopyTensor::data<int32_t>(
    PaddlePlace *place, int *size) const;
template PD_INFER_DECL uint8_t *ZeroCopyTensor::data<uint8_t>(
    PaddlePlace *place, int *size) const;
template PD_INFER_DECL float *ZeroCopyTensor::mutable_data<float>(
    PaddlePlace place);
template PD_INFER_DECL int64_t *ZeroCopyTensor::mutable_data<int64_t>(
    PaddlePlace place);
template PD_INFER_DECL int32_t *ZeroCopyTensor::mutable_data<int32_t>(
    PaddlePlace place);
template PD_INFER_DECL uint8_t *ZeroCopyTensor::mutable_data<uint8_t>(
    PaddlePlace place);

void *ZeroCopyTensor::FindTensor() const {
  PADDLE_ENFORCE_EQ(
      name_.empty(), false,
      platform::errors::PreconditionNotMet(
          "Need to SetName first, so that the corresponding tensor can "
          "be retrieved."));
  PADDLE_ENFORCE_NOT_NULL(scope_, platform::errors::PreconditionNotMet(
                                      "The scope should not be nullptr."));
  auto *scope = static_cast<framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::PreconditionNotMet(
               "No tensor called [%s] in the runtime scope", name_));
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  return tensor;
}

std::vector<int> ZeroCopyTensor::shape() const {
  EAGER_GET_TENSOR;
  PADDLE_ENFORCE_NOT_NULL(
      tensor_, platform::errors::PreconditionNotMet(
                   "Not found tensor called %s in the scope", name_));
  return framework::vectorize<int>(tensor->dims());
}

void ZeroCopyTensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
  EAGER_GET_TENSOR;
  framework::LoD lod;
  for (auto &level : x) {
    lod.emplace_back(level);
  }
  tensor->set_lod(lod);
}

std::vector<std::vector<size_t>> ZeroCopyTensor::lod() const {
  EAGER_GET_TENSOR;
  std::vector<std::vector<size_t>> res;
  for (auto &level : tensor->lod()) {
    res.emplace_back(level);
  }
  return res;
}

namespace experimental {

using float16 = paddle::platform::float16;

template <typename T>
void InternalUtils::CopyFromCpuWithIoStream(paddle::ZeroCopyTensor *t,
                                            const T *data,
                                            cudaStream_t stream) {
  if (t->tensor_ == nullptr) {
    PADDLE_ENFORCE_EQ(
        t->name_.empty(), false,
        paddle::platform::errors::PreconditionNotMet(
            "Need to SetName first, so that the corresponding tensor can "
            "be retrieved."));
    auto *scope = static_cast<paddle::framework::Scope *>(t->scope_);
    auto *var = scope->FindVar(t->name_);
    PADDLE_ENFORCE_NOT_NULL(
        var, paddle::platform::errors::PreconditionNotMet(
                 "No tensor called [%s] in the runtime scope", t->name_));
    auto *tensor = var->GetMutable<paddle::framework::LoDTensor>();
    t->tensor_ = tensor;
  }

  auto *tensor = static_cast<paddle::framework::LoDTensor *>(t->tensor_);
  PADDLE_ENFORCE_GE(tensor->numel(), 0,
                    paddle::platform::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);
  if (t->place_ == PlaceType::kCPU) {
    auto *t_data = tensor->mutable_data<T>(paddle::platform::CPUPlace());
    std::memcpy(static_cast<void *>(t_data), data, ele_size);
  } else if (t->place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    paddle::platform::CUDAPlace gpu_place(t->device_);
    auto *t_data = tensor->mutable_data<T>(gpu_place);
    paddle::memory::Copy(gpu_place, static_cast<void *>(t_data),
                         paddle::platform::CPUPlace(), data, ele_size, stream);
#else
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "CopyFromCpuWithIoStream only supports CPU and GPU now."));
  }
}

template <typename T>
void InternalUtils::CopyToCpuWithIoStream(paddle::ZeroCopyTensor *t, T *data,
                                          cudaStream_t stream) {
  if (t->tensor_ == nullptr) {
    PADDLE_ENFORCE_EQ(
        t->name_.empty(), false,
        paddle::platform::errors::PreconditionNotMet(
            "Need to SetName first, so that the corresponding tensor can "
            "be retrieved."));
    auto *scope = static_cast<paddle::framework::Scope *>(t->scope_);
    auto *var = scope->FindVar(t->name_);
    PADDLE_ENFORCE_NOT_NULL(
        var, paddle::platform::errors::PreconditionNotMet(
                 "No tensor called [%s] in the runtime scope", t->name_));
    auto *tensor = var->GetMutable<paddle::framework::LoDTensor>();
    t->tensor_ = tensor;
  }

  auto *tensor = static_cast<paddle::framework::LoDTensor *>(t->tensor_);
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();

  paddle::framework::Tensor out;
  auto mem_allocation =
      std::make_shared<paddle::memory::allocation::Allocation>(
          static_cast<void *>(data), ele_num * sizeof(T),
          paddle::platform::CPUPlace());
  out.ResetHolder(mem_allocation);

  if (paddle::platform::is_cpu_place(t_place)) {
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
  } else if (t->place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, t_place);
    paddle::memory::Copy(paddle::platform::CPUPlace(),
                         static_cast<void *>(data), gpu_place, t_data,
                         ele_num * sizeof(T), stream);
#else
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "CopyToCpuWithIoStream only supports CPU and GPU now."));
  }
}

template void InternalUtils::CopyFromCpuWithIoStream<float>(
    paddle::ZeroCopyTensor *t, const float *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int64_t>(
    paddle::ZeroCopyTensor *t, const int64_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int32_t>(
    paddle::ZeroCopyTensor *t, const int32_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<uint8_t>(
    paddle::ZeroCopyTensor *t, const uint8_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int8_t>(
    paddle::ZeroCopyTensor *t, const int8_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<float16>(
    paddle::ZeroCopyTensor *t, const float16 *data, cudaStream_t stream);

template void InternalUtils::CopyToCpuWithIoStream<float>(
    paddle::ZeroCopyTensor *t, float *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int64_t>(
    paddle::ZeroCopyTensor *t, int64_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int32_t>(
    paddle::ZeroCopyTensor *t, int32_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<uint8_t>(
    paddle::ZeroCopyTensor *t, uint8_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int8_t>(
    paddle::ZeroCopyTensor *t, int8_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<float16>(
    paddle::ZeroCopyTensor *t, float16 *data, cudaStream_t stream);

}  // namespace experimental
}  // namespace paddle
