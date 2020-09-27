//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void FusedSeqpoolKernel(
    T **input_values, T **seqpool_output_values, size_t **lods_values,
    const int64_t *data_lens, const int batch_size, const int embedding_size,
    const float pad_value, bool need_filter, float show_coeff, float clk_coeff,
    float threshold) {
  int bId = blockIdx.y * gridDim.x + blockIdx.x;
  int x = bId / batch_size;
  int y = bId - (x ? data_lens[x - 1] : 0);
  int start = *(lods_values[x] + y);
  int end = *(lods_values[x] + y + 1);

  for (int tid = threadIdx.x; tid < embedding_size; tid += blockDim.x) {
    if (start == end) {
      *(seqpool_output_values[x] + y * embedding_size + tid) = pad_value;
    } else {
      if (need_filter) {
        T val = static_cast<T>(0);
        for (int k = start; k < end; k++) {
          float show = *(input_values[x] + k * embedding_size);
          float click = *(input_values[x] + k * embedding_size + 1);
          if ((show - click) * show_coeff + click * clk_coeff < threshold) {
            continue;
          }
          if (tid <= 1) {  // show & click
            val += *(input_values[x] + k * embedding_size + tid);
          } else {
            val += ((int)(*(input_values[x] + k * embedding_size + tid) * 128 +
                          0.5)) /
                   128.0;
          }
        }
        *(seqpool_output_values[x] + y * embedding_size + tid) = val;
      } else {
        T val = static_cast<T>(0);
        for (int k = start; k < end; k++) {
          val += *(input_values[x] + k * embedding_size + tid);
        }
        *(seqpool_output_values[x] + y * embedding_size + tid) = val;
      }
    }
  }
}

template <typename T>
__global__ void FusedCVMKernel(T **output_values, T **seqpool_output_values,
                               const int64_t *data_lens, const int batch_size,
                               int64_t total_len, const int embedding_size,
                               bool use_cvm) {
  CUDA_KERNEL_LOOP(i, total_len * embedding_size) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;
    int y = key - (x ? data_lens[x - 1] : 0);
    int cvm_offset = 2;
    if (use_cvm) {
      if (offset == 0) {
        *(output_values[x] + y * embedding_size) =
            log(*(seqpool_output_values[x] + y * embedding_size) + 1);
      } else if (offset == 1) {
        *(output_values[x] + y * embedding_size + offset) =
            log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1) -
            log(*(seqpool_output_values[x] + y * embedding_size) + 1);
      } else {
        *(output_values[x] + y * embedding_size + offset) =
            *(seqpool_output_values[x] + y * embedding_size + offset);
      }
    } else {
      if (offset >= cvm_offset) {
        *(output_values[x] + y * (embedding_size - cvm_offset) + offset -
          cvm_offset) =
            *(seqpool_output_values[x] + y * embedding_size + offset);
      }
    }
  }
}

template <typename T>
__global__ void FusedSeqpoolCVMGradKernel(
    T **out_grads_values, T **out_seqpool_grads_values, T **in_grads_values,
    T **cvm_values, size_t **lods_values, const int64_t *data_lens,
    const int batch_size, int64_t total_len, const int embedding_size,
    bool use_cvm) {
  CUDA_KERNEL_LOOP(i, total_len * embedding_size) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;
    int y = key - (x ? data_lens[x - 1] : 0);

    int cvm_offset = 2;

    if (offset < cvm_offset) {
      *(out_seqpool_grads_values[x] + y * embedding_size + offset) =
          *(cvm_values[x] + y * cvm_offset + offset);
    } else {
      if (use_cvm) {
        *(out_seqpool_grads_values[x] + y * embedding_size + offset) =
            *(out_grads_values[x] + y * embedding_size + offset);
      } else {
        *(out_seqpool_grads_values[x] + y * embedding_size + offset) =
            *(out_grads_values[x] + y * (embedding_size - cvm_offset) + offset -
              cvm_offset);
      }
    }

    int start = *(lods_values[x] + y);
    int end = *(lods_values[x] + y + 1);
    for (int k = start; k < end; k++) {
      *(in_grads_values[x] + k * embedding_size + offset) =
          *(out_seqpool_grads_values[x] + y * embedding_size + offset);
    }
  }
}

template <typename T>
void DoFusedSeqpoolCVM(const paddle::platform::Place &place,
                       T **gpu_input_values, T **gpu_output_values,
                       T **gpu_seqpool_output_values, size_t **lods_values,
                       const int64_t *data_lens, int slot_num,
                       int64_t total_len, const int embedding_size,
                       const float padding_value, bool use_cvm,
                       bool need_filter, float show_coeff, float clk_coeff,
                       float threshold) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  int batch_size = total_len / slot_num;
  dim3 grid(batch_size, slot_num);
  FusedSeqpoolKernel<<<grid, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      gpu_input_values, gpu_seqpool_output_values, lods_values, data_lens,
      batch_size, embedding_size, padding_value, need_filter, show_coeff,
      clk_coeff, threshold);

  FusedCVMKernel<<<(total_len * embedding_size + PADDLE_CUDA_NUM_THREADS - 1) /
                       PADDLE_CUDA_NUM_THREADS,
                   PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      gpu_output_values, gpu_seqpool_output_values, data_lens, batch_size,
      total_len, embedding_size, use_cvm);
}

template <typename T>
void FusedSeqpoolCVM(const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     const std::vector<T *> &seqpool_output_data,
                     std::vector<const size_t *> lods,
                     const std::vector<int64_t> &data_lengths,
                     const int embedding_size, const float padding_value,
                     const bool use_cvm, float need_filter, float show_coeff,
                     float clk_coeff, float threshold) {
  auto data_lengths_lod = data_lengths;
  int slot_num = static_cast<int>(data_lengths.size());
  for (int i = 1; i < slot_num; i++) {
    data_lengths_lod[i] += data_lengths_lod[i - 1];
  }

  int64_t total_length = data_lengths_lod[slot_num - 1];

  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  LoDTensor data_lens_tensor;
  int64_t *data_lens = reinterpret_cast<int64_t *>(
      data_lens_tensor.mutable_data<int64_t>({slot_num, 1}, place));
  cudaMemcpyAsync(data_lens, data_lengths_lod.data(),
                  data_lengths_lod.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream);

  auto gpu_input_ptr =
      memory::AllocShared(place, input_data.size() * sizeof(T *));
  T **gpu_input_values = reinterpret_cast<T **>(gpu_input_ptr->ptr());
  cudaMemcpyAsync(gpu_input_values, input_data.data(),
                  input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  auto gpu_output_ptr =
      memory::AllocShared(place, output_data.size() * sizeof(T *));
  T **gpu_output_values = reinterpret_cast<T **>(gpu_output_ptr->ptr());
  cudaMemcpyAsync(gpu_output_values, output_data.data(),
                  output_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  auto gpu_seqpool_output_ptr =
      memory::AllocShared(place, seqpool_output_data.size() * sizeof(T *));
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(gpu_seqpool_output_ptr->ptr());
  cudaMemcpyAsync(gpu_seqpool_output_values, seqpool_output_data.data(),
                  seqpool_output_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);

  auto lods_ptr = memory::AllocShared(place, lods.size() * sizeof(size_t *));
  size_t **lods_values = reinterpret_cast<size_t **>(lods_ptr->ptr());
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  DoFusedSeqpoolCVM(place, gpu_input_values, gpu_output_values,
                    gpu_seqpool_output_values, lods_values, data_lens, slot_num,
                    total_length, embedding_size, padding_value, use_cvm,
                    need_filter, show_coeff, clk_coeff, threshold);
}

template <typename T>
static void FusedSeqpoolCVMFunctor(const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<LoDTensor>("X");
  auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

  const auto slot_size = inputs.size();
  std::vector<const float *> input_data(slot_size);
  std::vector<int64_t> data_lens(slot_size);
  std::vector<const size_t *> lods_data(slot_size);
  std::vector<T *> output_data(slot_size);

  std::vector<LoDTensor> seqpool_outputs(slot_size);
  std::vector<T *> seqpool_output_data(slot_size);

  auto padding_value = ctx.Attr<float>("pad_value");
  auto use_cvm = ctx.Attr<bool>("use_cvm");
  bool need_filter = ctx.Attr<bool>("need_filter");
  float show_coeff = ctx.Attr<float>("show_coeff");
  float clk_coeff = ctx.Attr<float>("clk_coeff");
  float threshold = ctx.Attr<float>("threshold");

  int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];

  for (size_t i = 0; i < slot_size; ++i) {
    const auto *input = inputs[i];
    auto dims = input->dims();

    auto lod = input->lod();
    auto lod_level = lod.size();
    int batch_size = lod[lod_level - 1].size() - 1;  // -1 to real batch size

    input_data[i] = reinterpret_cast<const T *>(input->data<T>());
    auto *output = outputs[i];
    if (use_cvm) {
      output->Resize({batch_size, embedding_size});
    } else {
      output->Resize({batch_size, embedding_size - 2});
    }
    output_data[i] =
        reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
    data_lens[i] = lod[lod_level - 1].size() - 1;
    lods_data[i] = lod[lod_level - 1].CUDAData(ctx.GetPlace());

    seqpool_output_data[i] =
        reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
            {batch_size, embedding_size}, ctx.GetPlace()));
  }

  FusedSeqpoolCVM(ctx.GetPlace(), input_data, output_data, seqpool_output_data,
                  lods_data, data_lens, embedding_size, padding_value, use_cvm,
                  need_filter, show_coeff, clk_coeff, threshold);
}

template <typename T>
void DoFusedSeqpoolCVMGrad(const paddle::platform::Place &place,
                           T **out_grads_values, T **out_seqpool_grads_values,
                           T **in_grads_values, T **gpu_cvm_values,
                           size_t **lods_values, const int64_t *slot_lens,
                           int slot_num, int64_t total_len,
                           const int embedding_size, bool use_cvm) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  const int batch_size = total_len / slot_num;
  FusedSeqpoolCVMGradKernel<<<(total_len * embedding_size +
                               PADDLE_CUDA_NUM_THREADS - 1) /
                                  PADDLE_CUDA_NUM_THREADS,
                              PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      out_grads_values, out_seqpool_grads_values, in_grads_values,
      gpu_cvm_values, lods_values, slot_lens, batch_size, total_len,
      embedding_size, use_cvm);
}

template <typename T>
void FusedSeqpoolCVMGrad(const paddle::platform::Place &place,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &out_seqpool_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         std::vector<const size_t *> &lods,
                         const std::vector<int64_t> &data_lengths,
                         const int embedding_size, const bool use_cvm) {
  auto data_lengths_lod = data_lengths;
  int slot_num = static_cast<int>(data_lengths.size());
  for (int i = 1; i < slot_num; i++) {
    data_lengths_lod[i] += data_lengths_lod[i - 1];
  }

  int64_t total_length = data_lengths_lod[slot_num - 1];

  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  LoDTensor data_lens_tensor;
  int64_t *data_lens = reinterpret_cast<int64_t *>(
      data_lens_tensor.mutable_data<int64_t>({slot_num, 1}, place));
  cudaMemcpyAsync(data_lens, data_lengths_lod.data(),
                  data_lengths_lod.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream);

  auto gpu_out_grads_ptr =
      memory::AllocShared(place, out_grads_data.size() * sizeof(T *));
  T **gpu_out_grads_values = reinterpret_cast<T **>(gpu_out_grads_ptr->ptr());
  cudaMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                  out_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  auto gpu_out_seqpool_grads_ptr =
      memory::AllocShared(place, out_seqpool_grads_data.size() * sizeof(T *));
  T **gpu_out_seqpool_grads_values =
      reinterpret_cast<T **>(gpu_out_seqpool_grads_ptr->ptr());
  cudaMemcpyAsync(gpu_out_seqpool_grads_values, out_seqpool_grads_data.data(),
                  out_seqpool_grads_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);

  auto gpu_in_grads_ptr =
      memory::AllocShared(place, in_grads_data.size() * sizeof(T *));
  T **gpu_in_grads_values = reinterpret_cast<T **>(gpu_in_grads_ptr->ptr());
  cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                  in_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  auto gpu_cvm_ptr = memory::AllocShared(place, cvm_data.size() * sizeof(T *));
  T **gpu_cvm_values = reinterpret_cast<T **>(gpu_cvm_ptr->ptr());
  cudaMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                  cvm_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  auto lods_ptr = memory::AllocShared(place, lods.size() * sizeof(size_t *));
  size_t **lods_values = reinterpret_cast<size_t **>(lods_ptr->ptr());
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  DoFusedSeqpoolCVMGrad(place, gpu_out_grads_values,
                        gpu_out_seqpool_grads_values, gpu_in_grads_values,
                        gpu_cvm_values, lods_values, data_lens, slot_num,
                        total_length, embedding_size, use_cvm);
}

template <typename T>
static void FusedSeqpoolCVMGradFunctor(const framework::ExecutionContext &ctx) {
  auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
  auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
  auto *cvm = ctx.Input<LoDTensor>("CVM");

  std::string pooltype = ctx.Attr<std::string>("pooltype");
  auto use_cvm = ctx.Attr<bool>("use_cvm");

  const auto slot_size = in_grads.size();
  std::vector<const T *> out_grads_data(slot_size);
  std::vector<T *> in_grads_data(slot_size);
  std::vector<const T *> cvm_data(slot_size);
  std::vector<const size_t *> lods_data(slot_size);
  std::vector<int64_t> data_lengths(slot_size);

  std::vector<LoDTensor> out_seqpool_grads(slot_size);
  std::vector<T *> out_seqpool_grads_data(slot_size);

  int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];

  for (size_t i = 0; i < slot_size; ++i) {
    auto *in_grad = in_grads[i];
    auto dims = in_grad->dims();

    auto lod = in_grad->lod();
    auto lod_level = lod.size();
    int batch_size = lod[lod_level - 1].size() - 1;  // -1 to real batch size

    auto *out_grad = out_grads[i];
    out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

    in_grads_data[i] =
        reinterpret_cast<T *>(in_grad->mutable_data<T>(ctx.GetPlace()));
    lods_data[i] = lod[lod_level - 1].CUDAData(ctx.GetPlace());
    data_lengths[i] = lod[lod_level - 1].size() - 1;
    cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());

    out_seqpool_grads_data[i] =
        reinterpret_cast<T *>(out_seqpool_grads[i].mutable_data<T>(
            {batch_size, embedding_size}, ctx.GetPlace()));
  }

  FusedSeqpoolCVMGrad(ctx.GetPlace(), out_grads_data, out_seqpool_grads_data,
                      in_grads_data, cvm_data, lods_data, data_lengths,
                      embedding_size, use_cvm);
}

template <typename T>
class FusedSeqpoolCVMCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    FusedSeqpoolCVMFunctor<T>(ctx);
  }
};

template <typename T>
class FusedSeqpoolCVMGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    FusedSeqpoolCVMGradFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm,
                        ops::FusedSeqpoolCVMCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_grad,
                        ops::FusedSeqpoolCVMGradCUDAKernel<float>);
