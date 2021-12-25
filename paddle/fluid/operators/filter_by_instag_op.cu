// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/operators/filter_by_instag_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
using Vector = framework::Vector<T>;
#else
template <typename T>
using Vector = framework::CPUVector<T>;
#endif

#define THREADS 512
#define NUM_STREAMS 30

__global__ void filter_by_instag_cuda_kernel(
    const size_t N, const size_t* x2_lods_data, const int64_t* x2_data,
    const int64_t* x3_data, int64_t filter_tag_size, int* flag_data) {
  // N is instance num
  // one threads for one instance
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int ins_tag_start = x2_lods_data[idx];
  int ins_tag_end = x2_lods_data[idx + 1];

  // fileter logic
  int i = ins_tag_start;
  for (; i < ins_tag_end; i++) {
    int64_t ins_tag = x2_data[i];
    int j = 0;
    for (; j < filter_tag_size; j++) {
      if (x3_data[j] == ins_tag) break;
    }
    // if ins_tag in filter tag
    if (j < filter_tag_size) {
      flag_data[idx] = 1;
      break;
    }
  }
}

template <typename T>
class FilterByInstagGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    gpuStream_t current_stream = context.cuda_device_context().stream();

    // X1 is global FC output
    // Dim [batch size, embedding size]
    auto* x1 = context.Input<LoDTensor>("Ins");
    bool is_lod = context.Attr<bool>("is_lod");

    int is_x1_lod = -1;
    if (is_lod)
      is_x1_lod = 1;
    else
      is_x1_lod = 0;

    int64_t out_val_if_empty = context.Attr<int64_t>("out_val_if_empty");
    size_t x1_embed_size = x1->dims()[1];
    // X2 is ins tag list
    // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]
    auto* x2 = context.Input<LoDTensor>("Ins_tag");
    // expected auto = const int64_t
    auto* x2_data = x2->data<int64_t>();

    // X3 is local fc tag list
    // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
    auto* x3 = context.Input<Tensor>("Filter_tag");
    const int64_t* x3_data = x3->data<int64_t>();

    
    platform::Timer timeline_;
    std::cout << "============DEBUG=====================" <<  std::endl;

    timeline_.Start();

    Vector<size_t> x2_lods;
    // Vector, in GPU
    if (x2->lod().size() != 0) {  // lod_level = 1
      x2_lods = x2->lod()[0];
    } else {  // lod_level = 0
      const size_t x2_lods_size = x2->dims()[0];
      x2_lods.reserve(x2_lods_size + 1);
      x2_lods.push_back(0);
      const size_t instag_num_per_ins = x2->dims()[1];
      for (size_t i = 0; i < x2_lods_size; i++) {
        x2_lods.push_back(x2_lods.back() + instag_num_per_ins);
      }
    }
    const size_t* x2_lods_data = x2_lods.CUDAData(context.GetPlace());
    const size_t x2_lods_size = x2_lods.size() - 1;
    // Vector, in GPU
    Vector<size_t> x1_lods;
    if (!is_x1_lod) {
      x1_lods.reserve(x1->dims()[0] + 1);
      x1_lods.push_back(0);
      for (int i = 0; i < x1->dims()[0]; i++) {
        x1_lods.push_back(i + 1);
      }
    } else {
      // x1_lods = context.Input<LoDTensor>("Ins")->lod()[0];
      // new: lod_level=0 => lod() return {}
      if (x1->lod().size() != 0) {  // lod_level = 1
        x1_lods = x1->lod()[0];
      } else {  // lod_level = 0
        x1_lods.reserve(x1->dims()[0] + 1);
        x1_lods.push_back(0);
        const size_t feasign_num_per_ins = x1->dims()[1];
        for (int i = 0; i < x1->dims()[0]; i++) {
          x1_lods.push_back(x1_lods.back() + feasign_num_per_ins);
        }
      }
    }

    // const size_t* x1_lods_data = x1_lods.CUDAData(context.GetPlace());
    auto* x1_data = x1->data<T>();

    // set output value
    // for those whose ins been dropout, set 0 for whole lines.
    // otherwise, copy whole line
    // Dim [local fc count, batch size, embedding size]
    LoDTensor* out = context.Output<LoDTensor>("Out");
    LoDTensor* map = context.Output<LoDTensor>("IndexMap");
    LoDTensor* loss_weight = context.Output<LoDTensor>("LossWeight");

    timeline_.Pause();

    std::cout << "First part cost: " << timeline_.ElapsedSec() << std::endl;
    
    timeline_.Start();

    Vector<int> flag(x2_lods_size, 0);
    int* flag_data = flag.CUDAMutableData(context.GetPlace());

    int block_size = THREADS;
    dim3 block_dim(block_size);
    dim3 grid_dim((x2_lods_size + block_size - 1) / block_size);

    // fileter_logic
    filter_by_instag_cuda_kernel<<<grid_dim, block_dim, 0, current_stream>>>(
        x2_lods_size, x2_lods_data, x2_data, x3_data, x3->numel(), flag_data);

    platform::GpuStreamSync(current_stream);
    
    timeline_.Pause();
   
    std::cout << "GPU kernel part cost: " << timeline_.ElapsedSec() << std::endl;

    timeline_.Start();
    
    std::unordered_map<int64_t, int64_t> mmap_aux;
    
    Vector<size_t> out_lods;
    out_lods.reserve(x2_lods_size + 1);
    out_lods.push_back(0);

    int cnt = 0;
    for (auto it = flag.begin(); it != flag.end(); cnt++, it++) {
      if ((*it) == 1) {
        size_t batch_len = x1_lods[cnt + 1] - x1_lods[cnt];
        mmap_aux[out_lods.back()] = x1_lods[cnt];
        out_lods.push_back(out_lods.back() + batch_len);
      }
    }

    timeline_.Pause();
   
    std::cout << "outlods part cost: " << timeline_.ElapsedSec() << std::endl;

    timeline_.Start();
    
    if (out_lods.size() - 1 > 0) {
      out->Resize(framework::make_ddim(
          {(int64_t)out_lods.back(), (int64_t)x1_embed_size}));
      map->Resize(framework::make_ddim({(int64_t)out_lods.size() - 1, 3}));
      loss_weight->Resize(
          framework::make_ddim({(int64_t)out_lods.size() - 1, 1}));
    } else {
      out->Resize(framework::make_ddim({1, (int64_t)x1_embed_size}));
      map->Resize(framework::make_ddim({1, 3}));
      loss_weight->Resize(framework::make_ddim({1, 1}));
    }
    
    timeline_.Pause();
    
    std::cout << "resize part cost: " << timeline_.ElapsedSec() << std::endl;
    
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    auto* map_data = map->mutable_data<int64_t>(context.GetPlace());
    auto* loss_weight_data =
        loss_weight->mutable_data<float>(context.GetPlace());


    timeline_.Start();

    if (out_lods.size() - 1 > 0) {
      
      Vector<size_t> map_lods(out_lods.size(), 0);
      //map_lods.resize(out_lods.size());
      thrust::device_ptr<int64_t> map_data_ptr(map_data);

      // only one host -> device
      thrust::host_vector<int64_t> h_vec(3 * (out_lods.size() - 1));
      
      for (size_t i = 0; i < out_lods.size() - 1; i++) {
          h_vec[i * 3] = (int64_t)out_lods[i];
          h_vec[i * 3 + 1] = mmap_aux[(int64_t)out_lods[i]];
          h_vec[i * 3 + 2] = out_lods[i + 1] - out_lods[i];
          map_lods[i] = i;
      }

      map_lods[out_lods.size() - 1] = out_lods.size() - 1;
      // only one copy
      thrust::copy(h_vec.begin(), h_vec.end(), map_data_ptr);
 
   
      timeline_.Pause();
   
      std::cout << "copy1 part cost: " << timeline_.ElapsedSec() << std::endl;


      timeline_.Start();
   
      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.push_back(map_lods);

      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);

      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.push_back(out_lods);
      out->set_lod(out_lod_info);

      thrust::device_ptr<T> out_data_ptr(out_data);
      thrust::device_ptr<const T> x1_data_ptr(x1_data);

      thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);

      thrust::fill(out_data_ptr, out_data_ptr + out->numel(), 0);
      thrust::fill(loss_weight_data_ptr,
                   loss_weight_data_ptr + loss_weight->numel(), 1.0);

      // multi stream copy
      // how to optimizer further
      //
      std::vector<gpuStream_t> copy_streams;

      for(int i = 0; i < NUM_STREAMS; i++) {
          cudaStream_t stream;
          cudaStreamCreate(&stream);
          copy_streams.push_back(stream);
      }
      
      for (size_t i = 0; i < out_lods.size() - 1; i++) {
          auto s = copy_streams[i % NUM_STREAMS];
          size_t pos = out_lods[i];
          thrust::copy(thrust::cuda::par.on(s), x1_data_ptr + h_vec[i * 3 + 1] * x1_embed_size,
                     x1_data_ptr +
                         (h_vec[i * 3 + 1] + h_vec[i * 3 + 2]) *
                             x1_embed_size,
                     out_data_ptr + pos * x1_embed_size);
      }

      for (auto& stream : copy_streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
      }   
      copy_streams.clear();

      timeline_.Pause();
      std::cout << "copy output part cost: " << timeline_.ElapsedSec() << std::endl;

    } else {

      Vector<size_t> map_lods(3,0);
      thrust::device_ptr<int64_t> map_data_ptr(map_data);
      map_data_ptr[0] = 0;
      map_data_ptr[1] = 1;
      map_data_ptr[2] = 1;
      map_lods[0] = 0;
      map_lods[1] = 1;
      map_lods[1] = 1;
      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.push_back(map_lods);
      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);
      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.push_back(out_lods);
      out->set_lod(out_lod_info);

      thrust::device_ptr<T> out_data_ptr(out_data);

      // gpu kernel
      if (std::is_same<T, int32_t>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int32_t>(out_val_if_empty));
      } else if (std::is_same<T, int64_t>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int64_t>(out_val_if_empty));
      } else if (std::is_same<T, float>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<float>(out_val_if_empty));
      } else {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<double>(out_val_if_empty));
      }

      thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);
      loss_weight_data_ptr[0] = 0;

    }
  }
};

template <typename T>
class FilterByInstagGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x1_grad = context.Output<LoDTensor>(framework::GradVarName("Ins"));
    auto* loss_weight = context.Input<LoDTensor>("LossWeight");
    auto* mmap = context.Input<LoDTensor>("IndexMap");
    auto* x1 = context.Input<LoDTensor>("Ins");

    x1_grad->set_lod(context.Input<LoDTensor>("Ins")->lod());
    x1_grad->Resize(x1->dims());

    auto* mmap_data = mmap->data<int64_t>();

    // expected auto = T
    auto* output_grad_data = output_grad->data<T>();
    auto* loss_weight_data = loss_weight->data<float>();

    // expected auto = T
    auto* x1_grad_data = x1_grad->mutable_data<T>(context.GetPlace());

    thrust::device_ptr<const float> loss_weight_data_ptr(loss_weight_data);
    thrust::device_ptr<T> x1_grad_data_ptr(x1_grad_data);
    thrust::device_ptr<const T> output_grad_data_ptr(output_grad_data);
    thrust::device_ptr<const int64_t> mmap_data_ptr(mmap_data);

    thrust::host_vector<int64_t> h_vec(mmap->numel());
    thrust::copy(mmap_data_ptr, mmap_data_ptr + mmap->numel(), h_vec.begin());

    thrust::fill(x1_grad_data_ptr,
                 x1_grad_data_ptr + x1->dims()[0] * x1->dims()[1], 0);

    if (loss_weight->numel() != 1 || loss_weight_data_ptr[0] != 0) {
    
      auto output_dims = output_grad->dims();
      // multi-stream copy
      std::vector<gpuStream_t> copy_streams;
      for(int i = 0; i < NUM_STREAMS; i++) {
          cudaStream_t stream;
          cudaStreamCreate(&stream);
          copy_streams.push_back(stream);
      }

      for (int i = 0; i < mmap->dims()[0]; i++) {
          auto& s = copy_streams[i % NUM_STREAMS]; 
          int src_ln = h_vec[i * 3];
          int dst_ln = h_vec[i * 3 + 1];
          int line_cnt = h_vec[i * 3 + 2];

          thrust::copy(thrust::cuda::par.on(s),
            output_grad_data_ptr + src_ln * output_dims[1],
            output_grad_data_ptr + (src_ln + line_cnt) * output_dims[1],
            x1_grad_data_ptr + dst_ln * output_dims[1]);

      }
      for (auto& stream : copy_streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
      }
      copy_streams.clear();
    }

  }

};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(filter_by_instag, ops::FilterByInstagGPUKernel<float>,
                        ops::FilterByInstagGPUKernel<double>,
                        ops::FilterByInstagGPUKernel<int32_t>,
                        ops::FilterByInstagGPUKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(filter_by_instag_grad,
                        ops::FilterByInstagGradGPUKernel<float>,
                        ops::FilterByInstagGradGPUKernel<double>,
                        ops::FilterByInstagGradGPUKernel<int32_t>,
                        ops::FilterByInstagGradGPUKernel<int64_t>);
