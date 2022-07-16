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

#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

HeterPsBase* HeterPsBase::get_instance(
    size_t capacity, std::shared_ptr<HeterPsResource> resource, std::string accessor_type, int optimizer_type) {
  if (accessor_type == "DownpourCtrDymfAccessor" && optimizer_type == 1) {
    auto* accessor_wrapper_ptr =
      GlobalAccessorTransfor::GetInstance().GetAccessorWrapper();
    CommonFeatureValueAccessor* gpu_accessor = ((AccessorWrapper<CommonFeatureValueAccessor>*)accessor_wrapper_ptr)->AccessorPtr();

/*
    // debug
    std::cout << "=============HeterPS GPUAccesor FeatureValue INFO=========" << std::endl;
    std::cout << "optimizer type:" << gpu_accessor->common_feature_value.optimizer_type_ << std::endl;
    std::cout << "Dim:" << gpu_accessor->common_feature_value.Dim() << std::endl;
    std::cout << "EmbedDim:" << gpu_accessor->common_feature_value.EmbedDim() << std::endl;
    std::cout << "EmbedXDim:" << gpu_accessor->common_feature_value.EmbedXDim() << std::endl;
    std::cout << "EmbedWDim:" << gpu_accessor->common_feature_value.EmbedWDim() << std::endl;
    std::cout << "CpuPtrIndex:" << gpu_accessor->common_feature_value.CpuPtrIndex() << std::endl;
    std::cout << "DeltaScoreIndex:" << gpu_accessor->common_feature_value.DeltaScoreIndex() << std::endl;
    std::cout << "ShowIndex:" << gpu_accessor->common_feature_value.ShowIndex() << std::endl;
    std::cout << "ClickIndex:" << gpu_accessor->common_feature_value.ClickIndex() << std::endl;
    std::cout << "EmbedWIndex:" << gpu_accessor->common_feature_value.EmbedWIndex() << std::endl;
    std::cout << "EmbedG2SumIndex:" << gpu_accessor->common_feature_value.EmbedG2SumIndex() << std::endl;
    std::cout << "SlotIndex:" << gpu_accessor->common_feature_value.SlotIndex() << std::endl;
    std::cout << "MfDimIndex:" << gpu_accessor->common_feature_value.MfDimIndex() << std::endl;
    std::cout << "MfSizeIndex:" << gpu_accessor->common_feature_value.MfSizeIndex() << std::endl;
    std::cout << "EmbedxG2SumIndex:" << gpu_accessor->common_feature_value.EmbedxG2SumIndex() << std::endl;
    std::cout << "EmbedxWIndex:" << gpu_accessor->common_feature_value.EmbedxWIndex() << std::endl;
    std::cout << "=============HeterPS GPUAccesor FeatureValue INFO=========" << std::endl;
*/
    return new HeterPs<CommonFeatureValueAccessor, SparseAdagradOptimizer>(capacity, resource, *gpu_accessor);
  } else {
    CHECK(0) << " HeterPsBase get_instance Warning: now only support "
               "DownpourCtrDymfAccessor && SparseAdagradOptimizer, but get accessor_type:"
            << accessor_type << " optimizer type: " << optimizer_type;
  }
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
HeterPs<FVAccessor, GPUOptimizer>::HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource, FVAccessor& gpu_accessor) {
  comm_ =
      std::make_shared<HeterComm<FeatureKey, float, float, FVAccessor>>(
          capacity, resource);
  comm_->set_gpu_accessor(gpu_accessor); // 后续去掉这个接口，放入heterComm构造函数
  opt_ = GPUOptimizer<FVAccessor>(gpu_accessor);
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
HeterPs<FVAccessor, GPUOptimizer>::~HeterPs() {}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::pull_sparse(int num, FeatureKey* d_keys, float* d_vals,
                          size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

// template <typename FVAccessor, template<typename T> class GPUOptimizer>
// void HeterPs<FVAccessor, GPUOptimizer>::build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals,
//                       size_t len, size_t chunk_size, int stream_num) {
//  comm_->build_ps(num, h_keys, h_vals, len, chunk_size, stream_num);
// }

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::build_ps(int num, FeatureKey* h_keys, char* pool,
                       size_t len, size_t feature_value_size, size_t chunk_size, int stream_num) {
  comm_->build_ps(num, h_keys, pool, len, feature_value_size, chunk_size, stream_num);
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
int HeterPs<FVAccessor, GPUOptimizer>::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::set_sparse_sgd(const OptimizerConfig& optimizer_config) {
  std::cout << "before heterps setsparse sgd" << std::endl;
  comm_->set_sparse_sgd(optimizer_config);
  std::cout << "after heterps setsparse sgd" << std::endl;
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::set_embedx_sgd(const OptimizerConfig& optimizer_config) {
  std::cout << "before heterps setembedx sgd" << std::endl;
  comm_->set_embedx_sgd(optimizer_config);
  std::cout << "after heterps setembedx sgd" << std::endl;
}

// template <typename FVAccessor, template<typename T> class GPUOptimizer>
// void HeterPs<FVAccessor, GPUOptimizer>::set_gpu_accessor(FVAccessor& gpu_accessor) {
//  gpu_accessor_ = gpu_accessor;
//  opt_ = GPUOptimizer<FVAccessor>(gpu_accessor_);
//}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::end_pass() { comm_->end_pass(); }

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::show_one_table(int gpu_num) { comm_->show_one_table(gpu_num); }

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::push_sparse(int num, FeatureKey* d_keys,
                                                    float* d_grads, size_t len) {
  comm_->push_sparse(num, d_keys, d_grads, len, opt_);
  // comm_->push_sparse_multi_node(num, d_keys, d_grads, len, opt_);
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                                     const std::vector<ncclComm_t>& inter_comms,
                                     int comm_size) {
  comm_->set_nccl_comm_and_size(inner_comms, inter_comms, comm_size);
}

template <typename FVAccessor, template<typename T> class GPUOptimizer>
void HeterPs<FVAccessor, GPUOptimizer>::set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
  comm_->set_multi_mf_dim(multi_mf_dim, max_mf_dim);
}

}  // end namespace framework
}  // end namespace paddle
#endif
