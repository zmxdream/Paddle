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
#include "paddle/fluid/framework/fleet/heter_ps/feature_value_inl.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_value_inl.h"
#include "paddle/fluid/framework/fleet/heter_ps/dy_gpu_value_inl.h"

#ifdef PADDLE_WITH_HETERPS

namespace paddle {
namespace framework {

HeterPsBase* HeterPsBase::get_instance(
    size_t capacity, std::shared_ptr<HeterPsResource> resource,
    std::string accessor_type, std::string gpu_value_type) {
  if (accessor_type == "DownpourCtrDymfAccessor" && gpu_value_type == "DyFeatureValue") {
    return new HeterPs<FeatureKey, DyGpuValue, DyGpuPushValue>(capacity, resource);
  } else if (accessor_type == "DownpourCtrAccessor" && gpu_value_type == "FeatureValue") {
    return new HeterPs<FeatureKey, GpuValue, GpuPushValue>(capacity, resource);
  }
  return nullptr;
}

template <typename KeyType, typename ValType, typename GradType>
HeterPs<KeyType, ValType, GradType>::HeterPs() {
}

template <typename KeyType, typename ValType, typename GradType>
HeterPs<KeyType, ValType, GradType>::HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource) {
  comm_ = std::make_shared<HeterComm<KeyType, ValType, GradType>>(capacity, resource);
  opt_ = Optimizer<ValType, GradType>();
}

template <typename KeyType, typename ValType, typename GradType>
void HeterPs<KeyType, ValType, GradType>::pull_sparse(int num, FeatureKey* d_keys, void* d_vals,
                           size_t len) {
  comm_->pull_sparse(num, d_keys, (ValType*)d_vals, len);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterPs<KeyType, ValType, GradType>::build_ps(int num, KeyType* h_keys, char* pool,
            size_t len, size_t feature_value_size, size_t chunk_size, int stream_num) {
  comm_->build_ps(num, h_keys, pool, len, feature_value_size, chunk_size, stream_num);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterPs<KeyType, ValType, GradType>::set_nccl_comm_and_size(
      const std::vector<ncclComm_t>& inner_comms,
      const std::vector<ncclComm_t>& inter_comms, int comm_size) {
  comm_->set_nccl_comm_and_size(inner_comms, inter_comms, comm_size);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterPs<KeyType, ValType, GradType>::set_multi_mf_dim(int max_mf_dim) {
  comm_->set_multi_mf_dim(max_mf_dim);
}

template <typename KeyType, typename ValType, typename GradType>
int HeterPs<KeyType, ValType, GradType>::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

template <typename KeyType, typename ValType, typename GradType>
void HeterPs<KeyType, ValType, GradType>::push_sparse(int num, FeatureKey* d_keys, void* d_grads, size_t len) {
   comm_->push_sparse(num, d_keys, (GradType*)d_grads, len, opt_);
}



}  // end namespace framework
}  // end namespace paddle
#endif
