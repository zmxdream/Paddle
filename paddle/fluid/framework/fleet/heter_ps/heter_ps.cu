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
    // NOTE(zhangminxu): gpups' sparse table optimizer type,
    // now only support embed&embedx 's sparse optimizer is the same
    // we will support using diff optimizer for embed & embedx
    auto* accessor_wrapper_ptr =
      GlobalAccessorFactory::GetInstance().GetAccessorWrapper();

  VLOG(0) << "in heterpsbase get_instance, optimizer_type:" << optimizer_type; 

  if (accessor_type == "DownpourCtrDymfAccessor" && optimizer_type == 1) { // adagrad
    // auto* accessor_wrapper_ptr =
    //  GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
    CommonFeatureValueAccessor* gpu_accessor =
      ((AccessorWrapper<CommonFeatureValueAccessor>*)accessor_wrapper_ptr)->AccessorPtr();
    return new HeterPs<CommonFeatureValueAccessor, SparseAdagradOptimizer>(capacity, resource, *gpu_accessor);
  } else if (accessor_type == "DownpourCtrDymfAccessor" && optimizer_type == 2) { // std_adagrad
    CommonFeatureValueAccessor* gpu_accessor =
      ((AccessorWrapper<CommonFeatureValueAccessor>*)accessor_wrapper_ptr)->AccessorPtr();
    return new HeterPs<CommonFeatureValueAccessor, StdAdagradOptimizer>(capacity, resource, *gpu_accessor);
  } else {
    CHECK(0) << " HeterPsBase get_instance Warning: now only support "
               "DownpourCtrDymfAccessor && SparseAdagradOptimizer/StdAdagradOptimizer, but get accessor_type:"
            << accessor_type << " optimizer type: " << optimizer_type;
  }
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
HeterPs<GPUAccessor, GPUOptimizer>::HeterPs(size_t capacity, std::shared_ptr<HeterPsResource> resource, GPUAccessor& gpu_accessor) {
/*
      std::make_shared<HeterComm<FeatureKey, FeatureValue, FeaturePushValue>>(
          capacity, resource);
  opt_ = Optimizer<FeatureValue, FeaturePushValue>();
  multi_node_ = resource->multi_node();
  std::cout << "yxf heterps::multinode: " <<  multi_node_ << std::endl;
*/
  comm_ =
      std::make_shared<HeterComm<FeatureKey, float, float, GPUAccessor>>(
          capacity, resource, gpu_accessor);
  opt_ = GPUOptimizer<GPUAccessor>(gpu_accessor);
  multi_node_ = resource->multi_node();
  std::cout << "yxf heterps::multinode: " <<  multi_node_ << std::endl;
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
HeterPs<GPUAccessor, GPUOptimizer>::~HeterPs() {}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::pull_sparse(int num, FeatureKey* d_keys, float* d_vals,
                          size_t len) {
  comm_->pull_sparse(num, d_keys, d_vals, len);
}

// template <typename GPUAccessor, template<typename T> class GPUOptimizer>
// void HeterPs<GPUAccessor, GPUOptimizer>::build_ps(int num, FeatureKey* h_keys, FeatureValue* h_vals,
//                       size_t len, size_t chunk_size, int stream_num) {
//  comm_->build_ps(num, h_keys, h_vals, len, chunk_size, stream_num);
// }
    
//   this->HeterPs_->build_ps(i, device_dim_keys.data(),
//                             cur_pool->mem(), len, feature_value_size,
//                             500000, 2);

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::build_ps(int num, FeatureKey* h_keys, char* pool,
                       size_t len, size_t feature_value_size, size_t chunk_size, int stream_num) {
  comm_->build_ps(num, h_keys, pool, len, feature_value_size, chunk_size, stream_num);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
int HeterPs<GPUAccessor, GPUOptimizer>::get_index_by_devid(int devid) {
  return comm_->get_index_by_devid(devid);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_sparse_sgd(const OptimizerConfig& optimizer_config) {
  comm_->set_sparse_sgd(optimizer_config);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_embedx_sgd(const OptimizerConfig& optimizer_config) {
  comm_->set_embedx_sgd(optimizer_config);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::end_pass() { comm_->end_pass(); }

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::show_one_table(int gpu_num) { comm_->show_one_table(gpu_num); }

/*
void HeterPs::push_sparse(int num, FeatureKey* d_keys,
                          FeaturePushValue* d_grads, size_t len) {
  if (multi_node_ <= 0) {
    comm_->push_sparse(num, d_keys, d_grads, len, opt_);
  } else {
    // check
    comm_->push_sparse_multi_node(num, d_keys, d_grads, len, opt_);
  }
}
*/

// d_grads: FeaturePushValue* -> float*
// 
template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::push_sparse(int num, FeatureKey* d_keys,
                                                    float* d_grads, size_t len) {
  if (multi_node_ <= 0) {
    comm_->push_sparse(num, d_keys, d_grads, len, opt_);
  } else {
    // check
    comm_->push_sparse_multi_node(num, d_keys, d_grads, len, opt_);
  }
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_nccl_comm_and_size(const std::vector<ncclComm_t>& inner_comms,
                                     const std::vector<ncclComm_t>& inter_comms,
                                     int comm_size) { // node_size??
  comm_->set_nccl_comm_and_size(inner_comms, inter_comms, comm_size);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_trans_inter_comm(const std::vector<ncclComm_t>& trans_inter_comms) {
  comm_->set_trans_inter_comm(trans_inter_comms);
}

template <typename GPUAccessor, template<typename T> class GPUOptimizer>
void HeterPs<GPUAccessor, GPUOptimizer>::set_multi_mf_dim(int multi_mf_dim, int max_mf_dim) {
  comm_->set_multi_mf_dim(multi_mf_dim, max_mf_dim);
}

/*
      uint32_t* d_restore_idx =
          reinterpret_cast<uint32_t*>(&key2slot[total_length]);
      uint32_t* d_sorted_idx =
          reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
      uint32_t* d_offset =
          reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
      uint32_t* d_merged_cnts =
          reinterpret_cast<uint32_t*>(&d_offset[total_length]);

      uint64_t* d_merged_keys =
          reinterpret_cast<uint64_t*>(&total_keys[total_length]);
      uint64_t* d_sorted_keys =
          reinterpret_cast<uint64_t*>(&d_merged_keys[total_length]);

      int dedup_size = HeterPs_->dedup_keys_and_fillidx(
          devid_2_index,
          static_cast<int>(total_length),
          total_keys,     // input
          d_merged_keys,  // output
          d_sorted_keys,  // sort keys
          d_restore_idx,  // pull fill idx
          d_sorted_idx,   // sort old idx
          d_offset,       // offset
          d_merged_cnts,
          FLAGS_gpups_dedup_pull_push_mode & 0x02);
*/
template <typename GPUAccessor, template<typename T> class GPUOptimizer>
int HeterPs<GPUAccessor, GPUOptimizer>::dedup_keys_and_fillidx(
    const int gpu_id,
    const int total_fea_num,
    const FeatureKey* d_keys,   // input
    FeatureKey* d_merged_keys,  // output
    FeatureKey* d_sorted_keys,
    uint32_t* d_restore_idx,
    uint32_t* d_sorted_idx,
    uint32_t* d_offset,
    uint32_t* d_merged_cnts,
    bool filter_zero) {
  return comm_->dedup_keys_and_fillidx(gpu_id,
                                       total_fea_num,
                                       d_keys,         // input
                                       d_merged_keys,  // output
                                       d_sorted_keys,
                                       d_restore_idx,
                                       d_sorted_idx,
                                       d_offset,
                                       d_merged_cnts,
                                       filter_zero);
}

}  // end namespace framework
}  // end namespace paddle
#endif
