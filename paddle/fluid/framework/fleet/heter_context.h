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

#pragma once

#ifdef PADDLE_WITH_HETERPS

#include <ThreadPool.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_WITH_PSLIB
#include "common_value.h"  // NOLINT
#endif

#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/table/depends/feature_value.h"
#endif

#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

class HeterContext {
 public:
  //保存去重后的待查table的key, 第一层对应table-shard, 第二层对应不同维度，第三层就是key集合
  std::vector<std::vector<std::vector<FeatureKey>>>feature_keys_;
  //保存查到的value数据，维度同feature_keys_
#ifdef PADDLE_WITH_PSLIB
  std::vector<std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>>>
      value_ptr_;
#endif
#ifdef PADDLE_WITH_PSCORE
  std::vector<std::vector<std::vector<paddle::distributed::FixedFeatureValue*>>>
      value_ptr_;
#endif
  //经过去重后的gpu-table中的key数据, 第一层设备，第二层维度，第三层具体的key
  std::vector<std::vector<std::vector<FeatureKey>>> device_keys_;

  //初始化
  void init(int shard_num, int device_num, int dim_num) {
    feature_keys_.resize(shard_num);
    for (auto& iter : feature_keys_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }
    value_ptr_.resize(shard_num);
    for (auto& iter : value_ptr_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }
    device_keys_.resize(device_num);
    for (auto& iter : device_keys_) {
      iter.resize(dim_num);
      for (auto& iter1: iter) {
        iter1.clear();
      }
    }

  }
  //将粗去重的key加入进来,后面再做精细化去重
  void batch_add_keys(int shard_num, int dim_id,
                      const robin_hood::unordered_set<uint64_t>& shard_keys) {
    int idx = feature_keys_[shard_num][dim_id].size();
    feature_keys_[shard_num][dim_id].resize(
        feature_keys_[shard_num][dim_id].size() + shard_keys.size());
    std::copy(shard_keys.begin(), shard_keys.end(),
              feature_keys_[shard_num][dim_id].begin() + idx);
  }
  void unique_keys() {
    std::vector<std::thread> threads;
    auto unique_func = [this](int i, int j) {
      auto& cur_keys = feature_keys_[i][j];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    for (size_t i = 0; i < feature_keys_.size(); i++) {
      for (size_t j = 0; j < feature_keys_[i].size(); j++) {
        threads.push_back(std::thread(unique_func, i, j));
      }
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  uint16_t pass_id_;
};


}  // end namespace framework
}  // end namespace paddle
#endif
