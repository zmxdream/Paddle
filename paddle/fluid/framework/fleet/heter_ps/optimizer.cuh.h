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

template <typename ValType, typename GradType>
class Optimizer {
 public:
  Optimizer() {}
  ~Optimizer() {}
  void initialize() {}
  __device__ void update_value(ValType* ptr, const GradType& grad, curandState& state) {
  }
};

}  // end namespace framework
}  // end namespace paddle
#endif
