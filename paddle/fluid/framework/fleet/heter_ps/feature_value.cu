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

#ifdef PADDLE_WITH_HETERPS

#include "paddle/fluid/framework/fleet/heter_ps/gpu_value_inl.h"
#include "paddle/fluid/framework/fleet/heter_ps/dy_gpu_value_inl.h"

namespace paddle {
namespace framework {

void GlobalValueTransfor::init(std::string accessor_type, std::string gpu_value_type) {
  if (transobj_ != nullptr) {
    return;
  }
  if (accessor_type == "DownpourCtrDymfAccessor" && gpu_value_type == "DyFeatureValue") {
    transobj_ = (ValueTransfor*)(new T_DyGpuValue_DownpourCtrDymfAccessor());
  } else if (accessor_type == "DownpourCtrAccessor" && gpu_value_type == "FeatureValue") {
    transobj_ = (ValueTransfor*)(new T_GpuValue_DownpourCtrAccessor());
  }
  return;
}

ValueTransfor* GlobalValueTransfor::get_value_transfor() {
  return transobj_;
}


}
}

#endif