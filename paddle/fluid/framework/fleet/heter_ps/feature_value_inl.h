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

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"

namespace paddle {
namespace framework {

template <typename ValueType>
__global__ void kernel_value_to_cvm(float** dest, ValueType* src, FeatureKey** keys, const int slot_num,
                              const int64_t* len, const int* slot_dim, int64_t total_len, int hidden_size, int value_size) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    int cur_dim =hidden_size - 3;
    //动态维度
    if (slot_dim != nullptr) {
      cur_dim = slot_dim[x] - 3;
    }  
    char* p_src = (char*)(src);
    ValueType* value_ptr = (ValueType*)(p_src + uint64_t(i) * uint64_t(value_size));
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * (cur_dim + 3)) = 0;
      *(dest[x] + y * (cur_dim + 3) + 1) = 0;
      *(dest[x] + y * (cur_dim + 3) + 2) = 0;
      for (int j = 0; j < cur_dim; j++) {
        *(dest[x] + y * (cur_dim + 3) + 3 + j) = 0;
      }
    } else {
      value_ptr->to_cvm(dest[x] + y * (cur_dim + 3), cur_dim);
    }
  }
}

template <typename PushValueType>
__global__ void kernel_grad_to_push(PushValueType* des, float** src, const int slot_num, const int64_t* len,
                                    const int* slot_dim, int64_t total_len, int hidden_size, int value_size,
                                    int batch_size, const int* slot_vector) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    char* d_src = (char*)(des);
    PushValueType* value_ptr = (PushValueType*)(d_src + i * value_size);
    int mf_dim = hidden_size - 3; 
    if (slot_dim != nullptr) {
      mf_dim = slot_dim[x];
    }
    int slot_id = slot_vector[x];
    value_ptr->from_grad(src[x] + y * (mf_dim + 3), mf_dim, slot_id, batch_size);
  }
}

class ValueTransforImp : public ValueTransfor {
protected:
  template <typename ValueType>
  void value_to_cvm_impl( float** gpu_cvm,
                          ValueType* gpu_value,
                          FeatureKey** gpu_keys,
                          const int slot_num,
                          const int64_t* key_len,
                          const int* slot_dim,
                          int64_t total_length,
                          int hidden_size,
                          int value_size,
                          cudaStream_t stream) {
    kernel_value_to_cvm<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
        gpu_cvm, gpu_value, gpu_keys, slot_num, key_len, slot_dim, total_length, hidden_size, value_size);
    cudaStreamSynchronize(stream);
  }
  template <typename PushValueType>
  void grad_to_push_impl(PushValueType* push_value,
                            float** grad_value,
                            const int slot_num,
                            const int64_t* grad_len,
                            const int* slot_dim,
                            int64_t total_length,
                            int hidden_size,
                            int value_size,
                            int batch_size,
                            const int* slot_vector,
                            cudaStream_t stream
                            ) {
    kernel_grad_to_push<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
          (PushValueType*)push_value, grad_value, slot_num, grad_len, slot_dim,
                       total_length, hidden_size, value_size, batch_size, slot_vector);
    cudaStreamSynchronize(stream);
  }
};

}
}

#endif