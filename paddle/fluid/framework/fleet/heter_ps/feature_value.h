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

#include <iostream>
#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/memory/memory.h"
#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#endif
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#endif

namespace paddle {
namespace framework {

#define TYPE_ALIGN(ALIGNVAL, LEN)  (((uint64_t)(LEN) + ((ALIGNVAL)-1)) & ~((uint64_t)((ALIGNVAL)-1)))

typedef uint64_t FeatureKey;

class ValueTransfor {
public:
  virtual int get_gpu_value_size(int dim_size) = 0;
  virtual int get_gpu_push_value_size(int dim_size) = 0;
  virtual void value_cpu_to_gpu(void* cpu, void* gpu, int dim_size) = 0;
  virtual void value_gpu_to_cpu(void* gpu) = 0;
  virtual void value_to_cvm(float** gpu_cvm, //写入的结果，cvm二维数组
                            const void* gpu_value, //查表出来的sparse数据
                            FeatureKey** gpu_keys, //对应的key的二维数组(内部需要用来判断是否为0)
                            const int slot_num, //一共有多少个slot
                            const int64_t* key_len, //每个slot下面有多少个key
                            const int* slot_dim, //每个slot的维度数据(可能为空,只有动态维度模式才会有值)
                            int64_t total_length, //总共有多少个key
                            int hidden_size, //非动态维度的情况下，cvm维度数
                            int value_size, //动态维度下，value的字节大小
                            cudaStream_t stream //流
                            ) = 0;
  virtual void grad_to_push(void* push_value, //写入的结果，连续的pushvalue类型值
                            float** grad_value, //梯度信息
                            const int slot_num, //一共有多少个slot
                            const int64_t* grad_len, //每个slot下面有多少个梯度
                            const int* slot_dim, //每个slot的维度数据(可能为空,只有动态维度模式才会有值)
                            int64_t total_length, //总共有多少个梯度
                            int hidden_size, //非动态维度的情况下，梯度维度数
                            int value_size, //动态维度下，value的字节大小
                            int batch_size, //mini-batch
                            const int* slot_vector, //slot的编号信息
                            cudaStream_t stream //流
                            ) = 0;
};

class GlobalValueTransfor {
public:
  static GlobalValueTransfor& get_instance() {
    static GlobalValueTransfor ins;
    return ins;
  }
  void init(std::string accessor_type, std::string gpu_value_type);
  ValueTransfor* get_value_transfor();
private:
  ValueTransfor* transobj_ = nullptr;
};
#define g_transfor GlobalValueTransfor::get_instance().get_value_transfor()

class PinnedVector {
public:
  template <typename Type>
  PinnedVector(const Type* buf, const size_t len, gpuStream_t& stream, const paddle::platform::Place& place) {
    mem_cpu_ = memory::Alloc(phi::GPUPinnedPlace(), len);
    memcpy(reinterpret_cast<char*>(mem_cpu_->ptr()), buf, len);
    mem_gpu_ = memory::Alloc(place, len);
    cudaMemcpyAsync(reinterpret_cast<char*>(mem_gpu_->ptr()), reinterpret_cast<char*>(mem_cpu_->ptr()),
                    len, cudaMemcpyHostToDevice, stream);
  }
  template <typename Type>
  Type* get_gpu_ptr() {
    return reinterpret_cast<Type*>(mem_gpu_->ptr());
  }
private:
  memory::allocation::AllocationPtr mem_cpu_;
  memory::allocation::AllocationPtr mem_gpu_;
};

}  // end namespace framework
}  // end namespace paddle


#endif
