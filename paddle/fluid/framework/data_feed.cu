/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
// fill slot values
__global__ void FillSlotValueOffsetKernel(
    const int ins_num, const int used_slot_num, size_t *slot_value_offsets,
    const int *uint64_offsets, const int uint64_slot_size,
    const int *float_offsets, const int float_slot_size,
    const UsedSlotGpuType *used_slots) {

  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(slot_idx, used_slot_num) {
    int value_off = slot_idx * col_num;
    slot_value_offsets[value_off] = 0;
    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * uint64_cols + info.slot_value_idx;
        int num = uint64_offsets[pos + 1] - uint64_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    } else {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * float_cols + info.slot_value_idx;
        int num = float_offsets[pos + 1] - float_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    }
  }
}

  // 填充gpu_slot_offset
  // uint64_use_slot_size_是uint64的slot数量
  // float_use_slot_size_是float的slot数量
  // value.d_uint64_offset存储每个ins的uint64 slot lod信息
  // value.d_float_offset存储每个ins的float slot lod信息
  // 比如 ins 10个，uint64 2个，float slot 3个
  // 那么 value.d_uint64_offset的shape就是10 * (2 + 1)
  // 那么 value.d_float_offset的shape就是10 * (3 + 1)
  // used_slot_gpu_types 每个slot的信息,包括是否为uint64, 以及slot_value_idx
  // 这个函数就是填充slot_value_offsets
  //
void SlotRecordInMemoryDataFeed::FillSlotValueOffset(
    const int ins_num, const int used_slot_num, size_t *slot_value_offsets,
    const int *uint64_offsets, const int uint64_slot_size,
    const int *float_offsets, const int float_slot_size,
    const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  FillSlotValueOffsetKernel<<<GET_BLOCKS(used_slot_num), CUDA_NUM_THREADS, 0,
                              stream>>>(
      ins_num, used_slot_num, slot_value_offsets, uint64_offsets,
      uint64_slot_size, float_offsets, float_slot_size, used_slots);
  cudaStreamSynchronize(stream);
}

  // uint64_feas保存的是所有样本的uint64 key
  // uint64_ins_lens shape (ins_num + 1), 保存每个ins的uint64 feasign num数量
  // uint64_offset shape(ins_num * (uint64_slot_num + 1)),保存每个样本的uint64_slot_offset 
__global__ void CopyForTensorKernel(
    const int used_slot_num, const int ins_num, void **dest,
    const size_t *slot_value_offsets, const uint64_t *uint64_feas,
    const int *uint64_offsets, const int *uint64_ins_lens,
    const int uint64_slot_size, const float *float_feas,
    const int *float_offsets, const int *float_ins_lens,
    const int float_slot_size, const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;
  CUDA_KERNEL_LOOP(i, ins_num * used_slot_num) {
    int slot_idx = i / ins_num;
    int ins_idx = i % ins_num;
    uint32_t value_offset = slot_value_offsets[slot_idx * col_num + ins_idx];
    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      uint64_t *up = reinterpret_cast<uint64_t *>(dest[slot_idx]);
      int index = info.slot_value_idx + uint64_cols * ins_idx;
      int old_off = uint64_offsets[index];
      int num = uint64_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int uint64_value_offset = uint64_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        up[k + value_offset] = uint64_feas[k + old_off + uint64_value_offset];
      }
    } else {
      float *fp = reinterpret_cast<float *>(dest[slot_idx]);
      int index = info.slot_value_idx + float_cols * ins_idx;
      int old_off = float_offsets[index];
      int num = float_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int float_value_offset = float_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        fp[k + value_offset] = float_feas[k + old_off + float_value_offset];
      }
    }
  }
}

  // pack->resize_gpu_slot_offsets(slot_total_num * sizeof(size_t));
  // gpu_slot_offset的shape是 use_slot_size * (ins_num + 1)
  // d_uint64_keys保存的是所有样本的uint64 key
  // d_uint64_lens shape (ins_num + 1), 保存每个ins的uint64 feasign num数量
  // d_uint64_offset shape(ins_num * (uint64_slot_num + 1)),保存每个样本的uint64_slot_offset 
void SlotRecordInMemoryDataFeed::CopyForTensor(
    const int ins_num, const int used_slot_num, void **dest,
    const size_t *slot_value_offsets, const uint64_t *uint64_feas,
    const int *uint64_offsets, const int *uint64_ins_lens,
    const int uint64_slot_size, const float *float_feas,
    const int *float_offsets, const int *float_ins_lens,
    const int float_slot_size, const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num), CUDA_NUM_THREADS,
                        0, stream>>>(
      used_slot_num, ins_num, dest, slot_value_offsets, uint64_feas,
      uint64_offsets, uint64_ins_lens, uint64_slot_size, float_feas,
      float_offsets, float_ins_lens, float_slot_size, used_slots);
  cudaStreamSynchronize(stream);
}

}  // namespace framework
}  // namespace paddle
#endif
