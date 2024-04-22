/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/xpu/xpu_info.h"

namespace paddle {
namespace platform {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() { return phi::backends::xpu::GetDriverVersion(); }

//! Get the version of XPU Runtime
int GetRuntimeVersion() { return phi::backends::xpu::GetRuntimeVersion(); }

/**************************** Device Management **************************/

int GetXPUDeviceCount() { return phi::backends::xpu::GetXPUDeviceCount(); }

int GetXPUCurrentDeviceId() {
  return phi::backends::xpu::GetXPUCurrentDeviceId();
}

void SetXPUDeviceId(int id) { phi::backends::xpu::SetXPUDeviceId(id); }

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  return phi::backends::xpu::GetXPUSelectedDevices();
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUL3Place& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUL3Place& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUL3Place& dst_place,
                   const void* src,
                   const platform::XPUL3Place& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUL3Place& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}
void MemcpySyncD2D(void* dst,
                   const platform::XPUL3Place& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::Place& dst_place) {
  if (dst_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncH2D(dst, src, count, place_dst);
  } else if(dst_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncH2D(dst, src, count, place_dst);
  }
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::Place& src_place) {
  if (src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    MemcpySyncD2H(dst, src, count, place_src);
  } else if(src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    MemcpySyncD2H(dst, src, count, place_src);
  }
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::Place& dst_place,
                   const void* src,
                   const platform::Place& src_place,
                   size_t count) {
  if (src_place.GetType() == phi::AllocationType::XPUL3 && src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPU && src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPU && src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPUL3 && src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  }
}

void XPUStreamSync(xpuStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}

/**************************** Others **************************/

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id) {
  return phi::backends::xpu::get_xpu_version(dev_id);
}

std::once_flag XPUMLHandler::init_flag_;

XPUMLHandler::XPUMLHandler() {
  std::call_once(XPUMLHandler::init_flag_, &XPUMLHandler::init_ml);
  xpumlDeviceGetCount(&device_nums_);
  device_handlers_.resize(device_nums_);
  mem_infos_.resize(device_nums_);
  for (unsigned int i = 0; i < device_nums_; ++i) {
      xpumlDeviceGetHandleByIndex(i, &device_handlers_[i]);
  }
}

void XPUMLHandler::init_ml() {
    xpumlInit();
}

bool XPUMLHandler::getMemoryUsageInfo(int dev_id, unsigned long long *total,
                                      unsigned long long* used, unsigned long long *free) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return false;
  }
  *total = mem_infos_[dev_id].totalGlobalMemory;
  *free = mem_infos_[dev_id].freeGlobalMemory;
  *used = mem_infos_[dev_id].usedGlobalMemory;
  return true;
}

bool XPUMLHandler::getL3UsageInfo(int dev_id, unsigned long long *total,
                                  unsigned long long *used, unsigned long long *free) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return false;
  }
  *total = mem_infos_[dev_id].totalL3Memory;
  *free = mem_infos_[dev_id].freeL3Memory;
  *used = mem_infos_[dev_id].usedL3Memory;
  return true;
}

std::tuple<unsigned long long, unsigned long long, unsigned long long> XPUMLHandler::getMemoryUsageTuple(int dev_id) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return {0, 0, 0};
  }
  return {mem_infos_[dev_id].totalGlobalMemory, 
          mem_infos_[dev_id].usedGlobalMemory, 
          mem_infos_[dev_id].freeGlobalMemory};

}

std::tuple<unsigned long long, unsigned long long, unsigned long long> XPUMLHandler::getL3UsageTuple(int dev_id) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return {0, 0, 0};
  }
  return {mem_infos_[dev_id].totalL3Memory, 
          mem_infos_[dev_id].usedL3Memory, 
          mem_infos_[dev_id].freeL3Memory};  
}

}  // namespace platform
}  // namespace paddle
