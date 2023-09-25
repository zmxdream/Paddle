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

#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
// #include "paddle/fluid/framework/mixed_vector.h"
namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void im2col(const T* data_im, int num_outs, int im_height,
                       int im_width, int dilation_h, int dilation_w,
                       int filter_height, int filter_width, int stride_height,
                       int stride_width, int padding_height, int padding_width,
                       int col_height, int col_width, T* data_col,
                       const DataLayout data_layout) {
  int input_channels = num_outs / col_height / col_width;
  int channels_col = input_channels * filter_height * filter_width;
  const int index =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  if (index < num_outs) {
    int w_out = (data_layout != DataLayout::kNHWC
                     ? index % col_width
                     : (index / input_channels) % col_width);
    int h_out = (data_layout != DataLayout::kNHWC
                     ? (index / col_width) % col_height
                     : (index / input_channels / col_width) % col_height);
    int channel_in =
        (data_layout != DataLayout::kNHWC ? index / col_width / col_height
                                          : index % input_channels);
    int channel_out = channel_in * filter_height * filter_width;
    int h_in = h_out * stride_height - padding_height;
    int w_in = w_out * stride_width - padding_width;

    data_col += (channel_out * col_height + h_out) * col_width + w_out;
    for (int i = 0; i < filter_height; ++i) {
      for (int j = 0; j < filter_width; ++j) {
        int rIdx = h_in + i * dilation_h;
        int cIdx = w_in + j * dilation_w;
        int im_idx;
        if (data_layout != DataLayout::kNHWC) {
          im_idx = (channel_in * im_height + rIdx) * im_width + cIdx;
        } else {
          im_idx = (rIdx * im_width + cIdx) * input_channels + channel_in;
        }
        *data_col =
            (rIdx >= im_height || rIdx < 0 || cIdx >= im_width || cIdx < 0)
                ? 0
                : data_im[im_idx];
        data_col += col_height * col_width;
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class DeviceContext, class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO, DeviceContext,
                    T> {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* col,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im.dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im.dims()));
    PADDLE_ENFORCE_EQ(col->dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col->dims()));

    int im_channels =
        (data_layout != DataLayout::kNHWC ? im.dims()[0] : im.dims()[2]);
    int im_height =
        (data_layout != DataLayout::kNHWC ? im.dims()[1] : im.dims()[0]);
    int im_width =
        (data_layout != DataLayout::kNHWC ? im.dims()[2] : im.dims()[1]);
    int filter_height = col->dims()[1];
    int filter_width = col->dims()[2];
    int col_height = col->dims()[3];
    int col_width = col->dims()[4];

    int num_outputs = im_channels * col_height * col_width;
    int num_thread = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &num_thread);
#endif
    int blocks = (num_outputs + num_thread - 1) / num_thread;
    int block_x = 512;
    int block_y = (blocks + 512 - 1) / 512;
    dim3 threads(num_thread, 1);
    dim3 grid(block_x, block_y);
    im2col<T><<<grid, threads, 0, context.stream()>>>(
        im.data<T>(), num_outputs, im_height, im_width, dilation[0],
        dilation[1], filter_height, filter_width, stride[0], stride[1],
        padding[0], padding[1], col_height, col_width, col->data<T>(),
        data_layout);
  }
};

template <class T>
__global__ void col2im(int n, const T* data_col, int im_height, int im_width,
                       int dilation_h, int dilation_w, int filter_height,
                       int filter_width, int stride_height, int stride_width,
                       int padding_height, int padding_width, int col_height,
                       int col_width, T* data_im,
                       const DataLayout data_layout) {
  const int index =
      (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

  const int d_filter_height = dilation_h * (filter_height - 1) + 1;
  const int d_filter_width = dilation_w * (filter_width - 1) + 1;

  int input_channels = n / im_height / im_width;

  if (index < n) {
    T val = 0;
    int w = (data_layout != DataLayout::kNHWC
                 ? index % im_width + padding_width
                 : (index / input_channels) % im_width + padding_width);
    int h = (data_layout != DataLayout::kNHWC
                 ? (index / im_width) % im_height + padding_height
                 : (index / input_channels / im_width) % im_height +
                       padding_height);
    int c = (data_layout != DataLayout::kNHWC ? index / im_width / im_height
                                              : index % input_channels);

    // compute the start and end of the output
    int w_col_start =
        (w < d_filter_width) ? 0 : (w - d_filter_width) / stride_width + 1;
    int w_col_end = min(w / stride_width + 1, col_width);
    int h_col_start =
        (h < d_filter_height) ? 0 : (h - d_filter_height) / stride_height + 1;
    int h_col_end = min(h / stride_height + 1, col_height);

    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_off = (h - h_col * stride_height);
        int w_off = (w - w_col * stride_width);
        if (h_off % dilation_h == 0 && w_off % dilation_w == 0) {
          h_off /= dilation_h;
          w_off /= dilation_w;
          int data_col_index =
              (((c * filter_height + h_off) * filter_width + w_off) *
                   col_height +
               h_col) *
                  col_width +
              w_col;

          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class DeviceContext, class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO, DeviceContext,
                    T> {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* im,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im->dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im->dims()));
    PADDLE_ENFORCE_EQ(col.dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col.dims()));

    int im_channels =
        (data_layout != DataLayout::kNHWC ? im->dims()[0] : im->dims()[2]);
    int im_height =
        (data_layout != DataLayout::kNHWC ? im->dims()[1] : im->dims()[0]);
    int im_width =
        (data_layout != DataLayout::kNHWC ? im->dims()[2] : im->dims()[1]);
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int col_height = col.dims()[3];
    int col_width = col.dims()[4];

    PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2] -
                       (dilation[0] * (filter_height - 1) + 1)) /
                              stride[0] +
                          1,
                      col_height, platform::errors::InvalidArgument(
                                      "Output_height and padding(padding_up, "
                                      "padding_down) are inconsistent."));
    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3] -
                       (dilation[1] * (filter_width - 1) + 1)) /
                              stride[1] +
                          1,
                      col_width, platform::errors::InvalidArgument(
                                     "col_width and padding(padding_left, "
                                     "padding_right) are inconsistent."));

    size_t num_kernels = im_channels * im_height * im_width;

    int num_thread = 1024;
#ifdef WITH_NV_JETSON
    platform::ChangeThreadNum(context, &num_thread);
#endif
    size_t blocks = (num_kernels + num_thread - 1) / num_thread;
    size_t block_x = 512;
    size_t block_y = (blocks + 512 - 1) / 512;
    dim3 threads(num_thread, 1);
    dim3 grid(block_x, block_y);

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    col2im<T><<<grid, threads, 0, context.stream()>>>(
        num_kernels, col.data<T>(), im_height, im_width, dilation[0],
        dilation[1], filter_height, filter_width, stride[0], stride[1],
        padding[0], padding[1], col_height, col_width, im->data<T>(),
        data_layout);
  }
};

template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CUDADeviceContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CUDADeviceContext, double>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             phi::GPUContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kCFO,
                             phi::GPUContext, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CUDADeviceContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             platform::CUDADeviceContext, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             phi::GPUContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kCFO,
                             phi::GPUContext, double>;

template <class T>
__global__ void im2colOCF(const T* im_data, int im_channels, int im_height,
                          int im_width, int filter_height, int filter_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width, int col_height,
                          int col_width, T* col_data) {
  int swid = blockIdx.x;
  int shid = blockIdx.y;
  for (int channelid = threadIdx.z; channelid < im_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * im_width +
                        channelid * im_height * im_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * col_width + swid) *
                             (im_channels * filter_height * filter_width);

        col_data[col_offset] =
            (height_offset >= im_height || height_offset < 0 ||
             width_offset >= im_width || width_offset < 0)
                ? T(0)
                : im_data[im_offset];
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class DeviceContext, class T>
class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF, DeviceContext,
                    T> {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* col,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im.dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im.dims()));
    PADDLE_ENFORCE_EQ(col->dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col->dims()));

    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[3];
    int filter_width = col->dims()[4];
    int col_height = col->dims()[0];
    int col_width = col->dims()[1];

    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y, std::min(block_dim_z, im_channels));
    dim3 grid(col_width, col_height);
    im2colOCF<T><<<grid, threads, 0, context.stream()>>>(
        im.data<T>(), im_channels, im_height, im_width, filter_height,
        filter_width, stride[0], stride[1], padding[0], padding[1], col_height,
        col_width, col->data<T>());
  }
};

/// Fuse

template <class T>
__global__ void im2colOCF_Fuse(T** im_data, const int size, const size_t* lod_level, int im_channels, int* im_height,
                          int im_width, int filter_height, int filter_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width, int* col_height,
                          int col_width, T** col_data) {
  int swid = blockIdx.x;
  int shid = blockIdx.y; 
  int sid = blockIdx.z;
  if (sid >= size) return;
  if (shid >= col_height[sid]) return;
  if (lod_level[sid] == lod_level[sid + 1]) return;

  for (int channelid = threadIdx.z; channelid < im_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * im_width +
                        channelid * im_height[sid] * im_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * col_width + swid) *
                             (im_channels * filter_height * filter_width);

        col_data[sid][col_offset] =
            (height_offset >= im_height[sid] || height_offset < 0 ||
             width_offset >= im_width || width_offset < 0)
                ? T(0)
                : im_data[sid][im_offset];
      }
    }
  }

}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class DeviceContext, class T>
class Im2ColFuseFunctor<paddle::operators::math::ColFormat::kOCF, DeviceContext,
                    T> {
 public:
  void operator()(const DeviceContext& context, T** im_data,
                  const int size, const int filter_height, const int filter_width,
                  const int im_width, const int col_width, const int max_col_height, const int im_channels,
                  int* col_height_data, int* im_height_data, size_t* lod_level_0_data,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, T** col_data,
                  const DataLayout data_layout) {
    // PADDLE_ENFORCE_EQ(im.dims().size(), 3,
    //                   platform::errors::InvalidArgument(
    //                       "The dimension of tensor 'im' should be 3. But got "
    //                       "the dims of tensor 'im' is [%s].",
    //                     im.dims()));
    // PADDLE_ENFORCE_EQ(col->dims().size(), 5,
    //                   platform::errors::InvalidArgument(
    //                       "The dimension of tensor 'col' should be 5. But got "
    //                       "the dims of tensor 'col' is [%s].",
    //                       col->dims()));

/*
    // 优化掉，放在第一个循环里
    int im_channels = -1;
    std::vector<int> im_height(size, 0);
    int im_width = -1;
    int filter_height = -1;
    int filter_width = -1;
    int col_width = -1;
    std::vector<int> col_height(size, 0);
    // framework::Vector<size_t> lod_level_0_cuda = lod_level_0;
    // im_data, col_data
    std::vector<T*> im_datas(size);
    std::vector<T*> col_datas(size);
    int max_col_height = -1;

    for (int i= 0; i < size; i++) {
      if (lod_level_0[i] == lod_level_0[i+1]) {
        im_datas[i] = nullptr;
        col_datas[i] = nullptr;
        continue;
      }
      // == 其实也可以去掉 ==
      PADDLE_ENFORCE_EQ(im[i].dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im[i].dims()));
      PADDLE_ENFORCE_EQ(col[i].dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col[i].dims()));
      // == 其实也可以去掉 ==
      im_height[i] = im[i].dims()[1];
      if (im_channels == -1)im_channels = im[i].dims()[0];
      if (im_width == -1) im_width = im[i].dims()[2];
      if (filter_height == -1) filter_height = col[i].dims()[3];
      if (filter_width == -1) filter_width = col[i].dims()[4];
      if (col_width == -1) col_width = col[i].dims()[1];
      col_height[i] = col[i].dims()[0];
      if (col_height[i] > max_col_height) max_col_height = col_height[i];

      im_datas[i] = im[i].data<T>(); 
      col_datas[i] = col[i].data<T>();
    }
*/

    // int im_channels = im.dims()[0];
    // int im_height = im.dims()[1];
    // int im_width = im.dims()[2];
    // int filter_height = col->dims()[3];
    // int filter_width = col->dims()[4];
    // int col_height = col->dims()[0];
    // int col_width = col->dims()[1];

    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    // 一次分配一块大显存
    // 一次拷贝到大块显存
    // auto gpu_place = context.GetPlace();
    // auto all_hbm = memory::Alloc(gpu_place, (4 * size  + 1 + (4 * size + 1) % 2) * sizeof(uint64_t));
    /*
    auto mixv_im_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_col_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_lod_level_0_cuda = memory::Alloc(gpu_place, (size + 1) * sizeof(size_t));
    auto mixv_im_data = memory::Alloc(gpu_place, size * sizeof(T*));
    auto mixv_col_data = memory::Alloc(gpu_place, size * sizeof(T*));

    int* im_height_data = reinterpret_cast<int*>(mixv_im_height->ptr());
    int* col_height_data = reinterpret_cast<int*>(mixv_col_height->ptr());
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(mixv_lod_level_0_cuda->ptr());
    T** im_data = reinterpret_cast<T**>(mixv_im_data->ptr());
    T** col_data = reinterpret_cast<T**>(mixv_col_data->ptr());
    */
    /*
    int* im_height_data = reinterpret_cast<int*>(all_hbm->ptr());
    int* col_height_data = reinterpret_cast<int*>(im_height_data + size);
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(col_height_data + size);
    T** im_data = reinterpret_cast<T**>(lod_level_0_data + size + 1);
    T** col_data = reinterpret_cast<T**>(im_data + size);

    // 其实im_height 就是col_height，这块可以继续优化 
    cudaMemcpy(im_height_data, im_height.data(), thread_size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_height_data, col_height.data(), size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lod_level_0_data, lod_level_0.data(), (size + 1)  * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(im_data, im_datas.data(), size * sizeof(T*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_data, col_datas.data(), size * sizeof(T*),
               cudaMemcpyHostToDevice);
    // paddle::framework::MixVector<int> mixv_im_height(&im_height);
    // paddle::framework::MixVector<int> mixv_col_height(&col_height);
    // paddle::framework::MixVector<size_t> mixv_lod_level_0_cuda(&lod_level_0);
    // paddle::framework::MixVector<T*> mixv_im_data(&im_datas);
    // paddle::framework::MixVector<T*> mixv_col_data(&col_datas);
    // auto gpu_place = context.GetPlace();
    // int* im_height_data = mixv_im_height.CUDAMutableData(gpu_place);
    // int* col_height_data = mixv_col_height.CUDAMutableData(gpu_place);
    // size_t* lod_level_0_data = mixv_lod_level_0_cuda.CUDAMutableData(gpu_place);
    // T** im_data = mixv_im_data.CUDAMutableData(gpu_place); 
    // T** col_data = mixv_col_data.CUDAMutableData(gpu_place);
*/

    // == Fuse ==
    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y, std::min(block_dim_z, im_channels));
    dim3 grid(col_width, max_col_height, size);

    // im2colOCF<T><<<grid, threads, 0, context.stream()>>>(
    //     im.data<T>(), im_channels, im_height, im_width, filter_height,
    //     filter_width, stride[0], stride[1], padding[0], padding[1], col_height,
    //     col_width, col->data<T>());

    im2colOCF_Fuse<T><<<grid, threads, 0, context.stream()>>>(
     im_data, size, lod_level_0_data, im_channels, im_height_data, im_width, filter_height,
     filter_width, stride[0], stride[1], padding[0], padding[1], col_height_data,
     col_width, col_data);
    // == Fuse ==

  }
};


/// Fuse



template <class T>
__global__ void col2imOCF(const T* col_data, int im_channels, int im_height,
                          int im_width, int filter_height, int filter_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width, int col_height,
                          int col_width, T* im_data) {
  int swid = blockIdx.x;
  int shid = blockIdx.y;
  for (int channelid = threadIdx.z; channelid < im_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * im_width +
                        channelid * im_height * im_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * col_width + swid) *
                             (im_channels * filter_height * filter_width);

        if (height_offset >= 0 && height_offset < im_height &&
            width_offset >= 0 && width_offset < im_width) {
          paddle::platform::CudaAtomicAdd(im_data + im_offset,
                                          col_data[col_offset]);
        }
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class DeviceContext, class T>
class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF, DeviceContext,
                    T> {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, framework::Tensor* im,
                  const DataLayout data_layout) {
    PADDLE_ENFORCE_EQ(im->dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im->dims()));
    PADDLE_ENFORCE_EQ(col.dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col.dims()));

    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int col_height = col.dims()[0];
    int col_width = col.dims()[1];

    PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2] -
                       (dilation[0] * (filter_height - 1) + 1)) /
                              stride[0] +
                          1,
                      col_height, platform::errors::InvalidArgument(
                                      "Output_height and padding(padding_up, "
                                      "padding_down) are inconsistent."));
    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3] -
                       (dilation[1] * (filter_width - 1) + 1)) /
                              stride[1] +
                          1,
                      col_width, platform::errors::InvalidArgument(
                                     "col_width and padding(padding_left, "
                                     "padding_right) are inconsistent."));

    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y, std::min(block_dim_z, im_channels));
    dim3 grid(col_width, col_height);
    col2imOCF<T><<<grid, threads, 0, context.stream()>>>(
        col.data<T>(), im_channels, im_height, im_width, filter_height,
        filter_width, stride[0], stride[1], padding[0], padding[1], col_height,
        col_width, im->data<T>());
  }
};


// === Fuse ===

template <class T>
__global__ void col2imOCF_Fuse(T** col_data, const int size, const size_t* lod_level, int im_channels, int* im_height,
                          int im_width, int filter_height, int filter_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width, int* col_height,
                          int col_width, T** im_data) {
  int swid = blockIdx.x;
  int shid = blockIdx.y;
  int sid = blockIdx.z;
  if (sid >= size) return;
  if (shid >= col_height[sid]) return;
  if (lod_level[sid] == lod_level[sid + 1]) return;

  for (int channelid = threadIdx.z; channelid < im_channels;
       channelid += blockDim.z) {
    for (int idy = threadIdx.y; idy < filter_height; idy += blockDim.y) {
      for (int idx = threadIdx.x; idx < filter_width; idx += blockDim.x) {
        int width_offset = idx + swid * stride_width - padding_width;
        int height_offset = idy + shid * stride_height - padding_height;
        int im_offset = width_offset + height_offset * im_width +
                        channelid * im_height[sid] * im_width;

        int col_offset = idx + idy * filter_width +
                         channelid * filter_height * filter_width +
                         (shid * col_width + swid) *
                             (im_channels * filter_height * filter_width);

        if (height_offset >= 0 && height_offset < im_height[sid] &&
            width_offset >= 0 && width_offset < im_width) {
          paddle::platform::CudaAtomicAdd(im_data[sid] + im_offset,
                                          col_data[sid][col_offset]);
        }
      }
    }
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class DeviceContext, class T>
class Col2ImFuseFunctor<paddle::operators::math::ColFormat::kOCF, DeviceContext,
                    T> {
 public:
  // void operator()(const DeviceContext& context, T** im_data,
  //                 const int size, const int filter_height, const int filter_width,
  //                 const int im_width, const int col_width, const int max_col_height, const int im_channels,
  //                 int* col_height_data, int* im_height_data, size_t* lod_level_0_data,
  //                 const std::vector<int>& dilation,
  //                 const std::vector<int>& stride,
  //                 const std::vector<int>& padding, T** col_data,
  //                 const DataLayout data_layout) {
  void operator()(const DeviceContext& context, T** col_data,
                  const int size, const int filter_height, const int filter_width,
                  const int im_width, const int col_width, const int max_col_height, const int im_channels,
                  int* col_height_data, int* im_height_data, size_t* lod_level_0_data,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding, T** im_data,
                  const DataLayout data_layout) {

/*
    PADDLE_ENFORCE_EQ(im->dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im->dims()));
    PADDLE_ENFORCE_EQ(col.dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col.dims()));
*/

    // int im_channels = im->dims()[0];
    // int im_height = im->dims()[1];
    // int im_width = im->dims()[2];
    // int filter_height = col.dims()[3];
    // int filter_width = col.dims()[4];
    // int col_height = col.dims()[0];
    // int col_width = col.dims()[1];

    // ===============
/*
    int im_channels = -1;
    std::vector<int> im_height(size, 0);
    int im_width = -1;
    int filter_height = -1;
    int filter_width = -1;
    int col_width = -1;
    std::vector<int> col_height(size, 0);
    // framework::Vector<size_t> lod_level_0_cuda = lod_level_0;
    // im_data, col_data
    std::vector<T*> im_datas(size);
    std::vector<T*> col_datas(size);
    int max_col_height = -1;

    for (int i= 0; i < size; i++) {
      if (lod_level_0[i] == lod_level_0[i+1]) {
        im_datas[i] = nullptr;
        col_datas[i] = nullptr;
        continue;
      }
      // == 其实也可以去掉 ==
      PADDLE_ENFORCE_EQ(im[i].dims().size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'im' should be 3. But got "
                          "the dims of tensor 'im' is [%s].",
                          im[i].dims()));
      PADDLE_ENFORCE_EQ(col[i].dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "The dimension of tensor 'col' should be 5. But got "
                          "the dims of tensor 'col' is [%s].",
                          col[i].dims()));
      // == 其实也可以去掉 ==
      im_height[i] = im[i].dims()[1];
      if (im_channels == -1)im_channels = im[i].dims()[0];
      if (im_width == -1) im_width = im[i].dims()[2];
      if (filter_height == -1) filter_height = col[i].dims()[3];
      if (filter_width == -1) filter_width = col[i].dims()[4];
      if (col_width == -1) col_width = col[i].dims()[1];
      col_height[i] = col[i].dims()[0];
      if (col_height[i] > max_col_height) max_col_height = col_height[i];

      im_datas[i] = im[i].data<T>(); 
      col_datas[i] = col[i].data<T>();

      PADDLE_ENFORCE_EQ((im_height[i] + padding[0] + padding[2] -
                         (dilation[0] * (filter_height - 1) + 1)) /
                                stride[0] +
                            1,
                        col_height[i], platform::errors::InvalidArgument(
                                      "Output_height and padding(padding_up, "
                                      "padding_down) are inconsistent."));




    }
*/

    // ===============


    // PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2] -
    //                    (dilation[0] * (filter_height - 1) + 1)) /
    //                           stride[0] +
    //                       1,
    //                   col_height, platform::errors::InvalidArgument(
    //                                   "Output_height and padding(padding_up, "
    //                                   "padding_down) are inconsistent."));
    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3] -
                       (dilation[1] * (filter_width - 1) + 1)) /
                              stride[1] +
                          1,
                      col_width, platform::errors::InvalidArgument(
                                     "col_width and padding(padding_left, "
                                     "padding_right) are inconsistent."));



/*
    paddle::framework::MixVector<int> mixv_im_height(&im_height);
    paddle::framework::MixVector<int> mixv_col_height(&col_height);
    paddle::framework::MixVector<size_t> mixv_lod_level_0_cuda(&lod_level_0);
    paddle::framework::MixVector<T*> mixv_im_data(&im_datas);
    paddle::framework::MixVector<T*> mixv_col_data(&col_datas);
    auto gpu_place = context.GetPlace();
    int* im_height_data = mixv_im_height.CUDAMutableData(gpu_place);
    int* col_height_data = mixv_col_height.CUDAMutableData(gpu_place);
    size_t* lod_level_0_data = mixv_lod_level_0_cuda.CUDAMutableData(gpu_place);
    T** im_data = mixv_im_data.CUDAMutableData(gpu_place); 
    T** col_data = mixv_col_data.CUDAMutableData(gpu_place);
*/

/*
    auto gpu_place = context.GetPlace();
    auto mixv_im_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_col_height = memory::Alloc(gpu_place, size * sizeof(int));
    auto mixv_lod_level_0_cuda = memory::Alloc(gpu_place, (size + 1) * sizeof(size_t));
    auto mixv_im_data = memory::Alloc(gpu_place, size * sizeof(T*));
    auto mixv_col_data = memory::Alloc(gpu_place, size * sizeof(T*));

    int* im_height_data = reinterpret_cast<int*>(mixv_im_height->ptr());
    int* col_height_data = reinterpret_cast<int*>(mixv_col_height->ptr());
    size_t* lod_level_0_data = reinterpret_cast<size_t*>(mixv_lod_level_0_cuda->ptr());
    T** im_data = reinterpret_cast<T**>(mixv_im_data->ptr());
    T** col_data = reinterpret_cast<T**>(mixv_col_data->ptr());
     
    // 其实im_height 就是col_height，这块可以继续优化 
    cudaMemcpy(im_height_data, im_height.data(), size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_height_data, col_height.data(), size * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lod_level_0_data, lod_level_0.data(), (size + 1)  * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(im_data, im_datas.data(), size * sizeof(T*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(col_data, col_datas.data(), size * sizeof(T*),
               cudaMemcpyHostToDevice);

*/


    int block_dim_x = 0;
    int block_dim_y = 0;
    if (filter_height <= 4 && filter_width <= 4) {
      block_dim_x = 4;
      block_dim_y = 4;
    } else if (filter_height <= 8 && filter_width <= 8) {
      block_dim_x = 8;
      block_dim_y = 8;
    } else if (filter_height <= 16 && filter_width <= 16) {
      block_dim_x = 16;
      block_dim_y = 16;
    } else {
      block_dim_x = 32;
      block_dim_y = 32;
    }

    int block_dim_z = 1024 / block_dim_x / block_dim_y;
    dim3 threads(block_dim_x, block_dim_y, std::min(block_dim_z, im_channels));
    dim3 grid(col_width, max_col_height, size);
    
    col2imOCF_Fuse<T><<<grid, threads, 0, context.stream()>>>(
         col_data, size, lod_level_0_data, im_channels, im_height_data, im_width, filter_height,
         filter_width, stride[0], stride[1], padding[0], padding[1], col_height_data,
         col_width, im_data);

    // col2imOCF<T><<<grid, threads, 0, context.stream()>>>(
    //     col.data<T>(), im_channels, im_height, im_width, filter_height,
    //     filter_width, stride[0], stride[1], padding[0], padding[1], col_height,
    //     col_width, im->data<T>());
  }
};

// === Fuse ===




template class Im2ColFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, float>;
template class Im2ColFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, double>;
template class Im2ColFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, float>;
template class Im2ColFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, double>;

template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, double>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, float>;
template class Im2ColFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, double>;

template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, double>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, float>;
template class Col2ImFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, double>;


template class Col2ImFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, float>;
template class Col2ImFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             platform::CUDADeviceContext, double>;
template class Col2ImFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, float>;
template class Col2ImFuseFunctor<paddle::operators::math::ColFormat::kOCF,
                             phi::GPUContext, double>;



}  // namespace math
}  // namespace operators
}  // namespace paddle
