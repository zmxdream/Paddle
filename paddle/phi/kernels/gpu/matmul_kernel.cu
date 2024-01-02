/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"
#include "paddle/phi/kernels/matmul_kernel.h"

namespace phi {
template <>
void MatMulFunction<phi::GPUContext, float>(const phi::GPUContext& dev_ctx,
                                            const DenseTensor& X,
                                            const DenseTensor& Y,
                                            DenseTensor* Out,
                                            bool trans_x,
                                            bool trans_y) {
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)
  if (X.dims().size() == 2 && Y.dims().size() == 2) {
    auto& x_dims = X.dims();  // M * K
    auto& y_dims = Y.dims();  // K * N
    const int M = trans_x ? x_dims[1] : x_dims[0];
    const int K = trans_x ? x_dims[0] : x_dims[1];
    const int N = trans_y ? y_dims[0] : y_dims[1];
    phi::funcs::LinearWithCublasLt<float>::Run(
        dev_ctx,
        &X,       // x
        &Y,       // y
        Out,      // out
        nullptr,  // bias
        nullptr,
        M,  // M  bsz_seqf
        N,  // N  output_size
        K,  // K  input_size
        trans_x,
        trans_y,
        phi::funcs::MatmulFusedType::kMatmul);
    return;
  }
#endif
  const std::vector<std::int64_t> x_dims = vectorize(X.dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y.dims());
  MatMulFunction<phi::GPUContext, float>(
      dev_ctx, X, Y, x_dims, y_dims, Out, trans_x, trans_y, false);
}
}  // namespace phi

PD_REGISTER_KERNEL(matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_with_flatten,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
