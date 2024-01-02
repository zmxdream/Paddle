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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"
namespace phi {
template <>
void MatMul<phi::GPUContext, float>(const phi::GPUContext& dev_ctx,
                                    const DenseTensor& a,
                                    bool trans_a,
                                    const DenseTensor& b,
                                    bool trans_b,
                                    DenseTensor* out) {
  dev_ctx.template Alloc<float>(out);
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060)
  if (a.dims().size() == 2 && b.dims().size() == 2) {
    auto& x_dims = a.dims();  // M * K
    auto& y_dims = b.dims();  // K * N
    const int M = trans_a ? x_dims[1] : x_dims[0];
    const int K = trans_a ? x_dims[0] : x_dims[1];
    const int N = trans_b ? y_dims[0] : y_dims[1];
    phi::funcs::LinearWithCublasLt<float>::Run(
        dev_ctx,
        &a,       // x
        &b,       // y
        out,      // out
        nullptr,  // bias
        nullptr,
        M,  // M  bsz_seqf
        N,  // N  output_size
        K,  // K  input_size
        trans_a,
        trans_b,
        phi::funcs::MatmulFusedType::kMatmul);
    return;
  }
#endif
  auto blas = phi::funcs::GetBlas<phi::GPUContext, float>(dev_ctx);
  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a.dims(), 0, trans_a);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b.dims(), 0, trans_b);
  if (a.dims().size() == 3 && b.dims().size() <= 2) {
    // the transpose_X must be false, if is true, the transpose cost much time
    if (!trans_a) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    }
  }
  blas.MatMul(a.data<float>(),
              mat_dim_a,
              b.data<float>(),
              mat_dim_b,
              static_cast<float>(1),
              dev_ctx.template Alloc<float>(out),
              static_cast<float>(false));
}
}  // namespace phi

PD_REGISTER_KERNEL(matmul_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_triple_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(matmul_with_flatten_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(matmul_with_flatten_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
