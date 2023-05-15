// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sfm/types.h>

#pragma once
namespace sfm {
namespace utils {
namespace internal {
/* chol_matinv_batch_async() inverts one or many real positive-definite
   matrices.

   A     pointer to an array of the real positive-definite matrices to be
         inverted
   Ainv  pointer to an array of the single-precision matrices which receive
         the inverses of the corresponding matrices pointed to by A, where
         each matrix is stored in column-major order
   size  number of rows and columns of the matrices in the arrays pointed to
         by A and Ainv. n must be greater than, or equal to 2. On sm_13 GPUs,
         n must be less than, or equal to, 62. On sm_2x and sm_3x GPUs, n must
         be less than, or equal to, 109.
   batch the number of matrices to be inverted. It must be greater than zero.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/
template <typename T>
int chol_matinv_batch_async(const T *A, T *Ainv, int_t size, int_t batch,
                            cudaStream_t stream = 0);

template <typename T>
int chol_matinv_batch_general_async(const T *A, T *Ainv, int_t size,
                                    int_t batch, cudaStream_t stream = 0);

template <typename T>
int chol_matinv_batch_per_thread_async(const T *A, T *Ainv, int_t size,
                                       int_t batch, cudaStream_t stream = 0);

template <typename T>
int chol_matinv_batch_async(T alpha, T beta, const T *A, T *Ainv, int_t size,
                            int_t batch, cudaStream_t stream = 0);

template <typename T>
int chol_matinv_batch_general_async(T alpha, T beta, const T *A, T *Ainv,
                                    int_t size, int_t batch,
                                    cudaStream_t stream = 0);

template <typename T>
int chol_matinv_batch_per_thread_async(T alpha, T beta, const T *A, T *Ainv,
                                       int_t size, int_t batch,
                                       cudaStream_t stream = 0);
} // namespace internal
} // namespace utils
} // namespace sfm