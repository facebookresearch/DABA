// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
namespace sfm {
namespace utils {
namespace internal {
/* lu_matinv_batch_async() inverts one or many square, non-singular and real
   matrices. Partial pivoting is employed in the inversion process for increased
   numerical stability.

   A     pointer to an array of the real matrices to be inverted
   Ainv  pointer to an array of the real matrices which receive the inverses of
         the corresponding matrices pointed to by A
   n     number of rows and columns of the matrices in the arrays pointed to
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
int lu_matinv_batch_async(const T *A, T *Ainv, int n, int batch,
                          cudaStream_t stream = 0);
} // namespace internal
} // namespace utils
} // namespace sfm