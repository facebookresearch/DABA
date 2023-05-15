// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "chol-inl.h"
#include "chol.cuh"

#include <cuda_runtime_api.h>
#include <sfm/utils/cuda_utils.h>
#include <sfm/utils/internal/shared_memory-inl.cuh>

#define A(i, j) A[(i * size + j) * batch + idx]
#define Ainv(i, j) Ainv[(i * size + j) * batch + idx]
#define mats(i) mats[i]
#define sols(i) sols[i]

namespace sfm {
namespace utils {
namespace internal {
template <typename T>
__global__ void chol_matinv_2x2_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 2;

  if (idx >= batch)
    return;

  T L00, L10, L11;
  T X00, X10, X11;

  L00 = A[0 * batch + idx] * alpha + beta;
  L10 = A[1 * batch + idx];
  L11 = A[3 * batch + idx] * alpha + beta;

  ComputeL_2x2(L00, L10, L11);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X10 = 0;
  ComputeCholInv_2x2(L00, L10, L11, X00, X10, X11);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
}

template <typename T>
__global__ void chol_matinv_3x3_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 3;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22;
  T X00, X10, X11, X20, X21, X22;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L22 = A(2, 2) * alpha + beta;

  ComputeL_3x3(L00, L10, L11, L20, L21, L22);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  ComputeCholInv_3x3(L00, L10, L11, L20, L21, L22, X00, X10, X11, X20, X21,
                     X22);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
}

template <typename T>
__global__ void chol_matinv_4x4_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 4;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L33 = A(3, 3) * alpha + beta;

  ComputeL_4x4(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  ComputeCholInv_4x4(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, X00, X10,
                     X11, X20, X21, X22, X30, X31, X32, X33);

  Ainv(0, 0) = L00;
  Ainv(1, 0) = L10;
  Ainv(2, 0) = L20;
  Ainv(3, 0) = L30;
  Ainv(0, 1) = L10;
  Ainv(1, 1) = L11;
  Ainv(2, 1) = L21;
  Ainv(3, 1) = L31;
  Ainv(0, 2) = L20;
  Ainv(1, 2) = L21;
  Ainv(2, 2) = L22;
  Ainv(3, 2) = L32;
  Ainv(0, 3) = L30;
  Ainv(1, 3) = L31;
  Ainv(2, 3) = L32;
  Ainv(3, 3) = L33;
}

template <typename T>
__global__ void chol_matinv_5x5_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 5;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L44 = A(4, 4) * alpha + beta;

  ComputeL_5x5(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42,
               L43, L44);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  ComputeCholInv_5x5(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41,
                     L42, L43, L44, X00, X10, X11, X20, X21, X22, X30, X31, X32,
                     X33, X40, X41, X42, X43, X44);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
}

template <typename T>
__global__ void chol_matinv_6x6_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 6;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
      X50, X51, X52, X53, X54, X55;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L50 = A(5, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L51 = A(5, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L52 = A(5, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L53 = A(5, 3);
  L44 = A(4, 4) * alpha + beta;
  L54 = A(5, 4);
  L55 = A(5, 5) * alpha + beta;

  ComputeL_6x6(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42,
               L43, L44, L50, L51, L52, L53, L54, L55);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X55 = 1 / L55;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  X50 = 0;
  X51 = 0;
  X52 = 0;
  X53 = 0;
  X54 = 0;
  ComputeCholInv_6x6(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41,
                     L42, L43, L44, L50, L51, L52, L53, L54, L55, X00, X10, X11,
                     X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
                     X50, X51, X52, X53, X54, X55);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(5, 0) = X50;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(5, 1) = X51;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(5, 2) = X52;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(5, 3) = X53;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
  Ainv(5, 4) = X54;
  Ainv(0, 5) = X50;
  Ainv(1, 5) = X51;
  Ainv(2, 5) = X52;
  Ainv(3, 5) = X53;
  Ainv(4, 5) = X54;
  Ainv(5, 5) = X55;
}

template <typename T>
__global__ void chol_matinv_7x7_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 7;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
      X50, X51, X52, X53, X54, X55, X60, X61, X62, X63, X64, X65, X66;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L50 = A(5, 0);
  L60 = A(6, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L51 = A(5, 1);
  L61 = A(6, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L52 = A(5, 2);
  L62 = A(6, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L53 = A(5, 3);
  L63 = A(6, 3);
  L44 = A(4, 4) * alpha + beta;
  L54 = A(5, 4);
  L64 = A(6, 4);
  L55 = A(5, 5) * alpha + beta;
  L65 = A(6, 5);
  L66 = A(6, 6) * alpha + beta;

  ComputeL_7x7(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42,
               L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64,
               L65, L66);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X55 = 1 / L55;
  X66 = 1 / L66;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  X50 = 0;
  X51 = 0;
  X52 = 0;
  X53 = 0;
  X54 = 0;
  X60 = 0;
  X61 = 0;
  X62 = 0;
  X63 = 0;
  X64 = 0;
  X65 = 0;
  ComputeCholInv_7x7(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41,
                     L42, L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62,
                     L63, L64, L65, L66, X00, X10, X11, X20, X21, X22, X30, X31,
                     X32, X33, X40, X41, X42, X43, X44, X50, X51, X52, X53, X54,
                     X55, X60, X61, X62, X63, X64, X65, X66);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(5, 0) = X50;
  Ainv(6, 0) = X60;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(5, 1) = X51;
  Ainv(6, 1) = X61;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(5, 2) = X52;
  Ainv(6, 2) = X62;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(5, 3) = X53;
  Ainv(6, 3) = X63;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
  Ainv(5, 4) = X54;
  Ainv(6, 4) = X64;
  Ainv(0, 5) = X50;
  Ainv(1, 5) = X51;
  Ainv(2, 5) = X52;
  Ainv(3, 5) = X53;
  Ainv(4, 5) = X54;
  Ainv(5, 5) = X55;
  Ainv(6, 5) = X65;
  Ainv(0, 6) = X60;
  Ainv(1, 6) = X61;
  Ainv(2, 6) = X62;
  Ainv(3, 6) = X63;
  Ainv(4, 6) = X64;
  Ainv(5, 6) = X65;
  Ainv(6, 6) = X66;
}

template <typename T>
__global__ void chol_matinv_8x8_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 8;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66, L70, L71,
      L72, L73, L74, L75, L76, L77;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
      X50, X51, X52, X53, X54, X55, X60, X61, X62, X63, X64, X65, X66, X70, X71,
      X72, X73, X74, X75, X76, X77;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L50 = A(5, 0);
  L60 = A(6, 0);
  L70 = A(7, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L51 = A(5, 1);
  L61 = A(6, 1);
  L71 = A(7, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L52 = A(5, 2);
  L62 = A(6, 2);
  L72 = A(7, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L53 = A(5, 3);
  L63 = A(6, 3);
  L73 = A(7, 3);
  L44 = A(4, 4) * alpha + beta;
  L54 = A(5, 4);
  L64 = A(6, 4);
  L74 = A(7, 4);
  L55 = A(5, 5) * alpha + beta;
  L65 = A(6, 5);
  L75 = A(7, 5);
  L66 = A(6, 6) * alpha + beta;
  L76 = A(7, 6);
  L77 = A(7, 7) * alpha + beta;

  ComputeL_8x8(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42,
               L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64,
               L65, L66, L70, L71, L72, L73, L74, L75, L76, L77);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X55 = 1 / L55;
  X66 = 1 / L66;
  X77 = 1 / L77;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  X50 = 0;
  X51 = 0;
  X52 = 0;
  X53 = 0;
  X54 = 0;
  X60 = 0;
  X61 = 0;
  X62 = 0;
  X63 = 0;
  X64 = 0;
  X65 = 0;
  X70 = 0;
  X71 = 0;
  X72 = 0;
  X73 = 0;
  X74 = 0;
  X75 = 0;
  X76 = 0;
  ComputeCholInv_8x8(
      L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66, L70, L71,
      L72, L73, L74, L75, L76, L77, X00, X10, X11, X20, X21, X22, X30, X31, X32,
      X33, X40, X41, X42, X43, X44, X50, X51, X52, X53, X54, X55, X60, X61, X62,
      X63, X64, X65, X66, X70, X71, X72, X73, X74, X75, X76, X77);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(5, 0) = X50;
  Ainv(6, 0) = X60;
  Ainv(7, 0) = X70;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(5, 1) = X51;
  Ainv(6, 1) = X61;
  Ainv(7, 1) = X71;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(5, 2) = X52;
  Ainv(6, 2) = X62;
  Ainv(7, 2) = X72;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(5, 3) = X53;
  Ainv(6, 3) = X63;
  Ainv(7, 3) = X73;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
  Ainv(5, 4) = X54;
  Ainv(6, 4) = X64;
  Ainv(7, 4) = X74;
  Ainv(0, 5) = X50;
  Ainv(1, 5) = X51;
  Ainv(2, 5) = X52;
  Ainv(3, 5) = X53;
  Ainv(4, 5) = X54;
  Ainv(5, 5) = X55;
  Ainv(6, 5) = X65;
  Ainv(7, 5) = X75;
  Ainv(0, 6) = X60;
  Ainv(1, 6) = X61;
  Ainv(2, 6) = X62;
  Ainv(3, 6) = X63;
  Ainv(4, 6) = X64;
  Ainv(5, 6) = X65;
  Ainv(6, 6) = X66;
  Ainv(7, 6) = X76;
  Ainv(0, 7) = X70;
  Ainv(1, 7) = X71;
  Ainv(2, 7) = X72;
  Ainv(3, 7) = X73;
  Ainv(4, 7) = X74;
  Ainv(5, 7) = X75;
  Ainv(6, 7) = X76;
  Ainv(7, 7) = X77;
}

template <typename T>
__global__ void chol_matinv_9x9_matrix_per_thread(T alpha, T beta, const T *A,
                                                  T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 9;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66, L70, L71,
      L72, L73, L74, L75, L76, L77, L80, L81, L82, L83, L84, L85, L86, L87, L88;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
      X50, X51, X52, X53, X54, X55, X60, X61, X62, X63, X64, X65, X66, X70, X71,
      X72, X73, X74, X75, X76, X77, X80, X81, X82, X83, X84, X85, X86, X87, X88;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L50 = A(5, 0);
  L60 = A(6, 0);
  L70 = A(7, 0);
  L80 = A(8, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L51 = A(5, 1);
  L61 = A(6, 1);
  L71 = A(7, 1);
  L81 = A(8, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L52 = A(5, 2);
  L62 = A(6, 2);
  L72 = A(7, 2);
  L82 = A(8, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L53 = A(5, 3);
  L63 = A(6, 3);
  L73 = A(7, 3);
  L83 = A(8, 3);
  L44 = A(4, 4) * alpha + beta;
  L54 = A(5, 4);
  L64 = A(6, 4);
  L74 = A(7, 4);
  L84 = A(8, 4);
  L55 = A(5, 5) * alpha + beta;
  L65 = A(6, 5);
  L75 = A(7, 5);
  L85 = A(8, 5);
  L66 = A(6, 6) * alpha + beta;
  L76 = A(7, 6);
  L86 = A(8, 6);
  L77 = A(7, 7) * alpha + beta;
  L87 = A(8, 7);
  L88 = A(8, 8) * alpha + beta;

  ComputeL_9x9(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42,
               L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64,
               L65, L66, L70, L71, L72, L73, L74, L75, L76, L77, L80, L81, L82,
               L83, L84, L85, L86, L87, L88);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X55 = 1 / L55;
  X66 = 1 / L66;
  X77 = 1 / L77;
  X88 = 1 / L88;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  X50 = 0;
  X51 = 0;
  X52 = 0;
  X53 = 0;
  X54 = 0;
  X60 = 0;
  X61 = 0;
  X62 = 0;
  X63 = 0;
  X64 = 0;
  X65 = 0;
  X70 = 0;
  X71 = 0;
  X72 = 0;
  X73 = 0;
  X74 = 0;
  X75 = 0;
  X76 = 0;
  X80 = 0;
  X81 = 0;
  X82 = 0;
  X83 = 0;
  X84 = 0;
  X85 = 0;
  X86 = 0;
  X87 = 0;
  ComputeCholInv_9x9(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41,
                     L42, L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62,
                     L63, L64, L65, L66, L70, L71, L72, L73, L74, L75, L76, L77,
                     L80, L81, L82, L83, L84, L85, L86, L87, L88, X00, X10, X11,
                     X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
                     X50, X51, X52, X53, X54, X55, X60, X61, X62, X63, X64, X65,
                     X66, X70, X71, X72, X73, X74, X75, X76, X77, X80, X81, X82,
                     X83, X84, X85, X86, X87, X88);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(5, 0) = X50;
  Ainv(6, 0) = X60;
  Ainv(7, 0) = X70;
  Ainv(8, 0) = X80;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(5, 1) = X51;
  Ainv(6, 1) = X61;
  Ainv(7, 1) = X71;
  Ainv(8, 1) = X81;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(5, 2) = X52;
  Ainv(6, 2) = X62;
  Ainv(7, 2) = X72;
  Ainv(8, 2) = X82;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(5, 3) = X53;
  Ainv(6, 3) = X63;
  Ainv(7, 3) = X73;
  Ainv(8, 3) = X83;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
  Ainv(5, 4) = X54;
  Ainv(6, 4) = X64;
  Ainv(7, 4) = X74;
  Ainv(8, 4) = X84;
  Ainv(0, 5) = X50;
  Ainv(1, 5) = X51;
  Ainv(2, 5) = X52;
  Ainv(3, 5) = X53;
  Ainv(4, 5) = X54;
  Ainv(5, 5) = X55;
  Ainv(6, 5) = X65;
  Ainv(7, 5) = X75;
  Ainv(8, 5) = X85;
  Ainv(0, 6) = X60;
  Ainv(1, 6) = X61;
  Ainv(2, 6) = X62;
  Ainv(3, 6) = X63;
  Ainv(4, 6) = X64;
  Ainv(5, 6) = X65;
  Ainv(6, 6) = X66;
  Ainv(7, 6) = X76;
  Ainv(8, 6) = X86;
  Ainv(0, 7) = X70;
  Ainv(1, 7) = X71;
  Ainv(2, 7) = X72;
  Ainv(3, 7) = X73;
  Ainv(4, 7) = X74;
  Ainv(5, 7) = X75;
  Ainv(6, 7) = X76;
  Ainv(7, 7) = X77;
  Ainv(8, 7) = X87;
  Ainv(0, 8) = X80;
  Ainv(1, 8) = X81;
  Ainv(2, 8) = X82;
  Ainv(3, 8) = X83;
  Ainv(4, 8) = X84;
  Ainv(5, 8) = X85;
  Ainv(6, 8) = X86;
  Ainv(7, 8) = X87;
  Ainv(8, 8) = X88;
}

template <typename T>
__global__ void chol_matinv_10x10_matrix_per_thread(T alpha, T beta, const T *A,
                                                    T *Ainv, int_t batch) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;
  const int_t size = 10;

  if (idx >= batch)
    return;

  T L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66, L70, L71,
      L72, L73, L74, L75, L76, L77, L80, L81, L82, L83, L84, L85, L86, L87, L88,
      L90, L91, L92, L93, L94, L95, L96, L97, L98, L99;
  T X00, X10, X11, X20, X21, X22, X30, X31, X32, X33, X40, X41, X42, X43, X44,
      X50, X51, X52, X53, X54, X55, X60, X61, X62, X63, X64, X65, X66, X70, X71,
      X72, X73, X74, X75, X76, X77, X80, X81, X82, X83, X84, X85, X86, X87, X88,
      X90, X91, X92, X93, X94, X95, X96, X97, X98, X99;

  L00 = A(0, 0) * alpha + beta;
  L10 = A(1, 0);
  L20 = A(2, 0);
  L30 = A(3, 0);
  L40 = A(4, 0);
  L50 = A(5, 0);
  L60 = A(6, 0);
  L70 = A(7, 0);
  L80 = A(8, 0);
  L90 = A(9, 0);
  L11 = A(1, 1) * alpha + beta;
  L21 = A(2, 1);
  L31 = A(3, 1);
  L41 = A(4, 1);
  L51 = A(5, 1);
  L61 = A(6, 1);
  L71 = A(7, 1);
  L81 = A(8, 1);
  L91 = A(9, 1);
  L22 = A(2, 2) * alpha + beta;
  L32 = A(3, 2);
  L42 = A(4, 2);
  L52 = A(5, 2);
  L62 = A(6, 2);
  L72 = A(7, 2);
  L82 = A(8, 2);
  L92 = A(9, 2);
  L33 = A(3, 3) * alpha + beta;
  L43 = A(4, 3);
  L53 = A(5, 3);
  L63 = A(6, 3);
  L73 = A(7, 3);
  L83 = A(8, 3);
  L93 = A(9, 3);
  L44 = A(4, 4) * alpha + beta;
  L54 = A(5, 4);
  L64 = A(6, 4);
  L74 = A(7, 4);
  L84 = A(8, 4);
  L94 = A(9, 4);
  L55 = A(5, 5) * alpha + beta;
  L65 = A(6, 5);
  L75 = A(7, 5);
  L85 = A(8, 5);
  L95 = A(9, 5);
  L66 = A(6, 6) * alpha + beta;
  L76 = A(7, 6);
  L86 = A(8, 6);
  L96 = A(9, 6);
  L77 = A(7, 7) * alpha + beta;
  L87 = A(8, 7);
  L97 = A(9, 7);
  L88 = A(8, 8) * alpha + beta;
  L98 = A(9, 8);
  L99 = A(9, 9) * alpha + beta;

  ComputeL_10x10(L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41,
                 L42, L43, L44, L50, L51, L52, L53, L54, L55, L60, L61, L62,
                 L63, L64, L65, L66, L70, L71, L72, L73, L74, L75, L76, L77,
                 L80, L81, L82, L83, L84, L85, L86, L87, L88, L90, L91, L92,
                 L93, L94, L95, L96, L97, L98, L99);

  X00 = 1 / L00;
  X11 = 1 / L11;
  X22 = 1 / L22;
  X33 = 1 / L33;
  X44 = 1 / L44;
  X55 = 1 / L55;
  X66 = 1 / L66;
  X77 = 1 / L77;
  X88 = 1 / L88;
  X99 = 1 / L99;
  X10 = 0;
  X20 = 0;
  X21 = 0;
  X30 = 0;
  X31 = 0;
  X32 = 0;
  X40 = 0;
  X41 = 0;
  X42 = 0;
  X43 = 0;
  X50 = 0;
  X51 = 0;
  X52 = 0;
  X53 = 0;
  X54 = 0;
  X60 = 0;
  X61 = 0;
  X62 = 0;
  X63 = 0;
  X64 = 0;
  X65 = 0;
  X70 = 0;
  X71 = 0;
  X72 = 0;
  X73 = 0;
  X74 = 0;
  X75 = 0;
  X76 = 0;
  X80 = 0;
  X81 = 0;
  X82 = 0;
  X83 = 0;
  X84 = 0;
  X85 = 0;
  X86 = 0;
  X87 = 0;
  X90 = 0;
  X91 = 0;
  X92 = 0;
  X93 = 0;
  X94 = 0;
  X95 = 0;
  X96 = 0;
  X97 = 0;
  X98 = 0;
  ComputeCholInv_10x10(
      L00, L10, L11, L20, L21, L22, L30, L31, L32, L33, L40, L41, L42, L43, L44,
      L50, L51, L52, L53, L54, L55, L60, L61, L62, L63, L64, L65, L66, L70, L71,
      L72, L73, L74, L75, L76, L77, L80, L81, L82, L83, L84, L85, L86, L87, L88,
      L90, L91, L92, L93, L94, L95, L96, L97, L98, L99, X00, X10, X11, X20, X21,
      X22, X30, X31, X32, X33, X40, X41, X42, X43, X44, X50, X51, X52, X53, X54,
      X55, X60, X61, X62, X63, X64, X65, X66, X70, X71, X72, X73, X74, X75, X76,
      X77, X80, X81, X82, X83, X84, X85, X86, X87, X88, X90, X91, X92, X93, X94,
      X95, X96, X97, X98, X99);

  Ainv(0, 0) = X00;
  Ainv(1, 0) = X10;
  Ainv(2, 0) = X20;
  Ainv(3, 0) = X30;
  Ainv(4, 0) = X40;
  Ainv(5, 0) = X50;
  Ainv(6, 0) = X60;
  Ainv(7, 0) = X70;
  Ainv(8, 0) = X80;
  Ainv(9, 0) = X90;
  Ainv(0, 1) = X10;
  Ainv(1, 1) = X11;
  Ainv(2, 1) = X21;
  Ainv(3, 1) = X31;
  Ainv(4, 1) = X41;
  Ainv(5, 1) = X51;
  Ainv(6, 1) = X61;
  Ainv(7, 1) = X71;
  Ainv(8, 1) = X81;
  Ainv(9, 1) = X91;
  Ainv(0, 2) = X20;
  Ainv(1, 2) = X21;
  Ainv(2, 2) = X22;
  Ainv(3, 2) = X32;
  Ainv(4, 2) = X42;
  Ainv(5, 2) = X52;
  Ainv(6, 2) = X62;
  Ainv(7, 2) = X72;
  Ainv(8, 2) = X82;
  Ainv(9, 2) = X92;
  Ainv(0, 3) = X30;
  Ainv(1, 3) = X31;
  Ainv(2, 3) = X32;
  Ainv(3, 3) = X33;
  Ainv(4, 3) = X43;
  Ainv(5, 3) = X53;
  Ainv(6, 3) = X63;
  Ainv(7, 3) = X73;
  Ainv(8, 3) = X83;
  Ainv(9, 3) = X93;
  Ainv(0, 4) = X40;
  Ainv(1, 4) = X41;
  Ainv(2, 4) = X42;
  Ainv(3, 4) = X43;
  Ainv(4, 4) = X44;
  Ainv(5, 4) = X54;
  Ainv(6, 4) = X64;
  Ainv(7, 4) = X74;
  Ainv(8, 4) = X84;
  Ainv(9, 4) = X94;
  Ainv(0, 5) = X50;
  Ainv(1, 5) = X51;
  Ainv(2, 5) = X52;
  Ainv(3, 5) = X53;
  Ainv(4, 5) = X54;
  Ainv(5, 5) = X55;
  Ainv(6, 5) = X65;
  Ainv(7, 5) = X75;
  Ainv(8, 5) = X85;
  Ainv(9, 5) = X95;
  Ainv(0, 6) = X60;
  Ainv(1, 6) = X61;
  Ainv(2, 6) = X62;
  Ainv(3, 6) = X63;
  Ainv(4, 6) = X64;
  Ainv(5, 6) = X65;
  Ainv(6, 6) = X66;
  Ainv(7, 6) = X76;
  Ainv(8, 6) = X86;
  Ainv(9, 6) = X96;
  Ainv(0, 7) = X70;
  Ainv(1, 7) = X71;
  Ainv(2, 7) = X72;
  Ainv(3, 7) = X73;
  Ainv(4, 7) = X74;
  Ainv(5, 7) = X75;
  Ainv(6, 7) = X76;
  Ainv(7, 7) = X77;
  Ainv(8, 7) = X87;
  Ainv(9, 7) = X97;
  Ainv(0, 8) = X80;
  Ainv(1, 8) = X81;
  Ainv(2, 8) = X82;
  Ainv(3, 8) = X83;
  Ainv(4, 8) = X84;
  Ainv(5, 8) = X85;
  Ainv(6, 8) = X86;
  Ainv(7, 8) = X87;
  Ainv(8, 8) = X88;
  Ainv(9, 8) = X98;
  Ainv(0, 9) = X90;
  Ainv(1, 9) = X91;
  Ainv(2, 9) = X92;
  Ainv(3, 9) = X93;
  Ainv(4, 9) = X94;
  Ainv(5, 9) = X95;
  Ainv(6, 9) = X96;
  Ainv(7, 9) = X97;
  Ainv(8, 9) = X98;
  Ainv(9, 9) = X99;
}

template <typename T, int_t size>
__global__ void chol_matinv_kernel_general_impl(T alpha, T beta, const T *A,
                                                T *Ainv, int_t batch) {
  T *shmem = sfm::utils::internal::SharedMemory<T>::get();
  T *chol = shmem;
  T *xs = shmem;
  T *L = shmem + blockDim.x * blockDim.y;

  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= batch)
    return;

  T vals[size + 1];
  T *mats = vals + 1;
  T *sols = vals;

  for (int i = 0; i <= threadIdx.y; i++) {
    mats(i) = A(threadIdx.y, i);
  }

  mats(threadIdx.y) = mats(threadIdx.y) * alpha + beta;

  for (int j = 0; j < size; j++) {
    // printf("%d: %d\n", j, threadIdx.y);
    __syncthreads();
    if (threadIdx.y == j) {
      mats(j) = sqrt(mats(j));
      chol[blockDim.x * threadIdx.y + threadIdx.x] = mats(j);
    }

    __syncthreads();
    if (threadIdx.y > j) {
      mats(j) /= chol[blockDim.x * j + threadIdx.x];
      chol[blockDim.x * threadIdx.y + threadIdx.x] = mats(j);
    }

    __syncthreads();
    if (threadIdx.y == j) {
      for (int k = j + 1; k < size; k++) {
        mats(k) = chol[blockDim.x * k + threadIdx.x];
      }
    } else if (threadIdx.y > j) {
      for (int k = j + 1; k <= threadIdx.y; k++) {
        mats(k) -= mats(j) * chol[blockDim.x * k + threadIdx.x];
      }
    }
  }

  for (int j = 0; j < threadIdx.y; j++) {
    sols(j) = 0;
  }
  sols(threadIdx.y) = 1 / mats(threadIdx.y);

  for (int j = size - 1; j >= 0; j--) {
    __syncthreads();
    if (threadIdx.y == j) {
      for (int k = 0; k <= j; k++) {
        xs[blockDim.x * k + threadIdx.x] = sols(k);
      }
    }

    T xsol = 0;
    for (int k = j; k >= 0; k--) {
      __syncthreads();

      if (threadIdx.y == k) {
        xs[blockDim.x * threadIdx.y + threadIdx.x] /= mats(k);
        xsol = xs[blockDim.x * threadIdx.y + threadIdx.x];
        L[threadIdx.x] = mats(j);
      }

      __syncthreads();
      if (threadIdx.y == j) {
        sols(k) = xs[blockDim.x * k + threadIdx.x];
      } else if (threadIdx.y >= k) {
        sols(k) -= L[threadIdx.x] * xsol;
      } else {
        xs[blockDim.x * threadIdx.y + threadIdx.x] -=
            mats(k) * xs[blockDim.x * k + threadIdx.x];
      }
    }
  }

  Ainv(threadIdx.y, threadIdx.y) = sols(threadIdx.y);

  for (int j = 0; j <= threadIdx.y; j++) {
    Ainv(j, threadIdx.y) = sols(j);
    Ainv(threadIdx.y, j) = sols(j);
  }
}

template <typename T>
int chol_matinv_batch_per_thread_async(const T *A, T *Ainv, int_t size,
                                       int_t batch, cudaStream_t stream) {
  typedef void (*func)(T alpha, T beta, const T *A, T *Ainv, int batch);

  const static func pf[11] = {nullptr,
                              nullptr,
                              chol_matinv_2x2_matrix_per_thread,
                              chol_matinv_3x3_matrix_per_thread,
                              chol_matinv_4x4_matrix_per_thread,
                              chol_matinv_5x5_matrix_per_thread,
                              chol_matinv_6x6_matrix_per_thread,
                              chol_matinv_7x7_matrix_per_thread,
                              chol_matinv_8x8_matrix_per_thread,
                              chol_matinv_9x9_matrix_per_thread,
                              chol_matinv_10x10_matrix_per_thread};

  const static int minBatchSize[11] = {0x7fffffff, 0x7fffffff, 1300, 1100,
                                       1100,       1100,       1100, 1100,
                                       1100,       1100,       1200};

  if (size < 2 || size > 10 || batch < minBatchSize[size]) {
    return 1;
  }

  int_t block_size = 128;
  int_t num_threads = batch;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  pf[size]<<<grid_size, block_size, 0, stream>>>(T(1.0), T(0.0), A, Ainv,
                                                 batch);

  if (cudaSuccess != cudaGetLastError()) {
    return -2;
  }

  return 0;
}

template <typename T>
int chol_matinv_batch_general_async(const T *A, T *Ainv, int_t size,
                                    int_t batch, cudaStream_t stream) {
  typedef void (*func)(T alpha, T beta, const T *A, T *Ainv, int batch);

  const static func pf[17] = {nullptr,
                              nullptr,
                              chol_matinv_kernel_general_impl<T, 2>,
                              chol_matinv_kernel_general_impl<T, 3>,
                              chol_matinv_kernel_general_impl<T, 4>,
                              chol_matinv_kernel_general_impl<T, 5>,
                              chol_matinv_kernel_general_impl<T, 6>,
                              chol_matinv_kernel_general_impl<T, 7>,
                              chol_matinv_kernel_general_impl<T, 8>,
                              chol_matinv_kernel_general_impl<T, 9>,
                              chol_matinv_kernel_general_impl<T, 10>,
                              chol_matinv_kernel_general_impl<T, 11>,
                              chol_matinv_kernel_general_impl<T, 12>,
                              chol_matinv_kernel_general_impl<T, 13>,
                              chol_matinv_kernel_general_impl<T, 14>,
                              chol_matinv_kernel_general_impl<T, 15>,
                              chol_matinv_kernel_general_impl<T, 16>};

  if (size > 16 || size < 2)
    return -1;

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, batch);
  block_size.y = size;

  int_t num_blocks = (batch + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  pf[size]<<<grid_size, block_size,
             block_size.x *(block_size.y + 1) * sizeof(T), stream>>>(
      T(1.0), T(0.0), A, Ainv, batch);

  if (cudaSuccess != cudaGetLastError()) {
    return -2;
  }

  return 0;
}

template <typename T>
int chol_matinv_batch_async(const T *A, T *Ainv, int_t size, int_t batch,
                            cudaStream_t stream) {
  int stat = chol_matinv_batch_per_thread_async(A, Ainv, size, batch, stream);

  if (stat <= 0)
    return stat;

  return chol_matinv_batch_general_async(A, Ainv, size, batch, stream);
}

template <typename T>
int chol_matinv_batch_per_thread_async(T alpha, T beta, const T *A, T *Ainv,
                                       int_t size, int_t batch,
                                       cudaStream_t stream) {
  typedef void (*func)(T alpha, T beta, const T *A, T *Ainv, int batch);

  const static func pf[11] = {nullptr,
                              nullptr,
                              chol_matinv_2x2_matrix_per_thread,
                              chol_matinv_3x3_matrix_per_thread,
                              chol_matinv_4x4_matrix_per_thread,
                              chol_matinv_5x5_matrix_per_thread,
                              chol_matinv_6x6_matrix_per_thread,
                              chol_matinv_7x7_matrix_per_thread,
                              chol_matinv_8x8_matrix_per_thread,
                              chol_matinv_9x9_matrix_per_thread,
                              chol_matinv_10x10_matrix_per_thread};

  const static int minBatchSize[11] = {0x7fffffff, 0x7fffffff, 1300, 1100,
                                       1100,       1100,       1100, 1100,
                                       1100,       1100,       1200};

  if (size < 2 || size > 10 || batch < minBatchSize[size]) {
    return 1;
  }

  int_t block_size = 128;
  int_t num_threads = batch;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  pf[size]<<<grid_size, block_size, 0, stream>>>(alpha, beta, A, Ainv, batch);

  if (cudaSuccess != cudaGetLastError()) {
    return -2;
  }

  return 0;
}

template <typename T>
int chol_matinv_batch_general_async(T alpha, T beta, const T *A, T *Ainv,
                                    int_t size, int_t batch,
                                    cudaStream_t stream) {
  typedef void (*func)(T alpha, T beta, const T *A, T *Ainv, int batch);

  const static func pf[17] = {nullptr,
                              nullptr,
                              chol_matinv_kernel_general_impl<T, 2>,
                              chol_matinv_kernel_general_impl<T, 3>,
                              chol_matinv_kernel_general_impl<T, 4>,
                              chol_matinv_kernel_general_impl<T, 5>,
                              chol_matinv_kernel_general_impl<T, 6>,
                              chol_matinv_kernel_general_impl<T, 7>,
                              chol_matinv_kernel_general_impl<T, 8>,
                              chol_matinv_kernel_general_impl<T, 9>,
                              chol_matinv_kernel_general_impl<T, 10>,
                              chol_matinv_kernel_general_impl<T, 11>,
                              chol_matinv_kernel_general_impl<T, 12>,
                              chol_matinv_kernel_general_impl<T, 13>,
                              chol_matinv_kernel_general_impl<T, 14>,
                              chol_matinv_kernel_general_impl<T, 15>,
                              chol_matinv_kernel_general_impl<T, 16>};

  if (size > 16 || size < 2)
    return -1;

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, batch);
  block_size.y = size;

  int_t num_blocks = (batch + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  pf[size]<<<grid_size, block_size,
             block_size.x *(block_size.y + 1) * sizeof(T), stream>>>(
      alpha, beta, A, Ainv, batch);

  if (cudaSuccess != cudaGetLastError()) {
    return -2;
  }

  return 0;
}

template <typename T>
int chol_matinv_batch_async(T alpha, T beta, const T *A, T *Ainv, int_t size,
                            int_t batch, cudaStream_t stream) {
  int stat = chol_matinv_batch_per_thread_async(alpha, beta, A, Ainv, size,
                                                batch, stream);

  if (stat <= 0)
    return stat;

  return chol_matinv_batch_general_async(alpha, beta, A, Ainv, size, batch,
                                         stream);
}

template int chol_matinv_batch_general_async(const float *A, float *Ainv,
                                             int_t n, int_t batch,
                                             cudaStream_t stream);

template int chol_matinv_batch_general_async(const double *A, double *Ainv,
                                             int_t n, int_t batch,
                                             cudaStream_t stream);

template int chol_matinv_batch_per_thread_async(const float *A, float *Ainv,
                                                int_t size, int_t batch,
                                                cudaStream_t stream);

template int chol_matinv_batch_per_thread_async(const double *A, double *Ainv,
                                                int_t size, int_t batch,
                                                cudaStream_t stream);

template int chol_matinv_batch_async(const float *A, float *Ainv, int_t size,
                                     int_t batch, cudaStream_t stream);

template int chol_matinv_batch_async(const double *A, double *Ainv, int_t size,
                                     int_t batch, cudaStream_t stream);

template int chol_matinv_batch_general_async(float alpha, float beta,
                                             const float *A, float *Ainv,
                                             int_t n, int_t batch,
                                             cudaStream_t stream);

template int chol_matinv_batch_general_async(double alpha, double beta,
                                             const double *A, double *Ainv,
                                             int_t n, int_t batch,
                                             cudaStream_t stream);

template int chol_matinv_batch_per_thread_async(float alpha, float beta,
                                                const float *A, float *Ainv,
                                                int_t size, int_t batch,
                                                cudaStream_t stream);

template int chol_matinv_batch_per_thread_async(double alpha, double beta,
                                                const double *A, double *Ainv,
                                                int_t size, int_t batch,
                                                cudaStream_t stream);

template int chol_matinv_batch_async(float alpha, float beta, const float *A,
                                     float *Ainv, int_t size, int_t batch,
                                     cudaStream_t stream);

template int chol_matinv_batch_async(double alpha, double beta, const double *A,
                                     double *Ainv, int_t size, int_t batch,
                                     cudaStream_t stream);
} // namespace internal
} // namespace utils
} // namespace sfm