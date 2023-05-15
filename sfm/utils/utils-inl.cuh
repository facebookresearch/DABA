// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sfm/ba/types.h>

namespace sfm {
namespace utils {
template <typename T, typename Operator>
__global__ void MatrixTransformKernel(const T *input, T *output, Operator op,
                                      int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input + threadIdx.y * N + idx;
  T *py = output + threadIdx.y * N + idx;

  T x = *px;
  *py = op(x);
}

template <typename T, typename BinaryOperator>
__global__ void MatrixTransformKernel(const T *input0, const T *input1,
                                      T *output, BinaryOperator op, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input0 + threadIdx.y * N + idx;
  T *py = input1 + threadIdx.y * N + idx;
  T *pz = output + threadIdx.y * N + idx;

  T x = *px;
  T y = *py;
  *pz = op(x, y);
}

template <typename T, typename Operator>
__global__ void DictedMatrixTransformKernel(const T *input, int_t input_cnt,
                                            const int_t *output_dicts,
                                            T *output, int_t output_cnt,
                                            Operator op, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input + threadIdx.y * input_cnt + idx;
  T *py = output + threadIdx.y * output_cnt + output_dicts[idx];

  T x = *px;
  *py = op(x);
}

template <typename T, typename Operator>
__global__ void
DictedMatrixTransformKernel(const int_t *input_dicts, const T *input,
                            int_t input_cnt, const int_t *output_dicts,
                            T *output, int_t output_cnt, Operator op, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input + threadIdx.y * input_cnt + input_dicts[idx];
  T *py = output + threadIdx.y * output_cnt + output_dicts[idx];

  T x = *px;
  *py = op(x);
}

template <typename T, typename BinaryOperator>
__global__ void DictedMatrixTransformKernel(const T *input0, int_t input_cnt0,
                                            const int_t *input_dicts1,
                                            const T *input1, int_t input_cnt1,
                                            const int_t *output_dicts,
                                            T *output, int_t output_cnt,
                                            BinaryOperator op, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input0 + threadIdx.y * input_cnt0 + idx;
  T *py = input1 + threadIdx.y * input_cnt1 + input_dicts1[idx];
  T *pz = output + threadIdx.y * output_cnt + output_dicts[idx];

  T x = *px;
  T y = *py;
  *pz = op(x, y);
}

template <typename T, typename BinaryOperator>
__global__ void
DictedMatrixTransformKernel(const int_t *input_dicts0, const T *input0,
                            int_t input_cnt0, const int_t *input_dicts1,
                            const T *input1, int_t input_cnt1,
                            const int_t *output_dicts, T *output,
                            int_t output_cnt, BinaryOperator op, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx > N) {
    return;
  }

  T *px = input0 + threadIdx.y * input_cnt0 + input_dicts0[idx];
  T *py = input1 + threadIdx.y * input_cnt1 + input_dicts1[idx];
  T *pz = output + threadIdx.y * output_cnt + output_dicts[idx];

  T x = *px;
  T y = *py;
  *pz = op(x, y);
}
}; // namespace utils
} // namespace sfm