// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

namespace sfm {
#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static void HandleError(cudaError_t err, const char *file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <typename T> void to_device(T *&device_vals, T *host_vals, int size) {
  HANDLE_ERROR(cudaMalloc((void **)&device_vals, sizeof(T) * size));
  HANDLE_ERROR(cudaMemcpy(device_vals, host_vals, sizeof(T) * size,
                          cudaMemcpyHostToDevice));
}

template <typename T> void to_host(T *&host_vals, T *device_vals, int size) {
  HANDLE_ERROR(cudaMemcpy(host_vals, device_vals, sizeof(T) * size,
                          cudaMemcpyDeviceToHost));
}

template <typename T> void copy(T *dst, T *src, int size, cudaMemcpyKind kind) {
  HANDLE_ERROR(cudaMemcpy(dst, src, sizeof(T) * size, kind));
}
} // namespace sfm