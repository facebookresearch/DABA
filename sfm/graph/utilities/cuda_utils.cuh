// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace sfm {
namespace graph {
#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
} // namespace graph
} // namespace sfm