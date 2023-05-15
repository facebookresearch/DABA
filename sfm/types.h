// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <nccl.h>
#include <type_traits>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <thrust/async/reduce.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace sfm {
enum Memory { kCPU = 0, kGPU = 1 };

enum RobustLoss { Trivial = 0, Huber = 1};

typedef int int_t;
typedef unsigned int uint_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

union unf {
  float f;
  uint32_t ui;
};

union und {
  double f;
  uint64_t ui;
};

template <typename T> struct traits;

template <> struct traits<float> {
  union Union {
    float f;
    int ui;
  };

  using Unsigned = int;
  static const MPI_Datatype MPI_INT_TYPE;
  static const MPI_Datatype MPI_FLOAT_TYPE;
  static const ncclDataType_t NCCL_INT_TYPE;
  static const ncclDataType_t NCCL_FLOAT_TYPE;
};

template <> struct traits<double> {
  union Union {
    double f;
    unsigned long long ui;
  };
  using Unsigned = unsigned long long;
  static const MPI_Datatype MPI_INT_TYPE;
  static const MPI_Datatype MPI_FLOAT_TYPE;
  static const ncclDataType_t NCCL_INT_TYPE;
  static const ncclDataType_t NCCL_FLOAT_TYPE;
};

enum SolverStatus {
  Skipped = -3,
  Exception = -2,
  Failure = -1,
  AcceptedIteration = 0,
  Gradient = 1,
  PreconditionedGradient = 2,
  RelativeDecrease = 3,
  TrustRegion = 4,
  Terminated = 5,
};

template <typename T>
using PinnedHostVector =
    std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

struct SquaredSum {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a * a + b * b;
  }
};
} // namespace sfm