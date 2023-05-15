// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace sfm {
namespace utils {
namespace internal {
template <typename T> struct SharedMemory {};

template <> struct SharedMemory<float> {
  static __device__ float *get() {
    extern __shared__ float floatPtr[];
    return floatPtr;
  }
};

template <> struct SharedMemory<double> {
  static __device__ double *get() {
    extern __shared__ double doublePtr[];
    return doublePtr;
  }
};

template <> struct SharedMemory<int> {
  static __device__ int *get() {
    extern __shared__ int intPtr[];
    return intPtr;
  }
};

template <> struct SharedMemory<size_t> {
  static __device__ int *get() {
    extern __shared__ int size_tPtr[];
    return size_tPtr;
  }
};
} // namespace internal
} // namespace utils
} // namespace sfm