// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_runtime.h>

namespace sfm {
namespace utils {
template <typename T> struct TrivialLoss {
  __host__ __device__ static void Evaluate(T x, T loss_radius, T &f) { f = x; }

  __host__ __device__ static void Linearize(T x, T loss_radius, T &f,
                                            T &rescale) {
    f = x;
    rescale = 1;
  }
};

template <typename T> struct HuberLoss {
  __host__ __device__ static void Evaluate(T x, T loss_radius, T &f) {
    f = (x <= loss_radius) ? x : 2 * sqrt(loss_radius * x) - loss_radius;
  }

  __host__ __device__ static void Linearize(T x, T loss_radius, T &f,
                                            T &rescale) {
    f = (x <= loss_radius) ? x : 2 * sqrt(loss_radius * x) - loss_radius;
    // rescale = (x <= loss_radius) ? 1 : sqrt(loss_radius / x);
    rescale = (x <= loss_radius) ? 1 : sqrt(0.5 * (f + loss_radius) / x);
  }
};
} // namespace utils
} // namespace sfm