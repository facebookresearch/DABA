// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Core>
#include <cuda_runtime.h>

#include <sfm/optimization/concepts.h>
#include <sfm/types.h>

namespace sfm {
namespace ba {
// struct Info {
//   Eigen::Vector<int, 3> index;
// };
struct Index {
  int_t node;
  int_t index;
};

template <typename T> struct Option {
  /** Regularizer for majorization minimization matrix */
  Eigen::Vector3<T> regularizer{1e-3, 1e-3, 1e-3};

  /** acceptable gradient norm over objective value */
  T accepted_delta = 5e-4;

  /** Whether to use Nesterov's acceleration */
  bool accelerated = true;

  /** Whether to print output as the algorithm runs */
  bool verbose = false;

  /** Robust loss function */
  RobustLoss robust_loss = Trivial;

  /** loss radius */
  T loss_radius = 1024;

  //------------------------------------------------------
  // NESTEROV PARAMETERS
  //------------------------------------------------------
  /** Parameter to initialize the exponential average (must greater or equal
   * to 1.0)*/
  T initial_nesterov_average_objective_value_ratio = 10.0;

  /** Parameter to computer the exponential average*/
  T initial_nesterov_eta = 1e-3;

  /** Parameter to increase eta*/
  T increasing_nesterov_eta_ratio = 5e-3;

  //------------------------------------------------------
  // Trust Region Method Option
  //------------------------------------------------------
  sfm::optimization::TrustRegionOption<T> trust_region_option{
      100, 1, 5e-3, 1e-4, 1e-6, 1e-4, 1e-2, 0.6, 0.5, 3, false};

  //------------------------------------------------------
  // Preconditioned Conjugate Graident Method Option
  //------------------------------------------------------
  sfm::optimization::PCGOption<T> pcg_option{100, 1e-2, 1e-1, 0.5, false};
};
} // namespace ba
} // namespace sfm