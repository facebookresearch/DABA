// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <sfm/types.h>

namespace sfm {
namespace optimization {
template <typename T> struct TrustRegionOption {
  //------------------------------------------------------
  // STOPPING CRITERIA
  //------------------------------------------------------
  /** Maximum permitted number of (outer) iterations of the Riemannian
   * trust-region method */
  int_t max_iterations = 100;

  /** Maximum permitted number of (outer) accepted iterations of the Riemannian
   * trust-region method */
  int_t max_accepted_iterations = 1;

  /** Stopping tolerance for the norm of the gradient */
  T gradient_norm_tolerance = 5e-3;

  /** Stopping tolerance for the norm of the update step */
  T update_step_norm_tolerance = 1e-4;

  /** Stopping criterion based upon the relative decrease in function value */
  T relative_function_decrease_tolerance = 1e-6;

  /** Stopping criterion based upon the norm of an accepted update step */
  T trust_region_radius_tolerance = 1e-4;

  /** Tolerance of gain ratio for an unsuccessful proposed update and decreasing
   * the trust region radius*/
  T gain_ratio_acceptance_tolerance = 0.01;

  /** Tolerance of gain ratio for a very successful proposed update step and
   * increasing the trust region radius */
  T gain_ratio_increasing_tolerance = 0.8;

  /** Multiplicative factor for decreasing the trust-region radius on an
   * unsuccessful iteration; this parameter should be in (0, 1) */
  T trust_region_radius_decreasing_ratio = 0.25;

  /** Multiplicative factor for decreasing the trust-region radius on a very
   * successful iteration; this parameter should be in (1, +inf) */
  T trust_region_radius_increasing_ratio = 2.5;

  /** Verbose for the Gauss-Newton method*/
  bool verbose = false;
};

template <typename T> struct PCGOption {
  /** Maximum iterations for preconditioned conjugate gradient method*/
  int_t max_iterations = 100;

  /** Relative residual norm reduction */
  T relative_residual_norm = 1e-2;
  /** Relative reduction of 0.5 * x^T*A*x + g^T */
  T relative_reduction_tol = 1e-1;
  T theta = 0.5;

  /** Verbose for preconditioned conjugate gradient method*/
  T verbose = false;
};

// y = H * x where H is a symmetric matrix
template <typename Vector, typename... Args>
using SymmetricLinearOperator =
    std::function<void(const Vector &x, Vector &y, Args...)>;

// y = M^-1 * x where M is a positive definite matrix
template <typename Vector, typename... Args>
using Preconditioner = std::function<void(const Vector &x, Vector &y, Args...)>;

// z = x + a * y
template <typename Vector, typename Scalar, typename... Args>
using Plus = std::function<void(const Vector &x, Scalar a, const Vector &y,
                                Vector &z, Args...)>;

template <typename Vector, typename Scalar, typename... Args>
using InnerProduct = std::function<void(const Vector &x, const Vector &y,
                                        Scalar &result, Args...)>;

// output = input
template <typename Vector, typename... Args>
using Equal = std::function<void(const Vector &input, Vector &output, Args...)>;

// output = -input
template <typename Vector, typename... Args>
using Negate =
    std::function<void(const Vector &input, Vector &output, Args...)>;

template <typename Vector, typename... Args>
using SetZero = std::function<void(Vector &x, Args...)>;

template <typename Scalar, typename... Args>
using QuadraticModel = std::function<void(Args...)>;

template <typename Scalar, typename... Args>
using PCGSolver =
    std::function<void(int_t &num_iters, Scalar &update_step_norm, Args...)>;

template <typename Scalar, typename... Args>
using Update =
    std::function<void(Scalar stepsize, Scalar &new_objective, Args...)>;

template <typename Scalar, typename... Args>
using PredictedReduction = std::function<void(Scalar &reduction, Args...)>;

template <typename Scalar, typename... Args>
using PredictedReductionWithDamping =
    std::function<void(Scalar &reduction, Scalar damping, Args...)>;

template <typename Scalar, typename... Args>
using Objective = std::function<void(Scalar &objective, Args...)>;

template <typename Scalar, typename... Args>
using GradientNorm = std::function<void(Scalar &gradient_norm, Args...)>;

template <typename Scalar, typename... Args>
using Accept = std::function<void(Args...)>;

template <typename Scalar, typename... Args>
using RescaleDiagonal =
    std::function<void(Scalar prev_damping, Scalar curr_damping, Args...)>;

template <typename... Args> using Synchronize = std::function<void(Args...)>;

template <typename... Args> using SetDevice = std::function<void(Args...)>;
} // namespace optimization
} // namespace sfm