// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iomanip>
#include <ios>
#include <iostream>

#include <glog/logging.h>
#include <sfm/optimization/concepts.h>
#include <sfm/types.h>

namespace sfm {
namespace optimization {
template <typename Vector, typename Scalar, typename... Args>
void PCG(const SymmetricLinearOperator<Vector, Args...> &H, const Vector &g,
         const Preconditioner<Vector, Args...> &preconditioner,
         const Plus<Vector, Scalar, Args...> &plus,
         const InnerProduct<Vector, Scalar, Args...> &inner_product,
         const Equal<Vector, Args...> &equal,
         const SetZero<Vector, Args...> &set_zero, Vector &s, Vector &residual,
         Vector &p, Vector &Hp, Vector &v, int_t &num_iters,
         Scalar &update_step_norm, Args &...args, int_t max_num_iters = 1000,
         Scalar relative_residual_norm = 0.1, Scalar theta = 0.5,
         Scalar relative_reduction_tol = 1e-4, bool verbose = false) {
  if (max_num_iters < 0) {
    LOG(ERROR)
        << "Maximum number of iterations must be a non-negative integer.";
    exit(-1);
  }

  if ((relative_residual_norm < 0) || (relative_residual_norm >= 1)) {
    LOG(ERROR) << "Target relative reduction of the residual norm must be a "
                  "real value in the range [0,1)";
    exit(-1);
  }

  if ((theta < 0) || (theta > 1)) {
    LOG(ERROR) << "Target superlinear convergence rate must be a real value in "
                  "the range [0,1]";
    exit(-1);
  }

  set_zero(s, args...);
  equal(g, residual, args...);
  preconditioner(residual, v, args...);
  equal(v, p, args...);

  Scalar residual_squared_norm;
  inner_product(residual, v, residual_squared_norm, args...);
  Scalar s_dot_p = 0;
  Scalar s_squared_norm = 0;
  Scalar p_squared_norm = residual_squared_norm;

  const Scalar initial_residual_norm = std::sqrt(residual_squared_norm);
  const Scalar target_residual_norm =
      initial_residual_norm *
      std::min(relative_residual_norm, std::pow(initial_residual_norm, theta));

  if (verbose) {
    std::cout << "-------------------------------------------------------"
              << std::endl;
    std::cout << "|       Preconditioned Conjugate Gradient Method      |"
              << std::endl;
    std::cout << "|-----------------------------------------------------|"
              << std::endl;
    std::cout << "| # of iterations |  residual norm  |    step norm    |"
              << std::endl;
    std::cout << "|-----------------|-----------------|-----------------|"
              << std::endl;
  }

  Scalar Q0 = 0;
  auto &residual_plus_g = Hp;
  plus(residual, Scalar(1.0), g, residual_plus_g, args...);
  inner_product(s, residual_plus_g, Q0, args...);

  num_iters = 0;

  if (verbose) {
    std::cout << "| " << std::setw(16) << std::fixed << num_iters << "| "
              << std::setw(16) << std::scientific
              << std::sqrt(residual_squared_norm) << "| " << std::setw(16)
              << std::scientific << std::sqrt(s_squared_norm) << "|"
              << std::endl;
  }

  if (std::sqrt(residual_squared_norm) < target_residual_norm) {
    return;
  }

  while (num_iters < max_num_iters) {
    H(p, Hp, args...);
    Scalar p_dot_Hp = 0;
    inner_product(p, Hp, p_dot_Hp, args...);

    if (p_dot_Hp < 0) {
      LOG(ERROR) << "The preconditioner is not positive definite." << std::endl;
      exit(-1);
    }

    Scalar alpha = residual_squared_norm / p_dot_Hp;

    plus(s, -alpha, p, s, args...);
    s_squared_norm += -2 * alpha * s_dot_p + alpha * alpha * p_squared_norm;

    plus(residual, -alpha, Hp, residual, args...);

    plus(residual, Scalar(1.0), g, Hp, args...);
    Scalar Q1 = 0;
    inner_product(s, residual_plus_g, Q1, args...);

    num_iters++;

    Scalar zeta = num_iters * (Q1 - Q0) / Q1;
    Q0 = Q1;

    preconditioner(residual, v, args...);
    Scalar residual_squared_norm_plus = 0;
    inner_product(residual, v, residual_squared_norm_plus, args...);
    Scalar beta = residual_squared_norm_plus / residual_squared_norm;

    residual_squared_norm = residual_squared_norm_plus;

    if (verbose) {
      std::cout << "| " << std::setw(16) << std::fixed << num_iters << "| "
                << std::setw(16) << std::scientific
                << std::sqrt(residual_squared_norm) << "| " << std::setw(16)
                << std::scientific << std::sqrt(s_squared_norm) << "|"
                << std::endl;
    }

    if (zeta < relative_reduction_tol) {
      break;
    }

    if (std::sqrt(residual_squared_norm) < target_residual_norm) {
      break;
    }

    s_dot_p = beta * (s_dot_p - alpha * p_squared_norm);
    p_squared_norm = residual_squared_norm + beta * beta * p_squared_norm;

    plus(v, beta, p, p, args...);
  }

  update_step_norm = std::sqrt(s_squared_norm);

  if (verbose) {
    std::cout << "|-----------------|-----------------|-----------------|"
              << std::endl;
    std::cout << "|      final      | " << std::setw(16) << std::scientific
              << std::sqrt(residual_squared_norm) << "| " << std::setw(16)
              << std::scientific << update_step_norm << "|" << std::endl;
    std::cout << "-------------------------------------------------------"
              << std::endl;
  }
}
} // namespace optimization
} // namespace sfm