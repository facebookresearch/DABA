// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <sfm/optimization/concepts.h>
#include <vector>

namespace sfm {
namespace optimization {
template <typename Scalar, typename... Args>
SolverStatus
LM(const Objective<Scalar, Args...> &objective,
   const QuadraticModel<Scalar, Args...> &quadratic_model,
   const GradientNorm<Scalar, Args...> &gradient_norm,
   const PCGSolver<Scalar, Args...> &pcg,
   const PredictedReductionWithDamping<Scalar, Args...> &predicted_reduction,
   const Update<Scalar, Args...> &update,
   const RescaleDiagonal<Scalar, Args...> &rescale_diagonal,
   const Accept<Scalar, Args...> &accept, int_t max_iterations,
   int_t max_accepted_iterations, Scalar gradient_norm_tolerance,
   Scalar update_step_norm_tolerance, Scalar relative_decrease_tolerance,
   Scalar trust_region_radius_tolerance, Scalar gain_ratio_acceptance_tolerance,
   Scalar gain_ratio_increasing_tolerance, Scalar initial_trust_region_radius,
   Scalar initial_trust_region_decreasing_ratio, Scalar &fobj,
   Scalar &grad_norm, Scalar &update_step_norm, Scalar &trust_region_radius,
   Scalar &trust_region_decreasing_ratio, Args &...args, bool verbose = false) {
  trust_region_radius = initial_trust_region_radius;
  trust_region_decreasing_ratio = initial_trust_region_decreasing_ratio;

  Scalar damping_ratio = 1.0 / trust_region_radius;

  objective(fobj, args...);
  quadratic_model(args...);
  rescale_diagonal(0.0, damping_ratio, args...);
  gradient_norm(grad_norm, args...);

  // verbose = false;

  if (verbose) {
    std::cout << "-----------------------------------------------------------"
                 "-------------------------------------------------"
              << std::endl;
    std::cout
        << "|                                           Levenberg–Marquardt "
           "Method                                      |"
        << std::endl;
    std::cout << "-----------------------------------------------------------"
                 "-------------------------------------------------"
              << std::endl;
  }

  int_t iteration = 0, accepted_iteration = 0;
  while (iteration < max_iterations) {
    if (grad_norm < gradient_norm_tolerance) {
      return SolverStatus::Gradient;
    }

    int_t inner_iterations;
    pcg(inner_iterations, update_step_norm, args...);

    if (verbose) {
      std::cout << "| " << iteration << " " << std::scientific << fobj << " "
                << grad_norm << " " << trust_region_radius << " "
                << inner_iterations << " " << update_step_norm << " ";
    }

    if (update_step_norm < update_step_norm_tolerance) {
      return SolverStatus::PreconditionedGradient;
    }

    Scalar fobj_plus = 0;
    update(Scalar(1.0), fobj_plus, args...);

    Scalar dfobj_pred = 0;
    predicted_reduction(dfobj_pred, damping_ratio, args...);
    Scalar dfobj = fobj - fobj_plus;
    Scalar relative_decrease = dfobj / (1e-6 + std::fabs(fobj));

    Scalar rho = dfobj / dfobj_pred;

    if (verbose) {
      std::cout << fobj_plus << " " << dfobj << " " << dfobj_pred << " " << rho
                << " |" << std::endl;
    }

    if (dfobj > 0) {
      accept(args...);
      accepted_iteration++;
      trust_region_radius =
          trust_region_radius /
          std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1, 3));
      trust_region_decreasing_ratio = 0.5;
      damping_ratio = 1.0 / trust_region_radius;

      if (relative_decrease < relative_decrease_tolerance) {
        return SolverStatus::RelativeDecrease;
      }

      if (accepted_iteration >= max_accepted_iterations) {
        return SolverStatus::AcceptedIteration;
      }

      objective(fobj, args...);
      quadratic_model(args...);
      rescale_diagonal(0.0, damping_ratio, args...);
      gradient_norm(grad_norm, args...);
    } else {
      trust_region_radius *= trust_region_decreasing_ratio;
      trust_region_decreasing_ratio *= 0.5;
      damping_ratio = 1.0 / trust_region_radius;
      if (trust_region_radius < trust_region_radius_tolerance) {
        return SolverStatus::TrustRegion;
      }

      rescale_diagonal(trust_region_decreasing_ratio / trust_region_radius,
                       damping_ratio, args...);
    }

    iteration++;
  }

  return SolverStatus::Failure;
}
} // namespace optimization
} // namespace sfm