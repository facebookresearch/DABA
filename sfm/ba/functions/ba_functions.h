// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sfm/ba/macro.h>
#include <sfm/ba/types.h>
#include <sfm/types.h>
#include <sfm/utils/internal/svd3x3-inl.cuh>
#include <sfm/utils/robust_loss.h>
#include <sfm/utils/utils.h>

#define SQRT_TWO 1.4142135623730950488016887242097
#define DELTA 1e-6
#define DELTA_SQUARED 1e-12

namespace sfm {
namespace ba {
template <typename T>
__host__ __device__ __forceinline__ void
EvaluateImpl(const Eigen::Matrix<T, 3, 4> &extrinsic,
             const Eigen::Vector<T, 3> &intrinsic,
             const Eigen::Vector<T, 3> &point,
             const Eigen::Vector<T, 2> &measurement, T sqrt_weight, T &fobj,
             RobustLoss robust_loss, T loss_radius) {
  const T delta = DELTA;
  const T delta_squared = DELTA_SQUARED;

  Eigen::Vector3<T> dist;
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> error;

  T radical_squared;
  T error_squared_norm;
  T dist_dot_rotated_ray;
  T dist_squared_norm_plus_delta_squared;

  radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_squared * radical_squared;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  dist = point - extrinsic.col(3);
  dist_dot_rotated_ray = dist.dot(rotated_ray);
  dist_squared_norm_plus_delta_squared = dist.squaredNorm() + delta_squared;

  error = rotated_ray;
  error -= (dist_dot_rotated_ray /
            (dist_squared_norm_plus_delta_squared +
             delta * sqrt(dist_squared_norm_plus_delta_squared))) *
           dist;
  error_squared_norm = sqrt_weight * sqrt_weight * error.squaredNorm();

  if (robust_loss == Trivial) {
    sfm::utils::TrivialLoss<T>::Evaluate(error_squared_norm, loss_radius, fobj);
  } else if (robust_loss == Huber) {
    sfm::utils::HuberLoss<T>::Evaluate(error_squared_norm, loss_radius, fobj);
  }
  fobj *= 0.5;
}

template <typename T>
__host__ __device__ __forceinline__ void LinearizeReprojectionLossFunctionImpl(
    const Eigen::Matrix<T, 3, 4> &extrinsic,
    const Eigen::Vector<T, 3> &intrinsic, const Eigen::Vector<T, 3> &point,
    const Eigen::Vector<T, 2> &measurement, T sqrt_weight,
    Eigen::Matrix<T, 3, 6> &jac_ext, Eigen::Matrix<T, 3, 3> &jac_int,
    Eigen::Matrix<T, 3, 1> &rescaled_error, RobustLoss robust_loss,
    T loss_radius) {
  const T delta = DELTA;
  const T delta_squared = DELTA_SQUARED;

  Eigen::Vector3<T> dist;
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> rotated_ray_cross_dist;
  Eigen::Vector3<T> rotated_dist;

  T radical_squared;
  T error_squared_norm;
  T dist_dot_rotated_ray;
  T sqrt_dist_squared_norm_plus_delta_squared;
  T dist_squared_norm_plus_delta_squared_plus_its_sqrt;
  T rescaled_sqrt_weight;
  T rescaled_sqrt_weight_over_dist_squared_norm;
  T fobj;

  radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_squared * radical_squared;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  dist = point - extrinsic.col(3);
  dist_dot_rotated_ray = dist.dot(rotated_ray);
  dist_squared_norm_plus_delta_squared_plus_its_sqrt =
      dist.squaredNorm() + delta_squared;
  sqrt_dist_squared_norm_plus_delta_squared =
      sqrt(dist_squared_norm_plus_delta_squared_plus_its_sqrt);
  dist_squared_norm_plus_delta_squared_plus_its_sqrt +=
      delta * sqrt_dist_squared_norm_plus_delta_squared;
  rescaled_error = rotated_ray;
  rescaled_error -= (dist_dot_rotated_ray /
                     dist_squared_norm_plus_delta_squared_plus_its_sqrt) *
                    dist;
  error_squared_norm = sqrt_weight * sqrt_weight * rescaled_error.squaredNorm();

  if (robust_loss == Trivial) {
    sfm::utils::TrivialLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                          rescaled_sqrt_weight);
  } else if (robust_loss == Huber) {
    sfm::utils::HuberLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                        rescaled_sqrt_weight);
  }

  rescaled_sqrt_weight *= sqrt_weight;
  rescaled_error *= rescaled_sqrt_weight;

  rotated_ray_cross_dist = rotated_ray.cross(dist);
  rescaled_sqrt_weight_over_dist_squared_norm =
      rescaled_sqrt_weight / dist_squared_norm_plus_delta_squared_plus_its_sqrt;

  auto jac_rot = jac_ext.template leftCols<3>();
  jac_rot.noalias() = -rescaled_sqrt_weight_over_dist_squared_norm * dist *
                      rotated_ray_cross_dist.transpose();
  jac_rot(0, 1) += rescaled_sqrt_weight * rotated_ray[2];
  jac_rot(1, 0) -= rescaled_sqrt_weight * rotated_ray[2];
  jac_rot(0, 2) -= rescaled_sqrt_weight * rotated_ray[1];
  jac_rot(2, 0) += rescaled_sqrt_weight * rotated_ray[1];
  jac_rot(1, 2) += rescaled_sqrt_weight * rotated_ray[0];
  jac_rot(2, 1) -= rescaled_sqrt_weight * rotated_ray[0];

  auto jac_lin = jac_ext.template rightCols<3>();
  jac_lin.setZero();
  jac_lin.diagonal().array() =
      dist_dot_rotated_ray * rescaled_sqrt_weight_over_dist_squared_norm;
  jac_lin.noalias() += rescaled_sqrt_weight_over_dist_squared_norm * dist *
                       rotated_ray.transpose();
  T rescale = rescaled_sqrt_weight_over_dist_squared_norm *
              (delta / sqrt_dist_squared_norm_plus_delta_squared + 2) /
              dist_squared_norm_plus_delta_squared_plus_its_sqrt;
  jac_lin.noalias() -=
      (rescale * dist_dot_rotated_ray) * dist * dist.transpose();

  jac_int.col(0) = extrinsic.col(2);
  jac_int.col(0) -= dist.dot(extrinsic.col(2)) /
                    (dist_squared_norm_plus_delta_squared_plus_its_sqrt)*dist;
  jac_int.col(0) *= rescaled_sqrt_weight;
  jac_int.col(1) = jac_int.col(0) * radical_squared;
  jac_int.col(2) = jac_int.col(1) * radical_squared;
}

template <typename T>
__host__ __device__ __forceinline__ void
EvaluateCameraSurrogateFunctionImpl(const Eigen::Matrix<T, 3, 4> &extrinsic,
                                    const Eigen::Vector<T, 3> &intrinsic,
                                    const Eigen::Vector<T, 2> &measurement,
                                    T rescaled_sqrt_weight, T rescaled_a,
                                    const Eigen::Vector<T, 3> &rescaled_g,
                                    T rescaled_constant, T &fobj) {
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> error;

  T radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_squared * radical_squared;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  error = rescaled_sqrt_weight * rotated_ray + rescaled_a * extrinsic.col(3) -
          rescaled_g;
  fobj = error.squaredNorm() + 0.5 * rescaled_constant;
}

template <typename T>
__host__ __device__ __forceinline__ void LinearizeCameraSurrogateFunctionImpl(
    const Eigen::Matrix<T, 3, 4> &extrinsic,
    const Eigen::Vector<T, 3> &intrinsic,
    const Eigen::Vector<T, 2> &measurement, T rescaled_sqrt_weight,
    T rescaled_a, const Eigen::Vector<T, 3> &rescaled_g, T rescaled_constant,
    Eigen::Matrix<T, 3, 6> &jac_ext, Eigen::Matrix<T, 3, 3> &jac_int,
    Eigen::Matrix<T, 3, 1> &rescaled_error) {
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;

  T radical_squared;
  T ray_norm;

  radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_squared * radical_squared;
  ray_norm = ray.norm();
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  rescaled_error = SQRT_TWO * (rescaled_sqrt_weight * rotated_ray +
                               rescaled_a * extrinsic.col(3) - rescaled_g);

  auto jac_rot = jac_ext.template leftCols<3>();
  jac_rot.diagonal().array() = 0;
  jac_rot(0, 1) = SQRT_TWO * rescaled_sqrt_weight * rotated_ray[2];
  jac_rot(1, 0) = -SQRT_TWO * rescaled_sqrt_weight * rotated_ray[2];
  jac_rot(0, 2) = -SQRT_TWO * rescaled_sqrt_weight * rotated_ray[1];
  jac_rot(2, 0) = SQRT_TWO * rescaled_sqrt_weight * rotated_ray[1];
  jac_rot(1, 2) = SQRT_TWO * rescaled_sqrt_weight * rotated_ray[0];
  jac_rot(2, 1) = -SQRT_TWO * rescaled_sqrt_weight * rotated_ray[0];

  auto jac_lin = jac_ext.template rightCols<3>();
  jac_lin.setZero();
  jac_lin.diagonal().array() = SQRT_TWO * rescaled_a;

  jac_int.col(0) = SQRT_TWO * rescaled_sqrt_weight * extrinsic.col(2);
  jac_int.col(1) = radical_squared * jac_int.col(0);
  jac_int.col(2) = radical_squared * jac_int.col(1);
}

template <typename T>
__host__ __device__ __forceinline__ void EvaluatePointSurrogateFunctionImpl(
    const Eigen::Vector<T, 3> &point, T rescaled_a,
    const Eigen::Vector<T, 3> &rescaled_g, T rescaled_constant, T &fobj) {
  Eigen::Vector3<T> error;

  error = rescaled_a * point - rescaled_g;
  fobj = error.squaredNorm() + 0.5 * rescaled_constant;
}

template <typename T>
__host__ __device__ __forceinline__ void LinearizePointSurrogateFunctionImpl(
    const Eigen::Vector<T, 3> &point, T rescaled_a,
    const Eigen::Vector<T, 3> &rescaled_g, T rescaled_constant,
    Eigen::Matrix<T, 3, 3> &jac_lmk, Eigen::Matrix<T, 3, 1> &rescaled_error) {
  rescaled_error = SQRT_TWO * (rescaled_a * point - rescaled_g);
  jac_lmk.setZero();
  jac_lmk.diagonal().array() = SQRT_TWO * rescaled_a;
}

template <typename T>
__host__ __device__ __forceinline__ void MajorizeReprojectionLossFunctionImpl(
    const Eigen::Matrix<T, 3, 4> &extrinsic,
    const Eigen::Vector<T, 3> &intrinsic, const Eigen::Vector<T, 3> &point,
    const Eigen::Vector<T, 2> &measurement, T sqrt_weight,
    Eigen::Vector<T, 3> &rescaled_g_vec, Eigen::Vector<T, 3> &rescaled_h_vec,
    Eigen::Vector<T, 5> &rescaled_f_vec, Eigen::Vector<T, 3> &rescaled_s_vec,
    T &rescaled_sqrt_weight, T &rescaled_a_val, T &rescaled_constant, T &fobj,
    RobustLoss robust_loss, T loss_radius) {
  const T delta = DELTA;
  const T delta_squared = DELTA_SQUARED;

  Eigen::Vector3<T> dist;
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> error;

  T ray_squared_norm;
  T error_squared_norm;
  T dist_dot_rotated_ray;
  T dist_squared_norm_plus_delta_squared;

  rescaled_f_vec[0] = 1;
  T &radical_squared = rescaled_f_vec[1];
  T &radical_fourth_order = rescaled_f_vec[2];
  radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];
  radical_fourth_order = radical_squared * radical_squared;
  rescaled_f_vec[3] = radical_squared * radical_fourth_order;
  rescaled_f_vec[4] = radical_fourth_order * radical_fourth_order;

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_fourth_order;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  dist = point - extrinsic.col(3);

  dist_dot_rotated_ray = dist.dot(rotated_ray);
  dist_squared_norm_plus_delta_squared = dist.squaredNorm() + delta_squared;
  rescaled_a_val = dist_dot_rotated_ray / dist_squared_norm_plus_delta_squared;
  error = rotated_ray;
  error -= (dist_dot_rotated_ray /
            (dist_squared_norm_plus_delta_squared +
             delta * sqrt(dist_squared_norm_plus_delta_squared))) *
           dist;
  error_squared_norm = sqrt_weight * sqrt_weight * error.squaredNorm();

  if (robust_loss == Trivial) {
    sfm::utils::TrivialLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                          rescaled_sqrt_weight);
  } else if (robust_loss == Huber) {
    sfm::utils::HuberLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                        rescaled_sqrt_weight);
  }
  fobj *= 0.5;
  rescaled_constant = fobj - 0.5 * rescaled_sqrt_weight * rescaled_sqrt_weight *
                                 error_squared_norm;
  rescaled_sqrt_weight *= sqrt_weight;

  rescaled_g_vec = 0.5 * rescaled_sqrt_weight *
                   (rotated_ray + rescaled_a_val * (extrinsic.col(3) + point));
  rescaled_a_val *= rescaled_sqrt_weight;

  rescaled_h_vec = 0.75 * rescaled_sqrt_weight * ray;
  rescaled_h_vec.noalias() += 0.25 * rescaled_a_val *
                              extrinsic.template leftCols<3>().transpose() *
                              dist;

  rescaled_s_vec = rescaled_f_vec.template head<3>() *
                   (rescaled_sqrt_weight * rescaled_h_vec[2]);
  rescaled_f_vec *= rescaled_sqrt_weight * rescaled_sqrt_weight;

  rescaled_constant += 0.5 * delta_squared * rescaled_a_val * rescaled_a_val;
}

template <typename T>
__host__ __device__ __forceinline__ void ConstructSurrogateFunctionImpl(
    const Eigen::Matrix<T, 3, 4> &extrinsic,
    const Eigen::Vector<T, 3> &intrinsic, const Eigen::Vector<T, 3> &point,
    const Eigen::Vector<T, 2> &measurement, T sqrt_weight, T &rescaled_a_val,
    Eigen::Vector<T, 3> &rescaled_g_vec, T &rescaled_sqrt_weight,
    T &rescaled_constant, T &fobj, RobustLoss robust_loss, T loss_radius) {
  const T delta = DELTA;
  const T delta_squared = DELTA_SQUARED;

  Eigen::Vector3<T> dist;
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> error;

  T error_squared_norm;
  T dist_dot_rotated_ray;
  T dist_squared_norm_plus_delta_squared;

  T radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];
  T radical_fourth_order = radical_squared * radical_squared;

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_fourth_order;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  dist = point - extrinsic.col(3);

  dist_dot_rotated_ray = dist.dot(rotated_ray);
  dist_squared_norm_plus_delta_squared = dist.squaredNorm() + delta_squared;
  rescaled_a_val = dist_dot_rotated_ray / dist_squared_norm_plus_delta_squared;
  error = rotated_ray;
  error -= (dist_dot_rotated_ray /
            (dist_squared_norm_plus_delta_squared +
             delta * sqrt(dist_squared_norm_plus_delta_squared))) *
           dist;
  error_squared_norm = sqrt_weight * sqrt_weight * error.squaredNorm();

  if (robust_loss == Trivial) {
    sfm::utils::TrivialLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                          rescaled_sqrt_weight);
  } else if (robust_loss == Huber) {
    sfm::utils::HuberLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                        rescaled_sqrt_weight);
  }
  fobj *= 0.5;
  rescaled_constant = fobj - 0.5 * rescaled_sqrt_weight * rescaled_sqrt_weight *
                                 error_squared_norm;
  rescaled_sqrt_weight *= sqrt_weight;

  rescaled_g_vec = 0.5 * rescaled_sqrt_weight *
                   (rotated_ray + rescaled_a_val * (extrinsic.col(3) + point));
  rescaled_a_val *= rescaled_sqrt_weight;

  rescaled_constant += 0.5 * delta_squared * rescaled_a_val * rescaled_a_val;
}

template <typename T>
__host__ __device__ __forceinline__ void ConstructSurrogateFunctionImpl(
    const Eigen::Matrix<T, 3, 4> &extrinsic,
    const Eigen::Vector<T, 3> &intrinsic, const Eigen::Vector<T, 3> &point,
    const Eigen::Vector<T, 2> &measurement, T sqrt_weight,
    Eigen::Vector<T, 3> &rescaled_h_vec, T &rescaled_a_val,
    Eigen::Vector<T, 3> &rescaled_g_vec, T &rescaled_sqrt_weight,
    T &rescaled_constant, T &fobj, RobustLoss robust_loss, T loss_radius) {
  const T delta = DELTA;
  const T delta_squared = DELTA_SQUARED;

  Eigen::Vector3<T> dist;
  Eigen::Vector3<T> ray;
  Eigen::Vector3<T> rotated_ray;
  Eigen::Vector3<T> error;

  T error_squared_norm;
  T dist_dot_rotated_ray;
  T dist_squared_norm_plus_delta_squared;

  T radical_squared =
      measurement[0] * measurement[0] + measurement[1] * measurement[1];
  T radical_fourth_order = radical_squared * radical_squared;

  ray.template head<2>() = measurement;
  ray[2] = intrinsic[0] + intrinsic[1] * radical_squared +
           intrinsic[2] * radical_fourth_order;
  rotated_ray.noalias() = extrinsic.template leftCols<3>() * ray;

  dist = point - extrinsic.col(3);

  dist_dot_rotated_ray = dist.dot(rotated_ray);
  dist_squared_norm_plus_delta_squared = dist.squaredNorm() + delta_squared;
  rescaled_a_val = dist_dot_rotated_ray / dist_squared_norm_plus_delta_squared;
  error = rotated_ray;
  error -= (dist_dot_rotated_ray /
            (dist_squared_norm_plus_delta_squared +
             delta * sqrt(dist_squared_norm_plus_delta_squared))) *
           dist;
  error_squared_norm = sqrt_weight * sqrt_weight * error.squaredNorm();

  if (robust_loss == Trivial) {
    sfm::utils::TrivialLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                          rescaled_sqrt_weight);
  } else if (robust_loss == Huber) {
    sfm::utils::HuberLoss<T>::Linearize(error_squared_norm, loss_radius, fobj,
                                        rescaled_sqrt_weight);
  }
  fobj *= 0.5;
  rescaled_constant = fobj - 0.5 * rescaled_sqrt_weight * rescaled_sqrt_weight *
                                 error_squared_norm;
  rescaled_sqrt_weight *= sqrt_weight;

  rescaled_g_vec = 0.5 * rescaled_sqrt_weight *
                   (rotated_ray + rescaled_a_val * (extrinsic.col(3) + point));
  rescaled_a_val *= rescaled_sqrt_weight;

  rescaled_h_vec = 0.75 * rescaled_sqrt_weight * ray;
  rescaled_h_vec.noalias() += 0.25 * rescaled_a_val *
                              extrinsic.template leftCols<3>().transpose() *
                              dist;

  rescaled_constant += 0.5 * delta_squared * rescaled_a_val * rescaled_a_val;
}

template <typename T>
__host__ __device__ __forceinline__ void SolveExtrinsicsProximalOperatorImpl(
    const T *data, T reg, const T *init_extrinsics, T *extrinsics,
    int_t num_extrinsics, int_t extrinsics_index) {
  Eigen::Matrix<T, 3, 4> ret;
  sfm::utils::GetMatrixOfArray(num_extrinsics, init_extrinsics,
                               extrinsics_index, ret);
  ret *= reg;

  Eigen::Vector3<T> hess_rot_t{0, 0, 0};
  T hess_tt = reg;
  Eigen::Matrix3<T> &grad_rot = *(Eigen::Matrix3<T> *)(ret.data());
  Eigen::Vector3<T> &grad_t = *(Eigen::Vector3<T> *)(ret.data() + 9);

  data += extrinsics_index;
#pragma unroll 3
  for (int_t i = 0; i < 3; i++) {
    hess_rot_t[i] = *data;
    data += num_extrinsics;

#pragma unroll 3
    for (int_t j = 0; j < 3; j++) {
      grad_rot(j, i) += *data;
      data += num_extrinsics;
    }
  }

  hess_tt += *data;
  data += num_extrinsics;

#pragma unroll 3
  for (int_t j = 0; j < 3; j++) {
    grad_t[j] += *data;
    data += num_extrinsics;
  }

  grad_rot.noalias() -= grad_t * hess_rot_t.transpose() / hess_tt;
  Eigen::Matrix3<T> U, V;
  Eigen::Vector3<T> S;

  sfm::utils::internal::svd3x3(
      grad_rot(0, 0), grad_rot(0, 1), grad_rot(0, 2), grad_rot(1, 0),
      grad_rot(1, 1), grad_rot(1, 2), grad_rot(2, 0), grad_rot(2, 1),
      grad_rot(2, 2), U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2),
      U(2, 0), U(2, 1), U(2, 2), S[0], S[1], S[2], V(0, 0), V(0, 1), V(0, 2),
      V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

  ret.template leftCols<3>().noalias() = U * V.transpose();
  ret.col(3).noalias() -= ret.template leftCols<3>() * hess_rot_t;
  ret.col(3) /= hess_tt;

  sfm::utils::SetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolveExtrinsicsProximalOperatorImpl(
    const T *data, T reg, const int_t *init_extrinsics_dicts,
    const T *init_extrinsics, T *extrinsics, int_t num_extrinsics,
    int_t extrinsics_index) {
  Eigen::Matrix<T, 3, 4> ret;
  sfm::utils::GetMatrixOfArray(num_extrinsics, init_extrinsics,
                               init_extrinsics_dicts[extrinsics_index], ret);
  ret *= reg;

  Eigen::Vector3<T> hess_rot_t{0, 0, 0};
  T hess_tt = reg;
  Eigen::Matrix3<T> &grad_rot = *(Eigen::Matrix3<T> *)(ret.data());
  Eigen::Vector3<T> &grad_t = *(Eigen::Vector3<T> *)(ret.data() + 9);

  data += extrinsics_index;
#pragma unroll 3
  for (int_t i = 0; i < 3; i++) {
    hess_rot_t[i] = *data;
    data += num_extrinsics;

#pragma unroll 3
    for (int_t j = 0; j < 3; j++) {
      grad_rot(j, i) += *data;
      data += num_extrinsics;
    }
  }

  hess_tt += *data;
  data += num_extrinsics;

#pragma unroll 3
  for (int_t j = 0; j < 3; j++) {
    grad_t[j] += *data;
    data += num_extrinsics;
  }

  grad_rot.noalias() -= grad_t * hess_rot_t.transpose() / hess_tt;
  Eigen::Matrix3<T> U, V;
  Eigen::Vector3<T> S;

  sfm::utils::internal::svd3x3(
      grad_rot(0, 0), grad_rot(0, 1), grad_rot(0, 2), grad_rot(1, 0),
      grad_rot(1, 1), grad_rot(1, 2), grad_rot(2, 0), grad_rot(2, 1),
      grad_rot(2, 2), U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2),
      U(2, 0), U(2, 1), U(2, 2), S[0], S[1], S[2], V(0, 0), V(0, 1), V(0, 2),
      V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

  ret.template leftCols<3>().noalias() = U * V.transpose();
  ret.col(3).noalias() -= ret.template leftCols<3>() * hess_rot_t;
  ret.col(3) /= hess_tt;

  sfm::utils::SetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolveExtrinsicsProximalOperatorImpl(
    const T *data, T reg, const int_t *init_extrinsics_dicts,
    const T *init_extrinsics, int_t num_init_extrinsics, T *extrinsics,
    int_t num_extrinsics, int_t extrinsics_index) {
  Eigen::Matrix<T, 3, 4> ret;
  sfm::utils::GetMatrixOfArray(num_init_extrinsics, init_extrinsics,
                               init_extrinsics_dicts[extrinsics_index], ret);
  ret *= reg;

  Eigen::Vector3<T> hess_rot_t{0, 0, 0};
  T hess_tt = reg;
  Eigen::Matrix3<T> &grad_rot = *(Eigen::Matrix3<T> *)(ret.data());
  Eigen::Vector3<T> &grad_t = *(Eigen::Vector3<T> *)(ret.data() + 9);

  data += extrinsics_index;
#pragma unroll 3
  for (int_t i = 0; i < 3; i++) {
    hess_rot_t[i] = *data;
    data += num_extrinsics;

#pragma unroll 3
    for (int_t j = 0; j < 3; j++) {
      grad_rot(j, i) += *data;
      data += num_extrinsics;
    }
  }

  hess_tt += *data;
  data += num_extrinsics;

#pragma unroll 3
  for (int_t j = 0; j < 3; j++) {
    grad_t[j] += *data;
    data += num_extrinsics;
  }

  grad_rot.noalias() -= grad_t * hess_rot_t.transpose() / hess_tt;
  Eigen::Matrix3<T> U, V;
  Eigen::Vector3<T> S;

  sfm::utils::internal::svd3x3(
      grad_rot(0, 0), grad_rot(0, 1), grad_rot(0, 2), grad_rot(1, 0),
      grad_rot(1, 1), grad_rot(1, 2), grad_rot(2, 0), grad_rot(2, 1),
      grad_rot(2, 2), U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2),
      U(2, 0), U(2, 1), U(2, 2), S[0], S[1], S[2], V(0, 0), V(0, 1), V(0, 2),
      V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

  ret.template leftCols<3>().noalias() = U * V.transpose();
  ret.col(3).noalias() -= ret.template leftCols<3>() * hess_rot_t;
  ret.col(3) /= hess_tt;

  sfm::utils::SetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void
UpdateIntrinsicsHelper(const Eigen::Vector<T, 8> &hess_grad, T reg,
                       Eigen::Vector3<T> &intrinsics) {
  const Eigen::Vector<T, 5> &hess = *(Eigen::Vector<T, 5> *)(hess_grad.data());
  const Eigen::Vector<T, 3> &grad =
      *(Eigen::Vector<T, 3> *)(hess_grad.data() + 5);

  Eigen::Matrix<T, 3, 3> H;
  H(0, 0) = hess[0];
  H(0, 1) = hess[1];
  H(0, 2) = hess[2];
  H(1, 0) = hess[1];
  H(1, 1) = hess[2];
  H(1, 2) = hess[3];
  H(2, 0) = hess[2];
  H(2, 1) = hess[3];
  H(2, 2) = hess[4];
  H.diagonal().array() += reg;

  intrinsics.noalias() = H.inverse() * grad;
}

template <typename T>
__host__ __device__ __forceinline__ void SolveIntrinsicsProximalOperatorImpl(
    const T *data, T reg, const T *init_intrinsics, T *intrinsics,
    int_t num_intrinsics, int_t intrinsics_index) {
  Eigen::Vector<T, 8> hess_grad = Eigen::Vector<T, 8>::Zero();
  Eigen::Vector<T, 3> &init = *(Eigen::Vector<T, 3> *)(hess_grad.data() + 5);
  sfm::utils::GetMatrixOfArray(num_intrinsics, init_intrinsics,
                               intrinsics_index, init);
  init *= reg;
  sfm::utils::AddMatrixOfArray(num_intrinsics, data, intrinsics_index,
                               hess_grad);

  Eigen::Vector<T, 3> ret;
  UpdateIntrinsicsHelper(hess_grad, reg, ret);

  sfm::utils::SetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolveIntrinsicsProximalOperatorImpl(
    const T *data, T reg, const int_t *init_intrinsics_dicts,
    const T *init_intrinsics, T *intrinsics, int_t num_intrinsics,
    int_t intrinsics_index) {
  Eigen::Vector<T, 8> hess_grad = Eigen::Vector<T, 8>::Zero();
  Eigen::Vector<T, 3> &init = *(Eigen::Vector<T, 3> *)(hess_grad.data() + 5);
  sfm::utils::GetMatrixOfArray(num_intrinsics, init_intrinsics,
                               init_intrinsics_dicts[intrinsics_index], init);
  init *= reg;
  sfm::utils::AddMatrixOfArray(num_intrinsics, data, intrinsics_index,
                               hess_grad);

  Eigen::Vector<T, 3> ret;
  UpdateIntrinsicsHelper(hess_grad, reg, ret);

  sfm::utils::SetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolveIntrinsicsProximalOperatorImpl(
    const T *data, T reg, const int_t *init_intrinsics_dicts,
    const T *init_intrinsics, int_t num_init_intrinsics, T *intrinsics,
    int_t num_intrinsics, int_t intrinsics_index) {
  Eigen::Vector<T, 8> hess_grad = Eigen::Vector<T, 8>::Zero();
  Eigen::Vector<T, 3> &init = *(Eigen::Vector<T, 3> *)(hess_grad.data() + 5);
  sfm::utils::GetMatrixOfArray(num_init_intrinsics, init_intrinsics,
                               init_intrinsics_dicts[intrinsics_index], init);
  init *= reg;
  sfm::utils::AddMatrixOfArray(num_intrinsics, data, intrinsics_index,
                               hess_grad);

  Eigen::Vector<T, 3> ret;
  UpdateIntrinsicsHelper(hess_grad, reg, ret);

  sfm::utils::SetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_index,
                               ret);
}

template <typename T>
__host__ __device__ __forceinline__ void
UpdatePointsHelper(const Eigen::Vector4<T> &hess_grad,
                   Eigen::Vector3<T> &point) {
  point = hess_grad.template tail<3>() / hess_grad[0];
}

template <typename T>
__host__ __device__ __forceinline__ void
SolvePointProximalOperatorImpl(const T *data, T reg, const T *init_points,
                               T *points, int_t num_points, int_t point_index) {
  Eigen::Vector4<T> hess_grad;
  hess_grad[0] = reg;
  Eigen::Vector3<T> &init = *(Eigen::Vector3<T> *)(hess_grad.data() + 1);
  sfm::utils::GetMatrixOfArray(num_points, init_points, point_index, init);
  // if (point_index == 5) {
  //   printf("------------Non Distributed-------------\n");
  //   printf("%f %f %f\n", init[0], init[1], init[2]);
  // }

  init *= reg;
  sfm::utils::AddMatrixOfArray(num_points, data, point_index, hess_grad);

  // if (point_index == 5) {
  //   printf("%f %f %f %f\n", hess_grad[0], hess_grad[1], hess_grad[2],
  //          hess_grad[3]);
  // }

  Eigen::Vector3<T> ret;
  UpdatePointsHelper(hess_grad, ret);

  // if (point_index == 5) {
  //   printf("%f %f %f\n\n", ret[0], ret[1], ret[2]);
  // }

  sfm::utils::SetMatrixOfArray(num_points, points, point_index, ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolvePointProximalOperatorImpl(
    const T *data, T reg, const int_t *init_point_dicts, const T *init_points,
    T *points, int_t num_points, int_t point_index) {
  Eigen::Vector4<T> hess_grad;
  hess_grad[0] = reg;
  Eigen::Vector3<T> &init = *(Eigen::Vector3<T> *)(hess_grad.data() + 1);
  sfm::utils::GetMatrixOfArray(num_points, init_points,
                               init_point_dicts[point_index], init);
  init *= reg;
  sfm::utils::AddMatrixOfArray(num_points, data, point_index, hess_grad);

  Eigen::Vector3<T> ret;
  UpdatePointsHelper(hess_grad, ret);

  sfm::utils::SetMatrixOfArray(num_points, points, point_index, ret);
}

template <typename T>
__host__ __device__ __forceinline__ void SolvePointProximalOperatorImpl(
    const T *data, T reg, const int_t *init_point_dicts, const T *init_points,
    int_t num_init_points, T *points, int_t num_points, int_t point_index) {
  Eigen::Vector4<T> hess_grad;
  hess_grad[0] = reg;
  Eigen::Vector3<T> &init = *(Eigen::Vector3<T> *)(hess_grad.data() + 1);
  sfm::utils::GetMatrixOfArray(num_init_points, init_points,
                               init_point_dicts[point_index], init);
  // if (point_index == 5) {
  //   printf("------------Distributed-------------\n");
  //   printf("%f %f %f\n", init[0], init[1], init[2]);
  // }

  init *= reg;
  sfm::utils::AddMatrixOfArray(num_points, data, point_index, hess_grad);

  // if (point_index == 5) {
  //   printf("%f %f %f %f\n", hess_grad[0], hess_grad[1], hess_grad[2],
  //          hess_grad[3]);
  // }

  Eigen::Vector3<T> ret;
  UpdatePointsHelper(hess_grad, ret);

  // if (point_index == 5) {
  //   printf("%f %f %f\n\n", ret[0], ret[1], ret[2]);
  // }

  sfm::utils::SetMatrixOfArray(num_points, points, point_index, ret);
}
} // namespace ba
} // namespace sfm
