// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifdef __INTELLISENSE___
void __syncthreads();
#endif

#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include <sfm/ba/functions/ba_functions.h>
#include <sfm/ba/macro.h>
#include <sfm/ba/types.h>
#include <sfm/math/SO3.h>
#include <sfm/types.h>
#include <sfm/utils/internal/atomic-inl.cuh>
#include <sfm/utils/internal/shared_memory-inl.cuh>
#include <sfm/utils/utils.h>

namespace sfm {
namespace ba {
template <typename T>
__global__ void EvaluateReprojectionLossFunctionKernel(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_measurements)
    return;

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, idx,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[idx], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[idx], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[idx], point);
  T sqrt_weight = sqrt_weights[idx];

  T fobj;
  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluateReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  T sqrt_weight = sqrt_weights[measurement_id];

  T fobj;

  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluateReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_id,
                               extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);
  T sqrt_weight = sqrt_weights[measurement_id];

  T fobj;

  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void LinearizeReprojectionLossFunctionKernel(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_measurements)
    return;

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, idx,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[idx], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[idx], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[idx], point);
  T sqrt_weight = sqrt_weights[idx];

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(
      num_measurements, jacobians_extrinsics_intrinsics, idx, jac_ext_int);
  sfm::utils::SetMatrixOfArray(num_measurements, rescaled_errors, idx,
                               rescaled_error);
}

template <typename T>
__global__ void LinearizeReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void LinearizeReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_id,
                               extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void EvaluateCameraSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);

  T rescaled_sqrt_weight = rescaled_sqrt_weights[idx];
  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  T fobj = 0;
  EvaluateCameraSurrogateFunctionImpl(extrinsic, intrinsic, measurement,
                                      rescaled_sqrt_weight, rescaled_a,
                                      rescaled_g, rescaled_const, fobj);
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluateCameraSurrogateFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_id,
                               extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);

  T rescaled_sqrt_weight = rescaled_sqrt_weights[idx];
  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  T fobj = 0;
  EvaluateCameraSurrogateFunctionImpl(extrinsic, intrinsic, measurement,
                                      rescaled_sqrt_weight, rescaled_a,
                                      rescaled_g, rescaled_const, fobj);
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void LinearizeCameraSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);

  T rescaled_sqrt_weight = rescaled_sqrt_weights[idx];
  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeCameraSurrogateFunctionImpl(
      extrinsic, intrinsic, measurement, rescaled_sqrt_weight, rescaled_a,
      rescaled_g, rescaled_const, jac_ext, jac_int, rescaled_error);

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void LinearizeCameraSurrogateFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics, extrinsics_id,
                               extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);

  T rescaled_sqrt_weight = rescaled_sqrt_weights[idx];
  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeCameraSurrogateFunctionImpl(
      extrinsic, intrinsic, measurement, rescaled_sqrt_weight, rescaled_a,
      rescaled_g, rescaled_const, jac_ext, jac_int, rescaled_error);

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void EvaluatePointSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Vector3<T> point;
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);

  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  T fobj = 0;
  EvaluatePointSurrogateFunctionImpl(point, rescaled_a, rescaled_g,
                                     rescaled_const, fobj);
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluatePointSurrogateFunctionKernel(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *fobjs,
    int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Vector3<T> point;
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);

  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  T fobj = 0;
  EvaluatePointSurrogateFunctionImpl(point, rescaled_a, rescaled_g,
                                     rescaled_const, fobj);
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void LinearizePointSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_point, T *rescaled_errors,
    int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Vector3<T> point;
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);

  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  Eigen::Matrix<T, 3, 3> jac_lmk;
  Eigen::Vector3<T> rescaled_error;

  LinearizePointSurrogateFunctionImpl(point, rescaled_a, rescaled_g,
                                      rescaled_const, jac_lmk, rescaled_error);

  sfm::utils::SetMatrixOfArray(N, jacobians_point, idx, jac_lmk);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void LinearizePointSurrogateFunctionKernel(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *jacobians_point,
    T *rescaled_errors, int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Vector3<T> point;
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);

  T rescaled_a = rescaled_a_vals[idx];
  T rescaled_const = rescaled_constants[idx];

  Eigen::Vector3<T> rescaled_g;
  sfm::utils::GetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g);

  Eigen::Matrix<T, 3, 3> jac_lmk;
  Eigen::Vector3<T> rescaled_error;

  LinearizePointSurrogateFunctionImpl(point, rescaled_a, rescaled_g,
                                      rescaled_const, jac_lmk, rescaled_error);

  sfm::utils::SetMatrixOfArray(N, jacobians_point, idx, jac_lmk);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void EvaluateAngleAxisReprojectionLossFunctionKernel(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_measurements)
    return;

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, idx,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_infos[idx], angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[idx], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[idx], point);
  T sqrt_weight = sqrt_weights[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  T fobj;

  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluateAngleAxisReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_infos[measurement_id],
                               angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  T sqrt_weight = sqrt_weights[measurement_id];

  T fobj;

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void EvaluateAngleAxisReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_id, angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  T fobj;

  EvaluateImpl(extrinsic, intrinsic, point, measurement, sqrt_weight, fobj,
               robust_loss, loss_radius);

  fobjs[idx] = fobj;
}

template <typename T>
__global__ void LinearizeAngleAxisReprojectionLossFunctionKernel(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_measurements)
    return;

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, idx,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_infos[idx], angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[idx], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[idx], point);
  T sqrt_weight = sqrt_weights[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  Eigen::Matrix<T, 3, 3> dexpR;
  sfm::math::SO3::Dexp(omega, dexpR);
  jac_ext.template leftCols<3>() = jac_ext.template leftCols<3>() * dexpR;

  sfm::utils::SetMatrixOfArray(
      num_measurements, jacobians_extrinsics_intrinsics, idx, jac_ext_int);
  sfm::utils::SetMatrixOfArray(num_measurements, rescaled_errors, idx,
                               rescaled_error);
}

template <typename T>
__global__ void LinearizeAngleAxisReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights,
    T *jacobians_extrinsics_intrinsics, T *rescaled_errors,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_infos[measurement_id],
                               angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  Eigen::Matrix<T, 3, 3> dexpR;
  sfm::math::SO3::Dexp(omega, dexpR);
  jac_ext.template leftCols<3>() = jac_ext.template leftCols<3>() * dexpR;

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void LinearizeAngleAxisReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];
  const int_t extrinsics_id =
      extrinsics_indices[extrinsics_infos[measurement_id]];
  const int_t intrinsics_id =
      intrinsics_indices[intrinsics_infos[measurement_id]];
  const int_t point_id = point_indices[point_infos[measurement_id]];

  Eigen::Vector<T, 6> angle_axis_extrinsic;
  Eigen::Vector<T, 3> intrinsic;
  Eigen::Vector<T, 3> point;
  Eigen::Vector<T, 2> measurement;

  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  sfm::utils::GetMatrixOfArray(num_extrinsics, angle_axis_extrinsics,
                               extrinsics_id, angle_axis_extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics, intrinsics_id,
                               intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_id, point);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Matrix<T, 3, 3> &rotation =
      *(Eigen::Matrix<T, 3, 3> *)(extrinsic.data());
  Eigen::Vector<T, 3> &omega =
      *(Eigen::Vector<T, 3> *)(angle_axis_extrinsic.data());
  sfm::math::SO3::Exp(omega, rotation);
  extrinsic.col(3) = angle_axis_extrinsic.template tail<3>();

  Eigen::Matrix<T, 3, 9> jac_ext_int;
  Eigen::Vector3<T> rescaled_error;

  Eigen::Matrix<T, 3, 6> &jac_ext =
      *(Eigen::Matrix<T, 3, 6> *)(jac_ext_int.data());
  Eigen::Matrix<T, 3, 3> &jac_int =
      *(Eigen::Matrix<T, 3, 3> *)(jac_ext_int.data() + 18);

  LinearizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, jac_ext, jac_int,
      rescaled_error, robust_loss, loss_radius);

  Eigen::Matrix<T, 3, 3> dexpR;
  sfm::math::SO3::Dexp(omega, dexpR);
  jac_ext.template leftCols<3>() = jac_ext.template leftCols<3>() * dexpR;

  sfm::utils::SetMatrixOfArray(N, jacobians_extrinsics_intrinsics, idx,
                               jac_ext_int);
  sfm::utils::SetMatrixOfArray(N, rescaled_errors, idx, rescaled_error);
}

template <typename T>
__global__ void MajorizeReprojectionLossFunctionKernel(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_measurements)
    return;

  Eigen::Vector2<T> measurement;
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, idx,
                               measurement);

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[idx], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[idx], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[idx], point);

  T sqrt_weight = sqrt_weights[idx];

  Eigen::Vector<T, 7> rescaled_h_a_g_vec;
  Eigen::Vector<T, 8> rescaled_f_s_vec;

  Eigen::Vector3<T> &rescaled_h_vec =
      *(Eigen::Vector3<T> *)(rescaled_h_a_g_vec.data());
  T &rescaled_a_val = *(rescaled_h_a_g_vec.data() + 3);
  Eigen::Vector3<T> &rescaled_g_vec =
      *(Eigen::Vector3<T> *)(rescaled_h_a_g_vec.data() + 4);
  Eigen::Vector<T, 5> &rescaled_f_vec =
      *(Eigen::Vector<T, 5> *)(rescaled_f_s_vec.data());
  Eigen::Vector3<T> &rescaled_s_vec =
      *(Eigen::Vector3<T> *)(rescaled_f_s_vec.data() + 5);

  T rescaled_sqrt_weight;
  T rescaled_constant, fobj;

  MajorizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, rescaled_g_vec,
      rescaled_h_vec, rescaled_f_vec, rescaled_s_vec, rescaled_sqrt_weight,
      rescaled_a_val, rescaled_constant, fobj, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(num_measurements, rescaled_h_a_g_vecs, idx,
                               rescaled_h_a_g_vec);
  sfm::utils::SetMatrixOfArray(num_measurements, rescaled_f_s_vecs, idx,
                               rescaled_f_s_vec);

  rescaled_sqrt_weights[idx] = rescaled_sqrt_weight;
  rescaled_constants[idx] = rescaled_constant;
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void MajorizeReprojectionLossFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Vector<T, 7> rescaled_h_a_g_vec;
  Eigen::Vector<T, 8> rescaled_f_s_vec;

  Eigen::Vector3<T> &rescaled_h_vec =
      *(Eigen::Vector3<T> *)(rescaled_h_a_g_vec.data());
  T &rescaled_a_val = *(rescaled_h_a_g_vec.data() + 3);
  Eigen::Vector3<T> &rescaled_g_vec =
      *(Eigen::Vector3<T> *)(rescaled_h_a_g_vec.data() + 4);
  Eigen::Vector<T, 5> &rescaled_f_vec =
      *(Eigen::Vector<T, 5> *)(rescaled_f_s_vec.data());
  Eigen::Vector3<T> &rescaled_s_vec =
      *(Eigen::Vector3<T> *)(rescaled_f_s_vec.data() + 5);

  T rescaled_sqrt_weight;
  T rescaled_constant, fobj;

  MajorizeReprojectionLossFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, rescaled_g_vec,
      rescaled_h_vec, rescaled_f_vec, rescaled_s_vec, rescaled_sqrt_weight,
      rescaled_a_val, rescaled_constant, fobj, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(N, rescaled_h_a_g_vecs, idx, rescaled_h_a_g_vec);
  sfm::utils::SetMatrixOfArray(N, rescaled_f_s_vecs, idx, rescaled_f_s_vec);

  rescaled_sqrt_weights[idx] = rescaled_sqrt_weight;
  rescaled_constants[idx] = rescaled_constant;
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void ConstructSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_a_vals, T *rescaled_g_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  T sqrt_weight = sqrt_weights[measurement_id];

  T rescaled_a_val;
  Eigen::Vector3<T> rescaled_g_vec;

  T rescaled_sqrt_weight;
  T rescaled_constant, fobj;

  ConstructSurrogateFunctionImpl(extrinsic, intrinsic, point, measurement,
                                 sqrt_weight, rescaled_a_val, rescaled_g_vec,
                                 rescaled_sqrt_weight, rescaled_constant, fobj,
                                 robust_loss, loss_radius);

  rescaled_a_vals[idx] = rescaled_a_val;
  sfm::utils::SetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g_vec);
  rescaled_sqrt_weights[idx] = rescaled_sqrt_weight;
  rescaled_constants[idx] = rescaled_constant;
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void ConstructSurrogateFunctionKernel(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_vecs, T *rescaled_a_vals,
    T *rescaled_g_vecs, T *rescaled_sqrt_weights, T *rescaled_constants,
    T *fobjs, RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  const int_t measurement_id = measurement_indices[idx];

  Eigen::Matrix<T, 3, 4> extrinsic;
  Eigen::Vector3<T> intrinsic;
  Eigen::Vector3<T> point;
  Eigen::Vector2<T> measurement;

  sfm::utils::GetMatrixOfArray(num_extrinsics, extrinsics,
                               extrinsics_infos[measurement_id], extrinsic);
  sfm::utils::GetMatrixOfArray(num_intrinsics, intrinsics,
                               intrinsics_infos[measurement_id], intrinsic);
  sfm::utils::GetMatrixOfArray(num_points, points, point_infos[measurement_id],
                               point);
  sfm::utils::GetMatrixOfArray(num_measurements, measurements, measurement_id,
                               measurement);
  T sqrt_weight = sqrt_weights[measurement_id];

  Eigen::Vector3<T> rescaled_h_vec;
  T rescaled_a_val;
  Eigen::Vector3<T> rescaled_g_vec;

  T rescaled_sqrt_weight;
  T rescaled_constant, fobj;

  ConstructSurrogateFunctionImpl(
      extrinsic, intrinsic, point, measurement, sqrt_weight, rescaled_h_vec,
      rescaled_a_val, rescaled_g_vec, rescaled_sqrt_weight, rescaled_constant,
      fobj, robust_loss, loss_radius);

  sfm::utils::SetMatrixOfArray(N, rescaled_h_vecs, idx, rescaled_h_vec);
  rescaled_a_vals[idx] = rescaled_a_val;
  sfm::utils::SetMatrixOfArray(N, rescaled_g_vecs, idx, rescaled_g_vec);
  rescaled_sqrt_weights[idx] = rescaled_sqrt_weight;
  rescaled_constants[idx] = rescaled_constant;
  fobjs[idx] = fobj;
}

template <typename T>
__global__ void UpdateExtrinsicsAndPointProximalOperatorKernel(
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    T *extrinsics_hess_grad, T *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_measurements)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * num_measurements + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * num_measurements + idx];
  }
  __syncthreads();

  int_t extrinsics_index = extrinsics_infos[idx];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

  if (threadIdx.y == 3) {
    int_t point_offset = point_infos[idx];
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      atomicAdd(points_hess_grad + point_offset, temp);
      extrinsics_offset += num_extrinsics;
      point_offset += num_points;
    }
  } else {
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      extrinsics_offset += num_extrinsics;
    }
  }
}

template <typename T>
__global__ void UpdateExtrinsicsAndPointProximalOperatorKernel(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t extrinsics_index = extrinsics_infos[measurement_index];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

  if (threadIdx.y == 3) {
    int_t point_offset = point_infos[measurement_index];
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      atomicAdd(points_hess_grad + point_offset, temp);
      extrinsics_offset += num_extrinsics;
      point_offset += num_points;
    }
  } else {
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      extrinsics_offset += num_extrinsics;
    }
  }
}

template <typename T>
__global__ void UpdateExtrinsicsAndPointProximalOperatorKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t extrinsics_index =
      extrinsics_indices[extrinsics_infos[measurement_index]];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

  if (threadIdx.y == 3) {
    int_t point_offset = point_indices[point_infos[measurement_index]];
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      atomicAdd(points_hess_grad + point_offset, temp);
      extrinsics_offset += num_extrinsics;
      point_offset += num_points;
    }
  } else {
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
      extrinsics_offset += num_extrinsics;
    }
  }
}

template <typename T>
__global__ void UpdateExtrinsicsProximalOperatorKernel(
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t extrinsics_index = extrinsics_infos[idx];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

#pragma unroll 4
  for (int_t i = 0; i < 4; i++) {
    T temp = rescaled_h_a_val *
             rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
    atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
    extrinsics_offset += num_extrinsics;
  }
}

template <typename T>
__global__ void UpdateExtrinsicsProximalOperatorKernel(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t extrinsics_index = extrinsics_infos[measurement_index];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

#pragma unroll 4
  for (int_t i = 0; i < 4; i++) {
    T temp = rescaled_h_a_val *
             rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
    atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
    extrinsics_offset += num_extrinsics;
  }
}

template <typename T>
__global__ void UpdateExtrinsicsProximalOperatorKernel(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t extrinsics_index =
      extrinsics_indices[extrinsics_infos[measurement_index]];
  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

#pragma unroll 4
  for (int_t i = 0; i < 4; i++) {
    T temp = rescaled_h_a_val *
             rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
    atomicAdd(extrinsics_hess_grad + extrinsics_offset, temp);
    extrinsics_offset += num_extrinsics;
  }
}

template <typename T>
__global__ void UpdateIntrinsicsProximalOperatorKernel(
    const T *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    T *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_measurements)
    return;

  T rescaled_f_s_val = rescaled_f_s_vecs[threadIdx.y * num_measurements + idx];

  int_t intrinsics_index = intrinsics_infos[idx];
  int_t intrinsics_offset = num_intrinsics * threadIdx.y + intrinsics_index;
  atomicAdd(intrinsics_hess_grad + intrinsics_offset, rescaled_f_s_val);
}

template <typename T>
__global__ void UpdateIntrinsicsProximalOperatorKernel(
    const int_t *measurement_indices, const T *rescaled_f_s_vecs,
    const int_t *intrinsics_infos, T *intrinsics_hess_grad,
    int_t num_intrinsics, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_f_s_val = rescaled_f_s_vecs[threadIdx.y * N + idx];

  int_t measurement_index = measurement_indices[idx];

  int_t intrinsics_index = intrinsics_infos[measurement_index];
  int_t intrinsics_offset = num_intrinsics * threadIdx.y + intrinsics_index;
  atomicAdd(intrinsics_hess_grad + intrinsics_offset, rescaled_f_s_val);
}

template <typename T>
__global__ void UpdateIntrinsicsProximalOperatorKernel(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const T *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    T *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_f_s_val = rescaled_f_s_vecs[threadIdx.y * N + idx];

  int_t measurement_index = measurement_indices[idx];

  int_t intrinsics_index =
      intrinsics_indices[intrinsics_infos[measurement_index]];

  assert(intrinsics_index >= 0 && intrinsics_index < num_intrinsics);

  int_t intrinsics_offset = num_intrinsics * threadIdx.y + intrinsics_index;
  atomicAdd(intrinsics_hess_grad + intrinsics_offset, rescaled_f_s_val);
}

template <typename T>
__global__ void UpdatePointProximalOperatorKernel(
    const T *rescaled_a_g_vecs, const int_t *point_infos, T *points_hess_grad,
    int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_a_g_val = rescaled_a_g_vecs[threadIdx.y * N + idx];

  T *rescaled_a_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  if (threadIdx.y == 0) {
    rescaled_a_vals_shared[threadIdx.x] = rescaled_a_g_val;
  }
  __syncthreads();

  int_t point_index = point_infos[idx];
  int_t point_offset = num_points * threadIdx.y + point_index;

  T temp = rescaled_a_g_val * rescaled_a_vals_shared[threadIdx.x];
  atomicAdd(points_hess_grad + point_offset, temp);
}

template <typename T>
__global__ void UpdatePointProximalOperatorKernel(
    const int_t *measurement_indices, const T *rescaled_a_g_vecs,
    const int_t *point_infos, T *points_hess_grad, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_a_g_val = rescaled_a_g_vecs[threadIdx.y * N + idx];

  T *rescaled_a_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  if (threadIdx.y == 0) {
    rescaled_a_vals_shared[threadIdx.x] = rescaled_a_g_val;
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t point_index = point_infos[measurement_index];
  int_t point_offset = num_points * threadIdx.y + point_index;

  T temp = rescaled_a_g_val * rescaled_a_vals_shared[threadIdx.x];
  atomicAdd(points_hess_grad + point_offset, temp);
}

template <typename T>
__global__ void UpdatePointProximalOperatorKernel(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *rescaled_a_g_vecs, const int_t *point_infos, T *points_hess_grad,
    int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_a_g_val = rescaled_a_g_vecs[threadIdx.y * N + idx];

  T *rescaled_a_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  if (threadIdx.y == 0) {
    rescaled_a_vals_shared[threadIdx.x] = rescaled_a_g_val;
  }
  __syncthreads();

  int_t measurement_index = measurement_indices[idx];

  int_t point_index = point_indices[point_infos[measurement_index]];
  int_t point_offset = num_points * threadIdx.y + point_index;

  assert(point_index >= 0 && point_index < num_points);

  T temp = rescaled_a_g_val * rescaled_a_vals_shared[threadIdx.x];
  atomicAdd(points_hess_grad + point_offset, temp);
}

template <typename T>
__global__ void ComputeExtrinsicsAndPointProximalOperatorProductKernel(
    const int_t *measurement_indices_by_extrinsics,
    const int_t *measurement_indices_by_points, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n,
    T *points_hess_grad_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index_by_extrinsics =
      measurement_indices_by_extrinsics[idx];

  int_t extrinsics_offset =
      4 * N * threadIdx.y + measurement_index_by_extrinsics;

  if (threadIdx.y == 3) {
    int_t measurement_index_by_points = measurement_indices_by_points[idx];
    int_t point_offset = measurement_index_by_points;
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      extrinsics_hess_grad_n[extrinsics_offset] = temp;
      points_hess_grad_n[point_offset] = temp;
      extrinsics_offset += N;
      point_offset += N;
    }
  } else {
#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val *
               rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
      extrinsics_hess_grad_n[extrinsics_offset] = temp;
      extrinsics_offset += N;
    }
  }
}

template <typename T>
__global__ void ComputeExtrinsicsAndPointProximalOperatorProductKernel(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *measurement_indices_by_points, const int_t *extrinsics_indices,
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    T *points_hess_grad_n, int_t num_extrinsics, int_t num_measurements,
    int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage[4];

  int_t start = measurement_offsets_by_extrinsics[blockId];
  int_t end = measurement_offsets_by_extrinsics[blockId + 1];
  int_t extrinsics_index = extrinsics_indices
      [extrinsics_infos[measurement_dicts_by_extrinsics[start]]];

  T extrinsics_hess_grad_sum[4] = {0};

  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * num_measurements + n];

    if (threadIdx.y == 3) {
      int_t point_offset = measurement_indices_by_points[n];
#pragma unroll 4
      for (int_t i = 0; i < 4; i++) {
        T temp = rescaled_h_a_val * rescaled_a_g_vecs[i * num_measurements + n];
        extrinsics_hess_grad_sum[i] += temp;
        points_hess_grad_n[point_offset] = temp;
        point_offset += num_measurements;
      }
    } else {
#pragma unroll 4
      for (int_t i = 0; i < 4; i++) {
        T temp = rescaled_h_a_val * rescaled_a_g_vecs[i * num_measurements + n];
        extrinsics_hess_grad_sum[i] += temp;
      }
    }
  }

  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

  for (int_t i = 0; i < 4; i++) {
    T ret =
        WarpReduce(temp_storage[threadIdx.y]).Sum(extrinsics_hess_grad_sum[i]);
    if (threadIdx.x == 0) {
      extrinsics_hess_grad[extrinsics_offset] += ret;
      extrinsics_offset += num_extrinsics;
    }
  }
}

template <typename T>
__global__ void ComputeExtrinsicsProximalOperatorProductKernel(
    const int_t *measurement_indices_by_extrinsics, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *rescaled_a_g_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * N + idx];
  if (threadIdx.y == 3) {
    rescaled_a_g_vals_shared[threadIdx.x] = rescaled_h_a_val;
  } else {
    rescaled_a_g_vals_shared[(threadIdx.y + 1) * blockDim.x + threadIdx.x] =
        rescaled_a_g_vecs[(threadIdx.y + 1) * N + idx];
  }
  __syncthreads();

  int_t measurement_index_by_extrinsics =
      measurement_indices_by_extrinsics[idx];

  int_t extrinsics_offset =
      4 * N * threadIdx.y + measurement_index_by_extrinsics;

#pragma unroll 4
  for (int_t i = 0; i < 4; i++) {
    T temp = rescaled_h_a_val *
             rescaled_a_g_vals_shared[i * blockDim.x + threadIdx.x];
    extrinsics_hess_grad_n[extrinsics_offset] = temp;
    extrinsics_offset += N;
  }
}

template <typename T>
__global__ void ComputeExtrinsicsProximalOperatorKernel(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *extrinsics_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage[4];

  int_t start = measurement_offsets_by_extrinsics[blockId];
  int_t end = measurement_offsets_by_extrinsics[blockId + 1];
  int_t extrinsics_index = extrinsics_indices
      [extrinsics_infos[measurement_dicts_by_extrinsics[start]]];

  T extrinsics_hess_grad_sum[4] = {0};

  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    T rescaled_h_a_val = rescaled_h_a_vecs[threadIdx.y * num_measurements + n];

#pragma unroll 4
    for (int_t i = 0; i < 4; i++) {
      T temp = rescaled_h_a_val * rescaled_a_g_vecs[i * num_measurements + n];
      extrinsics_hess_grad_sum[i] += temp;
    }
  }

  int_t extrinsics_offset = 4 * num_extrinsics * threadIdx.y + extrinsics_index;

  for (int_t i = 0; i < 4; i++) {
    T ret =
        WarpReduce(temp_storage[threadIdx.y]).Sum(extrinsics_hess_grad_sum[i]);
    if (threadIdx.x == 0) {
      extrinsics_hess_grad[extrinsics_offset] += ret;
      extrinsics_offset += num_extrinsics;
    }
  }
}

template <typename T>
__global__ void ComputeIntrinsicsProximalOperatorProductKernel(
    const int_t *measurement_indices_by_intrinsics, const T *rescaled_f_s_vecs,
    T *intrinsics_hess_grad_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_f_s_val = rescaled_f_s_vecs[threadIdx.y * N + idx];

  int_t measurement_index_by_intrinsics =
      measurement_indices_by_intrinsics[idx];

  int_t intrinsics_offset = N * threadIdx.y + measurement_index_by_intrinsics;
  intrinsics_hess_grad_n[intrinsics_offset] = rescaled_f_s_val;
}

template <typename T>
__global__ void ComputePointProximalOperatorProductKernel(
    const int_t *measurement_indices_by_points, const T *rescaled_a_g_vecs,
    T *points_hess_grad_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T rescaled_a_g_val = rescaled_a_g_vecs[threadIdx.y * N + idx];

  T *rescaled_a_vals_shared = sfm::utils::internal::SharedMemory<T>::get();

  if (threadIdx.y == 0) {
    rescaled_a_vals_shared[threadIdx.x] = rescaled_a_g_val;
  }
  __syncthreads();

  int_t measurement_index_by_points = measurement_indices_by_points[idx];
  int_t point_offset = N * threadIdx.y + measurement_index_by_points;

  T temp = rescaled_a_g_val * rescaled_a_vals_shared[threadIdx.x];
  points_hess_grad_n[point_offset] = temp;
}

template <typename T>
__global__ void SolveExtrinsicsProximalOperatorKernel(const T *data, T reg,
                                                      const T *init_extrinsics,
                                                      T *extrinsics,
                                                      int_t num_extrinsics) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_extrinsics)
    return;

  SolveExtrinsicsProximalOperatorImpl(data, reg, init_extrinsics, extrinsics,
                                      num_extrinsics, idx);
}

template <typename T>
__global__ void SolveExtrinsicsProximalOperatorKernel(
    const int_t *extrinsics_indices, const T *data, T reg,
    const T *init_extrinsics, T *extrinsics, int_t num_extrinsics, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t extrinsics_index = extrinsics_indices[idx];

  SolveExtrinsicsProximalOperatorImpl(data, reg, init_extrinsics, extrinsics,
                                      num_extrinsics, extrinsics_index);
}

template <typename T>
__global__ void SolveExtrinsicsProximalOperatorKernel(
    const T *data, T reg, const int_t *init_extrinsics_dicts,
    const T *init_extrinsics, int_t num_init_extrinsics, T *extrinsics,
    int_t num_extrinsics, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  SolveExtrinsicsProximalOperatorImpl(data, reg, init_extrinsics_dicts,
                                      init_extrinsics, num_init_extrinsics,
                                      extrinsics, num_extrinsics, idx);
}

template <typename T>
__global__ void SolveIntrinsicsProximalOperatorKernel(const T *data, T reg,
                                                      const T *init_intrinsics,
                                                      T *intrinsics,
                                                      int_t num_intrinsics) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_intrinsics)
    return;

  SolveIntrinsicsProximalOperatorImpl(data, reg, init_intrinsics, intrinsics,
                                      num_intrinsics, idx);
}

template <typename T>
__global__ void SolveIntrinsicsProximalOperatorKernel(
    const int_t *intrinsics_indices, const T *data, T reg,
    const T *init_intrinsics, T *intrinsics, int_t num_intrinsics, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t intrinsics_index = intrinsics_indices[idx];

  SolveIntrinsicsProximalOperatorImpl(data, reg, init_intrinsics, intrinsics,
                                      num_intrinsics, intrinsics_index);
}

template <typename T>
__global__ void SolveIntrinsicsProximalOperatorKernel(
    const T *data, T reg, const int_t *init_intrinsics_dicts,
    const T *init_intrinsics, int_t num_init_intrinsics, T *intrinsics,
    int_t num_intrinsics, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  SolveIntrinsicsProximalOperatorImpl(data, reg, init_intrinsics_dicts,
                                      init_intrinsics, num_init_intrinsics,
                                      intrinsics, num_intrinsics, idx);
}

template <typename T>
__global__ void SolvePointProximalOperatorKernel(const T *data, T reg,
                                                 const T *init_points,
                                                 T *points, int_t num_points) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_points)
    return;

  SolvePointProximalOperatorImpl(data, reg, init_points, points, num_points,
                                 idx);
}

template <typename T>
__global__ void
SolvePointProximalOperatorKernel(const int_t *point_indices, const T *data,
                                 T reg, const T *init_points, T *points,
                                 int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t point_index = point_indices[idx];

  SolvePointProximalOperatorImpl(data, reg, init_points, points, num_points,
                                 point_index);
}

template <typename T>
__global__ void SolvePointProximalOperatorKernel(
    const T *data, T reg, const int_t *init_point_dicts, const T *init_points,
    int_t num_init_points, T *points, int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  SolvePointProximalOperatorImpl(data, reg, init_point_dicts, init_points,
                                 num_init_points, points, num_points, idx);
}

template <typename T>
__global__ void UpdateReprojectionLossFunctionHessianGradientKernel(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_measurements)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) *
                                                num_measurements +
                                            idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }

    grad_sum += val * (rescaled_errors + n * num_measurements)[idx];
  }

  int_t camera_index = camera_infos[idx];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  const auto &measurement_index = idx;

  auto hess_cl = hessians_cl + LANDMARK_SIZE * threadIdx.y * num_measurements +
                 measurement_index;
#pragma unroll 3
  for (int_t i = 3; i < 6; i++) {
    *hess_cl = -hess_sum[i];
    hess_cl += num_measurements;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);

  if (threadIdx.y >= 3 && threadIdx.y < 6) {
    int_t point_index = point_infos[idx];

    auto hess_ll = hessians_ll +
                   (LANDMARK_SIZE + 1) * (threadIdx.y - 3) * num_points +
                   point_index;
    for (int_t i = threadIdx.y; i < 6; i++) {
      atomicAdd(hess_ll, hess_sum[i]);
      hess_ll += num_points;
    }

    auto grad_l = gradients_l + (threadIdx.y - 3) * num_points + +point_index;
    atomicAdd(grad_l, -grad_sum);
  }
}

template <typename T>
__global__ void UpdateReprojectionLossFunctionHessianGradientKernel(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }

    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t camera_index = camera_infos[measurement_index];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  auto hess_cl = hessians_cl + LANDMARK_SIZE * threadIdx.y * N + idx;
#pragma unroll 3
  for (int_t i = 3; i < 6; i++) {
    *hess_cl = -hess_sum[i];
    hess_cl += N;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);

  if (threadIdx.y >= 3 && threadIdx.y < 6) {
    int_t point_index = point_infos[measurement_index];

    auto hess_ll = hessians_ll +
                   (LANDMARK_SIZE + 1) * (threadIdx.y - 3) * num_points +
                   point_index;
    for (int_t i = threadIdx.y; i < 6; i++) {
      atomicAdd(hess_ll, hess_sum[i]);
      hess_ll += num_points;
    }

    auto grad_l = gradients_l + (threadIdx.y - 3) * num_points + point_index;
    atomicAdd(grad_l, -grad_sum);
  }
}

template <typename T>
__global__ void UpdateReprojectionLossFunctionHessianGradientKernel(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }

    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t camera_index = camera_indices[camera_infos[measurement_index]];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  auto hess_cl = hessians_cl + LANDMARK_SIZE * threadIdx.y * N + idx;
#pragma unroll 3
  for (int_t i = 3; i < 6; i++) {
    *hess_cl = -hess_sum[i];
    hess_cl += N;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);

  if (threadIdx.y >= 3 && threadIdx.y < 6) {
    int_t point_index = point_indices[point_infos[measurement_index]];

    auto hess_ll = hessians_ll +
                   (LANDMARK_SIZE + 1) * (threadIdx.y - 3) * num_points +
                   point_index;
    for (int_t i = threadIdx.y; i < 6; i++) {
      atomicAdd(hess_ll, hess_sum[i]);
      hess_ll += num_points;
    }

    auto grad_l = gradients_l + (threadIdx.y - 3) * num_points + point_index;
    atomicAdd(grad_l, -grad_sum);
  }
}

template <typename T>
__global__ void UpdateCameraSurrogateFunctionHessianGradientKernel(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  const auto &measurement_index = idx;

  int_t camera_index = camera_infos[measurement_index];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);
}

template <typename T>
__global__ void UpdateCameraSurrogateFunctionHessianGradientKernel(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t camera_index = camera_infos[measurement_index];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);
}

template <typename T>
__global__ void UpdateCameraSurrogateFunctionHessianGradientKernel(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t camera_index = camera_indices[camera_infos[measurement_index]];

  auto hess_cc = hessians_cc + (D_CAMERA_SIZE + 1) * threadIdx.y * num_cameras +
                 camera_index;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    atomicAdd(hess_cc, hess_sum[i]);
    hess_cc += num_cameras;
  }

  auto grad_c = gradients_c + threadIdx.y * num_cameras + camera_index;
  atomicAdd(grad_c, grad_sum);
}

template <typename T>
__global__ void UpdatePointSurrogateFunctionHessianGradientKernel(
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[LANDMARK_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_points[(3 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < LANDMARK_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  const auto &measurement_index = idx;

  int_t point_index = point_infos[measurement_index];

  auto hess_ll = hessians_ll + (LANDMARK_SIZE + 1) * threadIdx.y * num_points +
                 point_index;
  for (int_t i = threadIdx.y; i < LANDMARK_SIZE; i++) {
    atomicAdd(hess_ll, hess_sum[i]);
    hess_ll += num_points;
  }

  auto grad_l = gradients_l + threadIdx.y * num_points + point_index;
  atomicAdd(grad_l, grad_sum);
}

template <typename T>
__global__ void UpdatePointSurrogateFunctionHessianGradientKernel(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[LANDMARK_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_points[(3 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < LANDMARK_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t point_index = point_infos[measurement_index];

  auto hess_ll = hessians_ll + (LANDMARK_SIZE + 1) * threadIdx.y * num_points +
                 point_index;
  for (int_t i = threadIdx.y; i < LANDMARK_SIZE; i++) {
    atomicAdd(hess_ll, hess_sum[i]);
    hess_ll += num_points;
  }

  auto grad_l = gradients_l + threadIdx.y * num_points + point_index;
  atomicAdd(grad_l, grad_sum);
}

template <typename T>
__global__ void UpdatePointSurrogateFunctionHessianGradientKernel(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[LANDMARK_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_points[(3 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < LANDMARK_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index = measurement_indices[idx];

  int_t point_index = point_indices[point_infos[measurement_index]];

  auto hess_ll = hessians_ll + (LANDMARK_SIZE + 1) * threadIdx.y * num_points +
                 point_index;
  for (int_t i = threadIdx.y; i < LANDMARK_SIZE; i++) {
    atomicAdd(hess_ll, hess_sum[i]);
    hess_ll += num_points;
  }

  auto grad_l = gradients_l + threadIdx.y * num_points + point_index;
  atomicAdd(grad_l, grad_sum);
}

template <typename T>
__global__ void ComputeReprojectionLossFunctionHessianGradientProductKernel(
    const int_t *measurement_indices_by_cameras,
    const int_t *measurement_indices_by_points,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *hessians_cl_n, T *hessians_ll_n, T *gradients_c_n,
    T *gradients_l_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_extrinsics_intrinsics[(9 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }

    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index_by_cameras = measurement_indices_by_cameras[idx];

  auto hess_cc = hessians_cc_n +
                 (2 * D_CAMERA_SIZE - threadIdx.y + 1) * threadIdx.y * N / 2 +
                 measurement_index_by_cameras;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    *hess_cc = hess_sum[i];
    hess_cc += N;
  }

  auto hess_cl = hessians_cl_n + LANDMARK_SIZE * threadIdx.y * N + idx;
#pragma unroll 3
  for (int_t i = 3; i < 6; i++) {
    *hess_cl = -hess_sum[i];
    hess_cl += N;
  }

  auto grad_c = gradients_c_n + threadIdx.y * N + measurement_index_by_cameras;
  *grad_c = grad_sum;

  if (threadIdx.y >= 3 && threadIdx.y < 6) {
    int_t measurement_index_by_points = measurement_indices_by_points[idx];
    auto hess_ll =
        hessians_ll_n +
        (2 * D_LANDMARK_SIZE - threadIdx.y + 4) * (threadIdx.y - 3) * N / 2 +
        measurement_index_by_points;
    for (int_t i = threadIdx.y; i < 6; i++) {
      *hess_ll = hess_sum[i];
      hess_ll += N;
    }

    auto grad_l =
        gradients_l_n + (threadIdx.y - 3) * N + measurement_index_by_points;
    *grad_l = -grad_sum;
  }
}

template <typename T>
__global__ void ComputeReprojectionLossFunctionHessianGradientProductKernel(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras,
    const int_t *measurement_indices_by_points, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll_n,
    T *gradients_c, T *gradients_l_n, int_t num_cameras, int_t num_measurements,
    int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage[D_CAMERA_SIZE];

  int_t start = measurement_offsets_by_cameras[blockId];
  int_t end = measurement_offsets_by_cameras[blockId + 1];
  int_t camera_index =
      camera_indices[camera_infos[measurement_dicts_by_cameras[start]]];

  T hess_cc_sum[D_CAMERA_SIZE] = {0};
  T grad_c_sum = 0;

  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    T hess_sum[D_CAMERA_SIZE] = {0};
    T grad_sum = 0;

    for (int_t k = 0; k < 3; k++) {
      T val = jacobians_extrinsics_intrinsics[(9 * k + threadIdx.y) *
                                                  num_measurements +
                                              n];
#pragma unroll
      for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
        hess_sum[i] +=
            val *
            jacobians_extrinsics_intrinsics[(9 * k + i) * num_measurements + n];
      }

      grad_sum += val * (rescaled_errors + k * num_measurements)[n];
    }

    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_cc_sum[i] += hess_sum[i];
    }
    grad_c_sum += grad_sum;

    auto hess_cl =
        hessians_cl + LANDMARK_SIZE * threadIdx.y * num_measurements + n;
#pragma unroll 3
    for (int_t i = 3; i < 6; i++) {
      *hess_cl = -hess_sum[i];
      hess_cl += num_measurements;
    }

    if (threadIdx.y >= 3 && threadIdx.y < 6) {
      int_t measurement_index_by_points = measurement_indices_by_points[n];
      auto hess_ll = hessians_ll_n +
                     (2 * D_LANDMARK_SIZE - threadIdx.y + 4) *
                         (threadIdx.y - 3) * num_measurements / 2 +
                     measurement_index_by_points;
      for (int_t i = threadIdx.y; i < 6; i++) {
        *hess_ll = hess_sum[i];
        hess_ll += num_measurements;
      }

      auto grad_l = gradients_l_n + (threadIdx.y - 3) * num_measurements +
                    measurement_index_by_points;
      *grad_l = -grad_sum;
    }
  }

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    T ret = WarpReduce(temp_storage[threadIdx.y]).Sum(hess_cc_sum[i]);
    if (threadIdx.x == 0) {
      hessians_cc[(D_CAMERA_SIZE * threadIdx.y + i) * num_cameras +
                  camera_index] += ret;
    }
  }

  T ret = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_c_sum);
  if (threadIdx.x == 0) {
    gradients_c[threadIdx.y * num_cameras + camera_index] += ret;
  }
}

template <typename T>
__global__ void ComputeCameraSurrogateFunctionHessianGradientProductKernel(
    const int_t *measurement_indices_by_cameras,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *gradients_c_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[D_CAMERA_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val =
        jacobians_extrinsics_intrinsics[(D_CAMERA_SIZE * n + threadIdx.y) * N +
                                        idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }

    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index_by_cameras = measurement_indices_by_cameras[idx];

  auto hess_cc = hessians_cc_n +
                 (2 * D_CAMERA_SIZE - threadIdx.y + 1) * threadIdx.y * N / 2 +
                 measurement_index_by_cameras;
#pragma unroll
  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    *hess_cc = hess_sum[i];
    hess_cc += N;
  }

  auto grad_c = gradients_c_n + threadIdx.y * N + measurement_index_by_cameras;
  *grad_c = grad_sum;
}

template <typename T>
__global__ void ComputeCameraSurrogateFunctionHessianGradientKernel(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage[D_CAMERA_SIZE];

  int_t start = measurement_offsets_by_cameras[blockId];
  int_t end = measurement_offsets_by_cameras[blockId + 1];
  int_t camera_index =
      camera_indices[camera_infos[measurement_dicts_by_cameras[start]]];

  T hess_cc_sum[D_CAMERA_SIZE] = {0};
  T grad_c_sum = 0;

  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    int_t measurement_index = measurement_dicts_by_cameras[n];

    T hess_sum[D_CAMERA_SIZE] = {0};
    T grad_sum = 0;

    for (int_t k = 0; k < 3; k++) {
      T val =
          jacobians_extrinsics_intrinsics[(D_CAMERA_SIZE * k + threadIdx.y) *
                                              num_measurements +
                                          n];
#pragma unroll
      for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
        hess_sum[i] +=
            val * jacobians_extrinsics_intrinsics[(D_CAMERA_SIZE * k + i) *
                                                      num_measurements +
                                                  n];
      }

      grad_sum += val * (rescaled_errors + k * num_measurements)[n];
    }

    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      hess_cc_sum[i] += hess_sum[i];
    }
    grad_c_sum += grad_sum;
  }

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i++) {
    T ret = WarpReduce(temp_storage[threadIdx.y]).Sum(hess_cc_sum[i]);
    if (threadIdx.x == 0) {
      hessians_cc[(D_CAMERA_SIZE * threadIdx.y + i) * num_cameras +
                  camera_index] += ret;
    }
  }

  T ret = WarpReduce(temp_storage[threadIdx.y]).Sum(grad_c_sum);
  if (threadIdx.x == 0) {
    gradients_c[threadIdx.y * num_cameras + camera_index] += ret;
  }
}

template <typename T>
__global__ void ComputePointSurrogateFunctionHessianGradientProductKernel(
    const int_t *measurement_indices_by_points, const T *jacobians_points,
    const T *rescaled_errors, T *hessians_ll_n, T *gradients_l_n, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *jacs_shared = sfm::utils::internal::SharedMemory<T>::get();

  T hess_sum[LANDMARK_SIZE] = {0};
  T grad_sum = 0;

#pragma unroll 3
  for (int_t n = 0; n < 3; n++) {
    T val = jacobians_points[(3 * n + threadIdx.y) * N + idx];
    __syncthreads();
    jacs_shared[threadIdx.y * blockDim.x + threadIdx.x] = val;
    __syncthreads();

#pragma unroll
    for (int_t i = 0; i < LANDMARK_SIZE; i++) {
      hess_sum[i] += val * jacs_shared[i * blockDim.x + threadIdx.x];
    }
    grad_sum += val * (rescaled_errors + n * N)[idx];
  }

  int_t measurement_index_by_points = measurement_indices_by_points[idx];

  auto hess_ll = hessians_ll_n +
                 (2 * D_LANDMARK_SIZE - threadIdx.y + 1) * threadIdx.y * N / 2 +
                 measurement_index_by_points;
  for (int_t i = threadIdx.y; i < D_LANDMARK_SIZE; i++) {
    *hess_ll = hess_sum[i];
    hess_ll += N;
  }

  auto grad_l = gradients_l_n + threadIdx.y * N + measurement_index_by_points;
  *grad_l = grad_sum;
}

template <typename T>
__global__ void UpdateHessianSumForCameraKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const T *hess_cc_n, const int_t *camera_infos,
    T *hess_cc, int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  int_t index = threadIdx.y;

  const int_t b = 2 * D_CAMERA_SIZE + 1;
  const int_t row = 0.5 * (b - sqrt(float(b * b) - 8 * index));
  const int_t col = row + index - (b - row) * row / 2;

  int_t start = measurement_offsets[blockId];
  int_t end = measurement_offsets[blockId + 1];
  int_t camera_index = camera_indices[camera_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    sum += hess_cc_n[index * num_measurements + n];
  }

  __syncwarp();

  T ret = WarpReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    hess_cc[(D_CAMERA_SIZE * row + col) * num_cameras + camera_index] += ret;
  }
}

template <typename T>
__global__ void ComputeCameraDictedReductionKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, T alpha, const T *x, const int_t *camera_infos,
    T beta, T *y, int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;

  if (blockId >= N)
    return;

  int_t row = threadIdx.y;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  int_t start = measurement_offsets[blockId];
  int_t end = measurement_offsets[blockId + 1];
  int_t camera_index = camera_indices[camera_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    sum += x[row * num_measurements + n];
  }

  __syncwarp();

  T ret = WarpReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    y[row * num_cameras + camera_index] =
        alpha * ret + beta * y[row * num_cameras + camera_index];
  }
}

template <typename T>
__global__ void UpdateHessianSumForPointKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, const T *hess_ll_n, const int_t *point_infos,
    T *hess_ll, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t start = measurement_offsets[idx];
  int_t end = measurement_offsets[idx + 1];
  int_t point_index = point_indices[point_infos[measurement_dicts[start]]];

  const int_t b = 2 * D_LANDMARK_SIZE + 1;
  const int_t row = 0.5 * (b - sqrt(float(b * b) - 8 * threadIdx.y));
  const int_t col = row + threadIdx.y - (b - row) * row / 2;

  T sum = 0;
  for (int_t n = start; n < end; n++) {
    sum += hess_ll_n[threadIdx.y * num_measurements + n];
  }

  hess_ll[(D_LANDMARK_SIZE * row + col) * num_points + point_index] += sum;
}

template <typename T>
__global__ void ComputePointDictedReductionKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, T alpha, const T *x, const int_t *point_infos,
    T beta, T *y, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t start = measurement_offsets[idx];
  int_t end = measurement_offsets[idx + 1];
  int_t point_index = point_indices[point_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start; n < end; n++) {
    sum += x[threadIdx.y * num_measurements + n];
  }

  y[threadIdx.y * num_points + point_index] =
      alpha * sum + beta * y[threadIdx.y * num_points + point_index];
}

template <typename T>
__global__ void ComputeHessianCameraPointLeftMultiplicationKernel(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_measurements)
    return;

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();
  int_t camera_index = camera_infos[idx];

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i += blockDim.y) {
    x_shared[i * blockDim.x + threadIdx.x] =
        (x + i * num_cameras)[camera_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
    sum += (hessians_cl +
            (i * LANDMARK_SIZE + threadIdx.y) * num_measurements)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t point_index = point_infos[idx];
  atomicAdd(y + threadIdx.y * num_points + point_index, sum);
}

template <typename T>
__global__ void ComputeHessianCameraPointLeftMultiplicationKernel(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_indices[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();
  int_t camera_index = camera_infos[measurement_index];

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i += blockDim.y) {
    x_shared[i * blockDim.x + threadIdx.x] =
        (x + i * num_cameras)[camera_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
    sum += (hessians_cl + (i * LANDMARK_SIZE + threadIdx.y) * N)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t point_index = point_infos[measurement_index];
  atomicAdd(y + threadIdx.y * num_points + point_index, sum);
}

template <typename T>
__global__ void ComputeHessianCameraPointLeftMultiplicationKernel(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_indices[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();
  int_t camera_index = camera_indices[camera_infos[measurement_index]];

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i += blockDim.y) {
    x_shared[i * blockDim.x + threadIdx.x] =
        (x + i * num_cameras)[camera_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
    sum += (hessians_cl + (i * LANDMARK_SIZE + threadIdx.y) * N)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t point_index = point_indices[point_infos[measurement_index]];
  atomicAdd(y + threadIdx.y * num_points + point_index, sum);
}

template <typename T>
__global__ void ComputeHessianCameraPointRightMultiplicationKernel(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_measurements)
    return;

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();

  int_t point_index = point_infos[idx];

  for (int_t i = threadIdx.y; i < 3; i += blockDim.y) {
    x_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        (x + threadIdx.y * num_points)[point_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < LANDMARK_SIZE; i++) {
    sum += (hessians_cl +
            (threadIdx.y * LANDMARK_SIZE + i) * num_measurements)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t camera_index = camera_infos[idx];
  atomicAdd(y + threadIdx.y * num_cameras + camera_index, sum);
}

template <typename T>
__global__ void ComputeHessianCameraPointRightMultiplicationKernel(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_indices[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();

  int_t point_index = point_infos[measurement_index];

  for (int_t i = threadIdx.y; i < 3; i += blockDim.y) {
    x_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        (x + threadIdx.y * num_points)[point_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < LANDMARK_SIZE; i++) {
    sum += (hessians_cl + (threadIdx.y * LANDMARK_SIZE + i) * N)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t camera_index = camera_infos[measurement_index];
  atomicAdd(y + threadIdx.y * num_cameras + camera_index, sum);
}

template <typename T>
__global__ void ComputeHessianCameraPointRightMultiplicationKernel(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_indices[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();

  int_t point_index = point_indices[point_infos[measurement_index]];

  for (int_t i = threadIdx.y; i < 3; i += blockDim.y) {
    x_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        (x + threadIdx.y * num_points)[point_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < LANDMARK_SIZE; i++) {
    sum += (hessians_cl + (threadIdx.y * LANDMARK_SIZE + i) * N)[idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }
  sum *= beta;

  int_t camera_index = camera_indices[camera_infos[measurement_index]];
  atomicAdd(y + threadIdx.y * num_cameras + camera_index, sum);
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointLeftProductKernel(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, T alpha, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_indices[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();
  int_t camera_index = camera_indices[camera_infos[measurement_index]];

  for (int_t i = threadIdx.y; i < D_CAMERA_SIZE; i += blockDim.y) {
    x_shared[i * blockDim.x + threadIdx.x] =
        (x + i * num_cameras)[camera_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
    sum += hessians_cl[(i * LANDMARK_SIZE + threadIdx.y) * N + idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }

  y[threadIdx.y * num_measurements + idx] = alpha * sum;
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointLeftSumKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, T alpha, const T *x, const int_t *point_infos,
    T beta, T *y, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.y + threadIdx.y;

  if (idx >= N)
    return;

  int_t start = measurement_offsets[idx];
  int_t end = measurement_offsets[idx + 1];
  int_t point_index = point_indices[point_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start; n < end; n++) {
    sum += x[threadIdx.x * num_measurements + n];
  }

  y[threadIdx.x * num_points + point_index] =
      alpha * sum + beta * y[threadIdx.x * num_points + point_index];
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointLeftMultiplicationKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.y + threadIdx.y;

  if (idx >= N)
    return;

  int_t start = measurement_offsets[idx];
  int_t end = measurement_offsets[idx + 1];
  int_t point_index = point_indices[point_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start; n < end; n++) {
    int_t measurement_index = measurement_dicts[n];
    int_t camera_index = camera_indices[camera_infos[measurement_index]];

    for (int_t i = 0; i < D_CAMERA_SIZE; i++) {
      sum += hessians_cl[(i * LANDMARK_SIZE + threadIdx.x) * num_measurements +
                         n] *
             x[i * num_cameras + camera_index];
    }
  }

  y[threadIdx.x * num_points + point_index] =
      alpha * sum + beta * y[threadIdx.x * num_points + point_index];
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointRightProductKernel(
    const int_t *measurement_dicts, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, T alpha, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  int_t measurement_index = measurement_dicts[idx];

  T *x_shared = sfm::utils::internal::SharedMemory<T>::get();

  int_t point_index = point_indices[point_infos[measurement_index]];

  for (int_t i = threadIdx.y; i < 3; i += blockDim.y) {
    x_shared[threadIdx.y * blockDim.x + threadIdx.x] =
        (x + threadIdx.y * num_points)[point_index];
  }
  __syncthreads();

  T sum = 0;
  for (int_t i = 0; i < LANDMARK_SIZE; i++) {
    sum += hessians_cl[(threadIdx.y * LANDMARK_SIZE + i) * num_measurements +
                       idx] *
           x_shared[i * blockDim.x + threadIdx.x];
  }

  y[threadIdx.y * num_measurements + idx] = alpha * sum;
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointRightSumKernel(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, T alpha, const T *x, const int_t *camera_infos,
    T beta, T *y, int_t num_cameras, int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId / D_CAMERA_SIZE;

  if (idx >= N)
    return;

  int_t row = blockId % D_CAMERA_SIZE;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  int_t start = measurement_offsets[idx];
  int_t end = measurement_offsets[idx + 1];
  int_t camera_index = camera_indices[camera_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    sum += x[row * num_measurements + n];
  }

  __syncwarp();

  T ret = WarpReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    y[row * num_cameras + camera_index] =
        alpha * ret + beta * y[row * num_cameras + camera_index];
  }
}

template <typename T>
__global__ void ComputeBlockSparseHessianCameraPointRightMultiplicationKernel(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId / D_CAMERA_SIZE;

  if (idx >= N)
    return;

  int_t row = blockId % D_CAMERA_SIZE;

  typedef cub::WarpReduce<T> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  int_t start = measurement_index_offsets[idx];
  int_t end = measurement_index_offsets[idx + 1];
  int_t camera_index = camera_indices[camera_infos[measurement_dicts[start]]];

  T sum = 0;
  for (int_t n = start + threadIdx.x; n < end; n += 32) {
    int_t measurement_index = measurement_dicts[n];
    int_t point_index = point_indices[point_infos[measurement_index]];

    for (int_t i = 0; i < LANDMARK_SIZE; i++) {
      sum += hessians_cl[(row * LANDMARK_SIZE + i) * num_measurements + n] *
             x[i * num_points + point_index];
    }
  }

  __syncwarp();

  T ret = WarpReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    y[row * num_cameras + camera_index] =
        alpha * ret + beta * y[row * num_cameras + camera_index];
  }
}

template <typename T>
__global__ void ComputeHessianPointPointInverseKernel(const T *hessians_ll,
                                                      T *hessians_ll_inverse,
                                                      int_t num_points) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= num_points)
    return;

  const T H00 = hessians_ll[(0 * LANDMARK_SIZE + 0) * num_points + idx];
  const T H01 = hessians_ll[(0 * LANDMARK_SIZE + 1) * num_points + idx];
  const T H02 = hessians_ll[(0 * LANDMARK_SIZE + 2) * num_points + idx];
  const T H11 = hessians_ll[(1 * LANDMARK_SIZE + 1) * num_points + idx];
  const T H12 = hessians_ll[(1 * LANDMARK_SIZE + 2) * num_points + idx];
  const T H22 = hessians_ll[(2 * LANDMARK_SIZE + 2) * num_points + idx];

  T S00 = H11 * H22 - H12 * H12;
  T S01 = H02 * H12 - H01 * H22;
  T S02 = H01 * H12 - H02 * H11;
  T det = H00 * S00 + H01 * S01 + H02 * S02;
  S00 /= det;
  S01 /= det;
  S02 /= det;

  T S11 = (H00 * H22 - H02 * H02) / det;
  T S12 = (H01 * H02 - H00 * H12) / det;
  T S22 = (H00 * H11 - H01 * H01) / det;

  hessians_ll_inverse[(0 * LANDMARK_SIZE + 0) * num_points + idx] = S00;
  hessians_ll_inverse[(0 * LANDMARK_SIZE + 1) * num_points + idx] = S01;
  hessians_ll_inverse[(0 * LANDMARK_SIZE + 2) * num_points + idx] = S02;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 0) * num_points + idx] = S01;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 1) * num_points + idx] = S11;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 2) * num_points + idx] = S12;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 0) * num_points + idx] = S02;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 1) * num_points + idx] = S12;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 2) * num_points + idx] = S22;
}

template <typename T>
__global__ void ComputeHessianPointPointInverseKernel(
    const int_t *point_indices, const T *hessians_ll, T *hessians_ll_inverse,
    int_t num_points, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t point_index = point_indices[idx];

  const T H00 = hessians_ll[(0 * LANDMARK_SIZE + 0) * num_points + point_index];
  const T H01 = hessians_ll[(0 * LANDMARK_SIZE + 1) * num_points + point_index];
  const T H02 = hessians_ll[(0 * LANDMARK_SIZE + 2) * num_points + point_index];
  const T H11 = hessians_ll[(1 * LANDMARK_SIZE + 1) * num_points + point_index];
  const T H12 = hessians_ll[(1 * LANDMARK_SIZE + 2) * num_points + point_index];
  const T H22 = hessians_ll[(2 * LANDMARK_SIZE + 2) * num_points + point_index];

  T S00 = H11 * H22 - H12 * H12;
  T S01 = H02 * H12 - H01 * H22;
  T S02 = H01 * H12 - H02 * H11;
  T det = H00 * S00 + H01 * S01 + H02 * S02;
  S00 /= det;
  S01 /= det;
  S02 /= det;

  T S11 = (H00 * H22 - H02 * H02) / det;
  T S12 = (H01 * H02 - H00 * H12) / det;
  T S22 = (H00 * H11 - H01 * H01) / det;

  hessians_ll_inverse[(0 * LANDMARK_SIZE + 0) * num_points + point_index] = S00;
  hessians_ll_inverse[(0 * LANDMARK_SIZE + 1) * num_points + point_index] = S01;
  hessians_ll_inverse[(0 * LANDMARK_SIZE + 2) * num_points + point_index] = S02;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 0) * num_points + point_index] = S01;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 1) * num_points + point_index] = S11;
  hessians_ll_inverse[(1 * LANDMARK_SIZE + 2) * num_points + point_index] = S12;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 0) * num_points + point_index] = S02;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 1) * num_points + point_index] = S12;
  hessians_ll_inverse[(2 * LANDMARK_SIZE + 2) * num_points + point_index] = S22;
}
} // namespace ba
} // namespace sfm
