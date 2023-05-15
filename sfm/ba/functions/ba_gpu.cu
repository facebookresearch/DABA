// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sfm/ba/functions/ba.h>
#include <sfm/ba/functions/ba_async.cuh>

namespace sfm {
namespace ba {
template <>
void EvaluateReprojectionLossFunction<kGPU, float>(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements) {
  EvaluateReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, fobjs, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateReprojectionLossFunction<kGPU, double>(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements) {
  EvaluateReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, fobjs, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateReprojectionLossFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateReprojectionLossFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateReprojectionLossFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N) {
  EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateReprojectionLossFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N) {
  EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, float>(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *jacobians_extrinsics_intrinsics,
    float *rescaled_errors, RobustLoss robust_loss, float loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  LinearizeReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, double>(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *jacobians_extrinsics_intrinsics,
    double *rescaled_errors, RobustLoss robust_loss, double loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements) {
  LinearizeReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  LinearizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  LinearizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *jacobians_extrinsics_intrinsics,
    float *rescaled_errors, RobustLoss robust_loss, float loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  LinearizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeReprojectionLossFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *jacobians_extrinsics_intrinsics,
    double *rescaled_errors, RobustLoss robust_loss, double loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  LinearizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateCameraSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  EvaluateCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants, fobjs,
      num_extrinsics, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateCameraSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N) {
  EvaluateCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants, fobjs,
      num_extrinsics, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateCameraSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  EvaluateCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, fobjs, num_extrinsics, num_intrinsics,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluateCameraSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N) {
  EvaluateCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, fobjs, num_extrinsics, num_intrinsics,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeCameraSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  LinearizeCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants,
      jacobians_extrinsics_intrinsics, rescaled_errors, num_extrinsics,
      num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeCameraSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  LinearizeCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants,
      jacobians_extrinsics_intrinsics, rescaled_errors, num_extrinsics,
      num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeCameraSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  LinearizeCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, jacobians_extrinsics_intrinsics, rescaled_errors,
      num_extrinsics, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizeCameraSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  LinearizeCameraSurrogateFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, jacobians_extrinsics_intrinsics, rescaled_errors,
      num_extrinsics, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluatePointSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *points,
    const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_points, int_t N) {
  EvaluatePointSurrogateFunctionAsync(measurement_indices, points, point_infos,
                                      rescaled_a_vals, rescaled_g_vecs,
                                      rescaled_constants, fobjs, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluatePointSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *points,
    const int_t *point_infos, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_points, int_t N) {
  EvaluatePointSurrogateFunctionAsync(measurement_indices, points, point_infos,
                                      rescaled_a_vals, rescaled_g_vecs,
                                      rescaled_constants, fobjs, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluatePointSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *points, const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_points, int_t N) {
  EvaluatePointSurrogateFunctionAsync(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, fobjs, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void EvaluatePointSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *points, const int_t *point_infos,
    const double *rescaled_a_vals, const double *rescaled_g_vecs,
    const double *rescaled_constants, double *fobjs, int_t num_points,
    int_t N) {
  EvaluatePointSurrogateFunctionAsync(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, fobjs, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizePointSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *points,
    const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_point, float *rescaled_errors, int_t num_points, int_t N) {
  LinearizePointSurrogateFunctionAsync(measurement_indices, points, point_infos,
                                       rescaled_a_vals, rescaled_g_vecs,
                                       rescaled_constants, jacobians_point,
                                       rescaled_errors, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizePointSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *points,
    const int_t *point_infos, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_point, double *rescaled_errors, int_t num_points,
    int_t N) {
  LinearizePointSurrogateFunctionAsync(measurement_indices, points, point_infos,
                                       rescaled_a_vals, rescaled_g_vecs,
                                       rescaled_constants, jacobians_point,
                                       rescaled_errors, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizePointSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *points, const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_point, float *rescaled_errors, int_t num_points, int_t N) {
  LinearizePointSurrogateFunctionAsync(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, jacobians_point, rescaled_errors,
      num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void LinearizePointSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *points, const int_t *point_infos,
    const double *rescaled_a_vals, const double *rescaled_g_vecs,
    const double *rescaled_constants, double *jacobians_point,
    double *rescaled_errors, int_t num_points, int_t N) {
  LinearizePointSurrogateFunctionAsync(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, jacobians_point, rescaled_errors,
      num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void MajorizeReprojectionLossFunction<kGPU, float>(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *rescaled_h_a_g_vecs,
    float *rescaled_f_s_vecs, float *rescaled_sqrt_weights,
    float *rescaled_constants, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements) {
  MajorizeReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, rescaled_h_a_g_vecs,
      rescaled_f_s_vecs, rescaled_sqrt_weights, rescaled_constants, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void MajorizeReprojectionLossFunction<kGPU, double>(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *rescaled_h_a_g_vecs,
    double *rescaled_f_s_vecs, double *rescaled_sqrt_weights,
    double *rescaled_constants, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements) {
  MajorizeReprojectionLossFunctionAsync(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, rescaled_h_a_g_vecs,
      rescaled_f_s_vecs, rescaled_sqrt_weights, rescaled_constants, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void MajorizeReprojectionLossFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *rescaled_h_a_g_vecs, float *rescaled_f_s_vecs,
    float *rescaled_sqrt_weights, float *rescaled_constants, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  MajorizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_a_g_vecs, rescaled_f_s_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void MajorizeReprojectionLossFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_h_a_g_vecs, double *rescaled_f_s_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  MajorizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_a_g_vecs, rescaled_f_s_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *rescaled_h_vecs,
    float *rescaled_a_vals, float *rescaled_g_vecs,
    float *rescaled_sqrt_weights, float *rescaled_constants, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  ConstructSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_vecs, rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_h_vecs, double *rescaled_a_vals, double *rescaled_g_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  ConstructSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_vecs, rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructSurrogateFunction<kGPU, float>(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *rescaled_a_vals,
    float *rescaled_g_vecs, float *rescaled_sqrt_weights,
    float *rescaled_constants, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N) {
  ConstructSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructSurrogateFunction<kGPU, double>(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_a_vals, double *rescaled_g_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N) {
  ConstructSurrogateFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, float>(
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    float *extrinsics_hess_grad, float *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos, point_infos,
      extrinsics_hess_grad, points_hess_grad, num_extrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, double>(
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    double *extrinsics_hess_grad, double *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos, point_infos,
      extrinsics_hess_grad, points_hess_grad, num_extrinsics, num_points,
      num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, float *extrinsics_hess_grad,
    float *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, point_infos, extrinsics_hess_grad, points_hess_grad,
      num_extrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, double *extrinsics_hess_grad,
    double *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, point_infos, extrinsics_hess_grad, points_hess_grad,
      num_extrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, float *extrinsics_hess_grad,
    float *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      measurement_indices, extrinsics_indices, point_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, point_infos, extrinsics_hess_grad,
      points_hess_grad, num_extrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsAndPointProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, double *extrinsics_hess_grad,
    double *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructExtrinsicsAndPointProximalOperatorAsync(
      measurement_indices, extrinsics_indices, point_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, point_infos, extrinsics_hess_grad,
      points_hess_grad, num_extrinsics, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, float>(
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, float *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos,
      extrinsics_hess_grad, num_extrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, double>(
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, double *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos,
      extrinsics_hess_grad, num_extrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    float *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, extrinsics_hess_grad, num_extrinsics, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    double *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, extrinsics_hess_grad, num_extrinsics, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, float *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      measurement_indices, extrinsics_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, extrinsics_hess_grad, num_extrinsics,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructExtrinsicsProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, double *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N) {
  ConstructExtrinsicsProximalOperatorAsync(
      measurement_indices, extrinsics_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, extrinsics_hess_grad, num_extrinsics,
      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, float>(
    const float *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    float *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements) {
  ConstructIntrinsicsProximalOperatorAsync(rescaled_f_s_vecs, intrinsics_infos,
                                           intrinsics_hess_grad, num_intrinsics,
                                           num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, double>(
    const double *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    double *intrinsics_hess_grad, int_t num_intrinsics,
    int_t num_measurements) {
  ConstructIntrinsicsProximalOperatorAsync(rescaled_f_s_vecs, intrinsics_infos,
                                           intrinsics_hess_grad, num_intrinsics,
                                           num_measurements);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const float *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    float *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  ConstructIntrinsicsProximalOperatorAsync(
      measurement_indices, intrinsics_indices, rescaled_f_s_vecs,
      intrinsics_infos, intrinsics_hess_grad, num_intrinsics, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const double *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    double *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N) {
  ConstructIntrinsicsProximalOperatorAsync(
      measurement_indices, intrinsics_indices, rescaled_f_s_vecs,
      intrinsics_infos, intrinsics_hess_grad, num_intrinsics, num_measurements,
      N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const float *rescaled_f_s_vecs,
    const int_t *intrinsics_infos, float *intrinsics_hess_grad,
    int_t num_intrinsics, int_t num_measurements, int_t N) {
  ConstructIntrinsicsProximalOperatorAsync(
      measurement_indices, rescaled_f_s_vecs, intrinsics_infos,
      intrinsics_hess_grad, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructIntrinsicsProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const double *rescaled_f_s_vecs,
    const int_t *intrinsics_infos, double *intrinsics_hess_grad,
    int_t num_intrinsics, int_t num_measurements, int_t N) {
  ConstructIntrinsicsProximalOperatorAsync(
      measurement_indices, rescaled_f_s_vecs, intrinsics_infos,
      intrinsics_hess_grad, num_intrinsics, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, float>(const float *rescaled_a_g_vecs,
                                                 const int_t *point_infos,
                                                 float *points_hess_grad,
                                                 int_t num_points,
                                                 int_t num_measurements,
                                                 int_t N) {
  ConstructPointProximalOperatorAsync(rescaled_a_g_vecs, point_infos,
                                      points_hess_grad, num_points,
                                      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, double>(
    const double *rescaled_a_g_vecs, const int_t *point_infos,
    double *points_hess_grad, int_t num_points, int_t num_measurements,
    int_t N) {
  ConstructPointProximalOperatorAsync(rescaled_a_g_vecs, point_infos,
                                      points_hess_grad, num_points,
                                      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const float *rescaled_a_g_vecs,
    const int_t *point_infos, float *points_hess_grad, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructPointProximalOperatorAsync(measurement_indices, rescaled_a_g_vecs,
                                      point_infos, points_hess_grad, num_points,
                                      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const double *rescaled_a_g_vecs,
    const int_t *point_infos, double *points_hess_grad, int_t num_points,
    int_t num_measurements, int_t N) {
  ConstructPointProximalOperatorAsync(measurement_indices, rescaled_a_g_vecs,
                                      point_infos, points_hess_grad, num_points,
                                      num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, float>(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *rescaled_a_g_vecs, const int_t *point_infos,
    float *points_hess_grad, int_t num_points, int_t num_measurements,
    int_t N) {
  ConstructPointProximalOperatorAsync(
      measurement_indices, point_indices, rescaled_a_g_vecs, point_infos,
      points_hess_grad, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ConstructPointProximalOperator<kGPU, double>(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *rescaled_a_g_vecs, const int_t *point_infos,
    double *points_hess_grad, int_t num_points, int_t num_measurements,
    int_t N) {
  ConstructPointProximalOperatorAsync(
      measurement_indices, point_indices, rescaled_a_g_vecs, point_infos,
      points_hess_grad, num_points, num_measurements, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveExtrinsicsProximalOperator<kGPU, float>(const float *data, float reg,
                                                  const float *init_extrinsics,
                                                  float *extrinsics,
                                                  int_t num_extrinsics) {
  SolveExtrinsicsProximalOperatorAsync(data, reg, init_extrinsics, extrinsics,
                                       num_extrinsics);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveExtrinsicsProximalOperator<kGPU, double>(
    const double *data, double reg, const double *init_extrinsics,
    double *extrinsics, int_t num_extrinsics) {
  SolveExtrinsicsProximalOperatorAsync(data, reg, init_extrinsics, extrinsics,
                                       num_extrinsics);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveExtrinsicsProximalOperator<kGPU, float>(
    const int_t *extrinsics_indices, const float *data, float reg,
    const float *init_extrinsics, float *extrinsics, int_t num_extrinsics,
    int_t N) {
  SolveExtrinsicsProximalOperatorAsync(extrinsics_indices, data, reg,
                                       init_extrinsics, extrinsics,
                                       num_extrinsics, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveExtrinsicsProximalOperator<kGPU, double>(
    const int_t *extrinsics_indices, const double *data, double reg,
    const double *init_extrinsics, double *extrinsics, int_t num_extrinsics,
    int_t N) {
  SolveExtrinsicsProximalOperatorAsync(extrinsics_indices, data, reg,
                                       init_extrinsics, extrinsics,
                                       num_extrinsics, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveIntrinsicsProximalOperator<kGPU, float>(const float *data, float reg,
                                                  const float *init_intrinsics,
                                                  float *intrinsics,
                                                  int_t num_intrinsics) {
  SolveIntrinsicsProximalOperatorAsync(data, reg, init_intrinsics, intrinsics,
                                       num_intrinsics);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveIntrinsicsProximalOperator<kGPU, double>(
    const double *data, double reg, const double *init_intrinsics,
    double *intrinsics, int_t num_intrinsics) {
  SolveIntrinsicsProximalOperatorAsync(data, reg, init_intrinsics, intrinsics,
                                       num_intrinsics);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveIntrinsicsProximalOperator<kGPU, float>(
    const int_t *intrinsics_indices, const float *data, float reg,
    const float *init_intrinsics, float *intrinsics, int_t num_intrinsics,
    int_t N) {
  SolveIntrinsicsProximalOperatorAsync(intrinsics_indices, data, reg,
                                       init_intrinsics, intrinsics,
                                       num_intrinsics, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolveIntrinsicsProximalOperator<kGPU, double>(
    const int_t *intrinsics_indices, const double *data, double reg,
    const double *init_intrinsics, double *intrinsics, int_t num_intrinsics,
    int_t N) {
  SolveIntrinsicsProximalOperatorAsync(intrinsics_indices, data, reg,
                                       init_intrinsics, intrinsics,
                                       num_intrinsics, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolvePointProximalOperator<kGPU, float>(const float *data, float reg,
                                             const float *init_points,
                                             float *points, int_t num_points) {
  SolvePointProximalOperatorAsync(data, reg, init_points, points, num_points);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolvePointProximalOperator<kGPU, double>(const double *data, double reg,
                                              const double *init_points,
                                              double *points,
                                              int_t num_points) {
  SolvePointProximalOperatorAsync(data, reg, init_points, points, num_points);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolvePointProximalOperator<kGPU, float>(const int_t *point_indices,
                                             const float *data, float reg,
                                             const float *init_points,
                                             float *points, int_t num_points,
                                             int_t N) {
  SolvePointProximalOperatorAsync(point_indices, data, reg, init_points, points,
                                  num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void SolvePointProximalOperator<kGPU, double>(const int_t *point_indices,
                                              const double *data, double reg,
                                              const double *init_points,
                                              double *points, int_t num_points,
                                              int_t N) {
  SolvePointProximalOperatorAsync(point_indices, data, reg, init_points, points,
                                  num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, float>(
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset, float alpha, float beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, double>(
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, bool reset,
    double alpha, double beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, float alpha, float beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, point_infos, hessians_cc, hessians_cl, hessians_ll,
      gradients_c, gradients_l, num_cameras, num_points, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, double alpha, double beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, point_infos, hessians_cc, hessians_cl, hessians_ll,
      gradients_c, gradients_l, num_cameras, num_points, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *jacobians_extrinsics_intrinsics,
    const float *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, float *hessians_cc, float *hessians_cl,
    float *hessians_ll, float *gradients_c, float *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, float alpha, float beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, camera_indices, point_indices,
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, N, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeReprojectionLossFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, double alpha, double beta) {
  ComputeReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, camera_indices, point_indices,
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, N, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeCameraSurrogateFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta) {
  ComputeCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, hessians_cc, gradients_c, num_cameras, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeCameraSurrogateFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta) {
  ComputeCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, hessians_cc, gradients_c, num_cameras, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeCameraSurrogateFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta) {
  ComputeCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, camera_indices, jacobians_extrinsics_intrinsics,
      rescaled_errors, camera_infos, hessians_cc, gradients_c, num_cameras,
      num_measurements, N, reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeCameraSurrogateFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta) {
  ComputeCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, camera_indices, jacobians_extrinsics_intrinsics,
      rescaled_errors, camera_infos, hessians_cc, gradients_c, num_cameras,
      num_measurements, N, reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePointSurrogateFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices, const float *jacobians_points,
    const float *rescaled_errors, const int_t *point_infos, float *hessians_ll,
    float *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    bool reset, float alpha, float beta) {
  ComputePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_points, rescaled_errors, point_infos,
      hessians_ll, gradients_l, num_points, num_measurements, N, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePointSurrogateFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices, const double *jacobians_points,
    const double *rescaled_errors, const int_t *point_infos,
    double *hessians_ll, double *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta) {
  ComputePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_points, rescaled_errors, point_infos,
      hessians_ll, gradients_l, num_points, num_measurements, N, reset, alpha,
      beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePointSurrogateFunctionHessianGradient<kGPU, float>(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *jacobians_points, const float *rescaled_errors,
    const int_t *point_infos, float *hessians_ll, float *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta) {
  ComputePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, point_indices, jacobians_points, rescaled_errors,
      point_infos, hessians_ll, gradients_l, num_points, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePointSurrogateFunctionHessianGradient<kGPU, double>(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *jacobians_points, const double *rescaled_errors,
    const int_t *point_infos, double *hessians_ll, double *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, bool reset, double alpha,
    double beta) {
  ComputePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, point_indices, jacobians_points, rescaled_errors,
      point_infos, hessians_ll, gradients_l, num_points, num_measurements, N,
      reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianGradient<kGPU, float>(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const float *, 3> &jacobians,
    const std::array<const float *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    float alpha, float beta) {
  ComputeHessianGradientAsync(measurement_indices, jacobians, rescaled_errors,
                              camera_infos, point_infos, hessians_cc,
                              hessians_cl, hessians_ll, gradients_c,
                              gradients_l, num_cameras, num_points,
                              num_measurements, N, reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianGradient<kGPU, double>(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const double *, 3> &jacobians,
    const std::array<const double *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, double *hessians_cc,
    double *hessians_cl, double *hessians_ll, double *gradients_c,
    double *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    double alpha, double beta) {
  ComputeHessianGradientAsync(
      measurement_indices, jacobians, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, N, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianGradient<kGPU, float>(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const float *, 3> &jacobians,
    const std::array<const float *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    float alpha, float beta) {
  ComputeHessianGradientAsync(
      measurement_indices, camera_indices, point_indices, jacobians,
      rescaled_errors, camera_infos, point_infos, hessians_cc, hessians_cl,
      hessians_ll, gradients_c, gradients_l, num_cameras, num_points,
      num_measurements, N, reset, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianGradient<kGPU, double>(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const double *, 3> &jacobians,
    const std::array<const double *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, double *hessians_cc,
    double *hessians_cl, double *hessians_ll, double *gradients_c,
    double *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    double alpha, double beta) {
  ComputeHessianGradientAsync(
      measurement_indices, camera_indices, point_indices, jacobians,
      rescaled_errors, camera_infos, point_infos, hessians_cc, hessians_cl,
      hessians_ll, gradients_c, gradients_l, num_cameras, num_points,
      num_measurements, N, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, float>(
    const float *hessians_cl, const float *x, const int_t *camera_infos,
    const int_t *point_infos, float beta, float *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
      num_points, num_measurements, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, double>(
    const double *hessians_cl, const double *x, const int_t *camera_infos,
    const int_t *point_infos, double beta, double *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
      num_points, num_measurements, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, float>(
    const int_t *measurement_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices, hessians_cl, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, double>(
    const int_t *measurement_indices, const double *hessians_cl,
    const double *x, const int_t *camera_infos, const int_t *point_infos,
    double beta, double *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices, hessians_cl, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, float>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hessians_cl, x,
      camera_infos, point_infos, beta, y, num_cameras, num_points,
      num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointLeftMultiplication<kGPU, double>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *hessians_cl, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hessians_cl, x,
      camera_infos, point_infos, beta, y, num_cameras, num_points,
      num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, float>(
    const float *hessians_cl, const float *x, const int_t *camera_infos,
    const int_t *point_infos, float beta, float *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
      num_points, num_measurements, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, double>(
    const double *hessians_cl, const double *x, const int_t *camera_infos,
    const int_t *point_infos, double beta, double *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
      num_points, num_measurements, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, float>(
    const int_t *measurement_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices, hessians_cl, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, double>(
    const int_t *measurement_indices, const double *hessians_cl,
    const double *x, const int_t *camera_infos, const int_t *point_infos,
    double beta, double *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices, hessians_cl, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, float>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hessians_cl, x,
      camera_infos, point_infos, beta, y, num_cameras, num_points,
      num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianCameraPointRightMultiplication<kGPU, double>(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *hessians_cl, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset) {
  ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hessians_cl, x,
      camera_infos, point_infos, beta, y, num_cameras, num_points,
      num_measurements, N, reset);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianPointPointInverse<kGPU, float>(const float *hessians_ll,
                                                  float *hessians_ll_inverse,
                                                  int_t num_points) {
  ComputeHessianPointPointInverseAsync(hessians_ll, hessians_ll_inverse,
                                       num_points);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianPointPointInverse<kGPU, double>(const double *hessians_ll,
                                                   double *hessians_ll_inverse,
                                                   int_t num_points) {
  ComputeHessianPointPointInverseAsync(hessians_ll, hessians_ll_inverse,
                                       num_points);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianPointPointInverse<kGPU, float>(const int_t *point_indices,
                                                  const float *hessians_ll,
                                                  float *hessians_ll_inverse,
                                                  int_t num_points, int_t N) {
  ComputeHessianPointPointInverseAsync(point_indices, hessians_ll,
                                       hessians_ll_inverse, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeHessianPointPointInverse<kGPU, double>(const int_t *point_indices,
                                                   const double *hessians_ll,
                                                   double *hessians_ll_inverse,
                                                   int_t num_points, int_t N) {
  ComputeHessianPointPointInverseAsync(point_indices, hessians_ll,
                                       hessians_ll_inverse, num_points, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}
} // namespace ba
} // namespace sfm
