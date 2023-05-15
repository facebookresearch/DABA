// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/ba/types.h>
#include <sfm/types.h>
#include <sfm/utils/cuda_utils.h>
#include <sfm/utils/robust_loss.h>

#include <cuda_runtime.h>

namespace sfm {
namespace ba {
template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream = 0);

template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizeReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream = 0);

template <typename T>
void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_points, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *fobjs,
    int_t num_points, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_point, T *rescaled_errors,
    int_t num_points, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *jacobians_point,
    T *rescaled_errors, int_t num_points, int_t N, cudaStream_t stream = 0);

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream = 0);

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream = 0);

template <typename T>
void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights,
    T *jacobians_extrinsics_intrinsics, T *rescaled_errors,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void MajorizeReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream = 0);

template <typename T>
void MajorizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_vecs, T *rescaled_a_vals,
    T *rescaled_g_vecs, T *rescaled_sqrt_weights, T *rescaled_constants,
    T *fobjs, RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_a_vals, T *rescaled_g_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    T *extrinsics_hess_grad, T *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(const T *rescaled_h_a_vecs,
                                              const T *rescaled_a_g_vecs,
                                              const int_t *extrinsics_infos,
                                              T *extrinsics_hess_grad,
                                              int_t num_extrinsics,
                                              int_t num_measurements, int_t N,
                                              cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream = 0);

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(const T *rescaled_f_s_vecs,
                                              const int_t *intrinsics_infos,
                                              T *intrinsics_hess_grad,
                                              int_t num_intrinsics,
                                              int_t num_measurements,
                                              cudaStream_t stream = 0);

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(const int_t *measurement_indices,
                                              const T *rescaled_f_s_vecs,
                                              const int_t *intrinsics_infos,
                                              T *intrinsics_hess_grad,
                                              int_t num_intrinsics,
                                              int_t num_measurements, int_t N,
                                              cudaStream_t stream = 0);

template <typename T>
void ConstructPointProximalOperatorAsync(const T *rescaled_a_g_vecs,
                                         const int_t *point_infos,
                                         T *points_hess_grad, int_t num_points,
                                         int_t num_measurements, int_t N,
                                         cudaStream_t stream = 0);

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const T *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    T *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream = 0);

template <typename T>
void ConstructPointProximalOperatorAsync(const int_t *measurement_indices,
                                         const T *rescaled_a_g_vecs,
                                         const int_t *point_infos,
                                         T *points_hess_grad, int_t num_points,
                                         int_t num_measurements, int_t N,
                                         cudaStream_t stream = 0);

template <typename T>
void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *rescaled_a_g_vecs, const int_t *point_infos, T *points_hess_grad,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeExtrinsicsAndPointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const int_t *measurement_indices_by_points, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n,
    T *points_hess_grad_n, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeExtrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ComputeExtrinsicsProximalOperatorAsync(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *extrinsics_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeIntrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_intrinsics, const T *rescaled_f_s_vecs,
    T *intrinsics_hess_grad_n, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputePointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_points, const T *rescaled_a_g_vecs,
    T *points_hess_grad_n, int_t N, cudaStream_t stream = 0);

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const T *init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          cudaStream_t stream = 0);

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const int_t *extrinsics_indices,
                                          const T *data, T reg,
                                          const T *init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          int_t N, cudaStream_t stream = 0);

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const int_t *init_extrinsics_dicts,
                                          const T *init_extrinsics,
                                          int_t num_init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          int_t N, cudaStream_t stream = 0);

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const T *init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          cudaStream_t stream = 0);

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const int_t *intrinsics_indices,
                                          const T *data, T reg,
                                          const T *init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          int_t N, cudaStream_t stream = 0);

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const int_t *init_intrinsics_dicts,
                                          const T *init_intrinsics,
                                          int_t num_init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          int_t N, cudaStream_t stream = 0);

template <typename T>
void SolvePointProximalOperatorAsync(const T *data, T reg, const T *init_points,
                                     T *points, int_t num_points,
                                     cudaStream_t stream = 0);

template <typename T>
void SolvePointProximalOperatorAsync(const int_t *point_indices, const T *data,
                                     T reg, const T *init_points, T *points,
                                     int_t num_points, int_t N,
                                     cudaStream_t stream = 0);

template <typename T>
void SolvePointProximalOperatorAsync(const T *data, T reg,
                                     const int_t *init_point_dicts,
                                     const T *init_points,
                                     int_t num_init_points, T *points,
                                     int_t num_points, int_t N,
                                     cudaStream_t stream = 0);

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements,
    cudaStream_t stream = 0);

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, bool reset,
    T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0,
    cudaStream_t stream = 0);

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0,
    cudaStream_t stream = 0);

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N,
    bool reset, T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset,
    T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    bool reset, T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0,
    cudaStream_t stream = 0);

template <typename T>
void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    T alpha = 1.0, T beta = 0.0, cudaStream_t stream = 0);

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const int_t *measurement_indices_by_points,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *hessians_cl_n, T *hessians_ll_n, T *gradients_c_n,
    T *gradients_l_n, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *gradients_c_n, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputePointSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_points, const T *jacobians_points,
    const T *rescaled_errors, T *hessians_ll_n, T *gradients_l_n, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void UpdateHessianSumForCameraAsync(const int_t *measurement_dicts,
                                    const int_t *measurement_offsets,
                                    const int_t *camera_indices,
                                    const T *hess_cc_n,
                                    const int_t *camera_infos, T *hess_cc,
                                    int_t num_cameras, int_t num_measurements,
                                    int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeCameraDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, T alpha, const T *x, const int_t *camera_infos,
    T beta, T *y, int_t num_cameras, int_t num_measurements,
    int_t reduction_size, int_t N, cudaStream_t stream = 0);

template <typename T>
void UpdateHessianSumForPointAsync(const int_t *measurement_dicts,
                                   const int_t *measurement_offsets,
                                   const int_t *point_indices,
                                   const T *hess_ll_n, const int_t *point_infos,
                                   T *hess_ll, int_t num_points,
                                   int_t num_measurements, int_t N,
                                   cudaStream_t stream = 0);

template <typename T>
void ComputePointDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, T alpha, const T *x, const int_t *point_infos,
    T beta, T *y, int_t num_points, int_t num_measurements,
    int_t reduction_size, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras,
    const int_t *measurement_indices_by_points, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll_n,
    T *gradients_c, T *gradients_l_n, int_t num_cameras, int_t num_measurements,
    int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N,
    cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true, cudaStream_t stream = 0);

template <typename T>
void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, T *buffer, int_t num_cameras,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, T *buffer, int_t num_cameras,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputeHessianPointPointInverseAsync(const T *hessians_ll,
                                          T *hessians_ll_inverse,
                                          int_t num_points,
                                          cudaStream_t stream = 0);

template <typename T>
void ComputeHessianPointPointInverseAsync(const int_t *point_indices,
                                          const T *hessians_ll,
                                          T *hessians_ll_inverse,
                                          int_t num_points, int_t N,
                                          cudaStream_t stream = 0);
} // namespace ba
} // namespace sfm
