// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/ba/types.h>
#include <sfm/types.h>
#include <sfm/utils/robust_loss.h>

namespace sfm {
namespace ba {
template <Memory kMemory, typename T>
void EvaluateReprojectionLossFunction(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements);

template <Memory kMemory, typename T>
void EvaluateReprojectionLossFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void EvaluateReprojectionLossFunction(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void LinearizeReprojectionLossFunction(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements);

template <Memory kMemory, typename T>
void LinearizeReprojectionLossFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void LinearizeReprojectionLossFunction(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void EvaluateCameraSurrogateFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void EvaluateCameraSurrogateFunction(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void LinearizeCameraSurrogateFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void LinearizeCameraSurrogateFunction(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void EvaluatePointSurrogateFunction(const int_t *measurement_indices,
                                    const T *points, const int_t *point_infos,
                                    const T *rescaled_a_vals,
                                    const T *rescaled_g_vecs,
                                    const T *rescaled_constants, T *fobjs,
                                    int_t num_points, int_t N);

template <Memory kMemory, typename T>
void EvaluatePointSurrogateFunction(const int_t *measurement_indices,
                                    const int_t *point_indices, const T *points,
                                    const int_t *point_infos,
                                    const T *rescaled_a_vals,
                                    const T *rescaled_g_vecs,
                                    const T *rescaled_constants, T *fobjs,
                                    int_t num_points, int_t N);

template <Memory kMemory, typename T>
void LinearizePointSurrogateFunction(const int_t *measurement_indices,
                                     const T *points, const int_t *point_infos,
                                     const T *rescaled_a_vals,
                                     const T *rescaled_g_vecs,
                                     const T *rescaled_constants,
                                     T *jacobians_point, T *rescaled_errors,
                                     int_t num_points, int_t N);

template <Memory kMemory, typename T>
void LinearizePointSurrogateFunction(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *jacobians_point,
    T *rescaled_errors, int_t num_points, int_t N);

template <Memory kMemory, typename T>
void MajorizeReprojectionLossFunction(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements);

template <Memory kMemory, typename T>
void MajorizeReprojectionLossFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructSurrogateFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_vecs, T *rescaled_a_vals,
    T *rescaled_g_vecs, T *rescaled_sqrt_weights, T *rescaled_constants,
    T *fobjs, RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructSurrogateFunction(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_a_vals, T *rescaled_g_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructExtrinsicsAndPointProximalOperator(
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    T *extrinsics_hess_grad, T *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements);

template <Memory kMemory, typename T>
void ConstructExtrinsicsAndPointProximalOperator(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructExtrinsicsAndPointProximalOperator(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructExtrinsicsProximalOperator(const T *rescaled_h_a_vecs,
                                         const T *rescaled_a_g_vecs,
                                         const int_t *extrinsics_infos,
                                         T *extrinsics_hess_grad,
                                         int_t num_extrinsics,
                                         int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructExtrinsicsProximalOperator(const int_t *measurement_indices,
                                         const T *rescaled_h_a_vecs,
                                         const T *rescaled_a_g_vecs,
                                         const int_t *extrinsics_infos,
                                         T *extrinsics_hess_grad,
                                         int_t num_extrinsics,
                                         int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructExtrinsicsProximalOperator(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructIntrinsicsProximalOperator(const T *rescaled_f_s_vecs,
                                         const int_t *intrinsics_infos,
                                         T *intrinsics_hess_grad,
                                         int_t num_intrinsics,
                                         int_t num_measurements);

template <Memory kMemory, typename T>
void ConstructIntrinsicsProximalOperator(const int_t *measurement_indices,
                                         const int_t *intrinsics_indices,
                                         const T *rescaled_f_s_vecs,
                                         const int_t *intrinsics_infos,
                                         T *intrinsics_hess_grad,
                                         int_t num_intrinsics,
                                         int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructIntrinsicsProximalOperator(const int_t *measurement_indices,
                                         const T *rescaled_f_s_vecs,
                                         const int_t *intrinsics_infos,
                                         T *intrinsics_hess_grad,
                                         int_t num_intrinsics,
                                         int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructPointProximalOperator(const T *rescaled_a_g_vecs,
                                    const int_t *point_infos,
                                    T *points_hess_grad, int_t num_points,
                                    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructPointProximalOperator(const int_t *measurement_indices,
                                    const T *rescaled_a_g_vecs,
                                    const int_t *point_infos,
                                    T *points_hess_grad, int_t num_points,
                                    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void ConstructPointProximalOperator(const int_t *measurement_indices,
                                    const int_t *point_indices,
                                    const T *rescaled_a_g_vecs,
                                    const int_t *point_infos,
                                    T *points_hess_grad, int_t num_points,
                                    int_t num_measurements, int_t N);

template <Memory kMemory, typename T>
void SolveExtrinsicsProximalOperator(const T *data, T reg,
                                     const T *init_extrinsics, T *extrinsics,
                                     int_t num_extrinsics);

template <Memory kMemory, typename T>
void SolveExtrinsicsProximalOperator(const int_t *extrinsics_indices,
                                     const T *data, T reg,
                                     const T *init_extrinsics, T *extrinsics,
                                     int_t num_extrinsics, int_t N);

template <Memory kMemory, typename T>
void SolveIntrinsicsProximalOperator(const T *data, T reg,
                                     const T *init_intrinsics, T *intrinsics,
                                     int_t num_intrinsics);

template <Memory kMemory, typename T>
void SolveIntrinsicsProximalOperator(const int_t *intrinsics_indices,
                                     const T *data, T reg,
                                     const T *init_intrinsics, T *intrinsics,
                                     int_t num_intrinsics, int_t N);

template <Memory kMemory, typename T>
void SolvePointProximalOperator(const T *data, T reg, const T *init_points,
                                T *points, int_t num_points);

template <Memory kMemory, typename T>
void SolvePointProximalOperator(const int_t *point_indices, const T *data,
                                T reg, const T *init_points, T *points,
                                int_t num_points, int_t N);

template <Memory kMemory, typename T>
void ComputeReprojectionLossFunctionHessianGradient(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, bool reset,
    T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeReprojectionLossFunctionHessianGradient(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeReprojectionLossFunctionHessianGradient(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeCameraSurrogateFunctionHessianGradient(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N,
    bool reset, T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeCameraSurrogateFunctionHessianGradient(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset,
    T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputePointSurrogateFunctionHessianGradient(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    bool reset, T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputePointSurrogateFunctionHessianGradient(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeHessianGradient(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeHessianGradient(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    T alpha = 1.0, T beta = 0.0);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointLeftMultiplication(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointLeftMultiplication(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointLeftMultiplication(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointRightMultiplication(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointRightMultiplication(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianCameraPointRightMultiplication(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset = true);

template <Memory kMemory, typename T>
void ComputeHessianPointPointInverse(const T *hessians_ll,
                                     T *hessians_ll_inverse, int_t num_points);

template <Memory kMemory, typename T>
void ComputeHessianPointPointInverse(const int_t *point_indices,
                                     const T *hessians_ll,
                                     T *hessians_ll_inverse, int_t num_points,
                                     int_t N);
} // namespace ba
} // namespace sfm
