// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sfm/ba/functions/ba_async.cuh>
#include <sfm/ba/functions/ba_kernel.cuh>
#include <sfm/ba/macro.h>
#include <sfm/utils/utils.cuh>

namespace sfm {
namespace ba {
template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream) {
  int_t num_threads = num_measurements;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, fobjs, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
}

template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
}

template <typename T>
void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
}

template <typename T>
void LinearizeReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream) {
  int_t num_threads = num_measurements;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
}

template <typename T>
void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
}

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
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
}

template <typename T>
void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateCameraSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants, fobjs,
      num_extrinsics, num_intrinsics, num_measurements, N);
}

template <typename T>
void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateCameraSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, fobjs, num_extrinsics, num_intrinsics,
      num_measurements, N);
}

template <typename T>
void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeCameraSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_constants,
      jacobians_extrinsics_intrinsics, rescaled_errors, num_extrinsics,
      num_intrinsics, num_measurements, N);
}

template <typename T>
void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const T *extrinsics, const T *intrinsics,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const T *rescaled_sqrt_weights,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeCameraSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices, extrinsics,
      intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights, rescaled_a_vals, rescaled_g_vecs,
      rescaled_constants, jacobians_extrinsics_intrinsics, rescaled_errors,
      num_extrinsics, num_intrinsics, num_measurements, N);
}

template <typename T>
void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *fobjs, int_t num_points, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluatePointSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, fobjs, num_points, N);
}

template <typename T>
void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *fobjs,
    int_t num_points, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluatePointSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, fobjs, num_points, N);
}

template <typename T>
void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *points, const int_t *point_infos,
    const T *rescaled_a_vals, const T *rescaled_g_vecs,
    const T *rescaled_constants, T *jacobians_point, T *rescaled_errors,
    int_t num_points, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizePointSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, jacobians_point, rescaled_errors,
      num_points, N);
}

template <typename T>
void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *points, const int_t *point_infos, const T *rescaled_a_vals,
    const T *rescaled_g_vecs, const T *rescaled_constants, T *jacobians_point,
    T *rescaled_errors, int_t num_points, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizePointSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, point_indices, points, point_infos, rescaled_a_vals,
      rescaled_g_vecs, rescaled_constants, jacobians_point, rescaled_errors,
      num_points, N);
}

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream) {
  int_t num_threads = num_measurements;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                    stream>>>(
      angle_axis_extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, fobjs, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
}

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                    stream>>>(
      measurement_indices, angle_axis_extrinsics, intrinsics, points,
      measurements, extrinsics_infos, intrinsics_infos, point_infos,
      sqrt_weights, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
}

template <typename T>
void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *fobjs, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  EvaluateAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                    stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, angle_axis_extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
}

template <typename T>
void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const T *angle_axis_extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *jacobians_extrinsics_intrinsics,
    T *rescaled_errors, RobustLoss robust_loss, T loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream) {
  int_t num_threads = num_measurements;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                     stream>>>(
      angle_axis_extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
}

template <typename T>
void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *angle_axis_extrinsics,
    const T *intrinsics, const T *points, const T *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const T *sqrt_weights,
    T *jacobians_extrinsics_intrinsics, T *rescaled_errors,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                     stream>>>(
      measurement_indices, angle_axis_extrinsics, intrinsics, points,
      measurements, extrinsics_infos, intrinsics_infos, point_infos,
      sqrt_weights, jacobians_extrinsics_intrinsics, rescaled_errors,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements, N);
}

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
    int_t num_measurements, int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  LinearizeAngleAxisReprojectionLossFunctionKernel<<<grid_size, block_size, 0,
                                                     stream>>>(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, angle_axis_extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians_extrinsics_intrinsics, rescaled_errors, robust_loss,
      loss_radius, num_extrinsics, num_intrinsics, num_points, num_measurements,
      N);
}

template <typename T>
void MajorizeReprojectionLossFunctionAsync(
    const T *extrinsics, const T *intrinsics, const T *points,
    const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream) {
  int_t num_threads = num_measurements;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 32;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  MajorizeReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      extrinsics, intrinsics, points, measurements, extrinsics_infos,
      intrinsics_infos, point_infos, sqrt_weights, rescaled_h_a_g_vecs,
      rescaled_f_s_vecs, rescaled_sqrt_weights, rescaled_constants, fobjs,
      robust_loss, loss_radius, num_extrinsics, num_intrinsics, num_points,
      num_measurements);
}

template <typename T>
void MajorizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_a_g_vecs, T *rescaled_f_s_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 32;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  MajorizeReprojectionLossFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_a_g_vecs, rescaled_f_s_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
}

template <typename T>
void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_a_vals, T *rescaled_g_vecs,
    T *rescaled_sqrt_weights, T *rescaled_constants, T *fobjs,
    RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 32;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ConstructSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
}

template <typename T>
void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const T *extrinsics, const T *intrinsics,
    const T *points, const T *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const T *sqrt_weights, T *rescaled_h_vecs, T *rescaled_a_vals,
    T *rescaled_g_vecs, T *rescaled_sqrt_weights, T *rescaled_constants,
    T *fobjs, RobustLoss robust_loss, T loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 32;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ConstructSurrogateFunctionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_h_vecs, rescaled_a_vals, rescaled_g_vecs, rescaled_sqrt_weights,
      rescaled_constants, fobjs, robust_loss, loss_radius, num_extrinsics,
      num_intrinsics, num_points, num_measurements, N);
}

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    T *extrinsics_hess_grad, T *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_measurements);
  block_size.y = 4;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsAndPointProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos, point_infos,
      extrinsics_hess_grad, points_hess_grad, num_extrinsics, num_points,
      num_measurements);
}

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsAndPointProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, point_infos, extrinsics_hess_grad, points_hess_grad,
      num_extrinsics, num_points, num_measurements, N);
}

template <typename T>
void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, T *extrinsics_hess_grad, T *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsAndPointProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, extrinsics_indices, point_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, point_infos, extrinsics_hess_grad,
      points_hess_grad, num_extrinsics, num_points, num_measurements, N);
}

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(const T *rescaled_h_a_vecs,
                                              const T *rescaled_a_g_vecs,
                                              const int_t *extrinsics_infos,
                                              T *extrinsics_hess_grad,
                                              int_t num_extrinsics,
                                              int_t num_measurements, int_t N,
                                              cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_measurements);
  block_size.y = 4;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_infos,
      extrinsics_hess_grad, num_extrinsics, num_measurements, N);
}

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, extrinsics_hess_grad, num_extrinsics, num_measurements,
      N);
}

template <typename T>
void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const T *rescaled_h_a_vecs, const T *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, T *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateExtrinsicsProximalOperatorKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, extrinsics_indices, rescaled_h_a_vecs,
      rescaled_a_g_vecs, extrinsics_infos, extrinsics_hess_grad, num_extrinsics,
      num_measurements, N);
}

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(const T *rescaled_f_s_vecs,
                                              const int_t *intrinsics_infos,
                                              T *intrinsics_hess_grad,
                                              int_t num_intrinsics,
                                              int_t num_measurements,
                                              cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_measurements);
  block_size.y = 8;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      rescaled_f_s_vecs, intrinsics_infos, intrinsics_hess_grad, num_intrinsics,
      num_measurements);
}

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(const int_t *measurement_indices,
                                              const T *rescaled_f_s_vecs,
                                              const int_t *intrinsics_infos,
                                              T *intrinsics_hess_grad,
                                              int_t num_intrinsics,
                                              int_t num_measurements, int_t N,
                                              cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 8;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, rescaled_f_s_vecs, intrinsics_infos,
      intrinsics_hess_grad, num_intrinsics, num_measurements, N);
}

template <typename T>
void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const T *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    T *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 8;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_indices, intrinsics_indices, rescaled_f_s_vecs,
      intrinsics_infos, intrinsics_hess_grad, num_intrinsics, num_measurements,
      N);
}

template <typename T>
void ConstructPointProximalOperatorAsync(const T *rescaled_a_g_vecs,
                                         const int_t *point_infos,
                                         T *points_hess_grad, int_t num_points,
                                         int_t num_measurements, int_t N,
                                         cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointProximalOperatorKernel<<<grid_size, block_size,
                                      block_size.x * sizeof(T), stream>>>(
      rescaled_a_g_vecs, point_infos, points_hess_grad, num_points,
      num_measurements, N);
}

template <typename T>
void ConstructPointProximalOperatorAsync(const int_t *measurement_indices,
                                         const T *rescaled_a_g_vecs,
                                         const int_t *point_infos,
                                         T *points_hess_grad, int_t num_points,
                                         int_t num_measurements, int_t N,
                                         cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointProximalOperatorKernel<<<grid_size, block_size,
                                      block_size.x * sizeof(T), stream>>>(
      measurement_indices, rescaled_a_g_vecs, point_infos, points_hess_grad,
      num_points, num_measurements, N);
}

template <typename T>
void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *rescaled_a_g_vecs, const int_t *point_infos, T *points_hess_grad,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointProximalOperatorKernel<<<grid_size, block_size,
                                      block_size.x * sizeof(T), stream>>>(
      measurement_indices, point_indices, rescaled_a_g_vecs, point_infos,
      points_hess_grad, num_points, num_measurements, N);
}

template <typename T>
void ComputeExtrinsicsAndPointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const int_t *measurement_indices_by_points, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n,
    T *points_hess_grad_n, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeExtrinsicsAndPointProximalOperatorProductKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices_by_extrinsics, measurement_indices_by_points,
      rescaled_h_a_vecs, rescaled_a_g_vecs, extrinsics_hess_grad_n,
      points_hess_grad_n, N);
}

template <typename T>
void ComputeExtrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, T *extrinsics_hess_grad_n, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeExtrinsicsProximalOperatorProductKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices_by_extrinsics, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_hess_grad_n, N);
}

template <typename T>
void ComputeExtrinsicsProximalOperatorAsync(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *extrinsics_indices, const T *rescaled_h_a_vecs,
    const T *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    T *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = 32;
  block_size.y = 4;
  int_t num_blocks = N;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeExtrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_dicts_by_extrinsics, measurement_offsets_by_extrinsics,
      extrinsics_indices, rescaled_h_a_vecs, rescaled_a_g_vecs,
      extrinsics_infos, extrinsics_hess_grad, num_extrinsics, num_measurements,
      N);
}

template <typename T>
void ComputeIntrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_intrinsics, const T *rescaled_f_s_vecs,
    T *intrinsics_hess_grad_n, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 8;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeIntrinsicsProximalOperatorProductKernel<<<grid_size, block_size, 0,
                                                   stream>>>(
      measurement_indices_by_intrinsics, rescaled_f_s_vecs,
      intrinsics_hess_grad_n, N);
}

template <typename T>
void ComputePointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_points, const T *rescaled_a_g_vecs,
    T *points_hess_grad_n, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 4;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputePointProximalOperatorProductKernel<<<
      grid_size, block_size, block_size.x * sizeof(T), stream>>>(
      measurement_indices_by_points, rescaled_a_g_vecs, points_hess_grad_n, N);
}

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const T *init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          cudaStream_t stream) {
  int_t num_threads = num_extrinsics;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveExtrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_extrinsics, extrinsics, num_extrinsics);
}

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const int_t *extrinsics_indices,
                                          const T *data, T reg,
                                          const T *init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          int N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveExtrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      extrinsics_indices, data, reg, init_extrinsics, extrinsics,
      num_extrinsics, N);
}

template <typename T>
void SolveExtrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const int_t *init_extrinsics_dicts,
                                          const T *init_extrinsics,
                                          int_t num_init_extrinsics,
                                          T *extrinsics, int_t num_extrinsics,
                                          int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveExtrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_extrinsics_dicts, init_extrinsics, num_init_extrinsics,
      extrinsics, num_extrinsics, N);
}

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const T *init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          cudaStream_t stream) {
  int_t num_threads = num_intrinsics;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_intrinsics, intrinsics, num_intrinsics);
}

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const int_t *intrinsics_indices,
                                          const T *data, T reg,
                                          const T *init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      intrinsics_indices, data, reg, init_intrinsics, intrinsics,
      num_intrinsics, N);
}

template <typename T>
void SolveIntrinsicsProximalOperatorAsync(const T *data, T reg,
                                          const int_t *init_intrinsics_dicts,
                                          const T *init_intrinsics,
                                          int_t num_init_intrinsics,
                                          T *intrinsics, int_t num_intrinsics,
                                          int_t N, cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolveIntrinsicsProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_intrinsics_dicts, init_intrinsics, num_init_intrinsics,
      intrinsics, num_intrinsics, N);
}

template <typename T>
void SolvePointProximalOperatorAsync(const T *data, T reg, const T *init_points,
                                     T *points, int_t num_points,
                                     cudaStream_t stream) {
  int_t num_threads = num_points;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolvePointProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_points, points, num_points);
}

template <typename T>
void SolvePointProximalOperatorAsync(const int_t *point_indices, const T *data,
                                     T reg, const T *init_points, T *points,
                                     int_t num_points, int_t N,
                                     cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolvePointProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      point_indices, data, reg, init_points, points, num_points, N);
}

template <typename T>
void SolvePointProximalOperatorAsync(const T *data, T reg,
                                     const int_t *init_point_dicts,
                                     const T *init_points,
                                     int_t num_init_points, T *points,
                                     int_t num_points, int_t N,
                                     cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SolvePointProximalOperatorKernel<<<grid_size, block_size, 0, stream>>>(
      data, reg, init_point_dicts, init_points, num_init_points, points,
      num_points, N);
}

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements,
    cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_measurements);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateReprojectionLossFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements);
}

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateReprojectionLossFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, point_infos, hessians_cc, hessians_cl, hessians_ll,
      gradients_c, gradients_l, num_cameras, num_points, num_measurements, N);
}

template <typename T>
void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateReprojectionLossFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, camera_indices, point_indices,
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, N);
}

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateCameraSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      hessians_cc, gradients_c, num_cameras, num_measurements, N);
}

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateCameraSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, hessians_cc, gradients_c, num_cameras, num_measurements, N);
}

template <typename T>
void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateCameraSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, camera_indices, jacobians_extrinsics_intrinsics,
      rescaled_errors, camera_infos, hessians_cc, gradients_c, num_cameras,
      num_measurements, N);
}

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = LANDMARK_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      jacobians_points, rescaled_errors, point_infos, hessians_ll, gradients_l,
      num_points, num_measurements, N);
}

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = LANDMARK_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, jacobians_points, rescaled_errors, point_infos,
      hessians_ll, gradients_l, num_points, num_measurements, N);
}

template <typename T>
void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  dim3 grid_size, block_size;
  if (N <= 0) {
    return;
  }

  block_size.x = std::min((int_t)32, N);
  block_size.y = LANDMARK_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdatePointSurrogateFunctionHessianGradientKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices, point_indices, jacobians_points, rescaled_errors,
      point_infos, hessians_ll, gradients_l, num_points, num_measurements, N);
}

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, T *hessians_cc,
    T *hessians_cl, T *hessians_ll, T *gradients_c, T *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, bool reset,
    T alpha, T beta, cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdateReprojectionLossFunctionHessianGradientAsync(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha, T beta,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdateReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, point_infos, hessians_cc, hessians_cl, hessians_ll,
      gradients_c, gradients_l, num_cameras, num_points, num_measurements, N,
      stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha, T beta,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdateReprojectionLossFunctionHessianGradientAsync(
      measurement_indices, camera_indices, point_indices,
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
      gradients_l, num_cameras, num_points, num_measurements, N, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, T alpha,
    T beta, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
  }

  UpdateCameraSurrogateFunctionHessianGradientAsync(
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      hessians_cc, gradients_c, num_cameras, num_measurements, N, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);
}

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_extrinsics_intrinsics,
    const T *rescaled_errors, const int_t *camera_infos, T *hessians_cc,
    T *gradients_c, int_t num_cameras, int_t num_measurements, int_t N,
    bool reset, T alpha, T beta, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
  }

  UpdateCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, hessians_cc, gradients_c, num_cameras, num_measurements, N,
      stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);
}

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, T alpha,
    T beta, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
  }

  UpdateCameraSurrogateFunctionHessianGradientAsync(
      measurement_indices, camera_indices, jacobians_extrinsics_intrinsics,
      rescaled_errors, camera_infos, hessians_cc, gradients_c, num_cameras,
      num_measurements, N, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);
}

template <typename T>
void ComputePointSurrogateFunctionHessianGradientAsync(
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha, T beta,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdatePointSurrogateFunctionHessianGradientAsync(
      jacobians_points, rescaled_errors, point_infos, hessians_ll, gradients_l,
      num_points, num_measurements, N, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const T *jacobians_points,
    const T *rescaled_errors, const int_t *point_infos, T *hessians_ll,
    T *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    bool reset, T alpha, T beta, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdatePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, jacobians_points, rescaled_errors, point_infos,
      hessians_ll, gradients_l, num_points, num_measurements, N, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const T *jacobians_points, const T *rescaled_errors,
    const int_t *point_infos, T *hessians_ll, T *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, T alpha, T beta,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  UpdatePointSurrogateFunctionHessianGradientAsync(
      measurement_indices, point_indices, jacobians_points, rescaled_errors,
      point_infos, hessians_ll, gradients_l, num_points, num_measurements, N,
      stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset, T alpha,
    T beta, cudaStream_t stream) {
  if (N[0] <= 0 && N[1] <= 0 && N[2] <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  if (N[0] > 0) {
    UpdateReprojectionLossFunctionHessianGradientAsync(
        measurement_indices[0], jacobians[0], rescaled_errors[0], camera_infos,
        point_infos, hessians_cc, hessians_cl, hessians_ll, gradients_c,
        gradients_l, num_cameras, num_points, num_measurements, N[0], stream);
  }

  if (N[1] > 0) {
    UpdateCameraSurrogateFunctionHessianGradientAsync(
        measurement_indices[1], jacobians[1], rescaled_errors[1], camera_infos,
        hessians_cc, gradients_c, num_cameras, num_measurements, N[1], stream);
  }

  if (N[2] > 0) {
    UpdatePointSurrogateFunctionHessianGradientAsync(
        measurement_indices[2], jacobians[2], rescaled_errors[2], point_infos,
        hessians_ll, gradients_l, num_points, num_measurements, N[2], stream);
  }

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, D_LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const T *, 3> &jacobians,
    const std::array<const T *, 3> &rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll,
    T *gradients_c, T *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset, T alpha,
    T beta, cudaStream_t stream) {
  if (N[0] <= 0 && N[1] <= 0 && N[2] <= 0) {
    return;
  }

  if (reset) {
    cudaMemsetAsync(hessians_cc, 0,
                    D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(hessians_ll, 0,
                    D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_c, 0, D_CAMERA_SIZE * num_cameras * sizeof(T),
                    stream);
    cudaMemsetAsync(gradients_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                    stream);
  }

  if (N[0] > 0) {
    UpdateReprojectionLossFunctionHessianGradientAsync(
        measurement_indices[0], camera_indices, point_indices, jacobians[0],
        rescaled_errors[0], camera_infos, point_infos, hessians_cc, hessians_cl,
        hessians_ll, gradients_c, gradients_l, num_cameras, num_points,
        num_measurements, N[0], stream);
  }

  if (N[1] > 0) {
    UpdateCameraSurrogateFunctionHessianGradientAsync(
        measurement_indices[1], camera_indices, jacobians[1],
        rescaled_errors[1], camera_infos, hessians_cc, gradients_c, num_cameras,
        num_measurements, N[1], stream);
  }

  if (N[2] > 0) {
    UpdatePointSurrogateFunctionHessianGradientAsync(
        measurement_indices[2], point_indices, jacobians[2], rescaled_errors[2],
        point_infos, hessians_ll, gradients_l, num_points, num_measurements,
        N[2], stream);
  }

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_cc, D_CAMERA_SIZE,
                                      num_cameras, stream);

  sfm::utils::SetSymmetricMatrixAsync(alpha, beta, hessians_ll, LANDMARK_SIZE,
                                      num_points, stream);
}

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const int_t *measurement_indices_by_points,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *hessians_cl_n, T *hessians_ll_n, T *gradients_c_n,
    T *gradients_l_n, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeReprojectionLossFunctionHessianGradientProductKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices_by_cameras, measurement_indices_by_points,
      jacobians_extrinsics_intrinsics, rescaled_errors, hessians_cc_n,
      hessians_cl_n, hessians_ll_n, gradients_c_n, gradients_l_n, N);
}

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    T *hessians_cc_n, T *gradients_c_n, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeCameraSurrogateFunctionHessianGradientProductKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices_by_cameras, jacobians_extrinsics_intrinsics,
      rescaled_errors, hessians_cc_n, gradients_c_n, N);
}

template <typename T>
void ComputePointSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_points, const T *jacobians_points,
    const T *rescaled_errors, T *hessians_ll_n, T *gradients_l_n, int_t N,
    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = D_LANDMARK_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputePointSurrogateFunctionHessianGradientProductKernel<<<
      grid_size, block_size, block_size.x * block_size.y * sizeof(T), stream>>>(
      measurement_indices_by_points, jacobians_points, rescaled_errors,
      hessians_ll_n, gradients_l_n, N);
}

template <typename T>
void UpdateHessianSumForCameraAsync(const int_t *measurement_dicts,
                                    const int_t *measurement_offsets,
                                    const int_t *camera_indices,
                                    const T *hess_cc_n,
                                    const int_t *camera_infos, T *hess_cc,
                                    int_t num_cameras, int_t num_measurements,
                                    int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = 32;
  block_size.y = (D_CAMERA_SIZE + 1) * D_CAMERA_SIZE / 2;
  int_t num_blocks = N;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateHessianSumForCameraKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_offsets, camera_indices, hess_cc_n,
      camera_infos, hess_cc, num_cameras, num_measurements, N);
}

template <typename T>
void ComputeCameraDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, T alpha, const T *x, const int_t *camera_infos,
    T beta, T *y, int_t num_cameras, int_t num_measurements,
    int_t reduction_size, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = 32;
  block_size.y = reduction_size;
  int_t num_blocks = N;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeCameraDictedReductionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_offsets, camera_indices, alpha, x,
      camera_infos, beta, y, num_cameras, num_measurements, N);
}

template <typename T>
void UpdateHessianSumForPointAsync(const int_t *measurement_dicts,
                                   const int_t *measurement_offsets,
                                   const int_t *point_indices,
                                   const T *hess_ll_n, const int_t *point_infos,
                                   T *hess_ll, int_t num_points,
                                   int_t num_measurements, int_t N,
                                   cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = (D_LANDMARK_SIZE + 1) * D_LANDMARK_SIZE / 2;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  UpdateHessianSumForPointKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_offsets, point_indices, hess_ll_n,
      point_infos, hess_ll, num_points, num_measurements, N);
}

template <typename T>
void ComputePointDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, T alpha, const T *x, const int_t *point_infos,
    T beta, T *y, int_t num_points, int_t num_measurements,
    int_t reduction_size, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = reduction_size;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputePointDictedReductionKernel<<<grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_offsets, point_indices, alpha, x,
      point_infos, beta, y, num_points, num_measurements, N);
}

template <typename T>
void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras,
    const int_t *measurement_indices_by_points, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *hessians_cl, T *hessians_ll_n,
    T *gradients_c, T *gradients_l_n, int_t num_cameras, int_t num_measurements,
    int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = 32;
  block_size.y = D_CAMERA_SIZE;
  int_t num_blocks = N;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeReprojectionLossFunctionHessianGradientProductKernel<<<
      grid_size, block_size, 0, stream>>>(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      measurement_indices_by_points, camera_indices,
      jacobians_extrinsics_intrinsics, rescaled_errors, camera_infos,
      hessians_cc, hessians_cl, hessians_ll_n, gradients_c, gradients_l_n,
      num_cameras, num_measurements, N);
}

template <typename T>
void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras, const int_t *camera_indices,
    const T *jacobians_extrinsics_intrinsics, const T *rescaled_errors,
    const int_t *camera_infos, T *hessians_cc, T *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = 32;
  block_size.y = D_CAMERA_SIZE;
  int_t num_blocks = N;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeCameraSurrogateFunctionHessianGradientKernel<<<grid_size, block_size,
                                                        0, stream>>>(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      camera_indices, jacobians_extrinsics_intrinsics, rescaled_errors,
      camera_infos, hessians_cc, gradients_c, num_cameras, num_measurements, N);
}

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset, cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_measurements);
  block_size.y = 3;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_LANDMARK_SIZE * num_points * sizeof(T), stream);
  }

  ComputeHessianCameraPointLeftMultiplicationKernel<<<
      grid_size, block_size, block_size.x *(D_CAMERA_SIZE) * sizeof(T),
      stream>>>(hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
                num_points, num_measurements);
}

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 3;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_LANDMARK_SIZE * num_points * sizeof(T), stream);
  }

  ComputeHessianCameraPointLeftMultiplicationKernel<<<
      grid_size, block_size, block_size.x *(D_CAMERA_SIZE) * sizeof(T),
      stream>>>(measurement_indices, hessians_cl, x, camera_infos, point_infos,
                beta, y, num_cameras, num_points, num_measurements, N);
}

template <typename T>
void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = 3;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_LANDMARK_SIZE * num_points * sizeof(T), stream);
  }

  ComputeHessianCameraPointLeftMultiplicationKernel<<<
      grid_size, block_size, block_size.x *(D_CAMERA_SIZE) * sizeof(T),
      stream>>>(measurement_indices, camera_indices, point_indices, hessians_cl,
                x, camera_infos, point_infos, beta, y, num_cameras, num_points,
                num_measurements, N);
}

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const T *hessians_cl, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset, cudaStream_t stream) {
  if (num_measurements <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, num_measurements);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_CAMERA_SIZE * num_cameras * sizeof(T), stream);
  }

  ComputeHessianCameraPointRightMultiplicationKernel<<<
      grid_size, block_size, block_size.x * 3 * sizeof(T), stream>>>(
      hessians_cl, x, camera_infos, point_infos, beta, y, num_cameras,
      num_points, num_measurements);
}

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_CAMERA_SIZE * num_cameras * sizeof(T), stream);
  }

  ComputeHessianCameraPointRightMultiplicationKernel<<<
      grid_size, block_size, block_size.x * 3 * sizeof(T), stream>>>(
      measurement_indices, hessians_cl, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N);
}

template <typename T>
void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const T *hessians_cl, const T *x,
    const int_t *camera_infos, const int_t *point_infos, T beta, T *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  if (reset) {
    cudaMemsetAsync(y, 0, D_CAMERA_SIZE * num_cameras * sizeof(T), stream);
  }

  ComputeHessianCameraPointRightMultiplicationKernel<<<
      grid_size, block_size, block_size.x * 3 * sizeof(T), stream>>>(
      measurement_indices, camera_indices, point_indices, hessians_cl, x,
      camera_infos, point_infos, beta, y, num_cameras, num_points,
      num_measurements, N);
}

template <typename T>
void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = D_LANDMARK_SIZE;
  block_size.y = std::min((int)32, N);

  int_t num_blocks = (N + block_size.y - 1) / block_size.y;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeBlockSparseHessianCameraPointLeftMultiplicationKernel<<<
      grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_index_offsets, camera_indices,
      point_indices, hessians_cl, alpha, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N);
}

template <typename T>
void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, T *buffer, int_t num_cameras,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = D_LANDMARK_SIZE;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeBlockSparseHessianCameraPointLeftProductKernel<<<
      grid_size, block_size, block_size.x * D_CAMERA_SIZE * sizeof(T),
      stream>>>(measurement_dicts, camera_indices, point_indices, hessians_cl,
                T(1.0), x, camera_infos, point_infos, buffer, num_cameras,
                num_points, num_measurements, num_measurements);

  ComputePointDictedReductionAsync(measurement_dicts, measurement_offsets,
                                   point_indices, alpha, buffer, point_infos,
                                   beta, y, num_points, num_measurements,
                                   D_LANDMARK_SIZE, N, stream);

  // block_size.x = D_LANDMARK_SIZE;
  // block_size.y = std::min((int)32, N);

  // num_blocks = (N + block_size.y - 1) / block_size.y;
  // grid_size.y = (num_blocks - 1) / 65536 + 1;
  // grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  // ComputePointDictedReductionKernel<<<grid_size, block_size, 0, stream>>>(
  //     measurement_dicts, measurement_offsets, point_indices, alpha,
  //     buffer, point_infos, beta, y, num_points, num_measurements, N);
}

template <typename T>
void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = 32;

  int_t num_blocks = N * D_CAMERA_SIZE;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeBlockSparseHessianCameraPointRightMultiplicationKernel<<<
      grid_size, block_size, 0, stream>>>(
      measurement_dicts, measurement_index_offsets, camera_indices,
      point_indices, hessians_cl, alpha, x, camera_infos, point_infos, beta, y,
      num_cameras, num_points, num_measurements, N);
}

template <typename T>
void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const T *hessians_cl, T alpha, const T *x, const int_t *camera_infos,
    const int_t *point_infos, T beta, T *y, T *buffer, int_t num_cameras,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int)32, N);
  block_size.y = D_CAMERA_SIZE;

  int_t num_blocks = (num_measurements + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeBlockSparseHessianCameraPointRightProductKernel<<<
      grid_size, block_size, block_size.x * D_LANDMARK_SIZE * sizeof(T),
      stream>>>(measurement_dicts, camera_indices, point_indices, hessians_cl,
                T(1.0), x, camera_infos, point_infos, buffer, num_cameras,
                num_points, num_measurements, num_measurements);

  ComputeCameraDictedReductionAsync(measurement_dicts, measurement_offsets,
                                    camera_indices, alpha, buffer, camera_infos,
                                    beta, y, num_cameras, num_measurements,
                                    D_CAMERA_SIZE, N, stream);

  // block_size.x = 32;
  // block_size.y = 1;
  // num_blocks = N * D_CAMERA_SIZE;
  // grid_size.y = (num_blocks - 1) / 65536 + 1;
  // grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  // ComputeBlockSparseHessianCameraPointRightSumKernel<<<grid_size,
  // block_size,
  //                                                         0, stream>>>(
  //     measurement_dicts, measurement_offsets, camera_indices, alpha, buffer,
  //     camera_infos, beta, y, num_cameras, num_measurements, N);
}

template <typename T>
void ComputeHessianPointPointInverseAsync(const T *hessians_ll,
                                          T *hessians_ll_inverse,
                                          int_t num_points,
                                          cudaStream_t stream) {
  int_t num_threads = num_points;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeHessianPointPointInverseKernel<<<grid_size, block_size, 0, stream>>>(
      hessians_ll, hessians_ll_inverse, num_points);
}

template <typename T>
void ComputeHessianPointPointInverseAsync(const int_t *point_indices,
                                          const T *hessians_ll,
                                          T *hessians_ll_inverse,
                                          int_t num_points, int_t N,
                                          cudaStream_t stream) {
  int_t num_threads = N;
  if (num_threads <= 0) {
    return;
  }
  int_t block_size = 128;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeHessianPointPointInverseKernel<<<grid_size, block_size, 0, stream>>>(
      point_indices, hessians_ll, hessians_ll_inverse, num_points, N);
}

template void EvaluateReprojectionLossFunctionAsync(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream);

template void EvaluateReprojectionLossFunctionAsync(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream);

template void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void EvaluateReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *jacobians_extrinsics_intrinsics,
    float *rescaled_errors, RobustLoss robust_loss, float loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *jacobians_extrinsics_intrinsics,
    double *rescaled_errors, RobustLoss robust_loss, double loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *jacobians_extrinsics_intrinsics,
    float *rescaled_errors, RobustLoss robust_loss, float loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void LinearizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *jacobians_extrinsics_intrinsics,
    double *rescaled_errors, RobustLoss robust_loss, double loss_radius,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const float *extrinsics,
    const float *intrinsics, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const float *rescaled_sqrt_weights, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeCameraSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const double *extrinsics,
    const double *intrinsics, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const double *rescaled_sqrt_weights, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    int_t num_extrinsics, int_t num_intrinsics, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *points,
    const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_points, int_t N, cudaStream_t stream);

template void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *points,
    const int_t *point_infos, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *fobjs, int_t num_points, int_t N, cudaStream_t stream);

template void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *points, const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants, float *fobjs,
    int_t num_points, int_t N, cudaStream_t stream);

template void EvaluatePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *points, const int_t *point_infos,
    const double *rescaled_a_vals, const double *rescaled_g_vecs,
    const double *rescaled_constants, double *fobjs, int_t num_points, int_t N,
    cudaStream_t stream);

template void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *points,
    const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_point, float *rescaled_errors, int_t num_points, int_t N,
    cudaStream_t stream);

template void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *points,
    const int_t *point_infos, const double *rescaled_a_vals,
    const double *rescaled_g_vecs, const double *rescaled_constants,
    double *jacobians_point, double *rescaled_errors, int_t num_points, int_t N,
    cudaStream_t stream);

template void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *points, const int_t *point_infos, const float *rescaled_a_vals,
    const float *rescaled_g_vecs, const float *rescaled_constants,
    float *jacobians_point, float *rescaled_errors, int_t num_points, int_t N,
    cudaStream_t stream);

template void LinearizePointSurrogateFunctionAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *points, const int_t *point_infos,
    const double *rescaled_a_vals, const double *rescaled_g_vecs,
    const double *rescaled_constants, double *jacobians_point,
    double *rescaled_errors, int_t num_points, int_t N, cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const float *angle_axis_extrinsics, const float *intrinsics,
    const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const double *angle_axis_extrinsics, const double *intrinsics,
    const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const float *angle_axis_extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const double *angle_axis_extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *angle_axis_extrinsics, const float *intrinsics,
    const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void EvaluateAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *angle_axis_extrinsics, const double *intrinsics,
    const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const float *angle_axis_extrinsics, const float *intrinsics,
    const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const double *angle_axis_extrinsics, const double *intrinsics,
    const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const float *angle_axis_extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const double *angle_axis_extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const float *angle_axis_extrinsics, const float *intrinsics,
    const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *jacobians_extrinsics_intrinsics, float *rescaled_errors,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void LinearizeAngleAxisReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *intrinsics_indices, const int_t *point_indices,
    const double *angle_axis_extrinsics, const double *intrinsics,
    const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *jacobians_extrinsics_intrinsics, double *rescaled_errors,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void MajorizeReprojectionLossFunctionAsync(
    const float *extrinsics, const float *intrinsics, const float *points,
    const float *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const float *sqrt_weights, float *rescaled_h_a_g_vecs,
    float *rescaled_f_s_vecs, float *rescaled_sqrt_weights,
    float *rescaled_constants, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream);

template void MajorizeReprojectionLossFunctionAsync(
    const double *extrinsics, const double *intrinsics, const double *points,
    const double *measurements, const int_t *extrinsics_infos,
    const int_t *intrinsics_infos, const int_t *point_infos,
    const double *sqrt_weights, double *rescaled_h_a_g_vecs,
    double *rescaled_f_s_vecs, double *rescaled_sqrt_weights,
    double *rescaled_constants, double *fobjs, RobustLoss robust_loss,
    double loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream);

template void MajorizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights,
    float *rescaled_h_a_g_vecs, float *rescaled_f_s_vecs,
    float *rescaled_sqrt_weights, float *rescaled_constants, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void MajorizeReprojectionLossFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_h_a_g_vecs, double *rescaled_f_s_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *rescaled_h_vecs,
    float *rescaled_a_vals, float *rescaled_g_vecs,
    float *rescaled_sqrt_weights, float *rescaled_constants, float *fobjs,
    RobustLoss robust_loss, float loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_h_vecs, double *rescaled_a_vals, double *rescaled_g_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const float *extrinsics,
    const float *intrinsics, const float *points, const float *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const float *sqrt_weights, float *rescaled_a_vals,
    float *rescaled_g_vecs, float *rescaled_sqrt_weights,
    float *rescaled_constants, float *fobjs, RobustLoss robust_loss,
    float loss_radius, int_t num_extrinsics, int_t num_intrinsics,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructSurrogateFunctionAsync(
    const int_t *measurement_indices, const double *extrinsics,
    const double *intrinsics, const double *points, const double *measurements,
    const int_t *extrinsics_infos, const int_t *intrinsics_infos,
    const int_t *point_infos, const double *sqrt_weights,
    double *rescaled_a_vals, double *rescaled_g_vecs,
    double *rescaled_sqrt_weights, double *rescaled_constants, double *fobjs,
    RobustLoss robust_loss, double loss_radius, int_t num_extrinsics,
    int_t num_intrinsics, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    float *extrinsics_hess_grad, float *points_hess_grad, int_t num_extrinsics,
    int_t num_points, int_t num_measurements, cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, const int_t *point_infos,
    double *extrinsics_hess_grad, double *points_hess_grad,
    int_t num_extrinsics, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, float *extrinsics_hess_grad,
    float *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, double *extrinsics_hess_grad,
    double *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, float *extrinsics_hess_grad,
    float *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsAndPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const int_t *point_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    const int_t *point_infos, double *extrinsics_hess_grad,
    double *points_hess_grad, int_t num_extrinsics, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, float *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, double *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    float *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    double *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, float *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructExtrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *extrinsics_indices,
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    const int_t *extrinsics_infos, double *extrinsics_hess_grad,
    int_t num_extrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const float *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    float *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const double *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    double *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const float *rescaled_f_s_vecs,
    const int_t *intrinsics_infos, float *intrinsics_hess_grad,
    int_t num_intrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const double *rescaled_f_s_vecs,
    const int_t *intrinsics_infos, double *intrinsics_hess_grad,
    int_t num_intrinsics, int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const float *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    float *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ConstructIntrinsicsProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *intrinsics_indices,
    const double *rescaled_f_s_vecs, const int_t *intrinsics_infos,
    double *intrinsics_hess_grad, int_t num_intrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const float *rescaled_a_g_vecs, const int_t *point_infos,
    float *points_hess_grad, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const double *rescaled_a_g_vecs, const int_t *point_infos,
    double *points_hess_grad, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const float *rescaled_a_g_vecs,
    const int_t *point_infos, float *points_hess_grad, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const double *rescaled_a_g_vecs,
    const int_t *point_infos, double *points_hess_grad, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *rescaled_a_g_vecs, const int_t *point_infos,
    float *points_hess_grad, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ConstructPointProximalOperatorAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *rescaled_a_g_vecs, const int_t *point_infos,
    double *points_hess_grad, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ComputeExtrinsicsAndPointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const int_t *measurement_indices_by_points, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, float *extrinsics_hess_grad_n,
    float *points_hess_grad_n, int_t N, cudaStream_t stream);

template void ComputeExtrinsicsAndPointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const int_t *measurement_indices_by_points, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, double *extrinsics_hess_grad_n,
    double *points_hess_grad_n, int_t N, cudaStream_t stream);

template void ComputeExtrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const float *rescaled_h_a_vecs, const float *rescaled_a_g_vecs,
    float *extrinsics_hess_grad_n, int_t N, cudaStream_t stream);

template void ComputeExtrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_extrinsics,
    const double *rescaled_h_a_vecs, const double *rescaled_a_g_vecs,
    double *extrinsics_hess_grad_n, int_t N, cudaStream_t stream);

template void ComputeExtrinsicsProximalOperatorAsync(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *extrinsics_indices, const float *rescaled_h_a_vecs,
    const float *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    float *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeExtrinsicsProximalOperatorAsync(
    const int_t *measurement_dicts_by_extrinsics,
    const int_t *measurement_offsets_by_extrinsics,
    const int_t *extrinsics_indices, const double *rescaled_h_a_vecs,
    const double *rescaled_a_g_vecs, const int_t *extrinsics_infos,
    double *extrinsics_hess_grad, int_t num_extrinsics, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeIntrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_intrinsics,
    const float *rescaled_f_s_vecs, float *intrinsics_hess_grad_n, int_t N,
    cudaStream_t stream);

template void ComputeIntrinsicsProximalOperatorProductAsync(
    const int_t *measurement_indices_by_intrinsics,
    const double *rescaled_f_s_vecs, double *intrinsics_hess_grad_n, int_t N,
    cudaStream_t stream);

template void ComputePointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_points, const float *rescaled_a_g_vecs,
    float *points_hess_grad_n, int_t N, cudaStream_t stream);

template void ComputePointProximalOperatorProductAsync(
    const int_t *measurement_indices_by_points, const double *rescaled_a_g_vecs,
    double *points_hess_grad_n, int_t N, cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(const float *data, float reg,
                                                   const float *init_extrinsics,
                                                   float *extrinsics,
                                                   int_t num_extrinsics,
                                                   cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(
    const double *data, double reg, const double *init_extrinsics,
    double *extrinsics, int_t num_extrinsics, cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(
    const int_t *extrinsics_indices, const float *data, float reg,
    const float *init_extrinsics, float *extrinsics, int_t num_extrinsics,
    int_t N, cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(
    const int_t *extrinsics_indices, const double *data, double reg,
    const double *init_extrinsics, double *extrinsics, int_t num_extrinsics,
    int_t N, cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(
    const float *data, float reg, const int_t *init_extrinsics_dicts,
    const float *init_extrinsics, int_t num_init_extrinsics, float *extrinsics,
    int_t num_extrinsics, int_t N, cudaStream_t stream);

template void SolveExtrinsicsProximalOperatorAsync(
    const double *data, double reg, const int_t *init_extrinsics_dicts,
    const double *init_extrinsics, int_t num_init_extrinsics,
    double *extrinsics, int_t num_extrinsics, int_t N, cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(const float *data, float reg,
                                                   const float *init_intrinsics,
                                                   float *intrinsics,
                                                   int_t num_intrinsics,
                                                   cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(
    const double *data, double reg, const double *init_intrinsics,
    double *intrinsics, int_t num_intrinsics, cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(
    const int_t *intrinsics_indices, const float *data, float reg,
    const float *init_intrinsics, float *intrinsics, int_t num_intrinsics,
    int_t N, cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(
    const int_t *intrinsics_indices, const double *data, double reg,
    const double *init_intrinsics, double *intrinsics, int_t num_intrinsics,
    int_t N, cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(
    const float *data, float reg, const int_t *init_intrinsics_dicts,
    const float *init_intrinsics, int_t num_init_intrinsics, float *intrinsics,
    int_t num_intrinsics, int_t N, cudaStream_t stream);

template void SolveIntrinsicsProximalOperatorAsync(
    const double *data, double reg, const int_t *init_intrinsics_dicts,
    const double *init_intrinsics, int_t num_init_intrinsics,
    double *intrinsics, int_t num_intrinsics, int_t N, cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const float *data, float reg,
                                              const float *init_points,
                                              float *points, int_t num_points,
                                              cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const double *data, double reg,
                                              const double *init_points,
                                              double *points, int_t num_points,
                                              cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const int_t *point_indices,
                                              const float *data, float reg,
                                              const float *init_points,
                                              float *points, int_t num_points,
                                              int_t N, cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const int_t *point_indices,
                                              const double *data, double reg,
                                              const double *init_points,
                                              double *points, int_t num_points,
                                              int_t N, cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const float *data, float reg,
                                              const int_t *init_point_dicts,
                                              const float *init_points,
                                              int_t num_init_points,
                                              float *points, int_t num_points,
                                              int_t N, cudaStream_t stream);

template void SolvePointProximalOperatorAsync(const double *data, double reg,
                                              const int_t *init_point_dicts,
                                              const double *init_points,
                                              int_t num_init_points,
                                              double *points, int_t num_points,
                                              int_t N, cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements,
    cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *jacobians_extrinsics_intrinsics,
    const float *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, float *hessians_cc, float *hessians_cl,
    float *hessians_ll, float *gradients_c, float *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void UpdateReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const float *jacobians_points,
    const float *rescaled_errors, const int_t *point_infos, float *hessians_ll,
    float *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const double *jacobians_points,
    const double *rescaled_errors, const int_t *point_infos,
    double *hessians_ll, double *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *jacobians_points, const float *rescaled_errors,
    const int_t *point_infos, float *hessians_ll, float *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *jacobians_points, const double *rescaled_errors,
    const int_t *point_infos, double *hessians_ll, double *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const float *jacobians_points, const float *rescaled_errors,
    const int_t *point_infos, float *hessians_ll, float *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdatePointSurrogateFunctionHessianGradientAsync(
    const double *jacobians_points, const double *rescaled_errors,
    const int_t *point_infos, double *hessians_ll, double *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, bool reset, float alpha, float beta,
    cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, bool reset,
    double alpha, double beta, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, float alpha, float beta,
    cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, double alpha, double beta, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *jacobians_extrinsics_intrinsics,
    const float *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, float *hessians_cc, float *hessians_cl,
    float *hessians_ll, float *gradients_c, float *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, float alpha, float beta, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    const int_t *point_infos, double *hessians_cc, double *hessians_cl,
    double *hessians_ll, double *gradients_c, double *gradients_l,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, double alpha, double beta, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta,
    cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta,
    cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const float *jacobians_points,
    const float *rescaled_errors, const int_t *point_infos, float *hessians_ll,
    float *gradients_l, int_t num_points, int_t num_measurements, int_t N,
    bool reset, float alpha, float beta, cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const double *jacobians_points,
    const double *rescaled_errors, const int_t *point_infos,
    double *hessians_ll, double *gradients_l, int_t num_points,
    int_t num_measurements, int_t N, bool reset, double alpha, double beta,
    cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const float *jacobians_points, const float *rescaled_errors,
    const int_t *point_infos, float *hessians_ll, float *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, bool reset, float alpha,
    float beta, cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_indices, const int_t *point_indices,
    const double *jacobians_points, const double *rescaled_errors,
    const int_t *point_infos, double *hessians_ll, double *gradients_l,
    int_t num_points, int_t num_measurements, int_t N, bool reset, double alpha,
    double beta, cudaStream_t stream);

template void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const float *, 3> &jacobians,
    const std::array<const float *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    float alpha, float beta, cudaStream_t stream);

template void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const std::array<const double *, 3> &jacobians,
    const std::array<const double *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, double *hessians_cc,
    double *hessians_cl, double *hessians_ll, double *gradients_c,
    double *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    double alpha, double beta, cudaStream_t stream);

template void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const float *, 3> &jacobians,
    const std::array<const float *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, float *hessians_cc,
    float *hessians_cl, float *hessians_ll, float *gradients_c,
    float *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    float alpha, float beta, cudaStream_t stream);

template void ComputeHessianGradientAsync(
    const std::array<const int_t *, 3> &measurement_indices,
    const int_t *camera_indices, const int_t *point_indices,
    const std::array<const double *, 3> &jacobians,
    const std::array<const double *, 3> &rescaled_errors,
    const int_t *camera_infos, const int_t *point_infos, double *hessians_cc,
    double *hessians_cl, double *hessians_ll, double *gradients_c,
    double *gradients_l, int_t num_cameras, int_t num_points,
    int_t num_measurements, const std::array<int_t, 3> &N, bool reset,
    double alpha, double beta, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const int_t *measurement_indices_by_points,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    float *hessians_cc_n, float *hessians_cl_n, float *hessians_ll_n,
    float *gradients_c_n, float *gradients_l_n, int_t N, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const int_t *measurement_indices_by_points,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, double *hessians_cc_n, double *hessians_cl_n,
    double *hessians_ll_n, double *gradients_c_n, double *gradients_l_n,
    int_t N, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    float *hessians_cc_n, float *gradients_c_n, int_t N, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_cameras,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, double *hessians_cc_n, double *gradients_c_n,
    int_t N, cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_points, const float *jacobians_points,
    const float *rescaled_errors, float *hessians_ll_n, float *gradients_l_n,
    int_t N, cudaStream_t stream);

template void ComputePointSurrogateFunctionHessianGradientProductAsync(
    const int_t *measurement_indices_by_points, const double *jacobians_points,
    const double *rescaled_errors, double *hessians_ll_n, double *gradients_l_n,
    int_t N, cudaStream_t stream);

template void UpdateHessianSumForCameraAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const float *hess_cc_n,
    const int_t *camera_infos, float *hess_cc, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateHessianSumForCameraAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const double *hess_cc_n,
    const int_t *camera_infos, double *hess_cc, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeCameraDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, float alpha, const float *x,
    const int_t *camera_infos, float beta, float *y, int_t num_cameras,
    int_t num_measurements, int_t reduction_size, int_t N, cudaStream_t stream);

template void ComputeCameraDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, double alpha, const double *x,
    const int_t *camera_infos, double beta, double *y, int_t num_cameras,
    int_t num_measurements, int_t reduction_size, int_t N, cudaStream_t stream);

template void UpdateHessianSumForPointAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, const float *hess_ll_n,
    const int_t *point_infos, float *hess_ll, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void UpdateHessianSumForPointAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, const double *hess_ll_n,
    const int_t *point_infos, double *hess_ll, int_t num_points,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputePointDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, float alpha, const float *x,
    const int_t *point_infos, float beta, float *y, int_t num_points,
    int_t num_measurements, int_t reduction_size, int_t N, cudaStream_t stream);

template void ComputePointDictedReductionAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *point_indices, double alpha, const double *x,
    const int_t *point_infos, double beta, double *y, int_t num_points,
    int_t num_measurements, int_t reduction_size, int_t N, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras,
    const int_t *measurement_indices_by_points, const int_t *camera_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *hessians_cl,
    float *hessians_ll_n, float *gradients_c, float *gradients_l_n,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeReprojectionLossFunctionHessianGradientProductAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras,
    const int_t *measurement_indices_by_points, const int_t *camera_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *hessians_cl, double *hessians_ll_n,
    double *gradients_c, double *gradients_l_n, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras, const int_t *camera_indices,
    const float *jacobians_extrinsics_intrinsics, const float *rescaled_errors,
    const int_t *camera_infos, float *hessians_cc, float *gradients_c,
    int_t num_cameras, int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeCameraSurrogateFunctionHessianGradientAsync(
    const int_t *measurement_dicts_by_cameras,
    const int_t *measurement_offsets_by_cameras, const int_t *camera_indices,
    const double *jacobians_extrinsics_intrinsics,
    const double *rescaled_errors, const int_t *camera_infos,
    double *hessians_cc, double *gradients_c, int_t num_cameras,
    int_t num_measurements, int_t N, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const float *hessians_cl, const float *x, const int_t *camera_infos,
    const int_t *point_infos, float beta, float *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const double *hessians_cl, const double *x, const int_t *camera_infos,
    const int_t *point_infos, double beta, double *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const double *hessians_cl,
    const double *x, const int_t *camera_infos, const int_t *point_infos,
    double beta, double *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *hessians_cl, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const float *hessians_cl, const float *x, const int_t *camera_infos,
    const int_t *point_infos, float beta, float *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const double *hessians_cl, const double *x, const int_t *camera_infos,
    const int_t *point_infos, double beta, double *y, int_t num_cameras,
    int_t num_points, int_t num_measurements, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const double *hessians_cl,
    const double *x, const int_t *camera_infos, const int_t *point_infos,
    double beta, double *y, int_t num_cameras, int_t num_points,
    int_t num_measurements, int_t N, bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const float *hessians_cl, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_indices, const int_t *camera_indices,
    const int_t *point_indices, const double *hessians_cl, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    bool reset, cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const float *hessians_cl, float alpha, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const double *hessians_cl, double alpha, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const float *hessians_cl, float alpha, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const double *hessians_cl, double alpha, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    int_t num_cameras, int_t num_points, int_t num_measurements, int_t N,
    cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const float *hessians_cl, float alpha, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    float *buffer, int_t num_cameras, int_t num_points, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_index_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const double *hessians_cl, double alpha, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    double *buffer, int_t num_cameras, int_t num_points, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const float *hessians_cl, float alpha, const float *x,
    const int_t *camera_infos, const int_t *point_infos, float beta, float *y,
    float *buffer, int_t num_cameras, int_t num_points, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
    const int_t *measurement_dicts, const int_t *measurement_offsets,
    const int_t *camera_indices, const int_t *point_indices,
    const double *hessians_cl, double alpha, const double *x,
    const int_t *camera_infos, const int_t *point_infos, double beta, double *y,
    double *buffer, int_t num_cameras, int_t num_points, int_t num_measurements,
    int_t N, cudaStream_t stream);

template void ComputeHessianPointPointInverseAsync(const float *hessians_ll,
                                                   float *hessians_ll_inverse,
                                                   int_t num_points,
                                                   cudaStream_t stream);

template void ComputeHessianPointPointInverseAsync(const double *hessians_ll,
                                                   double *hessians_ll_inverse,
                                                   int_t num_points,
                                                   cudaStream_t stream);

template void ComputeHessianPointPointInverseAsync(const int_t *point_indices,
                                                   const float *hessians_ll,
                                                   float *hessians_ll_inverse,
                                                   int_t num_points, int_t N,
                                                   cudaStream_t stream);

template void ComputeHessianPointPointInverseAsync(const int_t *point_indices,
                                                   const double *hessians_ll,
                                                   double *hessians_ll_inverse,
                                                   int_t num_points, int_t N,
                                                   cudaStream_t stream);
} // namespace ba
} // namespace sfm
