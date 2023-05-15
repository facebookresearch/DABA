// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sfm/math/SO3.h>
#include <sfm/types.h>
#include <sfm/utils/cuda_utils.h>
#include <sfm/utils/internal/internal-inl.cuh>
#include <sfm/utils/internal/shared_memory-inl.cuh>
#include <sfm/utils/internal/solver/chol/chol.cuh>
#include <sfm/utils/internal/solver/lu/lu.cuh>
#include <sfm/utils/iterators.h>
#include <sfm/utils/utils.cuh>
#include <sfm/utils/utils.h>

namespace sfm {
namespace utils {
template <typename T>
__global__ void ProjectToSE3Impl(const T *data, T *se3, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  Eigen::Matrix<T, 3, 4> ret;

  Eigen::Vector3<T> hess_rot_t;
  T hess_tt;
  Eigen::Matrix3<T> &grad_rot = *(Eigen::Matrix3<T> *)(ret.data());
  Eigen::Vector3<T> &grad_t = *(Eigen::Vector3<T> *)(ret.data() + 9);

  data += idx;
  for (int i = 0; i < 3; i++) {
    hess_rot_t[i] = *data;
    data += N;

    for (int j = 0; j < 3; j++) {
      grad_rot(j, i) = *data;
      data += N;
    }
  }

  hess_tt = *data;
  data += N;

  for (int j = 0; j < 3; j++) {
    grad_t[j] = *data;
    data += N;
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

  sfm::utils::SetMatrixOfArray(N, se3, idx, ret);
}

template <typename T>
__global__ void PlusAsyncKernel(T a, const T *x, T b, const T *y, T *result,
                                int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx < N) {
    result[idx] = a * x[idx] + b * y[idx];
  }
}

template <typename T>
__global__ void NesterovExtrapolateSE3Kernel(const T *x0, const T *x1, T beta,
                                             T *y, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  Eigen::Matrix<T, 3, 4> X, Y;
  sfm::utils::GetMatrixOfArray(N, x0, idx, X);
  sfm::utils::GetMatrixOfArray(N, x1, idx, Y);
  Y += beta * (Y - X);

  Eigen::Matrix3<T> U, V;
  Eigen::Vector3<T> S;

  sfm::utils::internal::svd3x3(
      Y(0, 0), Y(0, 1), Y(0, 2), Y(1, 0), Y(1, 1), Y(1, 2), Y(2, 0), Y(2, 1),
      Y(2, 2), U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0),
      U(2, 1), U(2, 2), S[0], S[1], S[2], V(0, 0), V(0, 1), V(0, 2), V(1, 0),
      V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

  Y.template leftCols<3>().noalias() = U * V.transpose();

  sfm::utils::SetMatrixOfArray(N, y, idx, Y);
}

template <typename T>
__global__ void NesterovExtrapolateDictedSE3Kernel(const int_t *dicts,
                                                   const T *x0, const T *x1,
                                                   T beta, T *y, int_t x_size,
                                                   int_t y_size, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  Eigen::Matrix<T, 3, 4> X, Y;
  sfm::utils::GetMatrixOfArray(x_size, x0, idx, X);
  sfm::utils::GetMatrixOfArray(x_size, x1, idx, Y);
  Y += beta * (Y - X);

  Eigen::Matrix3<T> U, V;
  Eigen::Vector3<T> S;

  sfm::utils::internal::svd3x3(
      Y(0, 0), Y(0, 1), Y(0, 2), Y(1, 0), Y(1, 1), Y(1, 2), Y(2, 0), Y(2, 1),
      Y(2, 2), U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0),
      U(2, 1), U(2, 2), S[0], S[1], S[2], V(0, 0), V(0, 1), V(0, 2), V(1, 0),
      V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

  Y.template leftCols<3>().noalias() = U * V.transpose();

  sfm::utils::SetMatrixOfArray(y_size, y, dicts[idx], Y);
}

template <typename T>
__global__ void NesterovExtrapolateMatrixKernel(const T *x0, const T *x1,
                                                T beta, T *y, int_t rows,
                                                int_t cols, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  for (int cnt = 0, i = idx; cnt < rows * cols; cnt++, i += N) {
    y[i] = (1 + beta) * x1[i] - beta * x0[i];
  }
}

template <typename T>
__global__ void NesterovExtrapolateDictedMatrixKernel(const int_t *dicts,
                                                      const T *x0, const T *x1,
                                                      T beta, T *y, int_t rows,
                                                      int_t cols, int_t x_size,
                                                      int_t y_size, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  for (int cnt = 0, i = idx, j = dicts[idx]; cnt < rows * cols; cnt++) {
    y[j] = (1 + beta) * x1[i] - beta * x0[i];

    i += x_size;
    j += y_size;
  }
}

template <typename T>
__global__ void SetSymmetricMatrixKernel(T *matrix, int_t num_matrices) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_matrices)
    return;

  int_t offset = (threadIdx.y + 1) * num_matrices + idx;
  auto src = matrix + offset;
  auto dest =
      matrix + (threadIdx.y + 1) * (blockDim.y + 1) * num_matrices + idx;

  for (int_t i = 0; i <= threadIdx.y; i++) {
    *dest = *src;
    src += (blockDim.y + 1) * num_matrices;
    dest += num_matrices;
  }
}

template <typename T>
__global__ void SetSymmetricMatrixKernel(T alpha, T beta, T *matrix,
                                         int_t num_matrices) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_matrices)
    return;

  int_t offset = threadIdx.y * num_matrices + idx;
  int_t inc = blockDim.y * num_matrices;
  auto src = matrix + offset;
  auto dest = matrix + threadIdx.y * blockDim.y * num_matrices + idx;

  for (int_t i = 0; i < threadIdx.y; i++) {
    *dest = *src;
    src += inc;
    dest += num_matrices;
  }

  *dest = *src * alpha + beta;
}

template <typename T>
__global__ void RescaleSymmetricMatrixDiagonalKernel(T ratio, T *matrix,
                                                     int_t num_matrices) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= num_matrices)
    return;

  int_t offset = threadIdx.y * (blockDim.y + 1) * num_matrices + idx;
  auto src = matrix + offset;
  *src = *src * ratio;
}

template <typename T>
__global__ void
ComputeMatrixVectorMultiplicationKernel(T alpha, const T *matrix, const T *x,
                                        T beta, T *y, int_t N) {
  int_t blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int_t idx = blockId * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  T *xval = sfm::utils::internal::SharedMemory<T>::get();

  xval[blockDim.x * threadIdx.y + threadIdx.x] = x[N * threadIdx.y + idx];
  __syncthreads();

  T sum = 0;
  auto ptr = matrix + N * blockDim.y * threadIdx.y + idx;

  for (int j = 0; j < blockDim.y; j++) {
    sum += *ptr * xval[blockDim.x * j + threadIdx.x];
    ptr += N;
  }

  y[N * threadIdx.y + idx] = alpha * sum + beta * y[N * threadIdx.y + idx];
}

template <typename T>
__global__ void ComputeSE3RetractionKernel(const T *X, T stepsize, const T *dX,
                                           T *Xplus, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  Eigen::Matrix<T, 3, 4> Xk, Xkplus;
  Eigen::Vector<T, 6> dXk;
  Eigen::Matrix<T, 3, 3> dRk;
  sfm::utils::GetMatrixOfArray(N, X, idx, Xk);
  sfm::utils::GetMatrixOfArray(N, dX, idx, dXk);

  dXk *= stepsize;
  const Eigen::Vector3<T> &dOmega = *(Eigen::Vector3<T> *)(dXk.data());
  sfm::math::SO3::Exp(dOmega, dRk);
  Xkplus.template leftCols<3>().noalias() = dRk * Xk.template leftCols<3>();
  Xkplus.col(3) = Xk.col(3) + dXk.template tail<3>();

  sfm::utils::SetMatrixOfArray(N, Xplus, idx, Xkplus);
}

template <typename T>
__global__ void
CopyFromDictedMatrixOfArrayKernel(const int_t *dicts, int_t rows, int_t cols,
                                  const T *src, T *dst, int_t src_num_matrices,
                                  int_t dst_num_matrices, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  auto src_ptr = src + idx;
  auto dst_ptr = dst + dicts[idx];

  for (int_t i = 0; i < rows * cols; i++) {
    *dst_ptr = *src_ptr;
    src_ptr += src_num_matrices;
    dst_ptr += dst_num_matrices;
  }
}

template <typename T>
__global__ void
CopyToDictedMatrixOfArrayKernel(const int_t *dicts, int_t rows, int_t cols,
                                const T *src, T *dst, int_t src_num_matrices,
                                int_t dst_num_matrices, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  auto src_ptr = src + dicts[idx];
  auto dst_ptr = dst + idx;

  for (int_t i = 0; i < rows * cols; i++) {
    *dst_ptr = *src_ptr;
    src_ptr += src_num_matrices;
    dst_ptr += dst_num_matrices;
  }
}

template <typename T>
__global__ void
AddFromDictedMatrixOfArrayKernel(const int_t *dicts, int_t rows, int_t cols,
                                 const T *src, T *dst, int_t src_num_matrices,
                                 int_t dst_num_matrices, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  auto src_ptr = src + idx;
  auto dst_ptr = dst + dicts[idx];

  for (int_t i = 0; i < rows * cols; i++) {
    *dst_ptr += *src_ptr;
    src_ptr += src_num_matrices;
    dst_ptr += dst_num_matrices;
  }
}

template <typename T>
__global__ void
AddToDictedMatrixOfArrayKernel(const int_t *dicts, int_t rows, int_t cols,
                               const T *src, T *dst, int_t src_num_matrices,
                               int_t dst_num_matrices, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  auto src_ptr = src + dicts[idx];
  auto dst_ptr = dst + idx;

  for (int_t i = 0; i < rows * cols; i++) {
    *dst_ptr += *src_ptr;
    src_ptr += src_num_matrices;
    dst_ptr += dst_num_matrices;
  }
}

template <typename T>
__global__ void CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayKernel(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const T *src, T *dst, int_t src_num_matrices, int_t dst_num_matrices,
    int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  auto src_ptr = src + src_dicts[idx];
  auto dst_ptr = dst + dst_dicts[idx];

  for (int_t i = 0; i < rows * cols; i++) {
    *dst_ptr = *src_ptr;
    src_ptr += src_num_matrices;
    dst_ptr += dst_num_matrices;
  }
}

template <typename T>
__global__ void ApproximateKernel(const T *src, T *dst, int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  T src_val = src[idx];
  T dst_val = 0;

  Approximate(src_val, dst_val);
  dst[idx] = dst_val;
}

__global__ void SetIndexFromDictKernel(const int_t *dicts, int_t *indices,
                                       int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t index_idx = dicts[idx];

  indices[index_idx] = idx;
}

__global__ void ResetIndexFromDictKernel(const int_t *dicts, int_t *indices,
                                         int_t N) {
  int_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int_t idx = blockId * (blockDim.x * blockDim.y) + threadIdx.x;

  if (idx >= N)
    return;

  int_t index_idx = dicts[idx];

  indices[index_idx] = -1;
}

template <typename T>
void ProjectToSE3Async(const T *data, T *se3, int_t N, cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ProjectToSE3Impl<<<grid_size, block_size, 0, stream>>>(data, se3, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void InnerProductAsync(void *temp, size_t &temp_bytes, const T *input1, int_t N,
                       const T *input2, T *output, cudaStream_t stream) {
  TransformReduceAsync(temp, temp_bytes, input1, N, input2,
                       thrust::multiplies<T>(), output, T(0.0),
                       thrust::plus<T>(), stream);
}

template <typename T>
void SquaredNormAsync(void *temp, size_t &temp_bytes, const T *input, int_t N,
                      T *output, cudaStream_t stream) {
  TransformReduceAsync(temp, temp_bytes, input, N, thrust::square<T>(), output,
                       T(0.0), thrust::plus<T>(), stream);
}

template <typename T>
void DifferenceSquaredNormAsync(void *temp, size_t &temp_bytes, const T *input1,
                                int_t N, const T *input2, T *output,
                                cudaStream_t stream) {
  TransformReduceAsync(
      temp, temp_bytes, input1, N, input2,
      [input1, input2] __device__(T x, T y) -> T {
        T diff = x - y;
        return diff * diff;
      },
      output, T(0.0), cub::Sum(), stream);
}

template <typename T>
void MatrixDiagonalWeightedSquaredNormAsync(void *temp, size_t &temp_bytes,
                                            const T *v, int_t N, int_t size,
                                            const T *matrix, T *output,
                                            cudaStream_t stream) {
  cub::CountingInputIterator<int_t> index(0);
  TransformReduceAsync(
      temp, temp_bytes, v, N * size, index,
      [matrix, N, offset = (size + 1) * N] __device__(T x, int_t idx) -> T {
        int_t k = idx / N;
        int_t n = idx % N;
        return x * x * matrix[k * offset + n];
      },
      output, T(0.0), thrust::plus<T>(), stream);
}

template <typename T>
void PlusAsync(T a, const T *x, T b, const T *y, T *result, int_t N,
               cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  PlusAsyncKernel<<<grid_size, block_size, 0, stream>>>(a, x, b, y, result, N);
}

template <typename T>
void NesterovExtrapolateSE3Async(const T *x0, const T *x1, T beta, T *y,
                                 int_t N, cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  NesterovExtrapolateSE3Kernel<<<grid_size, block_size, 0, stream>>>(
      x0, x1, beta, y, N);
}

template <typename T>
void NesterovExtrapolateDictedSE3Async(const int_t *dicts, const T *x0,
                                       const T *x1, T beta, T *y, int_t x_size,
                                       int_t y_size, int_t N,
                                       cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  NesterovExtrapolateDictedSE3Kernel<<<grid_size, block_size, 0, stream>>>(
      dicts, x0, x1, beta, y, x_size, y_size, N);
}

template <typename T>
void NesterovExtrapolateMatrixAsync(const T *x0, const T *x1, T beta, T *y,
                                    int_t rows, int_t cols, int_t N,
                                    cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  NesterovExtrapolateMatrixKernel<<<grid_size, block_size, 0, stream>>>(
      x0, x1, beta, y, rows, cols, N);
}

template <typename T>
void NesterovExtrapolateDictedMatrixAsync(const int_t *dicts, const T *x0,
                                          const T *x1, T beta, T *y, int_t rows,
                                          int_t cols, int_t x_size,
                                          int_t y_size, int_t N,
                                          cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  NesterovExtrapolateDictedMatrixKernel<<<grid_size, block_size, 0, stream>>>(
      dicts, x0, x1, beta, y, rows, cols, x_size, y_size, N);
}

template <typename T>
void SetSymmetricMatrixAsync(T *matrix, int_t size, int_t num_matrices,
                             cudaStream_t stream) {
  if (num_matrices <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_matrices);
  block_size.y = size - 1;

  int_t num_blocks = (num_matrices + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SetSymmetricMatrixKernel<<<grid_size, block_size, 0, stream>>>(matrix,
                                                                 num_matrices);
}

template <typename T>
void SetSymmetricMatrixAsync(T alpha, T beta, T *matrix, int_t size,
                             int_t num_matrices, cudaStream_t stream) {
  if (num_matrices <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_matrices);
  block_size.y = size;

  int_t num_blocks = (num_matrices + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SetSymmetricMatrixKernel<<<grid_size, block_size, 0, stream>>>(
      alpha, beta, matrix, num_matrices);
}

template <typename T>
void RescaleSymmetricMatrixDiagonalAsync(T ratio, T *matrix, int_t size,
                                         int_t num_matrices,
                                         cudaStream_t stream) {
  if (num_matrices <= 0) {
    return;
  }
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, num_matrices);
  block_size.y = size;

  int_t num_blocks = (num_matrices + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  RescaleSymmetricMatrixDiagonalKernel<<<grid_size, block_size, 0, stream>>>(
      ratio, matrix, num_matrices);
}

template <typename T>
void ComputeMatrixInverseAsync(const T *matrix, T *matrix_inverse, int_t size,
                               int_t N, cudaStream_t stream) {
  int stat = sfm::utils::internal::lu_matinv_batch_async(matrix, matrix_inverse,
                                                         size, N, stream);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }
}

template <typename T>
void ComputePositiveDefiniteMatrixInverseAsync(const T *matrix,
                                               T *matrix_inverse, int_t size,
                                               int_t N, cudaStream_t stream) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      matrix, matrix_inverse, size, N, stream);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }
}

template <typename T>
void ComputePositiveDefiniteMatrixInverseAsync(T alpha, T beta, const T *matrix,
                                               T *matrix_inverse, int_t size,
                                               int_t N, cudaStream_t stream) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      alpha, beta, matrix, matrix_inverse, size, N, stream);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }
}

template <typename T>
void ComputeMatrixVectorMultiplicationAsync(T alpha, const T *matrix,
                                            const T *x, T beta, T *y,
                                            int_t size, int_t N,
                                            cudaStream_t stream) {
  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = size;
  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeMatrixVectorMultiplicationKernel<<<
      grid_size, block_size, block_size.x * size * sizeof(T), stream>>>(
      alpha, matrix, x, beta, y, N);
}

template <typename T>
void ComputeSE3RetractionAsync(const T *X, T stepsize, const T *dX, T *Xplus,
                               int_t N, cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ComputeSE3RetractionKernel<<<grid_size, block_size, 0, stream>>>(
      X, stepsize, dX, Xplus, N);
}

template <typename T>
void CopyFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                      int_t cols, const T *src, T *dst,
                                      int_t src_num_matrices,
                                      int_t dst_num_matrices, int_t N,
                                      cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  CopyFromDictedMatrixOfArrayKernel<<<grid_size, block_size, 0, stream>>>(
      dicts, rows, cols, src, dst, src_num_matrices, dst_num_matrices, N);
}

template <typename T>
void CopyToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                    const T *src, T *dst,
                                    int_t src_num_matrices,
                                    int_t dst_num_matrices, int_t N,
                                    cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  CopyToDictedMatrixOfArrayKernel<<<grid_size, block_size, 0, stream>>>(
      dicts, rows, cols, src, dst, src_num_matrices, dst_num_matrices, N);
}

template <typename T>
void AddToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                   const T *src, T *dst, int_t src_num_matrices,
                                   int_t dst_num_matrices, int_t N,
                                   cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  AddToDictedMatrixOfArrayKernel<<<grid_size, block_size, 0, stream>>>(
      dicts, rows, cols, src, dst, src_num_matrices, dst_num_matrices, N);
}

template <typename T>
void CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const T *src, T *dst, int_t src_num_matrices, int_t dst_num_matrices,
    int_t N, cudaStream_t stream) {
  if (N <= 0)
    return;

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayKernel<<<
      grid_size, block_size, 0, stream>>>(src_dicts, dst_dicts, rows, cols, src,
                                          dst, src_num_matrices,
                                          dst_num_matrices, N);
}

template <typename T>
void AddFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                     const T *src, T *dst,
                                     int_t src_num_matrices,
                                     int_t dst_num_matrices, int_t N,
                                     cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  AddFromDictedMatrixOfArrayKernel<<<grid_size, block_size, 0, stream>>>(
      dicts, rows, cols, src, dst, src_num_matrices, dst_num_matrices, N);
}

template <typename T>
void ApproximateAsync(const T *src, T *dst, int_t N, cudaStream_t stream) {
  if (N <= 0)
    return;

  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ApproximateKernel<<<grid_size, block_size, 0, stream>>>(src, dst, N);
}

void SetIndexFromDictAsync(const int_t *dicts, int_t *indices, int_t N,
                           cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  SetIndexFromDictKernel<<<grid_size, block_size, 0, stream>>>(dicts, indices,
                                                               N);
}

void ResetIndexFromDictAsync(const int_t *dicts, int_t *indices, int_t N,
                             cudaStream_t stream) {
  int_t block_size = 128;
  int_t num_threads = N;
  int_t num_blocks = (num_threads + block_size - 1) / block_size;

  dim3 grid_size;
  grid_size.y = (num_blocks - 1) / 65536 + 1;
  grid_size.x = (num_blocks + grid_size.y - 1) / grid_size.y;

  ResetIndexFromDictKernel<<<grid_size, block_size, 0, stream>>>(dicts, indices,
                                                                 N);
}

template <>
void ProjectToSE3<kGPU, float>(const float *data, float *se3, int_t N) {
  ProjectToSE3Async(data, se3, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ProjectToSE3<kGPU, double>(const double *data, double *se3, int_t N) {
  ProjectToSE3Async(data, se3, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void NesterovExtrapolateSE3<kGPU>(const float *x0, const float *x1, float beta,
                                  float *y, int_t N) {
  NesterovExtrapolateSE3Async(x0, x1, beta, y, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void NesterovExtrapolateSE3<kGPU>(const double *x0, const double *x1,
                                  double beta, double *y, int_t N) {
  NesterovExtrapolateSE3Async(x0, x1, beta, y, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void NesterovExtrapolateMatrix<kGPU>(const float *x0, const float *x1,
                                     float beta, float *y, int_t rows,
                                     int_t cols, int_t N) {
  NesterovExtrapolateMatrixAsync(x0, x1, beta, y, rows, cols, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void NesterovExtrapolateMatrix<kGPU>(const double *x0, const double *x1,
                                     double beta, double *y, int_t rows,
                                     int_t cols, int_t N) {
  NesterovExtrapolateMatrixAsync(x0, x1, beta, y, rows, cols, N, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeMatrixInverse<kGPU>(const float *matrix, float *matrix_inverse,
                                int_t size, int_t N) {
  int stat = sfm::utils::internal::lu_matinv_batch_async(matrix, matrix_inverse,
                                                         size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeMatrixInverse<kGPU>(const double *matrix, double *matrix_inverse,
                                int_t size, int_t N) {
  int stat = sfm::utils::internal::lu_matinv_batch_async(matrix, matrix_inverse,
                                                         size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePositiveDefiniteMatrixInverse<kGPU>(const float *matrix,
                                                float *matrix_inverse,
                                                int_t size, int_t N) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      matrix, matrix_inverse, size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePositiveDefiniteMatrixInverse<kGPU>(const double *matrix,
                                                double *matrix_inverse,
                                                int_t size, int_t N) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      matrix, matrix_inverse, size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePositiveDefiniteMatrixInverse<kGPU>(float alpha, float beta,
                                                const float *matrix,
                                                float *matrix_inverse,
                                                int_t size, int_t N) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      alpha, beta, matrix, matrix_inverse, size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputePositiveDefiniteMatrixInverse<kGPU>(double alpha, double beta,
                                                const double *matrix,
                                                double *matrix_inverse,
                                                int_t size, int_t N) {
  int stat = sfm::utils::internal::chol_matinv_batch_async(
      alpha, beta, matrix, matrix_inverse, size, N);

  if (stat < 0) {
    LOG(ERROR) << "Fail to compute batched matrix inverse. [ERROR CODE: "
               << stat << "]" << std::endl;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeMatrixVectorMultiplication<kGPU>(float alpha, const float *matrix,
                                             const float *x, float beta,
                                             float *y, int_t size, int_t N) {
  ComputeMatrixVectorMultiplicationAsync(alpha, matrix, x, beta, y, size, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeMatrixVectorMultiplication<kGPU>(double alpha, const double *matrix,
                                             const double *x, double beta,
                                             double *y, int_t size, int_t N) {
  ComputeMatrixVectorMultiplicationAsync(alpha, matrix, x, beta, y, size, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeSE3Retraction<kGPU>(const float *X, float stepsize, const float *dX,
                                float *Xplus, int_t N) {
  ComputeSE3RetractionAsync(X, stepsize, dX, Xplus, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void ComputeSE3Retraction<kGPU>(const double *X, double stepsize,
                                const double *dX, double *Xplus, int_t N) {
  ComputeSE3RetractionAsync(X, stepsize, dX, Xplus, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyFromDictedMatrixOfArray<kGPU>(const int_t *dicts, int_t rows,
                                       int_t cols, const float *src, float *dst,
                                       int_t src_num_matrices,
                                       int_t dst_num_matrices, int_t N) {
  CopyFromDictedMatrixOfArrayAsync(dicts, rows, cols, src, dst,
                                   src_num_matrices, dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyFromDictedMatrixOfArray<kGPU>(const int_t *dicts, int_t rows,
                                       int_t cols, const double *src,
                                       double *dst, int_t src_num_matrices,
                                       int_t dst_num_matrices, int_t N) {
  CopyFromDictedMatrixOfArrayAsync(dicts, rows, cols, src, dst,
                                   src_num_matrices, dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyToDictedMatrixOfArray<kGPU>(const int_t *dicts, int_t rows, int_t cols,
                                     const float *src, float *dst,
                                     int_t src_num_matrices,
                                     int_t dst_num_matrices, int_t N) {
  CopyToDictedMatrixOfArrayAsync(dicts, rows, cols, src, dst, src_num_matrices,
                                 dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyToDictedMatrixOfArray<kGPU>(const int_t *dicts, int_t rows, int_t cols,
                                     const double *src, double *dst,
                                     int_t src_num_matrices,
                                     int_t dst_num_matrices, int_t N) {
  CopyToDictedMatrixOfArrayAsync(dicts, rows, cols, src, dst, src_num_matrices,
                                 dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyFromDictedMatrixOfArrayToDictedMatrixOfArray<kGPU>(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const float *src, float *dst, int_t src_num_matrices,
    int_t dst_num_matrices, int_t N) {
  CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
      src_dicts, dst_dicts, rows, cols, src, dst, src_num_matrices,
      dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template <>
void CopyFromDictedMatrixOfArrayToDictedMatrixOfArray<kGPU>(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const double *src, double *dst, int_t src_num_matrices,
    int_t dst_num_matrices, int_t N) {
  CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
      src_dicts, dst_dicts, rows, cols, src, dst, src_num_matrices,
      dst_num_matrices, N);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

template void ProjectToSE3Async(const float *data, float *se3, int_t N,
                                cudaStream_t stream);

template void ProjectToSE3Async(const double *data, double *se3, int_t N,
                                cudaStream_t stream);

template void InnerProductAsync(void *temp, size_t &temp_bytes,
                                const float *input1, int_t N,
                                const float *input2, float *output,
                                cudaStream_t stream);

template void InnerProductAsync(void *temp, size_t &temp_bytes,
                                const double *input1, int_t N,
                                const double *input2, double *output,
                                cudaStream_t stream);

template void SquaredNormAsync(void *temp, size_t &temp_bytes,
                               const float *input, int_t N, float *output,
                               cudaStream_t stream);

template void SquaredNormAsync(void *temp, size_t &temp_bytes,
                               const double *input, int_t N, double *output,
                               cudaStream_t stream);

template void DifferenceSquaredNormAsync(void *temp, size_t &temp_bytes,
                                         const float *input1, int_t N,
                                         const float *input2, float *output,
                                         cudaStream_t stream);

template void DifferenceSquaredNormAsync(void *temp, size_t &temp_bytes,
                                         const double *input1, int_t N,
                                         const double *input2, double *output,
                                         cudaStream_t stream);

template void MatrixDiagonalWeightedSquaredNormAsync(
    void *temp, size_t &temp_bytes, const float *v, int_t N, int_t size,
    const float *matrix, float *output, cudaStream_t stream);

template void MatrixDiagonalWeightedSquaredNormAsync(
    void *temp, size_t &temp_bytes, const double *v, int_t N, int_t size,
    const double *matrix, double *output, cudaStream_t stream);

template void PlusAsync(float a, const float *x, float b, const float *y,
                        float *result, int_t N, cudaStream_t stream);

template void PlusAsync(double a, const double *x, double b, const double *y,
                        double *result, int_t N, cudaStream_t stream);

template void NesterovExtrapolateSE3Async(const float *x0, const float *x1,
                                          float beta, float *y, int_t N,
                                          cudaStream_t stream);

template void NesterovExtrapolateSE3Async(const double *x0, const double *x1,
                                          double beta, double *y, int_t N,
                                          cudaStream_t stream);

template void NesterovExtrapolateMatrixAsync(const float *x0, const float *x1,
                                             float beta, float *y, int_t rows,
                                             int_t cols, int_t N,
                                             cudaStream_t stream);

template void NesterovExtrapolateMatrixAsync(const double *x0, const double *x1,
                                             double beta, double *y, int_t rows,
                                             int_t cols, int_t N,
                                             cudaStream_t stream);

template void NesterovExtrapolateDictedSE3Async(
    const int_t *dicts, const float *x0, const float *x1, float beta, float *y,
    int_t x_size, int_t y_size, int_t N, cudaStream_t stream);

template void NesterovExtrapolateDictedSE3Async(
    const int_t *dicts, const double *x0, const double *x1, double beta,
    double *y, int_t x_size, int_t y_size, int_t N, cudaStream_t stream);

template void NesterovExtrapolateDictedMatrixAsync(
    const int_t *dicts, const float *x0, const float *x1, float beta, float *y,
    int_t rows, int_t cols, int_t x_size, int_t y_size, int_t N,
    cudaStream_t stream);

template void NesterovExtrapolateDictedMatrixAsync(
    const int_t *dicts, const double *x0, const double *x1, double beta,
    double *y, int_t rows, int_t cols, int_t x_size, int_t y_size, int_t N,
    cudaStream_t stream);

template void SetSymmetricMatrixAsync(float *matrix, int_t size,
                                      int_t num_matrices, cudaStream_t stream);

template void SetSymmetricMatrixAsync(double *matrix, int_t size,
                                      int_t num_matrices, cudaStream_t stream);

template void SetSymmetricMatrixAsync(float alpha, float beta, float *matrix,
                                      int_t size, int_t num_matrices,
                                      cudaStream_t stream);

template void SetSymmetricMatrixAsync(double alpha, double beta, double *matrix,
                                      int_t size, int_t num_matrices,
                                      cudaStream_t stream);

template void RescaleSymmetricMatrixDiagonalAsync(float ratio, float *matrix,
                                                  int_t size,
                                                  int_t num_matrices,
                                                  cudaStream_t stream);

template void RescaleSymmetricMatrixDiagonalAsync(double ratio, double *matrix,
                                                  int_t size,
                                                  int_t num_matrices,
                                                  cudaStream_t stream);

template void ComputeMatrixInverseAsync(const float *matrix,
                                        float *matrix_inverse, int_t size,
                                        int_t N, cudaStream_t stream);

template void ComputeMatrixInverseAsync(const double *matrix,
                                        double *matrix_inverse, int_t size,
                                        int_t N, cudaStream_t stream);

template void ComputePositiveDefiniteMatrixInverseAsync(const float *matrix,
                                                        float *matrix_inverse,
                                                        int_t size, int_t N,
                                                        cudaStream_t stream);

template void ComputePositiveDefiniteMatrixInverseAsync(const double *matrix,
                                                        double *matrix_inverse,
                                                        int_t size, int_t N,
                                                        cudaStream_t stream);

template void ComputePositiveDefiniteMatrixInverseAsync(float alpha, float beta,
                                                        const float *matrix,
                                                        float *matrix_inverse,
                                                        int_t size, int_t N,
                                                        cudaStream_t stream);

template void ComputePositiveDefiniteMatrixInverseAsync(
    double alpha, double beta, const double *matrix, double *matrix_inverse,
    int_t size, int_t N, cudaStream_t stream);

template void ComputeMatrixVectorMultiplicationAsync(
    float alpha, const float *matrix, const float *x, float beta, float *y,
    int_t size, int_t N, cudaStream_t stream);

template void ComputeMatrixVectorMultiplicationAsync(
    double alpha, const double *matrix, const double *x, double beta, double *y,
    int_t size, int_t N, cudaStream_t stream);

template void ComputeSE3RetractionAsync(const float *X, float stepsize,
                                        const float *dX, float *Xplus, int_t N,
                                        cudaStream_t stream);

template void ComputeSE3RetractionAsync(const double *X, double stepsize,
                                        const double *dX, double *Xplus,
                                        int_t N, cudaStream_t stream);

template void CopyFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                               int_t cols, const float *src,
                                               float *dst,
                                               int_t src_num_matrices,
                                               int_t dst_num_matrices, int_t N,
                                               cudaStream_t stream);

template void CopyFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                               int_t cols, const double *src,
                                               double *dst,
                                               int_t src_num_matrices,
                                               int_t dst_num_matrices, int_t N,
                                               cudaStream_t stream);

template void CopyToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                             int_t cols, const float *src,
                                             float *dst, int_t src_num_matrices,
                                             int_t dst_num_matrices, int_t N,
                                             cudaStream_t stream);

template void CopyToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                             int_t cols, const double *src,
                                             double *dst,
                                             int_t src_num_matrices,
                                             int_t dst_num_matrices, int_t N,
                                             cudaStream_t stream);

template void AddToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                            int_t cols, const float *src,
                                            float *dst, int_t src_num_matrices,
                                            int_t dst_num_matrices, int_t N,
                                            cudaStream_t stream);

template void AddToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                            int_t cols, const double *src,
                                            double *dst, int_t src_num_matrices,
                                            int_t dst_num_matrices, int_t N,
                                            cudaStream_t stream);

template void CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const float *src, float *dst, int_t src_num_matrices,
    int_t dst_num_matrices, int_t N, cudaStream_t stream);

template void CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const double *src, double *dst, int_t src_num_matrices,
    int_t dst_num_matrices, int_t N, cudaStream_t stream);

template void AddFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                              int_t cols, const float *src,
                                              float *dst,
                                              int_t src_num_matrices,
                                              int_t dst_num_matrices, int_t N,
                                              cudaStream_t stream);

template void AddFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                              int_t cols, const double *src,
                                              double *dst,
                                              int_t src_num_matrices,
                                              int_t dst_num_matrices, int_t N,
                                              cudaStream_t stream);

template void ApproximateAsync(const float *src, float *dst, int_t N,
                               cudaStream_t stream);

template void ApproximateAsync(const double *src, double *dst, int_t N,
                               cudaStream_t stream);
} // namespace utils
} // namespace sfm
