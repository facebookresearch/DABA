// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <sfm/types.h>
#include <thrust/device_vector.h>

namespace sfm {
namespace utils {
__host__ __device__ inline void Approximate(const double &x, double &y) {
  und &un = *(und *)(&y);
  un.f = x;
  un.ui &= 0xfffffffff0000000;
}

__host__ __device__ inline void Approximate(const float &x, float &y) {
  unf &un = *(unf *)(&y);
  un.f = x;
  un.ui &= 0xfffffe00;
  un.ui |= 0x000000ff;
}

template <typename T, int kRows, int kCols>
__host__ __device__ inline void
SetMatrixOfArray(int_t num_matrices, T *matrix_of_array, int_t n,
                 const Eigen::Matrix<T, kRows, kCols> &matrix) {
  assert(n >= 0 && n < num_matrices && "The index n must be 0<=n<N");

  auto ptr = matrix_of_array;

  for (int i = 0; i < kRows; i++) {
    for (int j = 0; j < kCols; j++, ptr += num_matrices) {
      ptr[n] = matrix(i, j);
    }
  }
}

template <typename T, int kRows, int kCols>
__host__ __device__ inline void
AddMatrixOfArray(int_t num_matrices, const T *matrix_of_array, int_t n,
                 Eigen::Matrix<T, kRows, kCols> &matrix) {
  assert(n >= 0 && n < num_matrices && "The index n must be 0<=n<N");

  auto ptr = matrix_of_array;

  for (int i = 0; i < kRows; i++) {
    for (int j = 0; j < kCols; j++, ptr += num_matrices) {
      matrix(i, j) += ptr[n];
    }
  }
}

template <typename T, int kRows, int kCols>
__host__ __device__ inline void
GetMatrixOfArray(int_t num_matrices, const T *matrix_of_array, int_t n,
                 Eigen::Matrix<T, kRows, kCols> &matrix) {
  assert(n >= 0 && n < num_matrices && "The index n must be 0<=n<N");

  auto ptr = matrix_of_array;

  for (int i = 0; i < kRows; i++) {
    for (int j = 0; j < kCols; j++, ptr += num_matrices) {
      matrix(i, j) = ptr[n];
    }
  }
}

template <typename T, int kRows, int kCols>
__host__ __device__ inline void ArrayOfMatrixToMatrixOfArray(
    int_t num_matrices, const Eigen::Matrix<T, kRows, kCols> *array_of_matrix,
    T *matrix_of_array) {
  auto ptr = array_of_matrix;

  for (int n = 0; n < num_matrices; n++, ptr++) {
    SetMatrixOfArray(num_matrices, matrix_of_array, n, *ptr);
  }
}

template <typename T, int kRows, int kCols>
__host__ __device__ inline void
MatrixOfArrayToArrayOfMatrix(int_t num_matrices, const T *matrix_of_array,
                             Eigen::Matrix<T, kRows, kCols> *array_of_matrix) {
  auto ptr = matrix_of_array;

  for (int i = 0; i < kRows; i++) {
    for (int j = 0; j < kCols; j++) {
      for (int id = 0; id < num_matrices; id++) {
        array_of_matrix[id](i, j) = *ptr++;
      }
    }
  }
}

template <typename T, int kRows, int kCols, cudaMemcpyKind kMemCopyKind>
__host__ inline void
MatrixOfArrayToMatrixOfArray(int_t num_matrices, const T *src_matrix_of_array,
                             int_t src_matrix_size, int_t src_matrix_offset,
                             T *dst_matrix_of_array, int_t dst_matrix_size,
                             int_t dst_matrix_offset, cudaStream_t stream = 0) {
  for (int_t n = 0; n < kRows * kCols; n++) {
    cudaMemcpyAsync(
        dst_matrix_of_array + dst_matrix_offset + n * dst_matrix_size,
        src_matrix_of_array + src_matrix_offset + n * src_matrix_size,
        sizeof(T) * num_matrices, kMemCopyKind, stream);
  }

  cudaStreamSynchronize(stream);
}

template <typename T, int kRows, int kCols>
__host__ inline void HostArrayOfMatrixToDeviceMatrixOfArray(
    const std::vector<Eigen::Matrix<T, kRows, kCols>> &host_array_of_matrix,
    thrust::device_vector<T> &device_matrix_of_array) {
  std::vector<T> host_matrix_of_array(host_array_of_matrix.size() * kRows *
                                      kCols);
  ArrayOfMatrixToMatrixOfArray(host_array_of_matrix.size(),
                               host_array_of_matrix.data(),
                               host_matrix_of_array.data());
  device_matrix_of_array.resize(host_matrix_of_array.size());
  thrust::copy(host_matrix_of_array.begin(), host_matrix_of_array.end(),
               device_matrix_of_array.data());
}

template <typename T, int kRows, int kCols>
__host__ inline void DeviceMatrixOfArrayToHostArrayOfMatrix(
    const thrust::device_vector<T> &device_matrix_of_array,
    std::vector<Eigen::Matrix<T, kRows, kCols>> &host_array_of_matrix) {
  assert(device_matrix_of_array.size() % (kRows * kCols) == 0 &&
         "Inconsistent data size.");

  const int_t matrix_size = kRows * kCols;

  if (device_matrix_of_array.size() % (kRows * kCols) != 0) {
    LOG(ERROR) << "The size of matrix of array is "
               << device_matrix_of_array.size()
               << ", but not divided by rows * cols=" << matrix_size
               << std::endl;
    exit(-1);
  }

  std::vector<T> host_matrix_of_array(device_matrix_of_array.size());
  thrust::copy(device_matrix_of_array.begin(), device_matrix_of_array.end(),
               host_matrix_of_array.data());

  host_array_of_matrix.resize(host_matrix_of_array.size() / (matrix_size));
  MatrixOfArrayToArrayOfMatrix(host_array_of_matrix.size(),
                               host_matrix_of_array.data(),
                               host_array_of_matrix.data());
}

template <typename T, int kRows, int kCols>
__host__ inline void DeviceMatrixOfArrayToHostArrayOfMatrix(
    const thrust::device_vector<T> &device_matrix_of_array,
    std::vector<Eigen::Matrix<T, kRows, kCols>> &host_array_of_matrix,
    int_t offset, int_t size) {
  const int_t matrix_size = kRows * kCols;
  const int_t matrix_of_array_size =
      device_matrix_of_array.size() / matrix_size;

  assert(device_matrix_of_array.size() % matrix_size == 0 &&
         "Inconsistent data size.");
  assert(offset >= 0 && size >= 0 &&
         "The offset and size of the array of matrix must be non-negative.");
  assert(matrix_of_array_size >= offset + size && "Inconsistent data size.");

  if (device_matrix_of_array.size() % (kRows * kCols) != 0) {
    LOG(ERROR) << "The size of matrix of array is "
               << device_matrix_of_array.size()
               << ", but not divided by rows * cols=" << matrix_size
               << std::endl;
    exit(-1);
  }

  if (offset < 0 || size < 0) {
    LOG(ERROR) << "The offset and count of the array of matrix is negative."
               << std::endl;
    exit(-1);
  }

  if (matrix_of_array_size < offset + size) {
    LOG(ERROR) << "The required data of the array of matrix is out of the "
                  "bound of the matrix of array."
               << std::endl;
    exit(-1);
  }

  std::vector<T> host_matrix_of_array(matrix_size * size);

  auto input = device_matrix_of_array.data();
  auto output = host_matrix_of_array.data();
  for (int_t i = 0; i < matrix_size; i++) {
    thrust::copy(input + offset, input + offset + size, output);
    input += matrix_of_array_size;
    output += size;
  }

  host_array_of_matrix.resize(host_matrix_of_array.size() / (matrix_size));
  MatrixOfArrayToArrayOfMatrix(host_array_of_matrix.size(),
                               host_matrix_of_array.data(),
                               host_array_of_matrix.data());
}

template <Memory kMemory, typename T>
void ProjectToSE3(const T *data, T *se3, int_t N);

template <Memory kMemory, typename T>
void NesterovExtrapolateSE3(const T *x0, const T *x1, T beta, T *y, int_t N);

template <Memory kMemory, typename T>
void NesterovExtrapolateMatrix(const T *x0, const T *x1, T beta, T *y,
                               int_t rows, int_t cols, int_t N);

template <Memory kMemory, typename T>
void ComputeMatrixInverse(const T *matrix, T *matrix_inverse, int_t size,
                          int_t N);

template <Memory kMemory, typename T>
void ComputePositiveDefiniteMatrixInverse(const T *matrix, T *matrix_inverse,
                                          int_t size, int_t N);

template <Memory kMemory, typename T>
void ComputePositiveDefiniteMatrixInverse(T alpha, T beta, const T *matrix,
                                          T *matrix_inverse, int_t size,
                                          int_t N);

template <Memory kMemory, typename T>
void ComputeMatrixVectorMultiplication(T alpha, const T *matrix, const T *x,
                                       T beta, T *y, int_t size, int_t N);

template <Memory kMemory, typename T>
void ComputeSE3Retraction(const T *X, T stepsize, const T *dX, T *Xplus,
                          int_t N);

template <Memory kMemory, typename T>
void CopyFromDictedMatrixOfArray(const int_t *dicts, int_t rows, int_t cols,
                                 const T *src, T *dst, int_t src_num_matrices,
                                 int_t dst_num_matrices, int_t N);

template <Memory kMemory, typename T>
void CopyToDictedMatrixOfArray(const int_t *dicts, int_t rows, int_t cols,
                               const T *src, T *dst, int_t src_num_matrices,
                               int_t dst_num_matrices, int_t N);

template <Memory kMemory, typename T>
void CopyFromDictedMatrixOfArrayToDictedMatrixOfArray(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const T *src, T *dst, int_t src_num_matrices, int_t dst_num_matrices,
    int_t N);
} // namespace utils
} // namespace sfm