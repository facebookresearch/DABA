// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sfm/types.h>
#include <sfm/utils/iterators.h>
#include <sfm/utils/utils-inl.cuh>
#include <thrust/async/for_each.h>
#include <thrust/device_vector.h>

namespace sfm {
namespace utils {
template <typename Value, typename Operator, typename Input>
__host__ __device__ cub::TransformInputIterator<Value, Operator, Input>
MakeTransformIterator(Input input, Operator op) {
  return cub::TransformInputIterator<Value, Operator, Input>(input, op);
}

template <typename Value, typename Offset, typename Operator, typename Input>
__host__ __device__ cub::TransformInputIterator<Value, Offset, Operator, Input>
MakeTransformIterator(Input input, Operator op) {
  return cub::TransformInputIterator<Value, Operator, Input>(input, op);
}

template <typename Value>
__host__ __device__ thrust::counting_iterator<Value>
MakeCountingIterator(Value val) {
  return thrust::counting_iterator<Value>(val);
}

template <typename Value, typename Offset = ptrdiff_t>
__host__ __device__ thrust::constant_iterator<Value>
MakeConstantIterator(Value val) {
  return thrust::counting_iterator<Value>(val);
}

template <typename... Iterators>
__host__ __device__ thrust::zip_iterator<thrust::tuple<Iterators...>>
MakeZipIterator(thrust::tuple<Iterators...> t) {
  return thrust::make_zip_iterator(t);
}

template <typename... Iterators>
__host__ __device__ thrust::zip_iterator<thrust::tuple<Iterators...>>
MakeZipIterator(Iterators... itrs) {
  return thrust::make_zip_iterator(thrust::make_tuple(itrs...));
}

template <typename T>
void ProjectToSE3Async(const T *data, T *se3, int_t N, cudaStream_t stream = 0);

template <typename InputIterator, typename OutputIterator,
          typename BinaryReduction, typename OutputType>
void ReduceAsync(void *temp, size_t &temp_bytes, InputIterator input, int_t N,
                 OutputIterator output, OutputType init,
                 BinaryReduction binary_reduction, cudaStream_t stream = 0) {
  if (temp == nullptr) {
    cub::DeviceReduce::Reduce(nullptr, temp_bytes, input, output, N,
                              binary_reduction, init, stream);
  } else {
    cub::DeviceReduce::Reduce(temp, temp_bytes, input, output, N,
                              binary_reduction, init, stream);
  }
}

template <typename InputIterator, typename Function>
void ForEachAsync(InputIterator input, int_t N, Function op,
                  cudaStream_t stream) {
  thrust::async::for_each_detail::for_each_fn::call(
      thrust::cuda::par.on(stream), input, input + N, op);
}

template <typename InputIterator, typename UnaryFunction,
          typename OutputIterator, typename BinaryFunction, typename OutputType>
void TransformReduceAsync(void *temp, size_t &temp_bytes, InputIterator input,
                          int_t N, UnaryFunction unary_op,
                          OutputIterator output, OutputType init,
                          BinaryFunction binary_op, cudaStream_t stream) {
  auto input_it =
      sfm::utils::MakeTransformIterator<OutputType>(input, unary_op);
  // using Input =
  //     sfm::TransformInputIterator<OutputType, InputIterator, UnaryFunction>;
  ReduceAsync(temp, temp_bytes, input_it, N, output, init, binary_op);
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryFunction1,
          typename BinaryFunction2, typename OutputType>
void TransformReduceAsync(void *temp, size_t &temp_bytes, InputIterator1 input1,
                          int_t N, InputIterator2 input2,
                          BinaryFunction1 binary_op1, OutputIterator output,
                          OutputType init, BinaryFunction2 binary_op2,
                          cudaStream_t stream) {

  using BinaryInput =
      sfm::BinaryTransformInputIterator<OutputType, InputIterator1,
                                        InputIterator2, BinaryFunction1>;
  ReduceAsync(temp, temp_bytes, BinaryInput(input1, input2, binary_op1), N,
              output, init, binary_op2, stream);
}

template <typename IndexIterator, typename InputIterator,
          typename OutputIterator, typename BinaryReduction,
          typename OutputType>
void ReduceByIndexAsync(void *temp, size_t &temp_bytes, IndexIterator index,
                        int_t N, InputIterator input, OutputIterator output,
                        OutputType init, BinaryReduction binary_reduction,
                        cudaStream_t stream = 0) {
  TransformReduceAsync(
      temp, temp_bytes, index, N,
      [input] __device__(auto n) -> OutputType { return input[n]; }, output,
      init, binary_reduction, stream);
}

template <typename T>
void InnerProductAsync(void *temp, size_t &temp_bytes, const T *input1, int_t N,
                       const T *input2, T *output, cudaStream_t stream);

template <typename T>
void SquaredNormAsync(void *temp, size_t &temp_bytes, const T *input, int_t N,
                      T *output, cudaStream_t stream);

template <typename T>
void DifferenceSquaredNormAsync(void *temp, size_t &temp_bytes, const T *input1,
                                int_t N, const T *input2, T *output,
                                cudaStream_t stream);

template <typename T>
void MatrixDiagonalWeightedSquaredNormAsync(void *temp, size_t &temp_bytes,
                                            const T *v, int_t N, int_t size,
                                            const T *matrix, T *output,
                                            cudaStream_t stream = 0);

template <typename T, typename Operator>
void MatrixTransformAsync(const T *input, T *output, int_t rows, int_t cols,
                          Operator op, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  MatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(input, output, op,
                                                              N);
}

template <typename T, typename BinaryOperator>
void MatrixTransformAsync(const T *input0, const T *input1, T *output,
                          int_t rows, int_t cols, BinaryOperator op, int_t N,
                          cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  DictedMatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(
      input0, input1, output, op, N);
}

template <typename T, typename Operator>
void DictedMatrixTransformAsync(const T *input, int_t input_cnt,
                                const int_t *output_dicts, T *output,
                                int_t output_cnt, int_t rows, int_t cols,
                                Operator op, int_t N, cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  DictedMatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(
      input, input_cnt, output_dicts, output, output_cnt, op, N);
}

template <typename T, typename Operator>
void DictedMatrixTransformAsync(const int_t *input_dicts, const T *input,
                                int_t input_cnt, const int_t *output_dicts,
                                T *output, int_t output_cnt, int_t rows,
                                int_t cols, Operator op, int_t N,
                                cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  DictedMatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(
      input_dicts, input, input_cnt, output_dicts, output, output_cnt, op, N);
}

template <typename T, typename BinaryOperator>
void DictedMatrixTransformAsync(const T *input0, int_t input_cnt0,
                                const int_t *input_dicts1, const T *input1,
                                int_t input_cnt1, const int_t *output_dicts,
                                T *output, int_t output_cnt, int_t rows,
                                int_t cols, BinaryOperator op, int_t N,
                                cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  DictedMatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(
      input0, input_cnt0, input_dicts1, input1, input_cnt1, output_dicts,
      output, output_cnt, op, N);
}

template <typename T, typename BinaryOperator>
void DictedMatrixTransformAsync(const int_t *input_dicts0, const T *input0,
                                int_t input_cnt0, const int_t *input_dicts1,
                                const T *input1, int_t input_cnt1,
                                const int_t *output_dicts, T *output,
                                int_t output_cnt, int_t rows, int_t cols,
                                BinaryOperator op, int_t N,
                                cudaStream_t stream) {
  if (N <= 0) {
    return;
  }

  dim3 grid_size, block_size;
  block_size.x = std::min((int_t)32, N);
  block_size.y = rows * cols;

  int_t num_blocks = (N + block_size.x - 1) / block_size.x;

  DictedMatrixTransformKernel<<<grid_size, block_size, 0, stream>>>(
      input_dicts0, input0, input_cnt0, input_dicts1, input1, input_cnt1,
      output_dicts, output, output_cnt, op, N);
}

template <typename T>
void PlusAsync(T a, const T *x, T b, const T *y, T *result, int_t N,
               cudaStream_t stream = 0);

template <typename T>
void NesterovExtrapolateSE3Async(const T *x0, const T *x1, T beta, T *y,
                                 int_t N, cudaStream_t stream = 0);

template <typename T>
void NesterovExtrapolateMatrixAsync(const T *x0, const T *x1, T beta, T *y,
                                    int_t rows, int_t cols, int_t N,
                                    cudaStream_t stream = 0);

template <typename T>
void NesterovExtrapolateDictedSE3Async(const int_t *dicts, const T *x0,
                                       const T *x1, T beta, T *y, int_t x_size,
                                       int_t y_size, int_t N,
                                       cudaStream_t stream = 0);

template <typename T>
void NesterovExtrapolateDictedMatrixAsync(const int_t *dicts, const T *x0,
                                          const T *x1, T beta, T *y, int_t rows,
                                          int_t cols, int_t x_size,
                                          int_t y_size, int_t N,
                                          cudaStream_t stream = 0);

template <typename T>
void SetSymmetricMatrixAsync(T *matrix, int_t size, int_t num_matrices,
                             cudaStream_t stream = 0);

template <typename T>
void SetSymmetricMatrixAsync(T alpha, T beta, T *matrix, int_t size,
                             int_t num_matrices, cudaStream_t stream = 0);

template <typename T>
void RescaleSymmetricMatrixDiagonalAsync(T ratio, T *matrix, int_t size,
                                         int_t num_matrices,
                                         cudaStream_t stream = 0);

template <typename T>
void ComputeMatrixInverseAsync(const T *matrix, T *matrix_inverse, int_t size,
                               int_t N, cudaStream_t stream = 0);

template <typename T>
void ComputePositiveDefiniteMatrixInverseAsync(const T *matrix,
                                               T *matrix_inverse, int_t size,
                                               int_t N,
                                               cudaStream_t stream = 0);

template <typename T>
void ComputePositiveDefiniteMatrixInverseAsync(T alpha, T beta, const T *matrix,
                                               T *matrix_inverse, int_t size,
                                               int_t N,
                                               cudaStream_t stream = 0);

template <typename T>
void ComputeMatrixVectorMultiplicationAsync(T alpha, const T *matrix,
                                            const T *x, T beta, T *y,
                                            int_t size, int_t N,
                                            cudaStream_t stream = 0);

template <typename T>
void ComputeSE3RetractionAsync(const T *X, T stepsize, const T *dX, T *Xplus,
                               int_t N, cudaStream_t stream = 0);

template <typename T>
void CopyFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows,
                                      int_t cols, const T *src, T *dst,
                                      int_t src_num_matrices,
                                      int_t dst_num_matrices, int_t N,
                                      cudaStream_t stream = 0);

template <typename T>
void CopyToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                    const T *src, T *dst,
                                    int_t src_num_matrices,
                                    int_t dst_num_matrices, int_t N,
                                    cudaStream_t stream = 0);

template <typename T>
void CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
    const int_t *src_dicts, const int_t *dst_dicts, int_t rows, int_t cols,
    const T *src, T *dst, int_t src_num_matrices, int_t dst_num_matrices,
    int_t N, cudaStream_t stream = 0);

template <typename T>
void AddFromDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                     const T *src, T *dst,
                                     int_t src_num_matrices,
                                     int_t dst_num_matrices, int_t N,
                                     cudaStream_t stream = 0);

template <typename T>
void AddToDictedMatrixOfArrayAsync(const int_t *dicts, int_t rows, int_t cols,
                                   const T *src, T *dst, int_t src_num_matrices,
                                   int_t dst_num_matrices, int_t N,
                                   cudaStream_t stream = 0);

template <typename T>
void ApproximateAsync(const T *src, T *dst, int_t N, cudaStream_t stream = 0);

void SetIndexFromDictAsync(const int_t *dicts, int_t *indices, int_t N,
                           cudaStream_t stream = 0);

void ResetIndexFromDictAsync(const int_t *dicts, int_t *indices, int_t N,
                             cudaStream_t stream = 0);
} // namespace utils
} // namespace sfm