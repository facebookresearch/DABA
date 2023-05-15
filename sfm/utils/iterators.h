// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <thrust/system/cuda/detail/util.h>

namespace sfm {
template <class ValueType, class InputIt, class UnaryOp>
using TransformInputIterator =
    thrust::cuda_cub::transform_input_iterator_t<ValueType, InputIt, UnaryOp>;

template <class ValueType, class InputIt1, class InputIt2, class BinaryOp>
using BinaryTransformInputIterator =
    thrust::cuda_cub::transform_pair_of_input_iterators_t<ValueType, InputIt1,
                                                          InputIt2, BinaryOp>;
} // namespace sfm