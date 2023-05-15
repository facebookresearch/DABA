// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Dense>

namespace Ceres {
using Scalar = double;
template <int size>
using Vector = Eigen::Matrix<Scalar, size, 1, Eigen::ColMajor>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector6 = Vector<6>;
using VectorX = Vector<Eigen::Dynamic>;
template <int rows, int cols>
using Matrix = Eigen::Matrix<Scalar, rows, cols, Eigen::ColMajor>;
using Matrix3 = Matrix<3, 3>;
using Matrix6 = Matrix<6, 6>;
using MatrixX = Matrix<Eigen::Dynamic, Eigen::Dynamic>;
} // namespace ceres