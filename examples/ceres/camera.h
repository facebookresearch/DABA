// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ceres/ceres.h>
#include <examples/ceres/math.h>
#include <examples/ceres/types.h>

namespace Ceres {
class Camera : public ceres::Manifold {
public:
  virtual int AmbientSize() const override { return 15; }

  virtual int TangentSize() const override { return 9; }

  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const override {
    Eigen::Map<const Matrix<3, 5>> X(x);
    Eigen::Map<const Matrix<3, 3>> dx(delta);
    Eigen::Map<Matrix<3, 5>> Xplus(x_plus_delta);
    Matrix<3, 3> dR;

    math::SO3::exp(dx.col(0), dR);
    Xplus.leftCols<3>().noalias() = dR * X.leftCols<3>();
    Xplus.rightCols<2>() = X.rightCols<2>() + dx.rightCols<2>();

    return true;
  }

  virtual bool PlusJacobian(const double *x, double *jacobian) const override {
    Eigen::Map<const Matrix<3, 5>> X(x);
    Eigen::Map<Eigen::Matrix<Scalar, 15, 9, Eigen::RowMajor>> jac(jacobian);
    jac.setZero();
    jac.bottomRightCorner<6, 6>().setIdentity();

    for (int i = 0; i < 3; i++) {
      auto jac_rot = jac.block<3, 3>(3 * i, 0);
      math::SO3::hat(X.col(i), jac_rot);
      jac_rot *= -1;
    }

    return true;
  }

  virtual bool Minus(const double *y, const double *x,
                     double *y_minus_x) const override {
    Eigen::Map<const Matrix<3, 5>> X(x);
    Eigen::Map<const Matrix<3, 5>> Y(y);
    Eigen::Map<Matrix<3, 3>> ret(y_minus_x);

    Matrix<3, 3> dZ;
    dZ.noalias() = Y.leftCols<3>() * X.leftCols<3>().transpose();
    auto drot = ret.col(0);
    math::SO3::log(dZ, drot);
    ret.rightCols<2>() = Y.rightCols<2>() - X.rightCols<2>();

    return true;
  }

  virtual bool MinusJacobian(const double *x, double *jacobian) const override {
    Eigen::Map<const Matrix<3, 5>> X(x);
    Eigen::Map<Eigen::Matrix<Scalar, 9, 15, Eigen::RowMajor>> jac(jacobian);
    jac.setZero();
    jac.bottomRightCorner<6, 6>().setIdentity();

    for (int i = 0; i < 3; i++) {
      auto jac_rot = jac.block<3, 3>(0, 3 * i);
      math::SO3::hat(X.col(i), jac_rot);
      jac_rot *= 0.5;
    }

    return true;
  }
};
} // namespace Ceres
