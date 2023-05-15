// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace Ceres {
namespace math {
struct SO3 {
  template <typename T1, typename T2>
  static int exp(const Eigen::MatrixBase<T1> &w, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)
    using Scalar = typename T1::Scalar;

    static const double NEAR_ZERO_EPS = 5e-3;

    // Compute the rotation
    const Scalar theta = w.stableNorm();
    const Scalar theta2 = theta * theta;
    const Scalar theta3 = theta2 * theta;

    bool near_zero = theta < NEAR_ZERO_EPS;

    Scalar sine, cosine, sine_by_theta, one_minus_cosine_by_theta2;
    sine = sin(theta);
    cosine = cos(theta);
    if (near_zero) {
      cosine = (4.0 - theta2) / (4.0 + theta2);
      sine_by_theta = 0.5 * cosine + 0.5;
      one_minus_cosine_by_theta2 = 0.5 * sine_by_theta;
    } else {
      sine_by_theta = sine / theta;
      one_minus_cosine_by_theta2 = (1.0 - cosine) / theta2;
    }

    ret.setZero();
    ret.template leftCols<3>().noalias() =
        one_minus_cosine_by_theta2 * w * w.transpose();
    ret.diagonal().array() += cosine;

    Eigen::Vector<Scalar, 3> temp = sine_by_theta * w;
    ret(0, 1) -= temp[2];
    ret(1, 0) += temp[2];
    ret(0, 2) += temp[1];
    ret(2, 0) -= temp[1];
    ret(1, 2) -= temp[0];
    ret(2, 1) += temp[0];

    return 0;
  }

  template <typename T1, typename T2>
  static int dexp(const Eigen::MatrixBase<T1> &v, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)
    using Scalar = typename T1::Scalar;

    Scalar theta2 = v.squaredNorm();
    Scalar theta = sqrt(theta2);
    Scalar theta3 = theta2 * theta;

    Scalar cosine = theta < (Scalar)1e-6 ? 8 / (4 + theta2) - 1 : cos(theta);
    Scalar sine = sin(theta);
    Scalar sine_by_theta = theta < (Scalar)1e-6
                               ? (Scalar)0.5 * cosine + (Scalar)0.5
                               : sine / theta;
    Scalar one_minus_cosine_by_theta2 = theta < (Scalar)1e-6
                                            ? (Scalar)0.5 * sine_by_theta
                                            : (1 - cosine) / theta2;
    Scalar theta_minus_sine_by_theta3 =
        theta < (Scalar)1e-6 ? 0 : (theta - sine) / theta3;

    ret.noalias() = theta_minus_sine_by_theta3 * v * v.transpose();
    ret(0, 0) += sine_by_theta;
    ret(1, 1) += sine_by_theta;
    ret(2, 2) += sine_by_theta;

    Scalar s = 0;
    s = one_minus_cosine_by_theta2 * v[2];
    ret(0, 1) -= s;
    ret(1, 0) += s;
    s = one_minus_cosine_by_theta2 * v[1];
    ret(0, 2) += s;
    ret(2, 0) -= s;
    s = one_minus_cosine_by_theta2 * v[0];
    ret(1, 2) -= s;
    ret(2, 1) += s;

    return 0;
  }

  template <typename T1, typename T2>
  static int log(const Eigen::MatrixBase<T1> &R, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 3)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 3)
    using Scalar = typename T1::Scalar;

    static const double NEAR_PI_EPS = 1e-7;
    static const double NEAR_ZERO_EPS = 5e-3;

    Eigen::Vector<Scalar, 3> sine_axis{0.5 * (R(2, 1) - R(1, 2)),
                                       0.5 * (R(0, 2) - R(2, 0)),
                                       0.5 * (R(1, 0) - R(0, 1))};
    Scalar cosine = 0.5 * (R.trace() - 1);
    Scalar sine = sine_axis.stableNorm();
    Scalar theta = atan2(sine, cosine);
    Scalar theta2 = theta * theta;

    bool near_pi = 1.0 + cosine < NEAR_PI_EPS;
    bool near_zero = theta < NEAR_ZERO_EPS;

    if (near_pi) {
      Eigen::Vector<Scalar, 3> ddiag = R.diagonal();
      int major;
      if (ddiag[0] > ddiag[1] && ddiag[0] > ddiag[2]) {
        major = 0;
      } else if (ddiag[1] > ddiag[0] && ddiag[1] > ddiag[2]) {
        major = 1;
      } else {
        major = 2;
      }

      ret = 0.5 * (R.row(major) + R.col(major).transpose());
      ret[major] -= cosine;
      ret.normalize();
      ret *= sine_axis[major] > 0 ? theta : -theta;
    } else {
      Scalar theta_by_sine = near_zero ? 1.0 + sine * sine / 6.0 : theta / sine;
      ret = sine_axis * theta_by_sine;
    }

    return 0;
  }

  template <typename T1, typename T2>
  static int hat(const Eigen::MatrixBase<T1> &w, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)
    ret.setZero();
    ret(0, 1) = -w[2];
    ret(1, 0) = w[2];
    ret(0, 2) = w[1];
    ret(2, 0) = -w[1];
    ret(1, 2) = -w[0];
    ret(2, 1) = w[0];

    return 0;
  }
};

struct SE3 {
  template <typename T1, typename T2>
  static int exp(const Eigen::MatrixBase<T1> &xi, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 6)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 4)
    using Scalar = typename T1::Scalar;

    static const double NEAR_ZERO_EPS = 5e-3;

    // Compute the rotation
    auto w = xi.template tail<3>();

    const Scalar theta = w.stableNorm();
    const Scalar theta2 = theta * theta;
    const Scalar theta3 = theta2 * theta;

    bool near_zero = theta < NEAR_ZERO_EPS;

    Scalar sine, cosine, sine_by_theta, one_minus_cosine_by_theta2;
    sine = sin(theta);
    cosine = cos(theta);

    if (near_zero) {
      cosine = (4.0 - theta2) / (4.0 + theta2);
      sine_by_theta = 0.5 * cosine + 0.5;
      one_minus_cosine_by_theta2 = 0.5 * sine_by_theta;
    } else {
      sine_by_theta = sine / theta;
      one_minus_cosine_by_theta2 = (1.0 - cosine) / theta2;
    }

    ret.setZero();
    ret.template leftCols<3>().noalias() =
        one_minus_cosine_by_theta2 * w * w.transpose();
    ret.diagonal().array() += cosine;

    Eigen::Vector<Scalar, 3> temp = sine_by_theta * w;
    ret(0, 1) -= temp[2];
    ret(1, 0) += temp[2];
    ret(0, 2) += temp[1];
    ret(2, 0) -= temp[1];
    ret(1, 2) -= temp[0];
    ret(2, 1) += temp[0];

    // compute the translation
    auto v = xi.template head<3>();

    Scalar sine_by_theta_t = sine_by_theta;
    Scalar one_minus_cosine_by_theta2_t = one_minus_cosine_by_theta2;
    Scalar theta_minus_sine_by_theta3_t;

    if (near_zero) {
      sine_by_theta_t = 1.0 - theta2 / 6.0;
      one_minus_cosine_by_theta2_t = 0.5 - theta2 / 24.0;
      theta_minus_sine_by_theta3_t = 1.0 / 6 - theta2 / 120.0;
    } else {
      theta_minus_sine_by_theta3_t = (theta - sine) / theta3;
    }

    ret.col(3) = sine_by_theta_t * v;
    ret.col(3).noalias() += one_minus_cosine_by_theta2_t * w.cross(v);
    ret.col(3) += theta_minus_sine_by_theta3_t * w.dot(v) * w;

    return 0;
  }

  template <typename T1, typename T2>
  static int log(const Eigen::MatrixBase<T1> &g, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 4)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 6)
    using Scalar = typename T1::Scalar;

    static const double NEAR_PI_EPS = 1e-7;
    static const double NEAR_ZERO_EPS = 5e-3;

    const auto &R = g.template leftCols<3>();
    const auto &t = g.col(3);

    auto v = ret.template head<3>();
    auto w = ret.template tail<3>();

    Eigen::Vector<Scalar, 3> sine_axis{0.5 * (g(2, 1) - g(1, 2)),
                                       0.5 * (g(0, 2) - g(2, 0)),
                                       0.5 * (g(1, 0) - g(0, 1))};
    Scalar cosine = 0.5 * (R.trace() - 1.0);
    Scalar sine = sine_axis.stableNorm();
    Scalar theta = atan2(sine, cosine);
    Scalar theta2 = theta * theta;
    Scalar theta_by_sine;

    bool near_pi = 1.0 + cosine < NEAR_PI_EPS;
    bool near_zero = theta < NEAR_ZERO_EPS;

    if (near_pi) {
      Eigen::Vector<Scalar, 3> ddiag = R.diagonal();
      int major;
      if (ddiag[0] > ddiag[1] && ddiag[0] > ddiag[2]) {
        major = 0;
      } else if (ddiag[1] > ddiag[0] && ddiag[1] > ddiag[2]) {
        major = 1;
      } else {
        major = 2;
      }

      w = 0.5 * (g.template block<3, 1>(0, major) +
                 g.template block<1, 3>(major, 0).transpose());
      w[major] -= cosine;
      w.normalize();
      w *= sine_axis[major] > 0 ? theta : -theta;
    } else {
      theta_by_sine = near_zero ? 1.0 + sine * sine / 6.0 : theta / sine;
      w = sine_axis * theta_by_sine;
    }

    Scalar a, b;

    if (near_zero) {
      a = 1.0 - theta2 / 12.0;
      b = 1.0 / 12 + theta2 / 720.0;
    } else {
      Scalar sine_theta = sine * theta;
      Scalar two_cosine_minus_two = 2.0 * cosine - 2.0;

      a = -sine_theta / two_cosine_minus_two;
      b = (sine_theta + two_cosine_minus_two) / (theta2 * two_cosine_minus_two);
    }

    v = a * t;
    v -= 0.5 * w.cross(t);
    v.noalias() += b * w.dot(t) * w;

    return 0;
  }

  template <typename T1, typename T2>
  static int dlog(const Eigen::MatrixBase<T1> &xi, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 6)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 6, 6)
    using Scalar = typename T1::Scalar;

    static const double NEAR_ZERO_EPS = 5e-3;

    ret.setZero();

    auto v = xi.template head<3>();
    auto w = xi.template tail<3>();

    Scalar theta = w.stableNorm();
    Scalar theta2 = theta * theta;

    Scalar sine, cosine;
    sincos(theta, &sine, &cosine);

    bool near_zero = theta < NEAR_ZERO_EPS;

    Scalar a, b, c, d, e;

    if (near_zero) {
      a = 1 - theta2 / 12;
      b = 1.0 / 12 + theta2 / 720;
      c = -1.0 / 360 - theta2 / 7560;
      d = -1.0 / 6 - theta2 / 180;
    } else {
      Scalar sine_theta = sine * theta;
      Scalar two_cosine_minus_two = 2 * cosine - 2;

      a = -sine_theta / two_cosine_minus_two;
      b = (sine_theta + two_cosine_minus_two) / (theta2 * two_cosine_minus_two);
      c = -(2 * two_cosine_minus_two + theta * sine + theta2) /
          (theta2 * theta2 * two_cosine_minus_two);
      d = (theta - sine) / (theta * two_cosine_minus_two);
    }

    e = w.dot(v);

    ret.template block<3, 3>(0, 0).noalias() = b * w * w.transpose();
    ret(0, 1) -= 0.5 * w[2];
    ret(1, 0) += 0.5 * w[2];
    ret(0, 2) += 0.5 * w[1];
    ret(2, 0) -= 0.5 * w[1];
    ret(1, 2) -= 0.5 * w[0];
    ret(2, 1) += 0.5 * w[0];
    ret.template block<3, 3>(0, 0).diagonal().array() += a;

    ret.template block<3, 3>(3, 3) = ret.template block<3, 3>(0, 0);

    ret.template block<3, 3>(0, 3).noalias() = c * e * w * w.transpose();
    ret.template block<3, 3>(0, 3).noalias() += b * w * v.transpose();
    ret.template block<3, 3>(0, 3).noalias() += b * v * w.transpose();
    ret.template block<3, 3>(0, 3).diagonal().array() += d * e;
    ret(0, 4) -= 0.5 * v[2];
    ret(1, 3) += 0.5 * v[2];
    ret(0, 5) += 0.5 * v[1];
    ret(2, 3) -= 0.5 * v[1];
    ret(1, 5) -= 0.5 * v[0];
    ret(2, 4) += 0.5 * v[0];

    return 0;
  }

  template <typename T1, typename T2, typename T3>
  static int compose(const Eigen::MatrixBase<T1> &g0,
                     const Eigen::MatrixBase<T2> &g1,
                     Eigen::MatrixBase<T3> &ret) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 4)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 4)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T3, 3, 4)

    ret.template leftCols<3>().noalias() =
        g0.template leftCols<3>() * g1.template leftCols<3>();
    ret.col(3) = g0.col(3);
    ret.col(3).noalias() += g0.template leftCols<3>() * g1.col(3);

    return 0;
  }

  template <typename T1, typename T2>
  static int inverse(const Eigen::MatrixBase<T1> &g,
                     Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 4)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 4)

    ret.template leftCols<3, 3>(0, 0) = g.template leftCols<3>().transpose();
    ret.col(3).noalias() = -g.template leftCols<3>() * g.col(3);

    return 0;
  }

  template <typename T1, typename T2>
  static int hat(const Eigen::MatrixBase<T1> &xi, Eigen::MatrixBase<T2> &ret) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 6)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 4)
    auto ret_rot = ret.template leftCols<3>();
    SO3::hat(xi.template tail<3>(), ret_rot);
    ret.col(3) = xi.template head<3>();

    return 0;
  }
};
} // namespace math
} // namespace Ceres