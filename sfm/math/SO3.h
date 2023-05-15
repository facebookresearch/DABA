// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cuda_runtime_api.h>

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b) { return a < b ? a : b; }

inline float fmaxf(float a, float b) { return a > b ? a : b; }

inline double fminf(double a, double b) { return a < b ? a : b; }

inline double fmaxf(double a, double b) { return a > b ? a : b; }

inline int max(int a, int b) { return a > b ? a : b; }

inline int min(int a, int b) { return a < b ? a : b; }

inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }

inline float rsqrtf(double x) { return 1.0 / sqrtf(x); }
#endif

namespace sfm {
namespace math {
struct SO3 {
  template <typename T>
  inline static __host__ __device__ void Exp(const Eigen::Vector3<T> &v,
                                             Eigen::Matrix3<T> &ret) {
    T theta2 = v.squaredNorm();
    T theta = sqrt(theta2);

    T cosine = theta < (T)1e-6 ? 8 / (4 + theta2) - 1 : cos(theta);
    T sine_by_theta =
        theta < (T)1e-6 ? (T)0.5 * cosine + (T)0.5 : sin(theta) / theta;
    T one_minus_cosine_by_theta2 =
        theta < (T)1e-6 ? (T)0.5 * sine_by_theta : (1 - cosine) / theta2;

    ret.noalias() = one_minus_cosine_by_theta2 * v * v.transpose();
    ret(0, 0) += cosine;
    ret(1, 1) += cosine;
    ret(2, 2) += cosine;

    T s = 0;
    s = sine_by_theta * v[2];
    ret(0, 1) -= s;
    ret(1, 0) += s;
    s = sine_by_theta * v[1];
    ret(0, 2) += s;
    ret(2, 0) -= s;
    s = sine_by_theta * v[0];
    ret(1, 2) -= s;
    ret(2, 1) += s;
  }

  template <typename T>
  inline static __host__ __device__ void Log(const Eigen::Matrix3<T> &R,
                                             Eigen::Vector3<T> &ret) {
    ret[0] = R(2, 1) - R(1, 2);
    ret[1] = R(0, 2) - R(2, 0);
    ret[2] = R(1, 0) - R(0, 1);

    T two_cosine = R.diagonal().sum() - 1;
    T two_sine = ret.norm();
    T theta = atan2(two_sine, two_cosine);
    T scale = 0;

    if (two_cosine >= (T)-1.9999) {
      scale = 1 / two_sine;
    } else {
      unsigned char major = R(1, 1) > R(0, 0);
      major = R(2, 2) > R(major, major) ? 2 : major;

      scale = ret[major] > 0 ? 1 : -1;

      ret[0] = R(major, 0) + R(0, major);
      ret[1] = R(major, 1) + R(1, major);
      ret[2] = R(major, 2) + R(2, major);
      ret[major] -= two_cosine;

      scale *= rsqrtf(ret.squaredNorm());
    }

    scale = theta < 1e-5 ? T(0.5) + theta * theta / 12 : theta * scale;
    ret *= scale;
  }

  template <typename T>
  inline static __host__ __device__ void Retract(Eigen::Vector3<T> v,
                                                 Eigen::Matrix3<T> &ret) {
    T theta2 = v.squaredNorm();
    T theta4 = theta2 * theta2;
    T theta6 = theta4 * theta2;
    T theta8 = theta4 * theta4;

    T scale = 1 + theta2 / 12 + theta4 / 120 + theta6 / (T)(1185.882352941177) +
              theta8 / (T)(11705.80645161290);

    theta2 *= scale * scale;

    v *= scale;

    T a = 2 / (4 + theta2);
    ret.noalias() = a * v * v.transpose();

    T b = 2 * a;
    T s = 0;
    s = b * v[2];
    ret(0, 1) -= s;
    ret(1, 0) += s;
    s = b * v[1];
    ret(0, 2) += s;
    ret(2, 0) -= s;
    s = b * v[0];
    ret(1, 2) -= s;
    ret(2, 1) += s;

    T c = 2 * b - 1;
    ret(0, 0) += c;
    ret(1, 1) += c;
    ret(2, 2) += c;
  }

  template <typename T>
  inline static __host__ __device__ void Dexp(const Eigen::Vector3<T> &v,
                                              Eigen::Matrix3<T> &ret) {
    T theta2 = v.squaredNorm();
    T theta = sqrt(theta2);
    T theta3 = theta2 * theta;

    T cosine = theta < (T)1e-6 ? 8 / (4 + theta2) - 1 : cos(theta);
    T sine = sin(theta);
    T sine_by_theta = theta < (T)1e-6 ? (T)0.5 * cosine + (T)0.5 : sine / theta;
    T one_minus_cosine_by_theta2 =
        theta < (T)1e-6 ? (T)0.5 * sine_by_theta : (1 - cosine) / theta2;
    T theta_minus_sine_by_theta3 =
        theta < (T)1e-6 ? 0 : (theta - sine) / theta3;

    ret.noalias() = theta_minus_sine_by_theta3 * v * v.transpose();
    ret(0, 0) += sine_by_theta;
    ret(1, 1) += sine_by_theta;
    ret(2, 2) += sine_by_theta;

    T s = 0;
    s = one_minus_cosine_by_theta2 * v[2];
    ret(0, 1) -= s;
    ret(1, 0) += s;
    s = one_minus_cosine_by_theta2 * v[1];
    ret(0, 2) += s;
    ret(2, 0) -= s;
    s = one_minus_cosine_by_theta2 * v[0];
    ret(1, 2) -= s;
    ret(2, 1) += s;
  }

  template <typename T>
  static __host__ __device__ void Local(const Eigen::Matrix3<T> &R,
                                        Eigen::Vector3<T> &ret) {}

  template <typename T>
  static __host__ __device__ void Fexp(const Eigen::Vector3<T> &v,
                                       Eigen::Matrix3<T> &ret) {
    T v_norm = v.norm();
    T signed_v_norm = v_norm - round(v_norm / 8) * 8;
    T rescale = signed_v_norm > 0 ? 2 - fabs(2 - signed_v_norm)
                                  : fabs(2 + signed_v_norm) - 2;
    T rescale_squared = rescale * rescale;

    Eigen::Vector3<T> rescaled_v = v_norm < 2 ? v : v / v_norm * rescale;
    ret.noalias() = 0.5 * rescaled_v * rescaled_v.transpose();

    T a = sqrtf(1 - 0.25 * rescale_squared);
    T s = 0;
    s = a * rescaled_v[2];
    ret(0, 1) -= s;
    ret(1, 0) += s;
    s = a * rescaled_v[1];
    ret(0, 2) += s;
    ret(2, 0) -= s;
    s = a * rescaled_v[0];
    ret(1, 2) -= s;
    ret(2, 1) += s;

    ret.diagonal().array() += 1 - 0.5 * rescale_squared;
  }

  template <typename T>
  static __host__ __device__ void Flog(const Eigen::Matrix3<T> &R,
                                       Eigen::Vector3<T> &ret) {
    T squared_norm = 3 - R.trace();
    bool mode = squared_norm < (T)1.999;

    char major = R(1, 1) > R(0, 0);
    major = R(2, 2) > R(major, major) ? 2 : major;

    ret[0] = R(2, 1) - R(1, 2);
    ret[1] = R(0, 2) - R(2, 0);
    ret[2] = R(1, 0) - R(0, 1);

    T scale = (mode || ret[major] >= 0) ? 1 : -1;

    ret[0] = mode ? ret[0] : R(major, 0) + R(0, major);
    ret[1] = mode ? ret[1] : R(major, 1) + R(1, major);
    ret[2] = mode ? ret[2] : R(major, 2) + R(2, major);
    ret[major] += mode ? 0 : squared_norm - 2;

    scale *= rsqrtf(mode ? 4 - squared_norm : ret[major]);
    ret *= scale;
  }
};
} // namespace math
} // namespace sfm