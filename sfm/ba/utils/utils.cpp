// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <fstream>
#include <iomanip>
#include <sstream>

#include <sfm/ba/dataset.h>
#include <sfm/ba/types.h>
#include <sfm/ba/utils/utils.h>
#include <sfm/math/SO3.h>

#include <Eigen/Geometry>

#include <glog/logging.h>

namespace sfm {
namespace ba {
int BundlerDatasetToBALDataset(std::string bundler_file, std::string bal_file,
                               int_t &num_cameras, int_t &num_points,
                               int_t &num_measurements) {
  using T = double;
  std::ifstream infile(bundler_file);

  if (infile.is_open() != true) {
    LOG(ERROR) << "Can not open " << bundler_file << std::endl;
    exit(-1);
  }

  std::vector<Eigen::Vector<T, 9>> cameras;
  std::vector<Eigen::Matrix<T, 3, 3>> rotations;
  std::vector<Eigen::Vector<T, 3>> points;
  std::vector<BADataset<T>::Measurement> measurements;

  num_cameras = 0;
  num_points = 0;
  num_measurements = 0;

  std::string line;
  std::getline(infile, line);

  std::getline(infile, line);
  std::stringstream sstream(line);

  sstream >> num_cameras;
  sstream >> num_points;

  cameras.resize(num_cameras);
  rotations.resize(num_cameras);

  for (int_t n = 0; n < num_cameras; n++) {
    Eigen::Vector<T, 3> &angle_axis =
        *(Eigen::Vector<T, 3> *)(cameras[n].data());
    Eigen::Vector<T, 3> &t = *(Eigen::Vector<T, 3> *)(cameras[n].data() + 3);
    Eigen::Vector<T, 3> &intrinsics =
        *(Eigen::Vector<T, 3> *)(cameras[n].data() + 6);
    auto &R = rotations[n];

    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    for (int_t i = 0; i < 3; i++) {
      sstream >> intrinsics[i];
    }

    for (int_t i = 0; i < 3; i++) {
      std::getline(infile, line);
      sstream.clear();
      sstream.str(line);

      for (int_t j = 0; j < 3; j++) {
        sstream >> R(i, j);
      }
    }

    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    for (int_t i = 0; i < 3; i++) {
      sstream >> t[i];
    }

    math::SO3::Log(R, angle_axis);
  }

  points.resize(num_points);
  std::vector<std::vector<int>> camera_stats(num_cameras);
  for (int_t n = 0; n < num_points; n++) {
    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    for (int_t i = 0; i < 3; i++) {
      sstream >> points[n][i];
    }

    std::getline(infile, line);

    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    int_t num_measurements;
    sstream >> num_measurements;

    for (int_t m = 0; m < num_measurements; m++) {
      typename BADataset<T>::Measurement measurement;
      int_t camera_index, sift_key;

      sstream >> camera_index;
      sstream >> sift_key;
      sstream >> measurement.measurement[0] >> measurement.measurement[1];

      measurement.extrinsics_index = camera_index;
      measurement.intrinsics_index = camera_index;
      measurement.point_index = n;
      measurement.sqrt_weight = 1;

      camera_stats[camera_index].push_back(measurements.size());
      measurements.push_back(measurement);
    }
  }

  T squared_error = 0;

  for (int_t n = 0; n < num_cameras; n++) {
    const auto &R = rotations[n];
    const auto &t = cameras[n].segment<3>(3);
    Eigen::Vector3<T> dist;
    Eigen::Vector2<T> feature;
    Eigen::Vector3<T> radial;
    Eigen::Matrix3<T> H = Eigen::Matrix3<T>::Zero();
    Eigen::Vector3<T> g = Eigen::Vector3<T>::Zero();

    for (const auto &index : camera_stats[n]) {
      const auto &measurement = measurements[index];
      dist.noalias() = R * points[measurement.point_index] + t;
      feature = -dist.head<2>() / dist[2];
      radial[0] = 1;
      radial[1] = feature.squaredNorm();
      radial[2] = radial[1] * radial[1];

      Eigen::Matrix<T, 2, 3> J = feature * radial.transpose();
      H.noalias() += J.transpose() * J;
      g.noalias() += J.transpose() * measurement.measurement;

      squared_error +=
          (radial.dot(cameras[n].tail<3>()) * feature - measurement.measurement)
              .squaredNorm();
    }

    auto intrinsics = cameras[n].tail<3>();
    intrinsics = H.ldlt().solve(g);
    intrinsics.tail<2>() /= intrinsics[0];
  }

  infile.close();

  num_measurements = measurements.size();

  std::ofstream outfile(bal_file);

  if (outfile.is_open() != true) {
    LOG(ERROR) << "Can not open " << bal_file << std::endl;
    exit(-1);
  }

  outfile << num_cameras << " " << num_points << " " << num_measurements
          << std::endl;

  for (const auto &measurement : measurements) {
    outfile << std::scientific << std::setprecision(25)
            << measurement.extrinsics_index << " " << measurement.point_index
            << " " << measurement.measurement.transpose() << std::endl;
  }

  for (int_t n = 0; n < cameras.size(); n++) {
    outfile << std::scientific << std::setprecision(25) << cameras[n]
            << std::endl;
  }

  for (int_t n = 0; n < points.size(); n++) {
    outfile << std::scientific << std::setprecision(25) << points[n]
            << std::endl;
  }

  outfile.close();

  return 0;
}
} // namespace ba
} // namespace sfm