// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/types.h>
#include <vector>

namespace sfm {
namespace ba {
template <typename T> class BADataset {
public:
  struct Measurement {
    Eigen::Vector2<T> measurement;

    T sqrt_weight;

    int_t extrinsics_index;
    int_t intrinsics_index;
    int_t point_index;
  };

  BADataset(const std::string &filename);

  int_t NumberOfExtrinsics() const;
  int_t NumberOfIntrinsics() const;
  int_t NumberOfPoints() const;

  const std::vector<Eigen::Matrix<T, 3, 4>> &Extrinsics() const;
  const std::vector<Eigen::Vector3<T>> &Intrinsics() const;
  const std::vector<Eigen::Vector3<T>> &Points() const;
  const std::vector<Measurement> &Measurements() const;

  virtual void Read(const std::string &filename, bool verbose = false) = 0;

protected:
  std::vector<Measurement> m_measurements;
  int_t m_num_extrinsics;
  int_t m_num_intrinsics;
  int_t m_num_points;
  std::vector<Eigen::Matrix<T, 3, 4>> m_extrinsics;
  std::vector<Eigen::Vector3<T>> m_intrinsics;
  std::vector<Eigen::Vector3<T>> m_points;
};

template <typename T> class BALDataset : public BADataset<T> {
public:
  BALDataset(const std::string &filename, bool verbose = false);
  virtual void Read(const std::string &filename, bool verbose = false) override;

protected:
  using BADataset<T>::m_measurements;
  using BADataset<T>::m_num_extrinsics;
  using BADataset<T>::m_num_intrinsics;
  using BADataset<T>::m_num_points;
  using BADataset<T>::m_extrinsics;
  using BADataset<T>::m_intrinsics;
  using BADataset<T>::m_points;
};

template <typename T> class BundlerDataset : public BADataset<T> {
public:
  BundlerDataset(const std::string &filename);
  virtual void Read(const std::string &filename, bool verbose = false) override;

protected:
  using BADataset<T>::m_measurements;
  using BADataset<T>::m_extrinsics;
  using BADataset<T>::m_intrinsics;
  using BADataset<T>::m_points;
};

template <typename T> class DBADataset {
public:
  struct Measurement {
    Eigen::Vector2<T> measurement;

    T sqrt_weight;

    std::array<int_t, 2> extrinsic_index;
    std::array<int_t, 2> intrinsic_index;
    std::array<int_t, 2> point_index;
  };

  DBADataset(const std::shared_ptr<BADataset<T>> dataset,
             int_t targeted_num_clusters, T initial_resolution = 1,
             T refined_resolution = 2, bool memory_efficient = false);

  void Setup(const std::shared_ptr<BADataset<T>> dataset,
             int_t targeted_num_clusters, T initial_resolution,
             T refined_resolution, bool memory_efficient);

  int_t NumberOfClusters() const;
  const std::vector<std::vector<Measurement>> &Measurements() const;
  const std::vector<std::vector<int_t>> &MeasurementIndices() const;
  const std::vector<std::array<int_t, 2>> &ExtrinsicsIndices() const;
  const std::vector<std::array<int_t, 2>> &IntrinsicsIndices() const;
  const std::vector<std::array<int_t, 2>> &PointIndices() const;
  const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &Extrinsics() const;
  const std::vector<std::vector<Eigen::Vector3<T>>> &Intrinsics() const;
  const std::vector<std::vector<Eigen::Vector3<T>>> &Points() const;

protected:
  int_t m_num_clusters;

  std::vector<std::vector<int_t>> m_measurement_indices;
  std::vector<std::vector<Measurement>> m_measurements;
  std::vector<std::array<int, 2>> m_extrinsics_indices;
  std::vector<std::array<int, 2>> m_intrinsics_indices;
  std::vector<std::array<int, 2>> m_point_indices;
  std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> m_extrinsics;
  std::vector<std::vector<Eigen::Vector3<T>>> m_intrinsics;
  std::vector<std::vector<Eigen::Vector3<T>>> m_points;
};
} // namespace ba
} // namespace sfm