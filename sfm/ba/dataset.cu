// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>

#include <sfm/ba/clustering/clustering.cuh>
#include <sfm/ba/clustering/common_methods.cuh>
#include <sfm/ba/dataset.h>
#include <sfm/ba/types.h>
#include <sfm/ba/utils/utils.h>
#include <sfm/math/SO3.h>
#include <sfm/types.h>
#include <sfm/utils/utils.cuh>

#include <thrust/device_vector.h>

#include <Eigen/Geometry>

#include <glog/logging.h>

namespace sfm {
namespace ba {
template <typename T> BADataset<T>::BADataset(const std::string &filename) {}

template <typename T>
const std::vector<Eigen::Matrix<T, 3, 4>> &BADataset<T>::Extrinsics() const {
  return m_extrinsics;
}

template <typename T>
const std::vector<Eigen::Vector3<T>> &BADataset<T>::Intrinsics() const {
  return m_intrinsics;
}

template <typename T>
const std::vector<Eigen::Vector3<T>> &BADataset<T>::Points() const {
  return m_points;
}

template <typename T>
const std::vector<typename BADataset<T>::Measurement> &
BADataset<T>::Measurements() const {
  return m_measurements;
}

template <typename T> int_t BADataset<T>::NumberOfExtrinsics() const {
  return m_num_extrinsics;
}

template <typename T> int_t BADataset<T>::NumberOfIntrinsics() const {
  return m_num_intrinsics;
}

template <typename T> int_t BADataset<T>::NumberOfPoints() const {
  return m_num_points;
}

template <typename T>
BALDataset<T>::BALDataset(const std::string &filename, bool verbose)
    : BADataset<T>(filename) {
  this->Read(filename, verbose);
}

template <typename T>
void BALDataset<T>::Read(const std::string &filename, bool verbose) {
  auto pfile = fopen(filename.c_str(), "r");

  if (pfile == nullptr) {
    LOG(ERROR) << "Can not open " << filename << std::endl;
    exit(-1);
  }

  m_extrinsics.clear();
  m_intrinsics.clear();
  m_points.clear();
  m_measurements.clear();

  int_t num_cameras = 0, num_points = 0, num_measurements = 0;
  fscanf(pfile, "%d %d %d", &num_cameras, &num_points, &num_measurements);
  m_num_extrinsics = num_cameras;
  m_num_intrinsics = num_cameras;
  m_num_points = num_points;

  m_measurements.reserve(num_measurements);
  m_extrinsics.resize(num_cameras);
  m_intrinsics.resize(num_cameras);

  for (int_t m = 0; m < num_measurements; m++) {
    if (verbose) {
      if (m % 1000 == 0) {
        printf("Load measurements %d of %d...       \r", m, num_measurements);
        fflush(stdout);
      }
    }
    typename BADataset<T>::Measurement measurement;
    double px, py;
    fscanf(pfile, "%d %d %lf %lf", &measurement.extrinsics_index,
           &measurement.point_index, &px, &py);
    measurement.measurement[0] = px;
    measurement.measurement[1] = py;
    measurement.intrinsics_index = measurement.extrinsics_index;
    measurement.sqrt_weight = 1;
    m_measurements.push_back(measurement);
  }

  if (verbose) {
    printf("Load measurements %d of %d...       \n", num_measurements,
           num_measurements);
  }

  for (int_t n = 0; n < num_cameras; n++) {
    if (verbose) {
      if (n % 50 == 0) {
        printf("Load cameras %d of %d...       \r", n, num_cameras);
        fflush(stdout);
      }
    }
    Eigen::Vector<double, 9> camera_raw_data;

    fscanf(pfile, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &camera_raw_data[0],
           &camera_raw_data[1], &camera_raw_data[2], &camera_raw_data[3],
           &camera_raw_data[4], &camera_raw_data[5], &camera_raw_data[6],
           &camera_raw_data[7], &camera_raw_data[8]);
    Eigen::Vector<T, 9> camera_data = camera_raw_data.cast<T>();

    Eigen::Matrix3<T> rotation;
    sfm::math::SO3::Exp(Eigen::Vector3<T>(camera_data.template head<3>()),
                        rotation);
    m_extrinsics[n].template leftCols<3>() = rotation.transpose();
    m_extrinsics[n].col(3).noalias() =
        rotation.transpose() * camera_data.template segment<3>(3);
    m_intrinsics[n] = camera_data.template tail<3>();
  }

  if (verbose) {
    printf("Load cameras %d of %d...       \n", num_cameras, num_cameras);
  }

  m_points.resize(num_points);

  for (int_t n = 0; n < num_points; n++) {
    if (verbose) {
      if (n % 250 == 0) {
        printf("Load points %d of %d...       \r", n, num_points);
        fflush(stdout);
      }
    }
    Eigen::Vector<double, 3> point_raw_data;

    fscanf(pfile, "%lf %lf %lf", &point_raw_data[0], &point_raw_data[1],
           &point_raw_data[2]);

    m_points[n] = -point_raw_data.cast<T>();
  }

  if (verbose) {
    printf("Load points %d of %d...       \n", num_points, num_points);
  }

  for (auto &measurement : m_measurements) {
    const auto &intrinsic = m_intrinsics[measurement.intrinsics_index];
    T rescale = intrinsic[0];
    measurement.measurement /= -rescale;
    measurement.sqrt_weight =
        rescale * std::sqrt(measurement.measurement.squaredNorm() + 1);
  }

  for (auto &intrinsic : m_intrinsics) {
    T rescale = intrinsic[0];
    intrinsic[1] = intrinsic[1];
    intrinsic[2] = intrinsic[2];
    intrinsic[1] *= rescale * rescale;
    intrinsic[2] *= rescale * rescale * rescale * rescale;
    intrinsic[0] = intrinsic[0] / rescale;
  }
}

template <typename T>
BundlerDataset<T>::BundlerDataset(const std::string &filename)
    : BADataset<T>(filename) {
  this->Read(filename);
}

template <typename T>
void BundlerDataset<T>::Read(const std::string &filename, bool verbose) {
  // Reference:
  // https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6

  std::ifstream infile(filename);

  if (infile.is_open() != true) {
    LOG(ERROR) << "Can not open " << filename << std::endl;
    exit(-1);
  }

  m_extrinsics.clear();
  m_intrinsics.clear();
  m_points.clear();
  m_measurements.clear();

  int_t num_cameras, num_points;

  std::string line;
  std::getline(infile, line);

  std::getline(infile, line);
  std::stringstream sstream(line);

  sstream >> num_cameras;
  sstream >> num_points;

  m_extrinsics.resize(num_cameras);
  m_intrinsics.resize(num_cameras);
  for (int_t n = 0; n < num_cameras; n++) {
    Eigen::Matrix3<T> R;
    Eigen::Vector3<T> t;

    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    for (int_t i = 0; i < 3; i++) {
      sstream >> m_intrinsics[n][i];
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

    m_extrinsics[n].template leftCols<3>() = R.transpose();
    m_extrinsics[n].col(3) = R.transpose() * t;
  }

  m_points.resize(num_points);

  for (int_t n = 0; n < num_points; n++) {
    std::getline(infile, line);
    sstream.clear();
    sstream.str(line);

    for (int_t i = 0; i < 3; i++) {
      sstream >> m_points[n][i];
    }

    m_points[n] = -m_points[n];

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

      m_measurements.push_back(measurement);
    }
  }

  for (auto &measurement : m_measurements) {
    const auto &intrinsic = m_intrinsics[measurement.intrinsics_index];
    T rescale = intrinsic[0];
    measurement.measurement /= -rescale;
    measurement.sqrt_weight =
        rescale * std::sqrt(measurement.measurement.squaredNorm() + 1);
  }

  for (auto &intrinsic : m_intrinsics) {
    T rescale = intrinsic[0];
    intrinsic[1] = intrinsic[1];
    // intrinsic[2] = intrinsic[2] - 2 * intrinsic[1] * intrinsic[1];
    intrinsic[2] = intrinsic[2];
    intrinsic[1] *= rescale * rescale;
    intrinsic[2] *= rescale * rescale * rescale * rescale;
    intrinsic[0] = intrinsic[0] / rescale;
  }
}
template <typename T>
DBADataset<T>::DBADataset(const std::shared_ptr<BADataset<T>> dataset,
                          int_t targeted_num_clusters, T initial_resolution,
                          T refined_resolution, bool memory_efficient) {
  this->Setup(dataset, targeted_num_clusters, initial_resolution,
              refined_resolution, memory_efficient);
}

template <typename T>
void DBADataset<T>::Setup(const std::shared_ptr<BADataset<T>> dataset,
                          int_t targeted_num_clusters, T initial_resolution,
                          T refined_resolution, bool memory_efficient) {
  if (targeted_num_clusters <= 0) {
    LOG(ERROR) << "The number of nodes must be greater or equal to 0, but the "
                  "given number of nodes is "
               << targeted_num_clusters << "." << std::endl;
    return;
  }

  const int_t num_extrinsics = dataset->Extrinsics().size();
  const int_t num_intrinsics = dataset->Intrinsics().size();
  const int_t num_points = dataset->Points().size();
  const int_t num_measurements = dataset->Measurements().size();
  const int_t num_cameras = num_extrinsics;

  sfm::graph::Handle handle;

  // clustering cameras and points
  thrust::device_vector<int_t> src_indices_v;
  thrust::device_vector<int_t> dst_indices_v;
  thrust::device_vector<uint64_t> weights_v;
  thrust::device_vector<int_t> clustering_v(num_cameras + num_points, -1);
  {
    sfm::ba::clustering::LoadGraph(handle, *dataset, src_indices_v,
                                   dst_indices_v, weights_v);

    // thrust::fill(handle.GetThrustPolicy(), weights_v.begin(),
    // weights_v.end(), 1);
    const int_t num_edges = src_indices_v.size();
    float_t modularity = 0;

    sfm::ba::clustering::Cluster(
        handle, num_cameras, num_points, src_indices_v.data().get(),
        dst_indices_v.data().get(), weights_v.data().get(), num_edges,
        targeted_num_clusters, clustering_v.data().get(),
        clustering_v.data().get() + num_cameras, modularity, m_num_clusters,
        (float)initial_resolution, (float)refined_resolution, memory_efficient);

    if (m_num_clusters != targeted_num_clusters) {
      // TODO: Assign unclustered cameras
      LOG(ERROR) << "Failed to cluster cameras in to " << targeted_num_clusters
                 << " clusters." << std::endl;
      exit(-1);
    }

    m_measurement_indices.clear();
    m_measurements.clear();
    m_extrinsics.clear();
    m_intrinsics.clear();
    m_points.clear();

    m_measurement_indices.resize(m_num_clusters);
    m_measurements.resize(m_num_clusters);
    m_extrinsics.resize(m_num_clusters);
    m_intrinsics.resize(m_num_clusters);
    m_points.resize(m_num_clusters);

    float_t total_weight = num_edges;

    {
      thrust::device_vector<int_t> selected_src_indices_v(
          src_indices_v.data() + num_edges / 2,
          src_indices_v.data() + num_edges);
      thrust::device_vector<int_t> selected_dst_indices_v(num_edges / 2);
      thrust::device_vector<uint64_t> selected_weights_v(
          weights_v.data() + num_edges / 2, weights_v.data() + num_edges);

      thrust::for_each_n(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
          num_edges / 2,
          [clustering = clustering_v.data().get(),
           dst_indices = dst_indices_v.data().get(),
           clustering_dst_indices = selected_dst_indices_v.data().get(),
           num_edges] __device__(sfm::int_t n) {
            clustering_dst_indices[n] =
                clustering[dst_indices[n + num_edges / 2]];
          });

      thrust::sort_by_key(
          handle.GetThrustPolicy(), selected_dst_indices_v.begin(),
          selected_dst_indices_v.end(),
          sfm::utils::MakeZipIterator(selected_src_indices_v.begin(),
                                      selected_weights_v.begin()));

      thrust::sort_by_key(
          handle.GetThrustPolicy(), selected_src_indices_v.begin(),
          selected_src_indices_v.end(),
          sfm::utils::MakeZipIterator(selected_dst_indices_v.begin(),
                                      selected_weights_v.begin()));

      thrust::device_vector<int_t> selected_reduced_src_indices_v(num_edges /
                                                                  2);
      thrust::device_vector<int_t> selected_reduced_dst_indices_v(num_edges /
                                                                  2);
      thrust::device_vector<uint64_t> selected_reduced_weights_v(num_edges / 2);

      auto selected_src_dst_begin = sfm::utils::MakeZipIterator(
          selected_src_indices_v.begin(), selected_dst_indices_v.begin());
      auto selected_reduced_src_dst_begin =
          sfm::utils::MakeZipIterator(selected_reduced_src_indices_v.begin(),
                                      selected_reduced_dst_indices_v.begin());
      int_t number_of_selected_reduced_edges =
          thrust::reduce_by_key(
              handle.GetThrustPolicy(), selected_src_dst_begin,
              selected_src_dst_begin + num_edges / 2,
              selected_weights_v.begin(), selected_reduced_src_dst_begin,
              selected_reduced_weights_v.begin(),
              thrust::equal_to<thrust::tuple<int_t, int_t>>(), cub::Sum())
              .second -
          selected_reduced_weights_v.begin();

      selected_reduced_src_indices_v.resize(number_of_selected_reduced_edges);
      selected_reduced_dst_indices_v.resize(number_of_selected_reduced_edges);
      selected_reduced_weights_v.resize(number_of_selected_reduced_edges);

      thrust::device_vector<int_t> point_indices_v(num_points);
      thrust::device_vector<uint64_t> point_cluster_weights_v(num_points);

      int_t point_cnts =
          thrust::reduce_by_key(
              handle.GetThrustPolicy(), selected_reduced_src_indices_v.begin(),
              selected_reduced_src_indices_v.end(),
              sfm::utils::MakeZipIterator(
                  selected_reduced_dst_indices_v.begin(),
                  selected_reduced_weights_v.data()),
              point_indices_v.data(),
              sfm::utils::MakeZipIterator(clustering_v.data() + num_cameras,
                                          point_cluster_weights_v.data()),
              thrust::equal_to<sfm::int_t>(),
              [] __device__(auto pair1, auto pair2) {
                auto cluster1 = thrust::get<0>(pair1);
                auto cluster2 = thrust::get<0>(pair2);
                auto weight1 = thrust::get<1>(pair1);
                auto weight2 = thrust::get<1>(pair2);

                if (weight1 > weight2) {
                  return pair1;
                } else if (weight1 == weight2 && cluster1 > cluster2) {
                  return pair1;
                } else {
                  return pair2;
                }
              })
              .first -
          point_indices_v.data();

      assert(point_cnts == num_points);

      if (point_cnts != num_points) {
        LOG(ERROR) << "Inconsistent results for the number of points."
                   << std::endl;
        exit(-1);
      }
    }
  }

  // Process extrinsics and intrinsics
  {
    thrust::device_vector<int_t> camera_indices_v(2 * num_cameras);
    thrust::device_vector<int_t> camera_indices_ref_v(num_cameras);
    thrust::device_vector<int_t> camera_clustering_ref_v(
        clustering_v.data(), clustering_v.data() + num_cameras);
    thrust::device_vector<int_t> camera_clustering_offsets_v(m_num_clusters + 1,
                                                             0);
    thrust::device_vector<int_t> camera_clustering_sizes_v(m_num_clusters, 0);

    thrust::sequence(handle.GetThrustPolicy(), camera_indices_ref_v.begin(),
                     camera_indices_ref_v.end(), 0);
    thrust::stable_sort_by_key(
        handle.GetThrustPolicy(), camera_clustering_ref_v.begin(),
        camera_clustering_ref_v.end(), camera_indices_ref_v.data());

    sfm::graph::detail::FillOffset(camera_clustering_ref_v.data().get(),
                                   camera_clustering_offsets_v.data().get(),
                                   targeted_num_clusters, num_cameras,
                                   handle.GetStream());

    thrust::for_each_n(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
        num_cameras,
        [camera_indices = camera_indices_v.data().get(),
         camera_clustering_offsets = camera_clustering_offsets_v.data().get(),
         camera_indices_ref = camera_indices_ref_v.data().get(),
         camera_clustering_ref =
             camera_clustering_ref_v.data().get()] __device__(int_t n) {
          int_t cluster = camera_clustering_ref[n];
          int_t index = camera_indices_ref[n];
          int_t offset = camera_clustering_offsets[cluster];
          camera_indices[2 * index] = cluster;
          camera_indices[2 * index + 1] = n - offset;
        });

    std::vector<std::array<int_t, 2>> &extrinsics_indices =
        m_extrinsics_indices;
    m_extrinsics_indices.resize(num_extrinsics);
    cudaMemcpy((void *)extrinsics_indices.data(),
               (void *)camera_indices_v.data().get(),
               sizeof(int_t) * camera_indices_v.size(), cudaMemcpyDeviceToHost);

    std::vector<std::array<int_t, 2>> &intrinsics_indices =
        m_intrinsics_indices;
    m_intrinsics_indices.resize(num_intrinsics);
    cudaMemcpy((void *)intrinsics_indices.data(),
               (void *)camera_indices_v.data().get(),
               sizeof(int_t) * camera_indices_v.size(), cudaMemcpyDeviceToHost);

    thrust::for_each_n(handle.GetThrustPolicy(),
                       sfm::utils::MakeCountingIterator(0), m_num_clusters,
                       [sizes = camera_clustering_sizes_v.data().get(),
                        offsets = camera_clustering_offsets_v.data()
                                      .get()] __device__(int_t cluster) {
                         sizes[cluster] =
                             offsets[cluster + 1] - offsets[cluster];
                       });

    std::vector<int_t> camera_clustering_sizes(m_num_clusters);
    cudaMemcpy((void *)camera_clustering_sizes.data(),
               (void *)camera_clustering_sizes_v.data().get(),
               sizeof(int_t) * camera_clustering_sizes_v.size(),
               cudaMemcpyDeviceToHost);

    for (int_t cluster = 0; cluster < m_num_clusters; cluster++) {
      m_extrinsics[cluster].reserve(camera_clustering_sizes[cluster]);
      m_intrinsics[cluster].reserve(camera_clustering_sizes[cluster]);
    }

    for (int_t i = 0; i < num_extrinsics; i++) {
      const auto &[cluster, index] = extrinsics_indices[i];

      assert(index == m_extrinsics[cluster].size() &&
             "Inconsistent extrinsics index.");

      if (index != m_extrinsics[cluster].size()) {
        LOG(ERROR) << "Inconsistent extrinsics index." << std::endl;
        exit(-1);
      }

      m_extrinsics[cluster].push_back(dataset->Extrinsics()[i]);
    }

    for (int_t i = 0; i < num_intrinsics; i++) {
      const auto &[cluster, index] = intrinsics_indices[i];

      assert(index == m_intrinsics[cluster].size() &&
             "Inconsistent intrinsics index.");

      if (index != m_intrinsics[cluster].size()) {
        LOG(ERROR) << "Inconsistent intrinsics index." << std::endl;
        exit(-1);
      }

      m_intrinsics[cluster].push_back(dataset->Intrinsics()[i]);
    }
  }

  // Process points
  {
    thrust::device_vector<int_t> point_indices_v(2 * num_points);
    thrust::device_vector<int_t> point_indices_ref_v(num_points);
    thrust::device_vector<int_t> point_clustering_ref_v(
        clustering_v.data() + num_cameras,
        clustering_v.data() + num_cameras + num_points);
    thrust::device_vector<int_t> point_clustering_offsets_v(m_num_clusters + 1,
                                                            0);
    thrust::device_vector<int_t> point_clustering_sizes_v(m_num_clusters, 0);

    thrust::sequence(handle.GetThrustPolicy(), point_indices_ref_v.begin(),
                     point_indices_ref_v.end(), 0);
    thrust::stable_sort_by_key(
        handle.GetThrustPolicy(), point_clustering_ref_v.begin(),
        point_clustering_ref_v.end(), point_indices_ref_v.data());

    sfm::graph::detail::FillOffset(point_clustering_ref_v.data().get(),
                                   point_clustering_offsets_v.data().get(),
                                   targeted_num_clusters, num_points,
                                   handle.GetStream());

    thrust::for_each_n(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
        num_points,
        [point_indices = point_indices_v.data().get(),
         point_clustering_offsets = point_clustering_offsets_v.data().get(),
         point_indices_ref = point_indices_ref_v.data().get(),
         point_clustering_ref =
             point_clustering_ref_v.data().get()] __device__(int_t n) {
          int_t cluster = point_clustering_ref[n];
          int_t index = point_indices_ref[n];
          int_t offset = point_clustering_offsets[cluster];
          point_indices[2 * index] = cluster;
          point_indices[2 * index + 1] = n - offset;
        });

    std::vector<std::array<int_t, 2>> &point_indices = m_point_indices;
    point_indices.resize(num_points);
    cudaMemcpy((void *)point_indices.data(),
               (void *)point_indices_v.data().get(),
               sizeof(int_t) * point_indices_v.size(), cudaMemcpyDeviceToHost);

    thrust::for_each_n(handle.GetThrustPolicy(),
                       sfm::utils::MakeCountingIterator(0), m_num_clusters,
                       [sizes = point_clustering_sizes_v.data().get(),
                        offsets = point_clustering_offsets_v.data()
                                      .get()] __device__(int_t cluster) {
                         sizes[cluster] =
                             offsets[cluster + 1] - offsets[cluster];
                       });

    std::vector<int_t> point_clustering_sizes(m_num_clusters);
    cudaMemcpy((void *)point_clustering_sizes.data(),
               (void *)point_clustering_sizes_v.data().get(),
               sizeof(int_t) * point_clustering_sizes_v.size(),
               cudaMemcpyDeviceToHost);

    for (int_t cluster = 0; cluster < m_num_clusters; cluster++) {
      m_points[cluster].reserve(point_clustering_sizes[cluster]);
    }

    for (int_t i = 0; i < num_points; i++) {
      const auto &[cluster, index] = point_indices[i];

      assert(index == m_points[cluster].size() && "Inconsistent point index.");

      if (index != m_points[cluster].size()) {
        LOG(ERROR) << "Inconsistent point index." << std::endl;
        exit(-1);
      }

      m_points[cluster].push_back(dataset->Points()[i]);
    }
  }

  {
    std::vector<int_t> measurement_sizes(m_num_clusters, 0);

    for (int_t cluster = 0; cluster < m_num_clusters; cluster++) {
      auto edge_begin = sfm::utils::MakeZipIterator(src_indices_v.data().get(),
                                                    dst_indices_v.data().get());
      measurement_sizes[cluster] = thrust::transform_reduce(
          handle.GetThrustPolicy(), edge_begin, edge_begin + num_measurements,
          [clustering = clustering_v.data().get(),
           cluster] __device__(auto edge) {
            int_t src = thrust::get<0>(edge);
            int_t dst = thrust::get<1>(edge);
            return clustering[src] == cluster || clustering[dst] == cluster;
          },
          int_t(0), cub::Sum());
    }

    for (int_t cluster = 0; cluster < m_num_clusters; cluster++) {
      m_measurements[cluster].reserve(measurement_sizes[cluster]);
      m_measurement_indices[cluster].reserve(measurement_sizes[cluster]);
    }

    for (int_t i = 0; i < dataset->Measurements().size(); i++) {
      const auto &measurement = dataset->Measurements()[i];
      const auto &extrinsics_index = measurement.extrinsics_index;
      const auto &intrinsics_index = measurement.intrinsics_index;
      const auto &point_index = measurement.point_index;
      const auto &extrinsics_cluster =
          m_extrinsics_indices[extrinsics_index][0];
      const auto &intrinsics_cluster =
          m_intrinsics_indices[intrinsics_index][0];
      const auto &point_cluster = m_point_indices[point_index][0];

      assert(extrinsics_index == intrinsics_index &&
             "Extrinsics and intrinsics indices must be the same.");

      if (extrinsics_index != intrinsics_index) {
        LOG(ERROR) << "Extrinsics and intrinsics indices must be the same."
                   << std::endl;
        exit(-1);
      }

      Measurement new_measurement{
          measurement.measurement, measurement.sqrt_weight,
          m_extrinsics_indices[extrinsics_index],
          m_intrinsics_indices[intrinsics_index], m_point_indices[point_index]};

      m_measurements[extrinsics_cluster].push_back(new_measurement);
      m_measurement_indices[extrinsics_cluster].push_back(i);

      if (intrinsics_cluster != extrinsics_cluster) {
        m_measurements[intrinsics_cluster].push_back(new_measurement);
        m_measurement_indices[intrinsics_cluster].push_back(i);
      }

      if (point_cluster != extrinsics_cluster &&
          point_cluster != intrinsics_cluster) {
        m_measurements[point_cluster].push_back(new_measurement);
        m_measurement_indices[point_cluster].push_back(i);
      }
    }
  }
}

template <typename T> int_t DBADataset<T>::NumberOfClusters() const {
  return m_num_clusters;
}

template <typename T>
const std::vector<std::vector<int_t>> &
DBADataset<T>::MeasurementIndices() const {
  return m_measurement_indices;
}

template <typename T>
const std::vector<std::array<int, 2>> &
DBADataset<T>::ExtrinsicsIndices() const {
  return m_extrinsics_indices;
}

template <typename T>
const std::vector<std::array<int, 2>> &
DBADataset<T>::IntrinsicsIndices() const {
  return m_intrinsics_indices;
}

template <typename T>
const std::vector<std::array<int, 2>> &DBADataset<T>::PointIndices() const {
  return m_point_indices;
}

template <typename T>
const std::vector<std::vector<typename DBADataset<T>::Measurement>> &
DBADataset<T>::Measurements() const {
  return m_measurements;
}

template <typename T>
const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &
DBADataset<T>::Extrinsics() const {
  return m_extrinsics;
}

template <typename T>
const std::vector<std::vector<Eigen::Vector3<T>>> &
DBADataset<T>::Intrinsics() const {
  return m_intrinsics;
}

template <typename T>
const std::vector<std::vector<Eigen::Vector3<T>>> &
DBADataset<T>::Points() const {
  return m_points;
}

template class BADataset<float>;
template class BADataset<double>;
template class BALDataset<float>;
template class BALDataset<double>;
template class BundlerDataset<float>;
template class BundlerDataset<double>;
template class DBADataset<float>;
template class DBADataset<double>;
} // namespace ba
} // namespace sfm