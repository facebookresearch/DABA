// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <mpi.h>
#include <nccl.h>

#include <sfm/ba/functions/ba_async.cuh>
#include <sfm/ba/macro.h>
#include <sfm/optimization/LM.h>
#include <sfm/optimization/PCG.h>
#include <sfm/types.h>
#include <sfm/utils/utils.cuh>
#include <sfm/utils/utils.h>

#include <examples/douglas_rachford/douglas_rachford_problem.h>

namespace sfm {
namespace ba {
template <typename T>
DouglasRachfordSubproblem<kGPU, T, false>::DouglasRachfordSubproblem(
    const DROption<T> &option, int_t rank, int_t num_ranks)
    : m_option(option), m_rank(rank), m_num_ranks(num_ranks),
      m_extrinsics_offsets(num_ranks, -1), m_extrinsics_sizes(num_ranks, -1),
      m_extrinsics_ids(num_ranks), m_intrinsics_offsets(num_ranks, -1),
      m_intrinsics_sizes(num_ranks, -1), m_intrinsics_ids(num_ranks),
      m_point_offsets(num_ranks, -1), m_point_sizes(num_ranks, -1),
      m_point_ids(num_ranks), m_trust_region_radius(-1.0),
      m_comm_streams(num_ranks) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);

  m_shared_point_sizes[0].resize(num_ranks, 0);
  m_shared_point_sizes[1].resize(num_ranks, 0);
  m_shared_point_dicts[0].resize(num_ranks);
  m_shared_point_dicts[1].resize(num_ranks);
  m_shared_point_data[0].resize(num_ranks);
  m_shared_point_data[1].resize(num_ranks);

  m_cpu_shared_point_sizes[0].resize(num_ranks, 0);
  m_cpu_shared_point_sizes[1].resize(num_ranks, 0);
  m_cpu_shared_point_dicts[0].resize(num_ranks);
  m_cpu_shared_point_dicts[1].resize(num_ranks);
  m_cpu_shared_point_data[0].resize(num_ranks);
  m_cpu_shared_point_data[1].resize(num_ranks);

  assert(num_devices > 0);
  if (num_devices <= 0) {
    LOG(ERROR) << "The are no GPUs available." << std::endl;
    exit(-1);
  }

  m_device = rank % num_devices;

  CHECK_CUDA(cudaSetDevice(m_device));
  CHECK_CUDA(cudaStreamCreate(&m_stream));

  for (auto &comm_stream : m_comm_streams) {
    CHECK_CUDA(cudaStreamCreate(&comm_stream));
  }
}

template <typename T>
DouglasRachfordSubproblem<kGPU, T, false>::DouglasRachfordSubproblem(
    const DROption<T> &option, int_t rank, int_t num_ranks, int_t device)
    : m_option(option), m_rank(rank), m_num_ranks(num_ranks),
      m_extrinsics_offsets(num_ranks, -1), m_extrinsics_sizes(num_ranks, -1),
      m_extrinsics_ids(num_ranks), m_intrinsics_offsets(num_ranks, -1),
      m_intrinsics_sizes(num_ranks, -1), m_intrinsics_ids(num_ranks),
      m_point_offsets(num_ranks, -1), m_point_sizes(num_ranks, -1),
      m_point_ids(num_ranks), m_trust_region_radius(-1.0),
      m_comm_streams(num_ranks), m_device(device) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  assert(m_device <= num_devices && m_device >= 0);

  m_shared_point_sizes[0].resize(num_ranks, 0);
  m_shared_point_sizes[1].resize(num_ranks, 0);
  m_shared_point_dicts[0].resize(num_ranks);
  m_shared_point_dicts[1].resize(num_ranks);
  m_shared_point_data[0].resize(num_ranks);
  m_shared_point_data[1].resize(num_ranks);

  m_cpu_shared_point_sizes[0].resize(num_ranks, 0);
  m_cpu_shared_point_sizes[1].resize(num_ranks, 0);
  m_cpu_shared_point_dicts[0].resize(num_ranks);
  m_cpu_shared_point_dicts[1].resize(num_ranks);
  m_cpu_shared_point_data[0].resize(num_ranks);
  m_cpu_shared_point_data[1].resize(num_ranks);

  if (m_device >= num_devices || m_device < 0) {
    LOG(ERROR) << "The device id must be in the range of [0, " << num_devices
               << ")." << std::endl;
    exit(-1);
  }

  CHECK_CUDA(cudaSetDevice(m_device));
  CHECK_CUDA(cudaStreamCreate(&m_stream));

  for (auto &comm_stream : m_comm_streams) {
    CHECK_CUDA(cudaStreamCreate(&comm_stream));
  }
}

template <typename T>
DouglasRachfordSubproblem<kGPU, T, false>::~DouglasRachfordSubproblem() {
  CHECK_CUDA(cudaSetDevice(m_device));
  CHECK_CUDA(cudaStreamDestroy(m_stream));

  for (auto &comm_stream : m_comm_streams) {
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
  }
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetCost(T &cost) const {
  cost = m_cost;
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetSurrogateCost(
    std::array<T, 2> &surrogate_cost) const {
  surrogate_cost = m_surrogate_f;
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetOption(
    DROption<T> &option) const {
  option = m_option;
  return 0;
}

template <typename T>
const std::vector<std::unordered_map<int_t, int_t>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetExtrinsicsIds() const {
  return m_extrinsics_ids;
}

template <typename T>
const std::vector<std::unordered_map<int_t, int_t>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetIntrinsicsIds() const {
  return m_intrinsics_ids;
}

template <typename T>
const std::vector<std::unordered_map<int_t, int_t>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetPointIds() const {
  return m_point_ids;
}

template <typename T>
const std::vector<std::array<int_t, 2>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetExtrinsicsDicts() const {
  return m_extrinsics_dicts;
}

template <typename T>
const std::vector<std::array<int_t, 2>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetIntrinsicsDicts() const {
  return m_intrinsics_dicts;
}

template <typename T>
const std::vector<std::array<int_t, 2>> &
DouglasRachfordSubproblem<kGPU, T, false>::GetPointDicts() const {
  return m_point_dicts;
}

template <typename T>
int_t DouglasRachfordSubproblem<kGPU, T, false>::GetDevice() const {
  return m_device;
}

template <typename T>
T DouglasRachfordSubproblem<kGPU, T, false>::GetMemoryUsage() const {
  T memory = 0;

  memory += m_extrinsics_infos.size() * sizeof(int_t);
  memory += m_intrinsics_infos.size() * sizeof(int_t);
  memory += m_point_infos.size() * sizeof(int_t);
  memory += m_measurements.size() * sizeof(T);
  memory += m_sqrt_weights.size() * sizeof(T);

  memory += m_n_measurement_indices.size() * sizeof(int_t);

  memory += m_n_extrinsics_dicts.size() * sizeof(int_t);
  memory += m_n_extrinsics_indices.size() * sizeof(int_t);
  memory += m_n_intrinsics_dicts.size() * sizeof(int_t);
  memory += m_n_intrinsics_indices.size() * sizeof(int_t);
  memory += m_n_point_dicts.size() * sizeof(int_t);
  memory += m_n_point_indices.size() * sizeof(int_t);

  memory += m_extrinsics.size() * sizeof(T);
  memory += m_intrinsics.size() * sizeof(T);
  memory += m_points.size() * sizeof(T);

  memory += m_points_consensus.size() * sizeof(T);
  memory += m_points_corrected.size() * sizeof(T);
  memory += m_points_reference.size() * sizeof(T);

  memory += m_f_values.size() * sizeof(T);

  memory += m_point_consensus_cnts.size() * sizeof(int_t);

  memory += m_trust_region_extrinsics[0].size() * sizeof(T);
  memory += m_trust_region_intrinsics[0].size() * sizeof(T);
  memory += m_trust_region_points[0].size() * sizeof(T);
  memory += m_trust_region_extrinsics[1].size() * sizeof(T);
  memory += m_trust_region_intrinsics[1].size() * sizeof(T);
  memory += m_trust_region_points[1].size() * sizeof(T);

  for (int_t index = 0; index < 2; index++) {
    for (int_t rank = 0; rank < m_num_ranks; rank++) {
      memory += m_shared_point_dicts[index][rank].size() * sizeof(int_t);
      memory += m_shared_point_data[index][rank].size() * sizeof(T);
    }
  }

  memory += m_measurement_dicts_by_cameras.size() * sizeof(int_t);
  memory += m_measurement_indices_by_cameras.size() * sizeof(int_t);
  memory += m_measurement_offsets_by_cameras.size() * sizeof(int_t);
  memory += m_measurement_dicts_by_points.size() * sizeof(int_t);
  memory += m_measurement_indices_by_points.size() * sizeof(int_t);
  memory += m_measurement_offsets_by_points.size() * sizeof(int_t);

  memory += m_hess_cc.size() * sizeof(T);
  memory += m_hess_ll.size() * sizeof(T);
  memory += m_grad_c.size() * sizeof(T);
  memory += m_reduced_grad_c.size() * sizeof(T);

  for (int_t n = 0; n < 5; n++) {
    memory += m_buffer[n].size() * sizeof(T);
  }

  memory += m_pcg_x_c.size() * sizeof(T);
  memory += m_pcg_x_l.size() * sizeof(T);
  memory += m_pcg_r_c.size() * sizeof(T);
  memory += m_pcg_dx_c.size() * sizeof(T);
  memory += m_pcg_dr_c.size() * sizeof(T);
  memory += m_pcg_dz_c.size() * sizeof(T);
  memory += m_pcg_buffer.size() * sizeof(T);

  memory += m_hess_cc_inv.size() * sizeof(T);
  memory += m_hess_ll_inv.size() * sizeof(T);

  return memory / 1024 / 1024;
}

template <typename T>
T DouglasRachfordSubproblem<kGPU, T, false>::GetCommunicationLoad() const {
  T load = 0;

  for (int_t index = 0; index < 2; index++) {
    for (int_t rank = 0; rank < m_num_ranks; rank++) {
      load += m_shared_point_data[index][rank].size() * sizeof(T);
    }
  }

  return 3 * load / 1024 / 1024;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Setup(
    const std::vector<std::array<int_t, 2>> &extrinsics_infos,
    const std::vector<std::array<int_t, 2>> &intrinsics_infos,
    const std::vector<std::array<int_t, 2>> &point_infos,
    const std::vector<Eigen::Vector2<T>> &measurements,
    const std::vector<T> &sqrt_weights) {
  cudaSetDevice(m_device);

  m_num_measurements = extrinsics_infos.size();

  if (intrinsics_infos.size() != m_num_measurements) {
    LOG(ERROR) << "Inconsistent sizes of measurement information for "
                  "extrinsics and intrinsics."
               << std::endl;
    exit(-1);
  }

  if (point_infos.size() != m_num_measurements) {
    LOG(ERROR) << "Inconsistent sizes of measurement information for "
                  "extrinsics and landamrks."
               << std::endl;
    exit(-1);
  }

  std::vector<int_t> measurement_selections[2];

  for (int_t i = 0; i < m_num_measurements; i++) {
    const auto &extrinsics_info = extrinsics_infos[i];
    const auto &intrinsics_info = intrinsics_infos[i];
    const auto &point_info = point_infos[i];

    assert(extrinsics_info[0] == intrinsics_info[0] &&
           extrinsics_info[1] == intrinsics_info[1] &&
           "The extrinsics and intrinsics must have the same index for each "
           "measurement.");

    if (extrinsics_info[0] != intrinsics_info[0] ||
        extrinsics_info[1] != intrinsics_info[1]) {
      LOG(ERROR) << "The extrinsics and intrinsics must have the same index "
                    "for each measurement."
                 << std::endl;
      exit(-1);
    }

    if (extrinsics_info[0] == m_rank && point_info[0] == m_rank) {
      measurement_selections[0].push_back(i);
    } else if (extrinsics_info[0] == m_rank) {
      measurement_selections[0].push_back(i);
    } else if (point_info[0] == m_rank) {
      measurement_selections[1].push_back(i);
    } else {
      continue;
    }
  }

  m_num_measurements = measurement_selections[0].size();

  m_n_measurement_indices.resize(m_num_measurements);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_measurement_indices.begin(),
                   m_n_measurement_indices.end(), 0);

  m_extrinsics_ids.resize(m_num_ranks);
  m_intrinsics_ids.resize(m_num_ranks);
  m_point_ids.resize(m_num_ranks);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    m_extrinsics_ids[rank].clear();
    m_intrinsics_ids[rank].clear();
    m_point_ids[rank].clear();
  }

  for (const auto &index : measurement_selections[0]) {
    const auto &extrinsics_info = extrinsics_infos[index];
    const auto &intrinsics_info = intrinsics_infos[index];
    const auto &point_info = point_infos[index];

    assert(extrinsics_info[0] == m_rank && intrinsics_info[0] == m_rank);

    if (extrinsics_info[0] != m_rank || intrinsics_info[0] != m_rank) {
      LOG(ERROR) << "Incorrect extrinsics or intrinsics infomation."
                 << std::endl;
      exit(-1);
    }

    if (m_extrinsics_ids[extrinsics_info[0]].count(extrinsics_info[1]) == 0) {
      m_extrinsics_ids[extrinsics_info[0]].insert({extrinsics_info[1], -1});
    }

    if (m_intrinsics_ids[intrinsics_info[0]].count(intrinsics_info[1]) == 0) {
      m_intrinsics_ids[intrinsics_info[0]].insert({intrinsics_info[1], -1});
    }

    if (m_point_ids[point_info[0]].count(point_info[1]) == 0) {
      m_point_ids[point_info[0]].insert({point_info[1], -1});
    }
  }

  std::vector<std::unordered_set<int_t>> shared_point_dict_sets(m_num_ranks);

  for (const auto &index : measurement_selections[1]) {
    const auto &extrinsics_info = extrinsics_infos[index];
    const auto &intrinsics_info = intrinsics_infos[index];
    const auto &point_info = point_infos[index];

    assert(extrinsics_info[0] != m_rank && intrinsics_info[0] != m_rank &&
           point_info[0] == m_rank);

    if (extrinsics_info[0] == m_rank || intrinsics_info[0] == m_rank ||
        point_info[0] != m_rank) {
      LOG(ERROR) << "Incorrect extrinsics, intrinsics or point infomation."
                 << std::endl;
      exit(-1);
    }

    if (m_point_ids[point_info[0]].count(point_info[1]) == 0) {
      m_point_ids[point_info[0]].insert({point_info[1], -1});
    }

    if (extrinsics_info[0] != m_rank && point_info[0] == m_rank) {
      shared_point_dict_sets[extrinsics_info[0]].insert(point_info[1]);
    }

    if (intrinsics_info[0] != m_rank &&
        intrinsics_info[0] != extrinsics_info[0] && point_info[0] == m_rank) {
      shared_point_dict_sets[intrinsics_info[0]].insert(point_info[1]);
    }
  }

  m_num_extrinsics = 0;
  m_num_intrinsics = 0;
  m_num_points = 0;

  m_extrinsics_offsets.resize(m_num_ranks);
  m_intrinsics_offsets.resize(m_num_ranks);
  m_point_offsets.resize(m_num_ranks);

  m_extrinsics_sizes.resize(m_num_ranks);
  m_intrinsics_sizes.resize(m_num_ranks);
  m_point_sizes.resize(m_num_ranks);

  for (int_t i = 0; i < m_num_ranks; i++) {
    int_t rank = i == 0 ? m_rank : i - (i <= m_rank);

    m_extrinsics_offsets[rank] = m_num_extrinsics;
    m_intrinsics_offsets[rank] = m_num_intrinsics;
    m_point_offsets[rank] = m_num_points;

    m_extrinsics_sizes[rank] = m_extrinsics_ids[rank].size();
    m_intrinsics_sizes[rank] = m_intrinsics_ids[rank].size();
    m_point_sizes[rank] = m_point_ids[rank].size();

    m_num_extrinsics += m_extrinsics_sizes[rank];
    m_num_intrinsics += m_intrinsics_sizes[rank];
    m_num_points += m_point_sizes[rank];
  }

  assert(m_num_extrinsics == m_extrinsics_sizes[m_rank]);
  assert(m_num_intrinsics == m_intrinsics_sizes[m_rank]);

  if (m_num_extrinsics != m_extrinsics_sizes[m_rank] ||
      m_num_intrinsics != m_intrinsics_sizes[m_rank]) {
    LOG(ERROR) << "Inconsistent setup information." << std::endl;
    exit(-1);
  }

  m_num_cameras = m_num_extrinsics;

  sfm::PinnedHostVector<T> point_consensus_cnts(m_num_points, 1);
  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    for (const auto &index : shared_point_dict_sets[rank]) {
      point_consensus_cnts[index] += 1;
    }
  }

  m_point_consensus_cnts.resize(m_num_points);
  cudaMemcpyAsync(m_point_consensus_cnts.data().get(),
                  point_consensus_cnts.data(), m_num_points * sizeof(T),
                  cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

  m_extrinsics_dicts.clear();
  m_intrinsics_dicts.clear();
  m_point_dicts.clear();

  m_extrinsics_dicts.reserve(m_num_extrinsics);
  m_intrinsics_dicts.reserve(m_num_intrinsics);
  m_point_dicts.reserve(m_num_points);

  for (int_t i = 0; i < m_num_ranks; i++) {
    int_t rank = i == 0 ? m_rank : i - (i <= m_rank);

    for (const auto &info : m_extrinsics_ids[rank]) {
      m_extrinsics_dicts.push_back({rank, info.first});
    }

    for (const auto &info : m_intrinsics_ids[rank]) {
      m_intrinsics_dicts.push_back({rank, info.first});
    }

    for (const auto &info : m_point_ids[rank]) {
      m_point_dicts.push_back({rank, info.first});
    }

    auto cmp = [](const std::array<int_t, 2> &a,
                  const std::array<int_t, 2> &b) -> bool {
      return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
    };

    std::sort(m_extrinsics_dicts.begin() + m_extrinsics_offsets[rank],
              m_extrinsics_dicts.begin() + m_extrinsics_offsets[rank] +
                  m_extrinsics_sizes[rank],
              cmp);

    std::sort(m_intrinsics_dicts.begin() + m_intrinsics_offsets[rank],
              m_intrinsics_dicts.begin() + m_intrinsics_offsets[rank] +
                  m_intrinsics_sizes[rank],
              cmp);

    std::sort(m_point_dicts.begin() + m_point_offsets[rank],
              m_point_dicts.begin() + m_point_offsets[rank] +
                  m_point_sizes[rank],
              cmp);
  }

  assert(m_extrinsics_dicts[m_extrinsics_sizes[m_rank] - 1][1] ==
         m_extrinsics_sizes[m_rank] - 1);
  assert(m_intrinsics_dicts[m_intrinsics_sizes[m_rank] - 1][1] ==
         m_intrinsics_sizes[m_rank] - 1);
  assert(m_point_dicts[m_point_sizes[m_rank] - 1][1] ==
         m_point_sizes[m_rank] - 1);

  if (m_extrinsics_dicts[m_extrinsics_sizes[m_rank] - 1][1] !=
          m_extrinsics_sizes[m_rank] - 1 ||
      m_intrinsics_dicts[m_intrinsics_sizes[m_rank] - 1][1] !=
          m_intrinsics_sizes[m_rank] - 1 ||
      m_point_dicts[m_point_sizes[m_rank] - 1][1] !=
          m_point_sizes[m_rank] - 1) {
    LOG(ERROR) << "There are missing extrinsics, intrinsics or points for rank "
               << m_rank << "." << std::endl;
    exit(-1);
  }

  for (int_t i = 0; i < m_extrinsics_dicts.size(); i++) {
    const auto &info = m_extrinsics_dicts[i];
    assert(m_extrinsics_ids[info[0]].count(info[1]) == 1);
    m_extrinsics_ids[info[0]][info[1]] = i;
  }

  for (int_t i = 0; i < m_intrinsics_dicts.size(); i++) {
    const auto &info = m_intrinsics_dicts[i];
    assert(m_intrinsics_ids[info[0]].count(info[1]) == 1);
    m_intrinsics_ids[info[0]][info[1]] = i;
  }

  for (int_t i = 0; i < m_point_dicts.size(); i++) {
    const auto &info = m_point_dicts[i];
    assert(m_point_ids[info[0]].count(info[1]) == 1);
    m_point_ids[info[0]][info[1]] = i;
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank) {
      assert(shared_point_dict_sets[rank].size() == 0);

      if (shared_point_dict_sets[rank].size() != 0) {
        LOG(ERROR) << "No points should be sent/received to/from rank "
                   << m_rank << " itself." << std::endl;
        exit(-1);
      }

      continue;
    }

    m_shared_point_sizes[0][rank] = shared_point_dict_sets[rank].size();
    m_shared_point_dicts[0][rank].resize(m_shared_point_sizes[0][rank]);
    m_cpu_shared_point_dicts[0][rank].clear();
    m_cpu_shared_point_dicts[0][rank].reserve(m_shared_point_sizes[0][rank]);
    m_cpu_shared_point_dicts[0][rank].insert(
        m_cpu_shared_point_dicts[0][rank].begin(),
        shared_point_dict_sets[rank].begin(),
        shared_point_dict_sets[rank].end());
    std::sort(m_cpu_shared_point_dicts[0][rank].begin(),
              m_cpu_shared_point_dicts[0][rank].end());
    cudaMemcpyAsync(m_shared_point_dicts[0][rank].data().get(),
                    m_cpu_shared_point_dicts[0][rank].data(),
                    m_shared_point_sizes[0][rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_shared_point_data[0][rank].resize(LANDMARK_SIZE *
                                        m_shared_point_sizes[0][rank]);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank) {
      continue;
    }

    m_shared_point_sizes[1][rank] = m_point_sizes[rank];
    m_shared_point_dicts[1][rank].resize(m_shared_point_sizes[1][rank]);
    m_cpu_shared_point_dicts[1][rank].resize(m_shared_point_sizes[1][rank]);
    std::iota(m_cpu_shared_point_dicts[1][rank].begin(),
              m_cpu_shared_point_dicts[1][rank].end(), m_point_offsets[rank]);
    cudaMemcpyAsync(m_shared_point_dicts[1][rank].data().get(),
                    m_cpu_shared_point_dicts[1][rank].data(),
                    m_shared_point_sizes[1][rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_shared_point_data[1][rank].resize(LANDMARK_SIZE *
                                        m_shared_point_sizes[1][rank]);
  }

  sfm::PinnedHostVector<int_t> unsorted_extrinsics_infos;
  sfm::PinnedHostVector<int_t> unsorted_intrinsics_infos;
  sfm::PinnedHostVector<int_t> unsorted_point_infos;

  for (const auto &index : measurement_selections[0]) {
    const auto &extrinsics_info = extrinsics_infos[index];
    assert(m_extrinsics_ids[extrinsics_info[0]].count(extrinsics_info[1]) == 1);
    unsorted_extrinsics_infos.push_back(
        m_extrinsics_ids[extrinsics_info[0]][extrinsics_info[1]]);

    const auto &intrinsics_info = intrinsics_infos[index];
    assert(m_intrinsics_ids[intrinsics_info[0]].count(intrinsics_info[1]) == 1);
    unsorted_intrinsics_infos.push_back(
        m_intrinsics_ids[intrinsics_info[0]][intrinsics_info[1]]);

    const auto &point_info = point_infos[index];
    assert(m_point_ids[point_info[0]].count(point_info[1]) == 1);
    unsorted_point_infos.push_back(m_point_ids[point_info[0]][point_info[1]]);
  }

  // sort measurements by cameras
  std::vector<int_t> measurement_selection_indices(m_num_measurements);
  std::iota(measurement_selection_indices.begin(),
            measurement_selection_indices.end(), 0);
  std::vector<int_t> measurement_n_counts_by_cameras(m_num_cameras, 0);

  for (int_t k = 0; k < m_num_measurements; k++) {
    measurement_n_counts_by_cameras[unsorted_extrinsics_infos[k]]++;
  }

  auto camera_cmp = [&](int a, int b) {
    const auto &camera_a = unsorted_extrinsics_infos[a];
    const auto &camera_b = unsorted_extrinsics_infos[b];
    const auto &point_a = unsorted_point_infos[a];
    const auto &point_b = unsorted_point_infos[b];
    const auto &camera_a_cnts = measurement_n_counts_by_cameras[camera_a];
    const auto &camera_b_cnts = measurement_n_counts_by_cameras[camera_b];

    return camera_a == camera_b
               ? (point_a == point_b ? a < b : point_a < point_b)
               : (camera_a_cnts == camera_b_cnts
                      ? camera_a < camera_b
                      : camera_a_cnts > camera_b_cnts);
  };

  sort(measurement_selection_indices.begin(),
       measurement_selection_indices.end(), camera_cmp);
  m_measurement_dicts_by_cameras.resize(m_num_measurements);
  m_measurement_indices_by_cameras.resize(m_num_measurements);
  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_measurement_dicts_by_cameras.begin(),
                   m_measurement_dicts_by_cameras.end(), 0);
  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_measurement_indices_by_cameras.begin(),
                   m_measurement_indices_by_cameras.end(), 0);

  std::vector<int_t> measurement_offsets_by_cameras;
  for (int_t k = 0; k < m_num_measurements; k++) {
    if (k == 0 ||
        unsorted_extrinsics_infos[measurement_selection_indices[k]] !=
            unsorted_extrinsics_infos[measurement_selection_indices[k - 1]]) {
      measurement_offsets_by_cameras.push_back(k);
    }
  }
  measurement_offsets_by_cameras.push_back(m_num_measurements);
  m_measurement_offsets_by_cameras.resize(
      measurement_offsets_by_cameras.size());
  cudaMemcpyAsync(m_measurement_offsets_by_cameras.data().get(),
                  measurement_offsets_by_cameras.data(),
                  measurement_offsets_by_cameras.size() * sizeof(int_t),
                  cudaMemcpyHostToDevice, m_stream);
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  sfm::PinnedHostVector<int_t> sel_extrinsics_infos;
  sfm::PinnedHostVector<int_t> sel_intrinsics_infos;
  sfm::PinnedHostVector<int_t> sel_point_infos;
  std::vector<Eigen::Vector<T, 2>> sel_measurements;
  std::vector<T> sel_sqrt_weights;
  sel_extrinsics_infos.reserve(m_num_measurements);
  sel_intrinsics_infos.reserve(m_num_measurements);
  sel_point_infos.reserve(m_num_measurements);
  sel_sqrt_weights.reserve(m_num_measurements);

  for (const auto &selection_index : measurement_selection_indices) {
    const auto &index = measurement_selections[0][selection_index];
    const auto &extrinsics_info = extrinsics_infos[index];
    assert(m_extrinsics_ids[extrinsics_info[0]].count(extrinsics_info[1]) == 1);
    sel_extrinsics_infos.push_back(
        m_extrinsics_ids[extrinsics_info[0]][extrinsics_info[1]]);

    const auto &intrinsics_info = intrinsics_infos[index];
    assert(m_intrinsics_ids[intrinsics_info[0]].count(intrinsics_info[1]) == 1);
    sel_intrinsics_infos.push_back(
        m_intrinsics_ids[intrinsics_info[0]][intrinsics_info[1]]);

    const auto &point_info = point_infos[index];
    assert(m_point_ids[point_info[0]].count(point_info[1]) == 1);
    sel_point_infos.push_back(m_point_ids[point_info[0]][point_info[1]]);

    sel_measurements.push_back(measurements[index]);

    sel_sqrt_weights.push_back(sqrt_weights[index]);
  }

  m_extrinsics_infos.resize(m_num_measurements);
  m_intrinsics_infos.resize(m_num_measurements);
  m_point_infos.resize(m_num_measurements);
  m_measurements.resize(m_num_measurements);
  m_sqrt_weights.resize(m_num_measurements);

  thrust::copy(sel_extrinsics_infos.begin(), sel_extrinsics_infos.end(),
               m_extrinsics_infos.begin());
  thrust::copy(sel_intrinsics_infos.begin(), sel_intrinsics_infos.end(),
               m_intrinsics_infos.begin());
  thrust::copy(sel_point_infos.begin(), sel_point_infos.end(),
               m_point_infos.begin());
  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(sel_measurements,
                                                     m_measurements);
  thrust::copy(sel_sqrt_weights.begin(), sel_sqrt_weights.end(),
               m_sqrt_weights.begin());

  m_extrinsics.resize(EXTRINSICS_SIZE * m_num_extrinsics);
  m_intrinsics.resize(INTRINSICS_SIZE * m_num_intrinsics);
  m_points.resize(LANDMARK_SIZE * m_num_points);

  m_points_consensus.resize(LANDMARK_SIZE * m_num_points);
  m_points_corrected.resize(LANDMARK_SIZE * m_num_points);
  m_points_reference.resize(LANDMARK_SIZE * m_num_points);

  // Allocate GPU memory for each iteration
  m_n_extrinsics_dicts.resize(m_num_extrinsics, -1);
  m_n_intrinsics_dicts.resize(m_num_intrinsics, -1);
  m_n_point_dicts.resize(m_num_points, -1);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_extrinsics_dicts.begin(),
                   m_n_extrinsics_dicts.end(), 0);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_intrinsics_dicts.begin(),
                   m_n_intrinsics_dicts.end(), 0);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_point_dicts.begin(),
                   m_n_point_dicts.end(), 0);

  m_n_extrinsics_indices.resize(m_num_extrinsics, -1);
  m_n_intrinsics_indices.resize(m_num_intrinsics, -1);
  m_n_point_indices.resize(m_num_points, -1);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_extrinsics_indices.begin(), m_n_extrinsics_indices.end(),
                   0);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_intrinsics_indices.begin(), m_n_intrinsics_indices.end(),
                   0);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_point_indices.begin(),
                   m_n_point_indices.end(), 0);

  m_f_values.resize(m_num_measurements);

  // sort measurements by points
  std::vector<int_t> measurement_n_counts_by_points(m_num_points, 0);
  std::vector<int_t> measurement_dicts_by_points(m_num_measurements);
  std::iota(measurement_dicts_by_points.begin(),
            measurement_dicts_by_points.end(), 0);

  for (const auto &measurement_index : measurement_dicts_by_points) {
    measurement_n_counts_by_points[sel_point_infos[measurement_index]]++;
  }

  auto point_cmp = [&](int a, int b) {
    const auto &point_a = sel_point_infos[a];
    const auto &point_b = sel_point_infos[b];
    const auto &camera_a = sel_extrinsics_infos[a];
    const auto &camera_b = sel_extrinsics_infos[b];
    const auto &point_a_cnts = measurement_n_counts_by_points[point_a];
    const auto &point_b_cnts = measurement_n_counts_by_points[point_b];

    return point_a == point_b
               ? (camera_a == camera_b ? a < b : camera_a < camera_b)
               : (point_a_cnts == point_b_cnts ? point_a < point_b
                                               : point_a_cnts > point_b_cnts);
  };

  sort(measurement_dicts_by_points.begin(), measurement_dicts_by_points.end(),
       point_cmp);
  m_measurement_dicts_by_points.resize(measurement_dicts_by_points.size());
  cudaMemcpyAsync(m_measurement_dicts_by_points.data().get(),
                  measurement_dicts_by_points.data(),
                  measurement_dicts_by_points.size() * sizeof(int_t),
                  cudaMemcpyHostToDevice, m_stream);

  std::vector<int_t> measurement_indices_by_points(m_num_measurements);
  for (int_t k = 0; k < m_num_measurements; k++) {
    measurement_indices_by_points[measurement_dicts_by_points[k]] = k;
  }
  m_measurement_indices_by_points.resize(measurement_indices_by_points.size());
  cudaMemcpyAsync(m_measurement_indices_by_points.data().get(),
                  measurement_indices_by_points.data(),
                  measurement_indices_by_points.size() * sizeof(int_t),
                  cudaMemcpyHostToDevice, m_stream);

  std::vector<int_t> measurement_offsets_by_points;
  for (int_t k = 0; k < measurement_dicts_by_points.size(); k++) {
    if (k == 0 || sel_point_infos[measurement_dicts_by_points[k]] !=
                      sel_point_infos[measurement_dicts_by_points[k - 1]]) {
      measurement_offsets_by_points.push_back(k);
    }
  }
  measurement_offsets_by_points.push_back(m_num_measurements);
  m_measurement_offsets_by_points.resize(measurement_offsets_by_points.size());
  cudaMemcpyAsync(m_measurement_offsets_by_points.data().get(),
                  measurement_offsets_by_points.data(),
                  measurement_offsets_by_points.size() * sizeof(int_t),
                  cudaMemcpyHostToDevice, m_stream);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    m_trust_region_extrinsics[0].resize(EXTRINSICS_SIZE * m_num_extrinsics);
    m_trust_region_extrinsics[1].resize(EXTRINSICS_SIZE * m_num_extrinsics);
    m_trust_region_intrinsics[0].resize(INTRINSICS_SIZE * m_num_intrinsics);
    m_trust_region_intrinsics[1].resize(INTRINSICS_SIZE * m_num_intrinsics);
    m_trust_region_points[0].resize(LANDMARK_SIZE * m_num_points);
    m_trust_region_points[1].resize(LANDMARK_SIZE * m_num_points);

    m_hess_cc.resize(D_CAMERA_SIZE * D_CAMERA_SIZE * m_num_cameras);
    m_hess_ll.resize(LANDMARK_SIZE * LANDMARK_SIZE * m_num_points);
    m_grad_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_grad_l.resize(LANDMARK_SIZE * m_num_points);
    m_reduced_grad_c.resize(D_CAMERA_SIZE * m_num_cameras);

    m_hess_cc_inv.resize(D_CAMERA_SIZE * D_CAMERA_SIZE * m_num_cameras);
    m_hess_ll_inv.resize(LANDMARK_SIZE * LANDMARK_SIZE * m_num_points);

    m_pcg_x_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_pcg_x_l.resize(LANDMARK_SIZE * m_num_points);
    m_pcg_r_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_pcg_dx_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_pcg_dr_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_pcg_dz_c.resize(D_CAMERA_SIZE * m_num_cameras);
    m_pcg_buffer.resize(LANDMARK_SIZE * m_num_points);

    m_buffer[0].resize(
        std::max(D_CAMERA_SIZE * D_LANDMARK_SIZE * m_num_measurements,
                 3 * D_CAMERA_SIZE * m_num_measurements));
    m_buffer[1].resize(D_CAMERA_SIZE * D_LANDMARK_SIZE * m_num_measurements);
    m_buffer[2].resize(D_CAMERA_SIZE * m_num_measurements);
    m_buffer[3].resize(std::max(
        {// buffer for hess_ll_n
         (D_LANDMARK_SIZE + 1) * D_LANDMARK_SIZE / 2 * m_num_measurements,
         // buffer for reducing objective/surrogate function
         m_num_measurements,
         // buffer for inner product
         std::max(D_CAMERA_SIZE, 6) * m_num_cameras,
         D_LANDMARK_SIZE * m_num_points}));
    m_buffer[4].resize(D_LANDMARK_SIZE * m_num_measurements);
  }

  m_future_surrogate_f.resize(2);
  m_future_objective_f.resize(1);
  m_future_inner_product.resize(3);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Accept(
    const Container<T> &extrinsics, const Container<T> &intrinsics,
    const Container<T> &points, const std::array<T, 2> &surrogate_f,
    T cost) const {
  AcceptAsync(extrinsics, intrinsics, points, surrogate_f, cost);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetExtrinsics(
    std::vector<Eigen::Matrix<T, 3, 4>> &extrinsics) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_extrinsics, extrinsics, m_extrinsics_offsets[m_rank],
      m_extrinsics_sizes[m_rank]);
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetIntrinsics(
    std::vector<Eigen::Matrix<T, 3, 1>> &intrinsics) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_intrinsics, intrinsics, m_intrinsics_offsets[m_rank],
      m_intrinsics_sizes[m_rank]);
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::GetPoints(
    std::vector<Eigen::Matrix<T, 3, 1>> &points) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_points, points, m_point_offsets[m_rank], m_point_sizes[m_rank]);
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Initialize(
    const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 4>>>
        &extrinsics_data,
    const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
        &intrinsics_data,
    const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
        &point_data) {
  std::vector<Eigen::Matrix<T, 3, 4>> initial_extrinsics;
  std::vector<Eigen::Matrix<T, 3, 1>> initial_intrinsics;
  std::vector<Eigen::Matrix<T, 3, 1>> initial_points;

  initial_extrinsics.reserve(m_num_extrinsics);
  initial_intrinsics.reserve(m_num_intrinsics);
  initial_points.reserve(m_num_points);

  for (const auto &index : m_extrinsics_dicts) {
    auto search = extrinsics_data[index[0]].find(index[1]);

    assert(search != extrinsics_data[index[0]].end());

    if (search == extrinsics_data[index[0]].end()) {
      LOG(ERROR) << "No initial guess for extrinsics (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_extrinsics.push_back((search->second));
  }

  for (const auto &index : m_intrinsics_dicts) {
    auto search = intrinsics_data[index[0]].find(index[1]);

    assert(search != intrinsics_data[index[0]].end());

    if (search == intrinsics_data[index[0]].end()) {
      LOG(ERROR) << "No initial guess for intrinsics (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_intrinsics.push_back((search->second));
  }

  for (const auto &index : m_point_dicts) {
    auto search = point_data[index[0]].find(index[1]);

    assert(search != point_data[index[0]].end());

    if (search == point_data[index[0]].end()) {
      LOG(ERROR) << "No initial guess for points (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_points.push_back((search->second));
  }

  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_extrinsics,
                                                     m_extrinsics);
  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_intrinsics,
                                                     m_intrinsics);
  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_points, m_points);

  cudaMemcpyAsync(m_points_consensus.data().get(), m_points.data().get(),
                  D_LANDMARK_SIZE * m_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);
  cudaMemsetAsync(m_points_corrected.data().get(), 0,
                  LANDMARK_SIZE * m_num_points * sizeof(T), m_stream);
  cudaMemcpyAsync(m_points_reference.data().get(), m_points.data().get(),
                  D_LANDMARK_SIZE * m_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  this->InitializeSurrogateFunction();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    m_trust_region_radius = std::numeric_limits<T>::max();
    m_trust_region_decreasing_ratio = 0.5;
  } else {
    m_trust_region_radius = -1.0;
    m_trust_region_decreasing_ratio = 0;
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Initialize(
    const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &extrinsics_data,
    const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &intrinsics_data,
    const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &point_data) {
  CHECK_CUDA(cudaSetDevice(m_device));

  std::vector<Eigen::Matrix<T, 3, 4>> initial_extrinsics;
  std::vector<Eigen::Matrix<T, 3, 1>> initial_intrinsics;
  std::vector<Eigen::Matrix<T, 3, 1>> initial_points;

  initial_extrinsics.reserve(m_num_extrinsics);
  initial_intrinsics.reserve(m_num_intrinsics);
  initial_points.reserve(m_num_points);

  for (const auto &index : m_extrinsics_dicts) {
    assert(index[1] >= 0 && index[1] < extrinsics_data[index[0]].size());

    if (index[1] < 0 || index[1] >= extrinsics_data[index[0]].size()) {
      LOG(ERROR) << "No initial guess for extrinsics (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_extrinsics.push_back(extrinsics_data[index[0]][index[1]]);
  }

  for (const auto &index : m_intrinsics_dicts) {
    assert(index[1] >= 0 && index[1] < intrinsics_data[index[0]].size());

    if (index[1] < 0 || index[1] >= intrinsics_data[index[0]].size()) {
      LOG(ERROR) << "No initial guess for intrinsics (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_intrinsics.push_back(intrinsics_data[index[0]][index[1]]);
  }

  for (const auto &index : m_point_dicts) {
    assert(index[1] >= 0 && index[1] < point_data[index[0]].size());

    if (index[1] < 0 || index[1] >= point_data[index[0]].size()) {
      LOG(ERROR) << "No initial guess for point (" << index[0] << ", "
                 << index[1] << ")." << std::endl;
      exit(-1);
    }

    initial_points.push_back(point_data[index[0]][index[1]]);
  }

  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_extrinsics,
                                                     m_extrinsics);
  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_intrinsics,
                                                     m_intrinsics);
  sfm::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_points, m_points);

  cudaMemcpyAsync(m_points_consensus.data().get(), m_points.data().get(),
                  D_LANDMARK_SIZE * m_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);
  cudaMemsetAsync(m_points_corrected.data().get(), 0,
                  LANDMARK_SIZE * m_num_points * sizeof(T), m_stream);
  cudaMemcpyAsync(m_points_reference.data().get(), m_points.data().get(),
                  D_LANDMARK_SIZE * m_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  this->InitializeSurrogateFunction();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    m_trust_region_radius = std::numeric_limits<T>::max();
  } else {
    m_trust_region_radius = -1.0;
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Iterate(T rho) const {
  CHECK_CUDA(cudaSetDevice(m_device));

  m_trust_region_radius = std::numeric_limits<T>::max();
  m_trust_region_decreasing_ratio = 0.5;
  EvaluateSurrogateFunction(m_extrinsics, m_intrinsics, m_points, rho,
                            m_surrogate_f, m_cost);
  Update(rho);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PreCommunicate(
    int_t round) const {
  PreCommunicateAsync(round);
  PreCommunicateSync(round);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PostCommunicate(
    int_t round) const {
  PostCommunicateAsync(round);
  PostCommunicateSync(round);
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::MPICommunicate(
    const MPI_Comm &comm, int_t round) const {
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  assert(mpi_rank == m_rank);

  if (mpi_rank != m_rank) {
    LOG(ERROR) << "The rank is not consistent with MPI." << std::endl;
    MPI_Abort(comm, -1);
    exit(-1);
  }

  const int_t send_index = round == 0 ? 1 : 0;
  const int_t recv_index = round == 0 ? 0 : 1;

  CHECK_CUDA(cudaSetDevice(m_device));

  PreCommunicate(round);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (m_shared_point_data[send_index][rank].size() !=
        LANDMARK_SIZE * m_shared_point_sizes[send_index][rank]) {
      LOG(ERROR) << "Inconsistent communicaiton data size." << std::endl;
      exit(-1);
    }

    if (rank == m_rank || !m_shared_point_sizes[send_index][rank]) {
      continue;
    }

    m_cpu_shared_point_data[send_index][rank].resize(
        m_shared_point_data[send_index][rank].size());

    cudaMemcpyAsync(
        m_cpu_shared_point_data[send_index][rank].data(),
        m_shared_point_data[send_index][rank].data().get(),
        m_cpu_shared_point_data[send_index][rank].size() * sizeof(T),
        cudaMemcpyKind::cudaMemcpyDeviceToHost, m_comm_streams[rank]);
  }

  std::vector<MPI_Request> send_requests(m_num_ranks);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank || !m_shared_point_sizes[send_index][rank]) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[rank]));
    MPI_Isend(m_cpu_shared_point_data[send_index][rank].data(),
              m_cpu_shared_point_data[send_index][rank].size(),
              traits<T>::MPI_FLOAT_TYPE, rank,
              rank * m_num_ranks + m_rank + round, comm, &send_requests[rank]);
  }

  std::vector<MPI_Request> recv_requests(m_num_ranks);
  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (m_shared_point_data[recv_index][rank].size() !=
        LANDMARK_SIZE * m_shared_point_sizes[recv_index][rank]) {
      LOG(ERROR) << "Inconsistent communicaiton data size." << std::endl;
      exit(-1);
    }

    if (rank == m_rank || !m_shared_point_sizes[recv_index][rank]) {
      continue;
    }

    m_cpu_shared_point_data[recv_index][rank].resize(
        m_shared_point_data[recv_index][rank].size());
    MPI_Irecv(m_cpu_shared_point_data[recv_index][rank].data(),
              m_cpu_shared_point_data[recv_index][rank].size(),
              traits<T>::MPI_FLOAT_TYPE, rank,
              m_rank * m_num_ranks + rank + round, comm, &recv_requests[rank]);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank || !m_shared_point_sizes[recv_index][rank]) {
      continue;
    }

    MPI_Status status;

    MPI_Wait(&recv_requests[rank], &status);
    cudaMemcpyAsync(m_shared_point_data[recv_index][rank].data().get(),
                    m_cpu_shared_point_data[recv_index][rank].data(),
                    m_cpu_shared_point_data[recv_index][rank].size() *
                        sizeof(T),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank || !m_shared_point_sizes[recv_index][rank]) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
  }

  PostCommunicate(round);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank || !m_shared_point_sizes[send_index][rank]) {
      continue;
    }

    MPI_Status status;
    MPI_Wait(&send_requests[rank], &status);
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::InitializeSurrogateFunction() {
  cudaEvent_t event;
  CHECK_CUDA(cudaEventCreate(&event));

  InitializeSurrogateFunctionAsync(m_future_surrogate_f, event);
  InitializeSurrogateFunctionSync(m_future_surrogate_f, event);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::AcceptAsync(
    const Container<T> &extrinsics, const Container<T> &intrinsics,
    const Container<T> &points, const std::array<T, 2> &surrogate_f,
    T cost) const {
  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_extrinsics_dicts.data().get(), EXTRINSICS_SIZE, 1,
      extrinsics.data().get(), m_extrinsics.data().get(), m_num_extrinsics,
      m_num_extrinsics, m_num_extrinsics, m_stream);

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_intrinsics_dicts.data().get(), INTRINSICS_SIZE, 1,
      intrinsics.data().get(), m_intrinsics.data().get(), m_num_intrinsics,
      m_num_intrinsics, m_num_intrinsics, m_stream);

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_point_dicts.data().get(), LANDMARK_SIZE, 1, points.data().get(),
      m_points.data().get(), m_num_points, m_num_points, m_num_points,
      m_stream);

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_point_dicts.data().get(), LANDMARK_SIZE, 1, m_points.data().get(),
      m_points_consensus.data().get(), m_num_points, m_num_points,
      m_point_sizes[m_rank], m_stream);

  sfm::utils::PlusAsync(
      T(2.0), m_points.data().get(), T(-1.0), m_points_reference.data().get(),
      m_points_corrected.data().get(), m_points.size(), m_stream);

  m_surrogate_f = surrogate_f;
  m_cost = cost;

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::EvaluateSurrogateFunction(
    const Container<T> &extrinsics_data, const Container<T> &intrinsics_data,
    const Container<T> &point_data, T rho, std::array<T, 2> &surrogate_f,
    T &cost) const {
  cudaEvent_t event;
  CHECK_CUDA(cudaEventCreate(&event));
  EvaluateSurrogateFunctionAsync(extrinsics_data, intrinsics_data, point_data,
                                 m_future_surrogate_f, event);
  EvaluateSurrogateFunctionSync(m_future_surrogate_f, rho, surrogate_f, cost,
                                event);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::LevenbergMarquardtMethod(
    T rho) const {
  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  sfm::optimization::Objective<T> objective = [this](T &objective) -> void {
    objective = this->m_trust_region_cost[0];
  };

  sfm::optimization::QuadraticModel<T> quadratic_model = [this, rho]() -> void {
    this->Linearize(rho);
  };

  sfm::optimization::RescaleDiagonal<T> rescale_diagonal =
      [this, rho](T prev_ratio, T curr_ratio) -> void {
    this->BuildLinearSystem((1 + curr_ratio) / (1 + prev_ratio));
  };

  sfm::optimization::PCGSolver<T> pcg = [this](int_t &num_iters,
                                               T &update_step_norm) -> void {
    this->PCG(num_iters, update_step_norm);
  };

  sfm::optimization::GradientNorm<T> gradient_norm =
      [this, reduce_buffer, &reduce_buffer_size](T &gradient_norm) -> void {
    auto future_inner_product = m_future_inner_product.data().get();
    sfm::utils::SquaredNormAsync(
        reduce_buffer, reduce_buffer_size, m_reduced_grad_c.data().get(),
        m_reduced_grad_c.size(), future_inner_product, m_stream);
    CHECK_CUDA(cudaStreamSynchronize(m_stream));

    cudaMemcpy(&gradient_norm, future_inner_product, sizeof(T),
               cudaMemcpyDeviceToHost);
    gradient_norm = std::sqrt(gradient_norm);
  };

  sfm::optimization::PredictedReductionWithDamping<T> predicted_reduction =
      [this, reduce_buffer, &reduce_buffer_size](T &predicted_reduction,
                                                 T ratio) -> void {
    auto future_inner_product = m_future_inner_product.data().get();
    auto pcg_buffer = m_pcg_buffer.data().get();

    sfm::utils::PlusAsync(
        T(1.0), m_pcg_r_c.data().get(), T(1.0), m_reduced_grad_c.data().get(),
        m_pcg_buffer.data().get(), m_pcg_r_c.size(), m_stream);

    sfm::utils::InnerProductAsync(reduce_buffer, reduce_buffer_size,
                                  m_pcg_x_c.data().get(), m_pcg_x_c.size(),
                                  pcg_buffer, future_inner_product, m_stream);

    sfm::utils::MatrixDiagonalWeightedSquaredNormAsync(
        reduce_buffer, reduce_buffer_size, m_pcg_x_c.data().get(),
        m_num_extrinsics, D_CAMERA_SIZE, m_hess_cc.data().get(),
        future_inner_product + 1, m_stream);

    sfm::utils::MatrixDiagonalWeightedSquaredNormAsync(
        reduce_buffer, reduce_buffer_size, m_pcg_x_l.data().get(), m_num_points,
        D_LANDMARK_SIZE, m_hess_ll.data().get(), future_inner_product + 2,
        m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));

    PinnedHostVector<T> inner_product(3);
    cudaMemcpy(inner_product.data(), m_future_inner_product.data().get(),
               3 * sizeof(T), cudaMemcpyDeviceToHost);

    predicted_reduction = 0.5 * (m_schur_reduction_l - inner_product[0]);

    predicted_reduction += 0.5 * ratio / (1 + ratio) * inner_product[1];

    predicted_reduction += 0.5 * ratio / (1 + ratio) * inner_product[2];
  };

  sfm::optimization::Update<T> update = [this, rho](T stepsize,
                                                    T &new_objective) -> void {
    this->Retract(stepsize);
    EvaluateSurrogateFunction(
        this->m_trust_region_extrinsics[1], this->m_trust_region_intrinsics[1],
        this->m_trust_region_points[1], rho,
        this->m_trust_region_surrogate_f[1], m_trust_region_cost[1]);

    new_objective = this->m_trust_region_cost[1];
  };

  sfm::optimization::Accept<T> accept = [this]() -> void {
    this->m_trust_region_extrinsics[0].swap(this->m_trust_region_extrinsics[1]);
    this->m_trust_region_intrinsics[0].swap(this->m_trust_region_intrinsics[1]);
    this->m_trust_region_points[0].swap(this->m_trust_region_points[1]);
    this->m_trust_region_surrogate_f[0].swap(
        this->m_trust_region_surrogate_f[1]);
    std::swap(this->m_trust_region_cost[0], this->m_trust_region_cost[1]);
  };

  T initial_trust_region_radius = std::min((T)1e2, m_trust_region_radius);
  T initial_trust_region_decreasing_ratio =
      std::min((T)0.5, m_trust_region_radius);
  T fobj = 0, grad_norm = 0, update_step_norm = 0, trust_region_radius = 0,
    trust_region_decreasing_ratio = 0;

  auto status = sfm::optimization::LM<T>(
      objective, quadratic_model, gradient_norm, pcg, predicted_reduction,
      update, rescale_diagonal, accept,
      m_option.trust_region_option.max_iterations,
      m_option.trust_region_option.max_accepted_iterations,
      m_option.trust_region_option.gradient_norm_tolerance,
      m_option.trust_region_option.update_step_norm_tolerance,
      m_option.trust_region_option.relative_function_decrease_tolerance,
      m_option.trust_region_option.trust_region_radius_tolerance,
      m_option.trust_region_option.gain_ratio_acceptance_tolerance,
      m_option.trust_region_option.gain_ratio_increasing_tolerance,
      initial_trust_region_radius, initial_trust_region_decreasing_ratio, fobj,
      grad_norm, update_step_norm, trust_region_radius,
      trust_region_decreasing_ratio, m_option.trust_region_option.verbose);

  m_trust_region_radius =
      std::min((T)2.5e2, std::max((T)0.1, trust_region_radius));

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Linearize(T rho) const {
  LinearizeAsync(rho);
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::BuildLinearSystem(
    T ratio) const {
  cudaEvent_t event;
  CHECK_CUDA(cudaEventCreate(&event));

  BuildLinearSystemAsync(ratio, m_future_inner_product, event);
  BuildLinearSystemSync(m_future_inner_product, event);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PCG(int_t &num_iters,
                                                   T &update_step_norm) const {
  const int_t *measurement_indices = m_n_measurement_indices.data().get();

  const int_t *n_camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  sfm::optimization::SymmetricLinearOperator<Container<T>> hessian =
      [this](const Container<T> &x, Container<T> &y) -> void {
    const T *hess_cc = m_hess_cc.data().get();
    const T *hess_cl[2] = {m_buffer[0].data().get(), m_buffer[1].data().get()};
    const T *hess_ll_inv = m_hess_ll_inv.data().get();
    const T *x_ptr = x.data().get();
    T *y_ptr = y.data().get();
    T *buffer = m_pcg_buffer.data().get();

    this->ComputeReducedCameraMatrixVectorMultiplicationAsync(
        hess_cc, {hess_cl[0], hess_cl[1]}, hess_ll_inv, x_ptr, y_ptr, buffer);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::Preconditioner<Container<T>> precondition =
      [this](const Container<T> &x, Container<T> &y) -> void {
    auto hess_cc_inv = m_hess_cc_inv.data().get();
    auto x_ptr = x.data().get();
    auto y_ptr = y.data().get();

    sfm::utils::ComputeMatrixVectorMultiplicationAsync(
        T(1.0), hess_cc_inv, x_ptr, T(0.0), y_ptr, D_CAMERA_SIZE, m_num_cameras,
        m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::Plus<Container<T>, T> plus =
      [this](const Container<T> &x, T alpha, const Container<T> &y,
             Container<T> &z) -> void {
    sfm::utils::PlusAsync(T(1.0), x.data().get(), alpha, y.data().get(),
                          z.data().get(), x.size(), m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::InnerProduct<Container<T>, T> inner_product =
      [this, reduce_buffer, &reduce_buffer_size](
          const Container<T> &x, const Container<T> &y, T &result) -> void {
    auto future_inner_product = m_future_inner_product.data().get();

    sfm::utils::InnerProductAsync(reduce_buffer, reduce_buffer_size,
                                  x.data().get(), x.size(), y.data().get(),
                                  future_inner_product, m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(&result, future_inner_product, sizeof(T),
               cudaMemcpyDeviceToHost);
  };

  sfm::optimization::Equal<Container<T>> equal =
      [this](const Container<T> &x, Container<T> &y) -> void {
    auto x_ptr = x.data().get();
    auto y_ptr = y.data().get();

    cudaMemcpyAsync(y_ptr, x_ptr, D_CAMERA_SIZE * m_num_cameras * sizeof(T),
                    cudaMemcpyDeviceToDevice, m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::SetZero<Container<T>> set_zero =
      [this](Container<T> &x) -> void {
    auto x_ptr = x.data().get();

    cudaMemsetAsync(x_ptr, 0, D_CAMERA_SIZE * m_num_cameras * sizeof(T),
                    m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  num_iters = 0;
  update_step_norm = 0;
  sfm::optimization::PCG(
      hessian, m_reduced_grad_c, precondition, plus, inner_product, equal,
      set_zero, m_pcg_x_c, m_pcg_r_c, m_pcg_dx_c, m_pcg_dr_c, m_pcg_dz_c,
      num_iters, update_step_norm, m_option.pcg_option.max_iterations,
      m_option.pcg_option.relative_residual_norm, m_option.pcg_option.theta,
      m_option.pcg_option.relative_reduction_tol, m_option.pcg_option.verbose);

  // schur complement to recover points
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  T *hess_cl[2] = {m_buffer[0].data().get(), m_buffer[1].data().get()};
  T *buffer = m_buffer[2].data().get();
  T *hess_ll_inv = m_hess_ll_inv.data().get();
  T *grad_l = m_grad_l.data().get();
  T *pcg_x_c = m_pcg_x_c.data().get();
  T *pcg_x_l = m_pcg_x_l.data().get();

  cudaMemcpyAsync(pcg_x_l, grad_l, LANDMARK_SIZE * m_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  const auto &measurement_dicts_by_points =
      m_measurement_dicts_by_points.data().get();
  const auto &measurement_offsets_by_points =
      m_measurement_offsets_by_points.data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
      measurement_dicts_by_points, measurement_offsets_by_points,
      n_camera_indices, n_point_indices, hess_cl[0], T(1.0), pcg_x_c,
      camera_infos, point_infos, T(1.0), pcg_x_l, buffer, m_num_cameras,
      m_num_points, m_num_measurements,
      m_measurement_offsets_by_points.size() - 1, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(-1.0), hess_ll_inv, pcg_x_l, T(0.0), pcg_x_l, LANDMARK_SIZE,
      m_num_points, m_stream);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Retract(T stepsize) const {
  RetractAsync(stepsize);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::Update(T rho) const {
  const int_t *extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *point_dicts = m_n_point_dicts.data().get();

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        extrinsics_dicts, EXTRINSICS_SIZE, 1, m_extrinsics.data().get(),
        m_trust_region_extrinsics[0].data().get(), m_num_extrinsics,
        m_num_extrinsics, m_num_extrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        intrinsics_dicts, INTRINSICS_SIZE, 1, m_intrinsics.data().get(),
        m_trust_region_intrinsics[0].data().get(), m_num_intrinsics,
        m_num_intrinsics, m_num_intrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        point_dicts, LANDMARK_SIZE, 1, m_points.data().get(),
        m_trust_region_points[0].data().get(), m_num_points, m_num_points,
        m_num_points, m_stream);

    m_trust_region_surrogate_f[0] = m_surrogate_f;
    m_trust_region_cost[0] = m_cost;

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    LevenbergMarquardtMethod(rho);

    Accept(m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
           m_trust_region_points[0], m_trust_region_surrogate_f[0],
           m_trust_region_cost[0]);
  }

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::InitializeSurrogateFunctionAsync(
    Container<T> &future_surrogate_f, cudaEvent_t event) {
  const auto extrinsics = m_extrinsics.data().get();
  const auto intrinsics = m_intrinsics.data().get();
  const auto points = m_points.data().get();
  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();

  T *f_values = m_f_values.data().get();

  if (future_surrogate_f.size() < 1) {
    future_surrogate_f.resize(1);
  }

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, f_values,
      m_option.robust_loss, m_option.loss_radius, m_num_extrinsics,
      m_num_intrinsics, m_num_points, m_num_measurements, m_num_measurements,
      m_stream);

  sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values,
                          m_num_measurements, future_surrogate_f.data().get(),
                          T(0.0), cub::Sum(), m_stream);

  CHECK_CUDA(cudaEventRecord(event, m_stream));

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::InitializeSurrogateFunctionSync(
    Container<T> &future_surrogate_f, cudaEvent_t event) {
  CHECK_CUDA(cudaEventSynchronize(event));

  cudaMemcpy(m_surrogate_f.data(), future_surrogate_f.data().get(), sizeof(T),
             cudaMemcpyDeviceToHost);

  m_surrogate_f[1] = 0;

  m_cost = m_surrogate_f[0];

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::RetractAsync(T stepsize) const {
  auto d_extrinsics = m_pcg_x_c.data().get();
  auto extrinsics = m_trust_region_extrinsics[0].data().get();
  auto extrinsics_plus = m_trust_region_extrinsics[1].data().get();
  sfm::utils::ComputeSE3RetractionAsync(extrinsics, stepsize, d_extrinsics,
                                        extrinsics_plus, m_num_cameras,
                                        m_stream);

  sfm::utils::PlusAsync(
      T(1.0), m_trust_region_intrinsics[0].data().get(), stepsize,
      m_pcg_x_c.data().get() + D_EXTRINSICS_SIZE * m_num_cameras,
      m_trust_region_intrinsics[1].data().get(),
      INTRINSICS_SIZE * m_num_cameras, m_stream);

  sfm::utils::PlusAsync(T(1.0), m_trust_region_points[0].data().get(), stepsize,
                        m_pcg_x_l.data().get(),
                        m_trust_region_points[1].data().get(),
                        LANDMARK_SIZE * m_num_points, m_stream);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::LinearizeAsync(T rho) const {
  T *extrinsics = m_trust_region_extrinsics[0].data().get();
  T *intrinsics = m_trust_region_intrinsics[0].data().get();
  T *points = m_trust_region_points[0].data().get();

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();
  const int_t *extrinsics_indices = m_n_extrinsics_indices.data().get();
  const int_t *intrinsics_indices = m_n_intrinsics_indices.data().get();
  const int_t *point_indices = m_n_point_indices.data().get();

  const auto &num_extrinsics = m_num_extrinsics;
  const auto &num_intrinsics = m_num_intrinsics;
  const auto &num_cameras = num_extrinsics;
  const auto &num_points = m_num_points;

  const int_t *measurement_indices_by_cameras =
      m_measurement_indices_by_cameras.data().get();
  const int_t *measurement_indices_by_points =
      m_measurement_indices_by_points.data().get();
  const int_t *measurement_dicts_by_cameras =
      m_measurement_dicts_by_cameras.data().get();
  const int_t *measurement_dicts_by_points =
      m_measurement_dicts_by_points.data().get();
  const int_t *measurement_offsets_by_cameras =
      m_measurement_offsets_by_cameras.data().get();
  const int_t *measurement_offsets_by_points =
      m_measurement_offsets_by_points.data().get();

  T *jacobians = m_buffer[0].data().get();

  T *rescaled_errs = m_buffer[2].data().get();

  auto hess_cc = m_hess_cc.data().get();
  auto hess_cl = m_buffer[1].data().get();
  auto hess_ll = m_hess_ll.data().get();
  auto grad_c = m_grad_c.data().get();
  auto grad_l = m_grad_l.data().get();

  auto hess_ll_n = m_buffer[3].data().get();
  auto hess_cl_n = hess_cl;
  auto grad_l_n = m_buffer[4].data().get();

  cudaMemsetAsync(hess_cc, 0,
                  D_CAMERA_SIZE * D_CAMERA_SIZE * num_cameras * sizeof(T),
                  m_stream);
  cudaMemsetAsync(hess_cl, 0,
                  D_CAMERA_SIZE * D_LANDMARK_SIZE * m_num_measurements *
                      sizeof(T),
                  m_stream);
  cudaMemsetAsync(hess_ll, 0,
                  D_LANDMARK_SIZE * D_LANDMARK_SIZE * num_points * sizeof(T),
                  m_stream);
  cudaMemsetAsync(grad_c, 0, D_CAMERA_SIZE * num_extrinsics * sizeof(T),
                  m_stream);
  cudaMemsetAsync(grad_l, 0, D_LANDMARK_SIZE * num_points * sizeof(T),
                  m_stream);

  sfm::ba::LinearizeReprojectionLossFunctionAsync(
      measurement_indices, extrinsics_indices, intrinsics_indices,
      point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, jacobians,
      rescaled_errs, m_option.robust_loss, m_option.loss_radius, num_extrinsics,
      num_intrinsics, num_points, m_num_measurements, m_num_measurements,
      m_stream);

  sfm::ba::ComputeReprojectionLossFunctionHessianGradientProductAsync(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      measurement_indices_by_points, extrinsics_indices, jacobians,
      rescaled_errs, extrinsics_infos, hess_cc, hess_cl_n, hess_ll_n, grad_c,
      grad_l_n, num_cameras, m_num_measurements,
      m_measurement_offsets_by_cameras.size() - 1, m_stream);

  sfm::ba::UpdateHessianSumForPointAsync(
      measurement_dicts_by_points, measurement_offsets_by_points, point_indices,
      hess_ll_n, point_infos, hess_ll, num_points, m_num_measurements,
      m_measurement_offsets_by_points.size() - 1, m_stream);

  sfm::ba::ComputePointDictedReductionAsync(
      measurement_dicts_by_points, measurement_offsets_by_points, point_indices,
      T(1.0), grad_l_n, point_infos, T(1.0), grad_l, num_points,
      m_num_measurements, D_LANDMARK_SIZE,
      m_measurement_offsets_by_points.size() - 1, m_stream);

  sfm::utils::PlusAsync(T(1.0), grad_l, rho, points, grad_l, m_points.size(),
                        m_stream);

  sfm::utils::PlusAsync(T(1.0), grad_l, -rho, m_points_reference.data().get(),
                        grad_l, m_points_reference.size(), m_stream);

  sfm::utils::SetSymmetricMatrixAsync(T(1.00001), T(1e-5), hess_cc,
                                      D_CAMERA_SIZE, num_cameras, m_stream);

  sfm::utils::SetSymmetricMatrixAsync(T(1.00001), T(1e-5) + rho, hess_ll,
                                      LANDMARK_SIZE, num_points, m_stream);

  sfm::utils::CopyToDictedMatrixOfArrayAsync(
      m_measurement_dicts_by_points.data().get(), D_CAMERA_SIZE,
      D_LANDMARK_SIZE, hess_cl, m_buffer[0].data().get(), m_num_measurements,
      m_num_measurements, m_num_measurements, m_stream);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::BuildLinearSystemAsync(
    T ratio, Container<T> &future_schur_reduction_l, cudaEvent_t event) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();

  const int_t *camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *point_indices = m_n_point_indices.data().get();

  T *hess_cc = m_hess_cc.data().get();
  T *hess_cl[2] = {m_buffer[0].data().get(), m_buffer[1].data().get()};
  T *hess_ll = m_hess_ll.data().get();
  T *grad_c = m_grad_c.data().get();
  T *grad_l = m_grad_l.data().get();
  T *reduced_grad_c = m_reduced_grad_c.data().get();
  T *pcg_buffer = m_pcg_buffer.data().get();
  T *buffer = m_buffer[2].data().get();

  auto hess_cc_inv = m_hess_cc_inv.data().get();
  auto hess_ll_inv = m_hess_ll_inv.data().get();

  sfm::utils::RescaleSymmetricMatrixDiagonalAsync(ratio, hess_cc, D_CAMERA_SIZE,
                                                  m_num_cameras, m_stream);

  sfm::utils::RescaleSymmetricMatrixDiagonalAsync(
      ratio, hess_ll, D_LANDMARK_SIZE, m_num_points, m_stream);

  sfm::utils::ComputePositiveDefiniteMatrixInverseAsync(
      T(1.001), T(1e-5), hess_cc, hess_cc_inv, D_CAMERA_SIZE, m_num_cameras,
      m_stream);

  sfm::ba::ComputeHessianPointPointInverseAsync(hess_ll, hess_ll_inv,
                                                m_num_points, m_stream);

  cudaMemcpyAsync(reduced_grad_c, grad_c,
                  D_CAMERA_SIZE * m_num_cameras * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, grad_l, T(0.0), pcg_buffer, LANDMARK_SIZE,
      m_num_points, m_stream);

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  if (future_schur_reduction_l.size() < 1) {
    future_schur_reduction_l.resize(1);
  }

  sfm::utils::InnerProductAsync(
      reduce_buffer, reduce_buffer_size, grad_l, m_grad_l.size(), pcg_buffer,
      future_schur_reduction_l.data().get(), m_stream);

  const auto measurement_dicts_by_cameras =
      m_measurement_dicts_by_cameras.data().get();
  const auto measurement_offsets_by_cameras =
      m_measurement_offsets_by_cameras.data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      camera_indices, point_indices, hess_cl[1], T(-1.0), pcg_buffer,
      camera_infos, point_infos, T(1.0), reduced_grad_c, buffer, m_num_cameras,
      m_num_points, m_num_measurements,
      m_measurement_offsets_by_cameras.size() - 1, m_stream);

  CHECK_CUDA(cudaEventRecord(event, m_stream));

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::BuildLinearSystemSync(
    Container<T> &future_schur_reduction_l, cudaEvent_t event) const {
  CHECK_CUDA(cudaEventSynchronize(event));
  m_schur_reduction_l = future_schur_reduction_l[0];
  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::EvaluateSurrogateFunctionAsync(
    const Container<T> &extrinsics_data, const Container<T> &intrinsics_data,
    const Container<T> &point_data, Container<T> &future_surrogate_f,
    cudaEvent_t event) const {
  assert(extrinsics_data.size() == EXTRINSICS_SIZE * m_num_extrinsics);
  assert(intrinsics_data.size() == INTRINSICS_SIZE * m_num_intrinsics);
  assert(point_data.size() == LANDMARK_SIZE * m_num_points);

  if (extrinsics_data.size() != EXTRINSICS_SIZE * m_num_extrinsics) {
    LOG(ERROR) << "Inconsistent data size of extrinsics" << std::endl;
    exit(-1);
  }

  if (intrinsics_data.size() != INTRINSICS_SIZE * m_num_intrinsics) {
    LOG(ERROR) << "Inconsistent data size of intrinsics" << std::endl;
    exit(-1);
  }

  if (point_data.size() != LANDMARK_SIZE * m_num_points) {
    LOG(ERROR) << "Inconsistent data size of points" << std::endl;
    exit(-1);
  }

  const auto extrinsics = extrinsics_data.data().get();
  const auto intrinsics = intrinsics_data.data().get();
  const auto points = point_data.data().get();

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  T *f_values = m_f_values.data().get();

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights, f_values,
      m_option.robust_loss, m_option.loss_radius, m_num_extrinsics,
      m_num_intrinsics, m_num_points, m_num_measurements, m_num_measurements,
      m_stream);

  if (future_surrogate_f.size() < 2) {
    future_surrogate_f.resize(2);
  }

  sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values,
                          m_num_measurements, future_surrogate_f.data().get(),
                          T(0.0), cub::Sum(), m_stream);

  sfm::utils::TransformReduceAsync(
      reduce_buffer, reduce_buffer_size,
      sfm::utils::MakeCountingIterator<sfm::int_t>(0), point_data.size(),
      [points, points_reference = m_points_reference.data().get()] __device__(
          int_t index) -> T {
        T error = points[index] - points_reference[index];
        return error * error;
      },
      m_future_surrogate_f.data().get() + 1, T(0.0), cub::Sum(), m_stream);

  CHECK_CUDA(cudaEventRecord(event, m_stream));

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::EvaluateSurrogateFunctionSync(
    Container<T> &future_surrogate_f, T rho, std::array<T, 2> &surrogate_f,
    T &cost, cudaEvent_t event) const {
  CHECK_CUDA(cudaEventSynchronize(event));

  cudaMemcpy(surrogate_f.data(), future_surrogate_f.data().get(),
             surrogate_f.size() * sizeof(T), cudaMemcpyDeviceToHost);

  surrogate_f[1] *= 0.5 * rho;

  cost = surrogate_f[0] + surrogate_f[1];

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::
    ComputeReducedCameraMatrixVectorMultiplicationAsync(const T *hess_cc,
                                                        const T *hess_cl,
                                                        const T *hess_ll_inv,
                                                        const T *x, T *y,
                                                        T *buffer) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();
  const int_t *camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *point_indices = m_n_point_indices.data().get();

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_cc, x, T(0.0), y, D_CAMERA_SIZE, m_num_cameras, m_stream);

  sfm::ba::ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hess_cl, x,
      camera_infos, point_infos, T(1.0), buffer, m_num_cameras, m_num_points,
      m_num_measurements, m_num_measurements, true, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, buffer, T(0.0), buffer, LANDMARK_SIZE, m_num_points,
      m_stream);

  sfm::ba::ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices, camera_indices, point_indices, hess_cl, buffer,
      camera_infos, point_infos, T(-1.0), y, m_num_cameras, m_num_points,
      m_num_measurements, m_num_measurements, false, m_stream);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::
    ComputeReducedCameraMatrixVectorMultiplicationAsync(
        const T *hess_cc, std::array<const T *, 2> hess_cl,
        const T *hess_ll_inv, const T *x, T *y, T *temp) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  const int_t *measurement_indices = m_n_measurement_indices.data().get();
  const int_t *camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *point_indices = m_n_point_indices.data().get();

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_cc, x, T(0.0), y, D_CAMERA_SIZE, m_num_cameras, m_stream);

  T *buffer = m_buffer[2].data().get();
  const auto &measurement_dicts_by_points =
      m_measurement_dicts_by_points.data().get();
  const auto &measurement_offsets_by_points =
      m_measurement_offsets_by_points.data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
      measurement_dicts_by_points, measurement_offsets_by_points,
      camera_indices, point_indices, hess_cl[0], T(1.0), x, camera_infos,
      point_infos, T(0.0), temp, buffer, m_num_cameras, m_num_points,
      m_num_measurements, m_measurement_offsets_by_points.size() - 1, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, temp, T(0.0), temp, LANDMARK_SIZE, m_num_points,
      m_stream);

  const auto measurement_dicts_by_cameras =
      m_measurement_dicts_by_cameras.data().get();
  const auto measurement_offsets_by_cameras =
      m_measurement_offsets_by_cameras.data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      camera_indices, point_indices, hess_cl[1], T(-1.0), temp, camera_infos,
      point_infos, T(1.0), y, buffer, m_num_cameras, m_num_points,
      m_num_measurements, m_measurement_offsets_by_cameras.size() - 1,
      m_stream);

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PreCommunicateAsync(
    int_t round) const {
  const int_t index = round == 0 ? 1 : 0;
  const T *src =
      round < 2 ? m_points_corrected.data().get() : m_points.data().get();

  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank || (!m_shared_point_sizes[index][neighbor])) {
      continue;
    }

    T *dst = m_shared_point_data[index][neighbor].data().get();
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        m_shared_point_dicts[index][neighbor].data().get(), LANDMARK_SIZE, 1,
        src, dst, m_num_points, m_shared_point_sizes[index][neighbor],
        m_shared_point_sizes[index][neighbor], m_comm_streams[neighbor]);
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PreCommunicateSync(
    int_t round) const {
  const int_t index = (round + 1) % 2;

  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank || (!m_shared_point_sizes[index][neighbor])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[neighbor]));
    CHECK_CUDA(cudaGetLastError());
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PostCommunicateAsync(
    int_t round) const {
  const int_t index = round == 0 ? 0 : 1;
  T *dst = round < 2 ? m_points_corrected.data().get()
                     : m_points_consensus.data().get();

  if (round == 0) {
    for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
      if (neighbor == m_rank || (!m_shared_point_sizes[index][neighbor])) {
        continue;
      }

      T *src = m_shared_point_data[index][neighbor].data().get();
      sfm::utils::AddFromDictedMatrixOfArrayAsync(
          m_shared_point_dicts[index][neighbor].data().get(), LANDMARK_SIZE, 1,
          src, dst, m_shared_point_sizes[index][neighbor], m_num_points,
          m_shared_point_sizes[index][neighbor], m_stream);
    }

    thrust::divides<T> op;
    auto points_consensus = m_points_corrected.data().get();
    auto consensus_cnts = m_point_consensus_cnts.data().get();

    for (int k = 0; k < 3; k++) {
      thrust::transform(thrust::cuda::par.on(m_stream), points_consensus,
                        points_consensus + m_num_points, consensus_cnts,
                        points_consensus, op);
      points_consensus += m_num_points;
    }
  } else {
    for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
      if (neighbor == m_rank || (!m_shared_point_sizes[index][neighbor])) {
        continue;
      }

      T *src = m_shared_point_data[index][neighbor].data().get();
      sfm::utils::CopyFromDictedMatrixOfArrayAsync(
          m_shared_point_dicts[index][neighbor].data().get(), LANDMARK_SIZE, 1,
          src, dst, m_shared_point_sizes[index][neighbor], m_num_points,
          m_shared_point_sizes[index][neighbor], m_stream);
    }

    if (round == 1) {
      sfm::utils::PlusAsync(T(1.0), m_points_reference.data().get(), T(-1.0),
                            m_points.data().get(),
                            m_points_reference.data().get(),
                            m_points_reference.size(), m_stream);

      sfm::utils::PlusAsync(T(1.0), m_points_reference.data().get(), T(1.0),
                            m_points_corrected.data().get(),
                            m_points_reference.data().get(),
                            m_points_reference.size(), m_stream);
    }
  }

  return 0;
}

template <typename T>
int DouglasRachfordSubproblem<kGPU, T, false>::PostCommunicateSync(
    int_t round) const {
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template class DouglasRachfordSubproblem<kGPU, float, false>;
template class DouglasRachfordSubproblem<kGPU, double, false>;
} // namespace ba
} // namespace sfm
