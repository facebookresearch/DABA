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

#include <glog/logging.h>
#include <mpi.h>
#include <nccl.h>

#include <cuda_device_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <sfm/ba/distributed_problem-inl.h>
#include <sfm/ba/distributed_problem.h>
#include <sfm/ba/functions/ba_async.cuh>
#include <sfm/ba/macro.h>
#include <sfm/optimization/LM.h>
#include <sfm/optimization/PCG.h>
#include <sfm/types.h>
#include <sfm/utils/utils.cuh>
#include <sfm/utils/utils.h>

namespace sfm {
namespace ba {
template <typename T>
DBASubproblem<kGPU, T, false>::DBASubproblem(const Option<T> &option,
                                             int_t rank, int_t num_ranks)
    : DBASubproblemBase<kGPU, T, false>(option, rank, num_ranks),
      m_comm_streams(num_ranks) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);

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
DBASubproblem<kGPU, T, false>::DBASubproblem(const Option<T> &option,
                                             int_t node, int_t num_ranks,
                                             int_t device)
    : DBASubproblemBase<kGPU, T, false>(option, node, num_ranks),
      m_comm_streams(num_ranks), m_device(device) {
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  assert(m_device <= num_devices && m_device >= 0);

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

template <typename T> DBASubproblem<kGPU, T, false>::~DBASubproblem() {
  CHECK_CUDA(cudaSetDevice(m_device));
  CHECK_CUDA(cudaStreamDestroy(m_stream));

  for (auto &comm_stream : m_comm_streams) {
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
  }
}

template <typename T> T DBASubproblem<kGPU, T, false>::GetMemoryUsage() const {
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

  memory += m_rescaled_h_a_g_vecs.size() * sizeof(T);
  memory += m_rescaled_f_s_vecs.size() * sizeof(T);
  memory += m_rescaled_sqrt_weights.size() * sizeof(T);
  memory += m_rescaled_constants.size() * sizeof(T);
  memory += m_f_values.size() * sizeof(T);

  memory += m_nesterov_extrinsics.size() * sizeof(T);
  memory += m_nesterov_intrinsics.size() * sizeof(T);
  memory += m_nesterov_points.size() * sizeof(T);
  memory += m_nesterov_rescaled_h_a_g_vecs.size() * sizeof(T);
  memory += m_nesterov_rescaled_sqrt_weights.size() * sizeof(T);
  memory += m_nesterov_rescaled_constants.size() * sizeof(T);
  memory += m_nesterov_f_values.size() * sizeof(T);

  memory += m_proximal_extrinsics.size() * sizeof(T);
  memory += m_proximal_intrinsics.size() * sizeof(T);
  memory += m_proximal_points.size() * sizeof(T);

  memory += m_trust_region_extrinsics[0].size() * sizeof(T);
  memory += m_trust_region_intrinsics[0].size() * sizeof(T);
  memory += m_trust_region_points[0].size() * sizeof(T);
  memory += m_trust_region_extrinsics[1].size() * sizeof(T);
  memory += m_trust_region_intrinsics[1].size() * sizeof(T);
  memory += m_trust_region_points[1].size() * sizeof(T);

  memory += m_extrinsics_proximal_operator.size() * sizeof(T);
  memory += m_intrinsics_proximal_operator.size() * sizeof(T);
  memory += m_point_proximal_operator.size() * sizeof(T);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    memory += m_send_extrinsics_dicts[rank].size() * sizeof(int_t);
    memory += m_send_intrinsics_dicts[rank].size() * sizeof(int_t);
    memory += m_send_point_dicts[rank].size() * sizeof(int_t);
    memory += m_send_data[rank].size() * sizeof(T);

    memory += m_recv_extrinsics_dicts[rank].size() * sizeof(int_t);
    memory += m_recv_intrinsics_dicts[rank].size() * sizeof(int_t);
    memory += m_recv_point_dicts[rank].size() * sizeof(int_t);
    memory += m_recv_data[rank].size() * sizeof(T);
  }

  for (int_t n = 0; n < 3; n++) {
    memory += m_n_measurement_dicts_by_cameras[n].size() * sizeof(int_t);
    memory += m_n_measurement_indices_by_cameras[n].size() * sizeof(int_t);
    memory += m_n_measurement_offsets_by_cameras[n].size() * sizeof(int_t);
    memory += m_n_measurement_dicts_by_points[n].size() * sizeof(int_t);
    memory += m_n_measurement_indices_by_points[n].size() * sizeof(int_t);
    memory += m_n_measurement_offsets_by_points[n].size() * sizeof(int_t);
  }

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

  memory += m_hess_cc_inv.size() * sizeof(T);
  memory += m_hess_ll_inv.size() * sizeof(T);

  return memory / 1024 / 1024;
}

template <typename T>
T DBASubproblem<kGPU, T, false>::GetCommunicationLoad() const {
  T load = 0;

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    load += m_send_data[rank].size() * sizeof(T);
    load += m_recv_data[rank].size() * sizeof(T);
  }

  return load / 1024 / 1024;
}

template <typename T> int_t DBASubproblem<kGPU, T, false>::GetDevice() const {
  return m_device;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Setup(
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

  std::vector<int_t> measurement_selection;
  std::vector<int_t> measurement_selections[3];

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
      measurement_selections[1].push_back(i);
    } else if (point_info[0] == m_rank) {
      measurement_selections[2].push_back(i);
    } else {
      continue;
    }
  }

  for (int_t n = 0; n < 3; n++) {
    measurement_selection.insert(measurement_selection.end(),
                                 measurement_selections[n].begin(),
                                 measurement_selections[n].end());
  }

  m_n_measurement_sizes[0] = measurement_selections[0].size();
  m_n_measurement_sizes[1] = measurement_selections[1].size();
  m_n_measurement_sizes[2] = measurement_selections[2].size();

  m_n_measurement_offsets[0] = 0;
  m_n_measurement_offsets[1] =
      m_n_measurement_offsets[0] + m_n_measurement_sizes[0];
  m_n_measurement_offsets[2] =
      m_n_measurement_offsets[1] + m_n_measurement_sizes[1];

  m_n_num_measurements = m_n_measurement_sizes[0] + m_n_measurement_sizes[1] +
                         m_n_measurement_sizes[2];

  m_n_measurement_indices.resize(m_n_num_measurements);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_measurement_indices.begin(),
                   m_n_measurement_indices.end(), 0);

  m_num_measurements = measurement_selections[0].size() +
                       measurement_selections[1].size() +
                       measurement_selections[2].size();

  m_extrinsics_ids.resize(m_num_ranks);
  m_intrinsics_ids.resize(m_num_ranks);
  m_point_ids.resize(m_num_ranks);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    m_extrinsics_ids[rank].clear();
    m_intrinsics_ids[rank].clear();
    m_point_ids[rank].clear();
  }

  std::vector<std::unordered_set<int_t>> send_extrinsics_dict_sets(m_num_ranks);
  std::vector<std::unordered_set<int_t>> send_intrinsics_dict_sets(m_num_ranks);
  std::vector<std::unordered_set<int_t>> send_point_dict_sets(m_num_ranks);

  for (const auto &index : measurement_selection) {
    const auto &extrinsics_info = extrinsics_infos[index];
    const auto &intrinsics_info = intrinsics_infos[index];
    const auto &point_info = point_infos[index];

    if (m_extrinsics_ids[extrinsics_info[0]].count(extrinsics_info[1]) == 0) {
      m_extrinsics_ids[extrinsics_info[0]].insert({extrinsics_info[1], -1});
    }

    if (m_intrinsics_ids[intrinsics_info[0]].count(intrinsics_info[1]) == 0) {
      m_intrinsics_ids[intrinsics_info[0]].insert({intrinsics_info[1], -1});
    }

    if (m_point_ids[point_info[0]].count(point_info[1]) == 0) {
      m_point_ids[point_info[0]].insert({point_info[1], -1});
    }

    if (extrinsics_info[0] != m_rank) {
      if (intrinsics_info[0] == m_rank) {
        send_intrinsics_dict_sets[extrinsics_info[0]].insert(
            intrinsics_info[1]);
      }

      if (point_info[0] == m_rank) {
        send_point_dict_sets[extrinsics_info[0]].insert(point_info[1]);
      }
    }

    if (intrinsics_info[0] != m_rank &&
        intrinsics_info[0] != extrinsics_info[0]) {
      if (extrinsics_info[0] == m_rank) {
        send_extrinsics_dict_sets[intrinsics_info[0]].insert(
            extrinsics_info[1]);
      }

      if (point_info[0] == m_rank) {
        send_point_dict_sets[intrinsics_info[0]].insert(point_info[1]);
      }
    }

    if (point_info[0] != m_rank && point_info[0] != extrinsics_info[0] &&
        point_info[0] != intrinsics_info[0]) {
      if (extrinsics_info[0] == m_rank) {
        send_extrinsics_dict_sets[point_info[0]].insert(extrinsics_info[1]);
      }

      if (intrinsics_info[0] == m_rank) {
        send_intrinsics_dict_sets[point_info[0]].insert(intrinsics_info[1]);
      }
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

  m_num_cameras = m_num_extrinsics;

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
      assert(send_extrinsics_dict_sets[rank].size() == 0);
      assert(send_intrinsics_dict_sets[rank].size() == 0);
      assert(send_point_dict_sets[rank].size() == 0);

      if (send_extrinsics_dict_sets[rank].size() != 0 ||
          send_intrinsics_dict_sets[rank].size() != 0 ||
          send_point_dict_sets[rank].size() != 0) {
        LOG(ERROR)
            << "No extrinsics, intrinsics or points should be sent to rank "
            << m_rank << " itself." << std::endl;
        exit(-1);
      }

      continue;
    }

    m_send_extrinsics_sizes[rank] = send_extrinsics_dict_sets[rank].size();
    m_send_extrinsics_dicts[rank].resize(m_send_extrinsics_sizes[rank]);
    m_cpu_send_extrinsics_dicts[rank].clear();
    m_cpu_send_extrinsics_dicts[rank].reserve(m_send_extrinsics_sizes[rank]);
    m_cpu_send_extrinsics_dicts[rank].insert(
        m_cpu_send_extrinsics_dicts[rank].begin(),
        send_extrinsics_dict_sets[rank].begin(),
        send_extrinsics_dict_sets[rank].end());
    std::sort(m_cpu_send_extrinsics_dicts[rank].begin(),
              m_cpu_send_extrinsics_dicts[rank].end());
    cudaMemcpyAsync(m_send_extrinsics_dicts[rank].data().get(),
                    m_cpu_send_extrinsics_dicts[rank].data(),
                    m_send_extrinsics_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_send_intrinsics_sizes[rank] = send_intrinsics_dict_sets[rank].size();
    m_send_intrinsics_dicts[rank].resize(m_send_intrinsics_sizes[rank]);
    m_cpu_send_intrinsics_dicts[rank].clear();
    m_cpu_send_intrinsics_dicts[rank].reserve(m_send_intrinsics_sizes[rank]);
    m_cpu_send_intrinsics_dicts[rank].insert(
        m_cpu_send_intrinsics_dicts[rank].begin(),
        send_intrinsics_dict_sets[rank].begin(),
        send_intrinsics_dict_sets[rank].end());
    std::sort(m_cpu_send_intrinsics_dicts[rank].begin(),
              m_cpu_send_intrinsics_dicts[rank].end());
    cudaMemcpyAsync(m_send_intrinsics_dicts[rank].data().get(),
                    m_cpu_send_intrinsics_dicts[rank].data(),
                    m_send_intrinsics_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_send_point_sizes[rank] = send_point_dict_sets[rank].size();
    m_send_point_dicts[rank].resize(m_send_point_sizes[rank]);
    m_cpu_send_point_dicts[rank].clear();
    m_cpu_send_point_dicts[rank].reserve(m_send_point_sizes[rank]);
    m_cpu_send_point_dicts[rank].insert(m_cpu_send_point_dicts[rank].begin(),
                                        send_point_dict_sets[rank].begin(),
                                        send_point_dict_sets[rank].end());
    std::sort(m_cpu_send_point_dicts[rank].begin(),
              m_cpu_send_point_dicts[rank].end());
    cudaMemcpyAsync(m_send_point_dicts[rank].data().get(),
                    m_cpu_send_point_dicts[rank].data(),
                    m_send_point_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_send_data[rank].resize(EXTRINSICS_SIZE * m_send_extrinsics_sizes[rank] +
                             INTRINSICS_SIZE * m_send_intrinsics_sizes[rank] +
                             LANDMARK_SIZE * m_send_point_sizes[rank]);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank) {
      continue;
    }

    m_recv_extrinsics_sizes[rank] = m_extrinsics_sizes[rank];
    m_recv_extrinsics_dicts[rank].resize(m_recv_extrinsics_sizes[rank]);
    m_cpu_recv_extrinsics_dicts[rank].resize(m_recv_extrinsics_sizes[rank]);
    std::iota(m_cpu_recv_extrinsics_dicts[rank].begin(),
              m_cpu_recv_extrinsics_dicts[rank].end(),
              m_extrinsics_offsets[rank]);
    cudaMemcpyAsync(m_recv_extrinsics_dicts[rank].data().get(),
                    m_cpu_recv_extrinsics_dicts[rank].data(),
                    m_recv_extrinsics_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_recv_intrinsics_sizes[rank] = m_intrinsics_sizes[rank];
    m_recv_intrinsics_dicts[rank].resize(m_recv_intrinsics_sizes[rank]);
    m_cpu_recv_intrinsics_dicts[rank].resize(m_recv_intrinsics_sizes[rank]);
    std::iota(m_cpu_recv_intrinsics_dicts[rank].begin(),
              m_cpu_recv_intrinsics_dicts[rank].end(),
              m_intrinsics_offsets[rank]);
    cudaMemcpyAsync(m_recv_intrinsics_dicts[rank].data().get(),
                    m_cpu_recv_intrinsics_dicts[rank].data(),
                    m_recv_intrinsics_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_recv_point_sizes[rank] = m_point_sizes[rank];
    m_recv_point_dicts[rank].resize(m_recv_point_sizes[rank]);
    m_cpu_recv_point_dicts[rank].resize(m_recv_point_sizes[rank]);
    std::iota(m_cpu_recv_point_dicts[rank].begin(),
              m_cpu_recv_point_dicts[rank].end(), m_point_offsets[rank]);
    cudaMemcpyAsync(m_recv_point_dicts[rank].data().get(),
                    m_cpu_recv_point_dicts[rank].data(),
                    m_recv_point_sizes[rank] * sizeof(int_t),
                    cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream);

    m_recv_data[rank].resize(EXTRINSICS_SIZE * m_recv_extrinsics_sizes[rank] +
                             INTRINSICS_SIZE * m_recv_intrinsics_sizes[rank] +
                             LANDMARK_SIZE * m_recv_point_sizes[rank]);
  }

  sfm::PinnedHostVector<int_t> unsorted_extrinsics_infos;
  sfm::PinnedHostVector<int_t> unsorted_intrinsics_infos;
  sfm::PinnedHostVector<int_t> unsorted_point_infos;

  for (const auto &index : measurement_selection) {
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
  std::vector<int_t> measurement_selection_indices(m_n_num_measurements);
  std::iota(measurement_selection_indices.begin(),
            measurement_selection_indices.end(), 0);
  for (int_t n = 0; n < 2; n++) {
    std::vector<int_t> measurement_n_counts_by_cameras(
        m_extrinsics_sizes[m_rank], 0);

    for (int_t k = m_n_measurement_offsets[n];
         k < m_n_measurement_offsets[n] + m_n_measurement_sizes[n]; k++) {
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

    sort(measurement_selection_indices.begin() + m_n_measurement_offsets[n],
         measurement_selection_indices.begin() + m_n_measurement_offsets[n] +
             m_n_measurement_sizes[n],
         camera_cmp);
    m_n_measurement_dicts_by_cameras[n].resize(m_n_measurement_sizes[n]);
    m_n_measurement_indices_by_cameras[n].resize(m_n_measurement_sizes[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_dicts_by_cameras[n].begin(),
                     m_n_measurement_dicts_by_cameras[n].end(),
                     m_n_measurement_offsets[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_indices_by_cameras[n].begin(),
                     m_n_measurement_indices_by_cameras[n].end(), 0);

    std::vector<int_t> measurement_offsets_by_cameras;
    for (int_t k = m_n_measurement_offsets[n];
         k < m_n_measurement_offsets[n] + m_n_measurement_sizes[n]; k++) {
      if (k == m_n_measurement_offsets[n] ||
          unsorted_extrinsics_infos[measurement_selection_indices[k]] !=
              unsorted_extrinsics_infos[measurement_selection_indices[k - 1]]) {
        measurement_offsets_by_cameras.push_back(k -
                                                 m_n_measurement_offsets[n]);
      }
    }
    measurement_offsets_by_cameras.push_back(m_n_measurement_sizes[n]);
    m_n_measurement_offsets_by_cameras[n].resize(
        measurement_offsets_by_cameras.size());
    cudaMemcpyAsync(m_n_measurement_offsets_by_cameras[n].data().get(),
                    measurement_offsets_by_cameras.data(),
                    measurement_offsets_by_cameras.size() * sizeof(int_t),
                    cudaMemcpyHostToDevice, m_stream);
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  }

  // sort measurements by points
  for (int_t n = 2; n < 3; n++) {
    std::vector<int_t> measurement_n_counts_by_points(m_point_sizes[m_rank], 0);

    for (int_t k = m_n_measurement_offsets[n];
         k < m_n_measurement_offsets[n] + m_n_measurement_sizes[n]; k++) {
      measurement_n_counts_by_points[unsorted_point_infos[k]]++;
    }

    auto point_cmp = [&](int a, int b) {
      const auto &point_a = unsorted_point_infos[a];
      const auto &point_b = unsorted_point_infos[b];
      const auto &camera_a = unsorted_extrinsics_infos[a];
      const auto &camera_b = unsorted_extrinsics_infos[b];
      const auto &point_a_cnts = measurement_n_counts_by_points[point_a];
      const auto &point_b_cnts = measurement_n_counts_by_points[point_b];

      return point_a == point_b
                 ? (camera_a == camera_b ? a < b : camera_a < camera_b)
                 : (point_a_cnts == point_b_cnts ? point_a < point_b
                                                 : point_a_cnts > point_b_cnts);
    };

    sort(measurement_selection_indices.begin() + m_n_measurement_offsets[n],
         measurement_selection_indices.begin() + m_n_measurement_offsets[n] +
             m_n_measurement_sizes[n],
         point_cmp);
    m_n_measurement_dicts_by_points[n].resize(m_n_measurement_sizes[n]);
    m_n_measurement_indices_by_points[n].resize(m_n_measurement_sizes[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_dicts_by_points[n].begin(),
                     m_n_measurement_dicts_by_points[n].end(),
                     m_n_measurement_offsets[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_indices_by_points[n].begin(),
                     m_n_measurement_indices_by_points[n].end(), 0);

    std::vector<int_t> measurement_offsets_by_points;
    for (int_t k = m_n_measurement_offsets[n];
         k < m_n_measurement_offsets[n] + m_n_measurement_sizes[n]; k++) {
      if (k == m_n_measurement_offsets[n] ||
          unsorted_point_infos[measurement_selection_indices[k]] !=
              unsorted_point_infos[measurement_selection_indices[k - 1]]) {
        measurement_offsets_by_points.push_back(k - m_n_measurement_offsets[n]);
      }
    }
    measurement_offsets_by_points.push_back(m_n_measurement_sizes[n]);
    m_n_measurement_offsets_by_points[n].resize(
        measurement_offsets_by_points.size());
    cudaMemcpyAsync(m_n_measurement_offsets_by_points[n].data().get(),
                    measurement_offsets_by_points.data(),
                    measurement_offsets_by_points.size() * sizeof(int_t),
                    cudaMemcpyHostToDevice, m_stream);
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  }

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
    const auto &index = measurement_selection[selection_index];
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

  // TODO: Refactor these dicts and indices for dynamic clustering
  // Allocate GPU memory for each iteration
  m_n_num_extrinsics = m_extrinsics_sizes[m_rank];
  m_n_num_intrinsics = m_intrinsics_sizes[m_rank];
  m_n_num_points = m_point_sizes[m_rank];

  m_n_extrinsics_indices.resize(m_num_extrinsics, -1);
  m_n_intrinsics_indices.resize(m_num_intrinsics, -1);
  m_n_point_indices.resize(m_num_points, -1);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_extrinsics_indices.data() + m_extrinsics_offsets[m_rank],
                   m_n_extrinsics_indices.data() +
                       m_extrinsics_offsets[m_rank] +
                       m_extrinsics_sizes[m_rank],
                   0);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_intrinsics_indices.data() + m_intrinsics_offsets[m_rank],
                   m_n_intrinsics_indices.data() +
                       m_intrinsics_offsets[m_rank] +
                       m_intrinsics_sizes[m_rank],
                   0);

  thrust::sequence(thrust::cuda::par.on(m_stream),
                   m_n_point_indices.data() + m_point_offsets[m_rank],
                   m_n_point_indices.data() + m_point_offsets[m_rank] +
                       m_point_sizes[m_rank],
                   0);

  m_n_extrinsics_dicts.resize(m_n_num_extrinsics, -1);
  m_n_intrinsics_dicts.resize(m_n_num_intrinsics, -1);
  m_n_point_dicts.resize(m_n_num_points, -1);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_extrinsics_dicts.data(),
                   m_n_extrinsics_dicts.data() + m_n_num_extrinsics, 0);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_intrinsics_dicts.data(),
                   m_n_intrinsics_dicts.data() + m_n_num_intrinsics, 0);

  thrust::sequence(thrust::cuda::par.on(m_stream), m_n_point_dicts.data(),
                   m_n_point_dicts.data() + m_n_num_points, 0);

  m_extrinsics_proximal_operator.resize(4 * 4 * m_n_num_extrinsics);
  m_intrinsics_proximal_operator.resize(1 * 8 * m_n_num_intrinsics);
  m_point_proximal_operator.resize(1 * 4 * m_n_num_points);

  m_rescaled_h_a_g_vecs.resize(
      7 * (m_n_num_measurements - m_n_measurement_sizes[0]));
  m_rescaled_f_s_vecs.resize(8 *
                             (m_n_num_measurements - m_n_measurement_sizes[0]));
  m_rescaled_sqrt_weights.resize(m_n_num_measurements -
                                 m_n_measurement_sizes[0]);
  m_rescaled_constants.resize(m_n_num_measurements - m_n_measurement_sizes[0]);
  m_f_values.resize(m_n_num_measurements);

  // sorted measurements by points
  for (int_t n = 0; n < 2; n++) {
    thrust::device_vector<int_t> measurement_n_counts_by_points(
        n == 1 ? m_num_points : m_point_sizes[m_rank], 0);

    m_n_measurement_dicts_by_points[n].resize(m_n_measurement_sizes[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_dicts_by_points[n].begin(),
                     m_n_measurement_dicts_by_points[n].end(),
                     m_n_measurement_offsets[n]);

    sfm::utils::ForEachAsync(
        m_n_measurement_dicts_by_points[n].data().get(),
        m_n_measurement_sizes[n],
        [point_infos = m_point_infos.data().get(),
         measurement_cnts = measurement_n_counts_by_points.data()
                                .get()] __device__(auto index) {
          atomicAdd(measurement_cnts + point_infos[index], 1);
        },
        m_stream);

    auto point_cmp =
        [point_infos = m_point_infos.data().get(),
         camera_infos = m_extrinsics_infos.data().get(),
         measurement_cnts =
             measurement_n_counts_by_points.data().get()] __device__(int_t a,
                                                                     int_t b) {
          const auto point_a = point_infos[a];
          const auto point_b = point_infos[b];
          const auto camera_a = camera_infos[a];
          const auto camera_b = camera_infos[b];
          const auto point_a_cnts = measurement_cnts[point_a];
          const auto point_b_cnts = measurement_cnts[point_b];

          return point_a == point_b
                     ? (camera_a == camera_b ? a < b : camera_a < camera_b)
                     : (point_a_cnts == point_b_cnts
                            ? point_a < point_b
                            : point_a_cnts > point_b_cnts);
        };

    thrust::stable_sort(thrust::cuda::par.on(m_stream),
                        m_n_measurement_dicts_by_points[n].begin(),
                        m_n_measurement_dicts_by_points[n].end(), point_cmp);

    m_n_measurement_indices_by_points[n].resize(m_n_measurement_sizes[n]);

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator(0), m_n_measurement_sizes[n],
        [measurment_indices = m_n_measurement_indices_by_points[n].data().get(),
         measurement_dicts = m_n_measurement_dicts_by_points[n].data().get(),
         offset = m_n_measurement_offsets[n]] __device__(auto k) {
          measurment_indices[measurement_dicts[k] - offset] = k;
        },
        m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    m_n_measurement_offsets_by_points[n].resize(m_n_measurement_sizes[n] + 1,
                                                m_n_measurement_sizes[n]);

    int_t offset_size =
        thrust::copy_if(
            thrust::cuda::par.on(m_stream),
            sfm::utils::MakeCountingIterator<int_t>(0),
            sfm::utils::MakeCountingIterator<int_t>(m_n_measurement_sizes[n]),
            m_n_measurement_offsets_by_points[n].data().get(),
            [point_infos = m_point_infos.data().get(),
             measurement_dicts = m_n_measurement_dicts_by_points[n]
                                     .data()
                                     .get()] __device__(auto k) {
              return k == 0 || point_infos[measurement_dicts[k]] !=
                                   point_infos[measurement_dicts[k - 1]];
            }) -
        m_n_measurement_offsets_by_points[n].data().get() + 1;
    m_n_measurement_offsets_by_points[n].resize(offset_size);
    m_n_measurement_offsets_by_points[n].shrink_to_fit();

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  }

  // sorted measurements by cameras
  for (int_t n = 2; n < 3; n++) {
    thrust::device_vector<int_t> measurement_n_counts_by_cameras(
        m_num_extrinsics, 0);

    m_n_measurement_dicts_by_cameras[n].resize(m_n_measurement_sizes[n]);
    thrust::sequence(thrust::cuda::par.on(m_stream),
                     m_n_measurement_dicts_by_cameras[n].begin(),
                     m_n_measurement_dicts_by_cameras[n].end(),
                     m_n_measurement_offsets[n]);

    sfm::utils::ForEachAsync(
        m_n_measurement_dicts_by_cameras[n].data().get(),
        m_n_measurement_sizes[n],
        [camera_infos = m_extrinsics_infos.data().get(),
         measurement_cnts = measurement_n_counts_by_cameras.data()
                                .get()] __device__(auto index) {
          atomicAdd(measurement_cnts + camera_infos[index], 1);
        },
        m_stream);

    auto camera_cmp =
        [camera_infos = m_extrinsics_infos.data().get(),
         point_infos = m_point_infos.data().get(),
         measurement_cnts =
             measurement_n_counts_by_cameras.data().get()] __device__(int_t a,
                                                                      int_t b) {
          const auto &camera_a = camera_infos[a];
          const auto &camera_b = camera_infos[b];
          const auto &point_a = point_infos[a];
          const auto &point_b = point_infos[b];
          const auto &camera_a_cnts = measurement_cnts[camera_a];
          const auto &camera_b_cnts = measurement_cnts[camera_b];

          return camera_a == camera_b
                     ? (point_a == point_b ? a < b : point_a < point_b)
                     : (camera_a_cnts == camera_b_cnts
                            ? camera_a < camera_b
                            : camera_a_cnts > camera_b_cnts);
        };

    thrust::stable_sort(thrust::cuda::par.on(m_stream),
                        m_n_measurement_dicts_by_cameras[n].begin(),
                        m_n_measurement_dicts_by_cameras[n].end(), camera_cmp);

    m_n_measurement_indices_by_cameras[n].resize(m_n_measurement_sizes[n]);

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator(0), m_n_measurement_sizes[n],
        [measurment_indices =
             m_n_measurement_indices_by_cameras[n].data().get(),
         measurement_dicts = m_n_measurement_dicts_by_cameras[n].data().get(),
         offset = m_n_measurement_offsets[n]] __device__(auto k) {
          measurment_indices[measurement_dicts[k] - offset] = k;
        },
        m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    m_n_measurement_offsets_by_cameras[n].resize(m_n_measurement_sizes[n] + 1,
                                                 m_n_measurement_sizes[n]);

    int_t offset_size =
        thrust::copy_if(
            thrust::cuda::par.on(m_stream),
            sfm::utils::MakeCountingIterator<int_t>(0),
            sfm::utils::MakeCountingIterator<int_t>(m_n_measurement_sizes[n]),
            m_n_measurement_offsets_by_cameras[n].data().get(),
            [camera_infos = m_extrinsics_infos.data().get(),
             measurement_dicts = m_n_measurement_dicts_by_cameras[n]
                                     .data()
                                     .get()] __device__(auto k) {
              return k == 0 || camera_infos[measurement_dicts[k]] !=
                                   camera_infos[measurement_dicts[k - 1]];
            }) -
        m_n_measurement_offsets_by_cameras[n].data().get() + 1;
    m_n_measurement_offsets_by_cameras[n].resize(offset_size);
    m_n_measurement_offsets_by_cameras[n].shrink_to_fit();

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  }

  if (m_option.accelerated) {
    m_nesterov_extrinsics.resize(EXTRINSICS_SIZE * m_num_extrinsics);
    m_nesterov_intrinsics.resize(INTRINSICS_SIZE * m_num_intrinsics);
    m_nesterov_points.resize(LANDMARK_SIZE * m_num_points);
    m_nesterov_rescaled_h_a_g_vecs.resize(
        7 * (m_n_num_measurements - m_n_measurement_sizes[0]));
    m_nesterov_rescaled_sqrt_weights.resize(m_n_num_measurements -
                                            m_n_measurement_sizes[0]);
    m_nesterov_rescaled_constants.resize(m_n_num_measurements -
                                         m_n_measurement_sizes[0]);
    m_nesterov_f_values.resize(m_n_num_measurements);
  }

  m_proximal_extrinsics.resize(EXTRINSICS_SIZE * m_n_num_extrinsics);
  m_proximal_intrinsics.resize(INTRINSICS_SIZE * m_n_num_intrinsics);
  m_proximal_points.resize(LANDMARK_SIZE * m_n_num_points);

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    m_trust_region_extrinsics[0].resize(EXTRINSICS_SIZE * m_n_num_extrinsics);
    m_trust_region_extrinsics[1].resize(EXTRINSICS_SIZE * m_n_num_extrinsics);
    m_trust_region_intrinsics[0].resize(INTRINSICS_SIZE * m_n_num_intrinsics);
    m_trust_region_intrinsics[1].resize(INTRINSICS_SIZE * m_n_num_intrinsics);
    m_trust_region_points[0].resize(LANDMARK_SIZE * m_n_num_points);
    m_trust_region_points[1].resize(LANDMARK_SIZE * m_n_num_points);

    const auto &m_n_num_cameras = m_n_num_extrinsics;
#if TEST
    m_jacobians[0].resize(3 * D_CAMERA_SIZE * m_n_measurement_sizes[0]);
    m_jacobians[1].resize(3 * D_CAMERA_SIZE * m_n_measurement_sizes[1]);
    m_jacobians[2].resize(3 * D_LANDMARK_SIZE * m_n_measurement_sizes[2]);
    m_rescaled_errors[0].resize(3 * m_n_measurement_sizes[0]);
    m_rescaled_errors[1].resize(3 * m_n_measurement_sizes[1]);
    m_rescaled_errors[2].resize(3 * m_n_measurement_sizes[2]);
#endif
    m_hess_cc.resize(D_CAMERA_SIZE * D_CAMERA_SIZE * m_n_num_cameras);
    m_hess_ll.resize(LANDMARK_SIZE * LANDMARK_SIZE * m_n_num_points);
    m_grad_c.resize(D_CAMERA_SIZE * m_n_num_cameras);
    m_grad_l.resize(LANDMARK_SIZE * m_n_num_points);
    m_reduced_grad_c.resize(D_CAMERA_SIZE * m_n_num_cameras);

    m_hess_cc_inv.resize(D_CAMERA_SIZE * D_CAMERA_SIZE * m_n_num_cameras);
    m_hess_ll_inv.resize(LANDMARK_SIZE * LANDMARK_SIZE * m_n_num_points);

    m_pcg_x_c.resize(D_CAMERA_SIZE * m_n_num_cameras);
    m_pcg_x_l.resize(LANDMARK_SIZE * m_n_num_points);
    m_pcg_r_c.resize(D_CAMERA_SIZE * m_n_num_cameras);
    m_pcg_dx_c.resize(D_CAMERA_SIZE * m_n_num_cameras);
    m_pcg_dr_c.resize(D_CAMERA_SIZE * m_n_num_cameras);
    m_pcg_dz_c.resize(D_CAMERA_SIZE * m_n_num_cameras);

    m_buffer[0].resize(
        std::max({// buffer for extrinsics_proximal_operator_n for full
                  // reprojection error
                  16 * m_n_measurement_sizes[0],
                  // buffer for hess_cl ordered by cameras
                  D_CAMERA_SIZE * D_LANDMARK_SIZE * m_n_measurement_sizes[0],
                  // buffer for jacobinas of linearization
                  3 * D_CAMERA_SIZE * m_n_measurement_sizes[0],
                  3 * D_CAMERA_SIZE * m_n_measurement_sizes[1],
                  3 * D_LANDMARK_SIZE * m_n_measurement_sizes[2]}));
    // buffer for hess_cl ordered by points
    m_buffer[1].resize(
        std::max({D_CAMERA_SIZE * D_LANDMARK_SIZE * m_n_measurement_sizes[0],
                  // buffer for rescaled_h_a_g_vecs, rescaled_f_s_vecs,
                  // rescaled_sqrt_weights, rescaled_constants related to
                  // camera-point reprojection error
                  (7 + 8 + 1 + 1) * m_n_measurement_sizes[0]}));
    m_buffer[2].resize(
        // buffer for camera block sparse matrix multiplication
        std::max({D_CAMERA_SIZE * m_n_measurement_sizes[0],
                  // buffer for rescaled errors of linearization
                  3 * m_n_measurement_sizes[0], 3 * m_n_measurement_sizes[1],
                  3 * m_n_measurement_sizes[2]}));
    m_buffer[3].resize(std::max(
        {// buffer for hess_ll_n
         (D_LANDMARK_SIZE + 1) * D_LANDMARK_SIZE / 2 *
             std::max(m_n_measurement_sizes[0], m_n_measurement_sizes[2]),
         // buffer for point_proximal_operator_n
         4 * std::max(m_n_measurement_sizes[0], m_n_measurement_sizes[2]),
         // buffer for reducing objective/surrogate function
         m_n_measurement_sizes[0], m_n_measurement_sizes[1],
         m_n_measurement_sizes[2],
         // buffer for inner product
         D_CAMERA_SIZE * m_n_num_cameras, D_LANDMARK_SIZE * m_n_num_points}));
    m_buffer[4].resize(std::max(
        {// buffer for gradient_n
         D_LANDMARK_SIZE * m_n_measurement_sizes[0],
         D_LANDMARK_SIZE * m_n_measurement_sizes[2],
         // buffer for preconditioned conjugate gradient method
         D_LANDMARK_SIZE * m_n_num_points, D_CAMERA_SIZE * m_n_num_cameras}));
  } else {
    // buffer for extrinsics_proximal_operator_n for full reprojection error
    m_buffer[0].resize(16 * m_n_measurement_sizes[0]);
    // buffer for rescaled_h_a_g_vecs, rescaled_f_s_vecs, rescaled_sqrt_weights,
    // rescaled_constants related to camera-point reprojection error
    m_buffer[1].resize((7 + 8 + 1 + 1) * m_n_measurement_sizes[0]);
    m_buffer[3].resize(std::max(
        {// buffer for point_proximal_operator_n
         4 * std::max(m_n_measurement_sizes[0], m_n_measurement_sizes[2]),
         // reduce buffer for objective/surrogate function
         m_n_measurement_sizes[0], m_n_measurement_sizes[1],
         m_n_measurement_sizes[2]}));
  }

  m_future_surrogate_f.resize(3);
  m_future_objective_f.resize(3);
  m_future_inner_product.resize(3);
  m_future_surrogate_f.shrink_to_fit();
  m_future_objective_f.shrink_to_fit();
  m_future_inner_product.shrink_to_fit();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  if (m_option.accelerated) {
    thrust::device_vector<int_t> camera_rank_infos(m_num_extrinsics -
                                                   m_n_num_extrinsics);
    thrust::device_vector<int_t> point_rank_infos(m_num_points -
                                                  m_n_num_points);

    for (int_t rank = 0; rank < m_num_ranks; rank++) {
      if (rank == m_rank) {
        continue;
      }

      if (m_recv_extrinsics_sizes[rank] > 0) {
        sfm::utils::ForEachAsync(
            m_recv_extrinsics_dicts[rank].data().get(),
            m_recv_extrinsics_sizes[rank],
            [rank, rank_infos = camera_rank_infos.data().get() -
                                m_n_num_extrinsics] __device__(auto index) {
              rank_infos[index] = rank;
            },
            m_stream);
      }

      if (m_recv_point_sizes[rank] > 0) {
        sfm::utils::ForEachAsync(
            m_recv_point_dicts[rank].data().get(), m_recv_point_sizes[rank],
            [rank, rank_infos = point_rank_infos.data().get() -
                                m_n_num_points] __device__(auto index) {
              rank_infos[index] = rank;
            },
            m_stream);
      }
    }

    thrust::device_vector<int_t> measurement_rank_infos(
        m_n_measurement_sizes[1] + m_n_measurement_sizes[2]);
    thrust::device_vector<int_t> measurement_offsets_by_ranks(
        m_num_ranks + 1, m_n_measurement_sizes[1] + m_n_measurement_sizes[2]);

    m_n_measurement_dicts_by_ranks.resize(m_n_measurement_sizes[1] +
                                          m_n_measurement_sizes[2]);
    thrust::sequence(
        thrust::cuda::par.on(m_stream), m_n_measurement_dicts_by_ranks.begin(),
        m_n_measurement_dicts_by_ranks.end(), m_n_measurement_offsets[1]);

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator(m_n_measurement_offsets[1]),
        m_n_measurement_sizes[1],
        [measurement_rank_infos =
             measurement_rank_infos.data().get() - m_n_measurement_offsets[1],
         point_rank_infos = point_rank_infos.data().get() - m_n_num_points,
         point_infos = m_point_infos.data().get()] __device__(auto n) {
          measurement_rank_infos[n] = point_rank_infos[point_infos[n]];
        },
        m_stream);

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator(m_n_measurement_offsets[2]),
        m_n_measurement_sizes[2],
        [measurement_rank_infos =
             measurement_rank_infos.data().get() - m_n_measurement_offsets[1],
         camera_rank_infos =
             camera_rank_infos.data().get() - m_n_num_extrinsics,
         camera_infos = m_extrinsics_infos.data().get()] __device__(auto n) {
          measurement_rank_infos[n] = camera_rank_infos[camera_infos[n]];
        },
        m_stream);

    thrust::stable_sort_by_key(
        thrust::cuda::par.on(m_stream), measurement_rank_infos.begin(),
        measurement_rank_infos.end(), m_n_measurement_dicts_by_ranks.begin());

    int_t num_neighbors =
        thrust::copy_if(
            thrust::cuda::par.on(m_stream),
            sfm::utils::MakeCountingIterator<int_t>(0),
            sfm::utils::MakeCountingIterator<int_t>(m_n_measurement_sizes[1] +
                                                    m_n_measurement_sizes[2]),
            measurement_offsets_by_ranks.data().get(),
            [rank_infos =
                 measurement_rank_infos.data().get()] __device__(auto k) {
              return k == 0 || rank_infos[k - 1] != rank_infos[k];
            }) -
        measurement_offsets_by_ranks.data().get();

    measurement_offsets_by_ranks.resize(
        num_neighbors + 1, m_n_measurement_sizes[1] + m_n_measurement_sizes[2]);
    m_n_measurement_offsets_by_ranks.resize(
        num_neighbors + 1, m_n_measurement_sizes[1] + m_n_measurement_sizes[2]);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    thrust::copy(measurement_offsets_by_ranks.begin(),
                 measurement_offsets_by_ranks.end(),
                 m_n_measurement_offsets_by_ranks.begin());

    m_n_measurement_sizes_by_ranks.resize(num_neighbors);
    for (int_t n = 0; n < num_neighbors; n++) {
      m_n_measurement_sizes_by_ranks[n] =
          m_n_measurement_offsets_by_ranks[n + 1] -
          m_n_measurement_offsets_by_ranks[n];
    }

    m_n_measurement_offsets_by_ranks.resize(num_neighbors);
    m_nesterov_reduce_buffer.resize(num_neighbors);
    m_nesterov_future_reduced_f.resize(3 * num_neighbors);
    m_nesterov_reduced_f.resize(3 * num_neighbors);

    for (int_t n = 0; n < m_n_measurement_offsets_by_ranks.size(); n++) {
      size_t buffer_size;
      sfm::utils::ReduceAsync(nullptr, buffer_size, m_f_values.data().get(),
                              m_n_measurement_sizes_by_ranks[n],
                              m_nesterov_future_reduced_f.data().get() + n,
                              T(0), cub::Sum(), m_stream);
      m_nesterov_reduce_buffer[n].resize(buffer_size);
    }
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Accept(const Container<T> &extrinsics,
                                          const Container<T> &intrinsics,
                                          const Container<T> &points,
                                          const Container<T> &f_values,
                                          const std::array<T, 3> &surrogate_f,
                                          T cost) const {
  if (m_option.accelerated) {
    m_extrinsics.swap(m_nesterov_extrinsics);
    m_intrinsics.swap(m_nesterov_intrinsics);
    m_points.swap(m_nesterov_points);

    auto num_neighbors = m_n_measurement_offsets_by_ranks.size();

    if (m_nesterov_future_reduced_f.size() < 3 * num_neighbors) {
      m_nesterov_future_reduced_f.resize(3 * num_neighbors);
    }

    for (int_t n = 0; n < num_neighbors; n++) {
      auto reduce_buffer_n = (void *)m_nesterov_reduce_buffer[n].data().get();
      auto reduce_buffer_size_n = m_nesterov_reduce_buffer[n].size();
      sfm::utils::ReduceByIndexAsync(
          reduce_buffer_n, reduce_buffer_size_n,
          m_n_measurement_dicts_by_ranks.data().get() +
              m_n_measurement_offsets_by_ranks[n],
          m_n_measurement_sizes_by_ranks[n], f_values.data().get(),
          m_nesterov_future_reduced_f.data().get() + num_neighbors + n,
          m_nesterov_reduced_f[num_neighbors + n] -
              0.5 * m_nesterov_reduced_f[n],
          cub::Sum(), m_stream);
    }
  }

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_extrinsics_dicts.data().get(), EXTRINSICS_SIZE, 1,
      extrinsics.data().get(), m_extrinsics.data().get(), m_n_num_extrinsics,
      m_num_extrinsics, m_n_num_extrinsics, m_stream);

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_intrinsics_dicts.data().get(), INTRINSICS_SIZE, 1,
      intrinsics.data().get(), m_intrinsics.data().get(), m_n_num_intrinsics,
      m_num_intrinsics, m_n_num_intrinsics, m_stream);

  sfm::utils::CopyFromDictedMatrixOfArrayAsync(
      m_n_point_dicts.data().get(), LANDMARK_SIZE, 1, points.data().get(),
      m_points.data().get(), m_n_num_points, m_num_points, m_n_num_points,
      m_stream);

  m_surrogate_f = surrogate_f;
  m_cost = cost;

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::GetExtrinsics(
    std::vector<Eigen::Matrix<T, 3, 4>> &extrinsics) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_extrinsics, extrinsics, m_extrinsics_offsets[m_rank],
      m_extrinsics_sizes[m_rank]);
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::GetIntrinsics(
    std::vector<Eigen::Matrix<T, 3, 1>> &intrinsics) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_intrinsics, intrinsics, m_intrinsics_offsets[m_rank],
      m_intrinsics_sizes[m_rank]);
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::GetPoints(
    std::vector<Eigen::Matrix<T, 3, 1>> &points) const {
  sfm::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(
      m_points, points, m_point_offsets[m_rank], m_point_sizes[m_rank]);
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Initialize(
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

  this->InitializeSurrogateFunction();
  this->InitializeTrustRegionMethod();
  this->InitializeNesterovAcceleration();

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Initialize(
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

  this->InitializeSurrogateFunction();
  this->InitializeTrustRegionMethod();
  this->InitializeNesterovAcceleration();

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::InitializeSurrogateFunction() {
  auto &future_surrogate_f = m_future_surrogate_f;
  const auto extrinsics = m_extrinsics.data().get();
  const auto intrinsics = m_intrinsics.data().get();
  const auto points = m_points.data().get();
  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  T *f_values[3] = {m_f_values.data().get() + m_n_measurement_offsets[0],
                    m_f_values.data().get() + m_n_measurement_offsets[1],
                    m_f_values.data().get() + m_n_measurement_offsets[2]};

  if (future_surrogate_f.size() < 3) {
    future_surrogate_f.resize(3);
  }

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  for (int_t n = 0; n < 3; n++) {
    sfm::ba::EvaluateReprojectionLossFunctionAsync(
        measurement_indices[n], extrinsics, intrinsics, points, measurements,
        extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
        f_values[n], m_option.robust_loss, m_option.loss_radius,
        m_num_extrinsics, m_num_intrinsics, m_num_points, m_num_measurements,
        m_n_measurement_sizes[n], m_stream);

    sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values[n],
                            m_n_measurement_sizes[n],
                            future_surrogate_f.data().get() + n, T(0.0),
                            cub::Sum(), m_stream);
  }

  if (m_option.accelerated) {
    auto num_neighbors = m_n_measurement_offsets_by_ranks.size();

    if (m_nesterov_reduced_f.size() < 3 * num_neighbors) {
      m_nesterov_reduced_f.resize(3 * num_neighbors);
    }

    for (int_t n = 0; n < num_neighbors; n++) {
      auto reduce_buffer_n = m_nesterov_reduce_buffer[n].data().get();
      auto reduce_buffer_size_n = m_nesterov_reduce_buffer[n].size();

      sfm::utils::ReduceByIndexAsync(
          reduce_buffer_n, reduce_buffer_size_n,
          m_n_measurement_dicts_by_ranks.data().get() +
              m_n_measurement_offsets_by_ranks[n],
          m_n_measurement_sizes_by_ranks[n], m_f_values.data().get(),
          m_nesterov_future_reduced_f.data().get() + n, T(0.0), cub::Sum(),
          m_comm_streams[n]);
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  cudaMemcpy(m_surrogate_f.data(), future_surrogate_f.data().get(),
             m_surrogate_f.size() * sizeof(T), cudaMemcpyDeviceToHost);

  m_surrogate_f[1] *= 0.5;
  m_surrogate_f[2] *= 0.5;
  m_cost = m_surrogate_f[0] + m_surrogate_f[1] + m_surrogate_f[2];
  m_surrogate_f_constant = 0;

  if (m_option.accelerated) {
    auto num_neighbors = m_n_measurement_offsets_by_ranks.size();

    for (int_t n = 0; n < num_neighbors; n++) {
      CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[n]));
      CHECK_CUDA(cudaGetLastError());
    }

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator<int_t>(0), num_neighbors,
        [objective_f = m_nesterov_future_reduced_f.data().get(),
         surrogate_f_0 =
             m_nesterov_future_reduced_f.data().get() + num_neighbors,
         surrogate_f_1 = m_nesterov_future_reduced_f.data().get() +
                         2 * num_neighbors] __device__(auto n) {
          surrogate_f_0[n] = 0.5 * objective_f[n];
          surrogate_f_1[n] = 0.5 * objective_f[n];
        },
        m_stream);

    cudaMemcpy(m_nesterov_reduced_f.data(),
               m_nesterov_future_reduced_f.data().get(),
               3 * sizeof(T) * num_neighbors, cudaMemcpyDeviceToHost);

    for (int_t n = 0; n < num_neighbors; n++) {
      m_nesterov_reduced_f[num_neighbors + n] = 0.5 * m_nesterov_reduced_f[n];
      m_nesterov_reduced_f[2 * num_neighbors + n] =
          0.5 * m_nesterov_reduced_f[n];
    }

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::InitializeNesterovAcceleration() {
  if (m_option.accelerated) {
    thrust::copy(thrust::cuda::par.on(m_stream), m_extrinsics.begin(),
                 m_extrinsics.end(), m_nesterov_extrinsics.begin());
    thrust::copy(thrust::cuda::par.on(m_stream), m_intrinsics.begin(),
                 m_intrinsics.end(), m_nesterov_intrinsics.begin());
    thrust::copy(thrust::cuda::par.on(m_stream), m_points.begin(),
                 m_points.end(), m_nesterov_points.begin());

    m_nesterov_eta = m_option.initial_nesterov_eta;
    m_nesterov_s.clear();
    m_nesterov_beta.clear();

    m_nesterov_s.push_back(1);
    m_nesterov_avg_objective_value =
        std::max(m_option.initial_nesterov_average_objective_value_ratio,
                 (T)1.0) *
        m_cost;
  }

  return 0;
}

template <typename T> int DBASubproblem<kGPU, T, false>::Iterate() const {
  CHECK_CUDA(cudaSetDevice(m_device));

  if (m_option.accelerated) {
    PreNesterovUpdate();
    NesterovUpdate();
    PostNesterovUpdate();
  } else {
    Update();
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PreCommunicate(bool nesterov) const {
  PreCommunicateAsync(nesterov);
  PreCommunicateSync();

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PostCommunicate(bool nesterov) const {
  PostCommunicateAsync(nesterov);
  PostCommunicateSync();
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::MPICommunicate(const MPI_Comm &comm,
                                                  bool nesterov) const {
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  assert(mpi_rank == m_rank);

  if (mpi_rank != m_rank) {
    LOG(ERROR) << "The rank is not consistent with MPI." << std::endl;
    MPI_Abort(comm, -1);
    exit(-1);
  }

  PreCommunicate(nesterov);

  CHECK_CUDA(cudaSetDevice(m_device));

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_send_extrinsics_sizes[rank] && !m_send_intrinsics_sizes[rank] &&
         !m_send_point_sizes[rank])) {
      continue;
    }

    m_cpu_send_data[rank].resize(m_send_data[rank].size());

    cudaMemcpyAsync(
        m_cpu_send_data[rank].data(), m_send_data[rank].data().get(),
        m_cpu_send_data[rank].size() * sizeof(T),
        cudaMemcpyKind::cudaMemcpyDeviceToHost, m_comm_streams[rank]);
  }

  std::vector<MPI_Request> send_requests(m_num_ranks);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_send_extrinsics_sizes[rank] && !m_send_intrinsics_sizes[rank] &&
         !m_send_point_sizes[rank])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[rank]));
    MPI_Isend(m_cpu_send_data[rank].data(), m_cpu_send_data[rank].size(),
              traits<T>::MPI_FLOAT_TYPE, rank,
              rank * m_num_ranks + m_rank + nesterov, comm,
              &send_requests[rank]);
  }

  std::vector<MPI_Request> recv_requests(m_num_ranks);
  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_recv_extrinsics_sizes[rank] && !m_recv_intrinsics_sizes[rank] &&
         !m_recv_point_sizes[rank])) {
      continue;
    }

    m_cpu_recv_data[rank].resize(m_recv_data[rank].size());
    MPI_Irecv(m_cpu_recv_data[rank].data(), m_cpu_recv_data[rank].size(),
              traits<T>::MPI_FLOAT_TYPE, rank,
              m_rank * m_num_ranks + rank + nesterov, comm,
              &recv_requests[rank]);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_recv_extrinsics_sizes[rank] && !m_recv_intrinsics_sizes[rank] &&
         !m_recv_point_sizes[rank])) {
      continue;
    }

    MPI_Status status;

    MPI_Wait(&recv_requests[rank], &status);
    cudaMemcpyAsync(
        m_recv_data[rank].data().get(), m_cpu_recv_data[rank].data(),
        m_cpu_recv_data[rank].size() * sizeof(T),
        cudaMemcpyKind::cudaMemcpyHostToDevice, m_comm_streams[rank]);
  }

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_recv_extrinsics_sizes[rank] && !m_recv_intrinsics_sizes[rank] &&
         !m_recv_point_sizes[rank])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[rank]));
  }

  PostCommunicate(nesterov);

  for (int_t rank = 0; rank < m_num_ranks; rank++) {
    if (rank == m_rank ||
        (!m_send_extrinsics_sizes[rank] && !m_send_intrinsics_sizes[rank] &&
         !m_send_point_sizes[rank])) {
      continue;
    }

    MPI_Status status;
    MPI_Wait(&send_requests[rank], &status);
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::NCCLCommunicate(const ncclComm_t &comm,
                                                   bool nesterov) const {
  int device;
  ncclCommCuDevice(comm, &device);

  assert(device == m_device && m_rank == m_device);

  if (device != m_device || m_rank != m_device) {
    LOG(ERROR) << "The communicator or rank is not consistent with the device."
               << std::endl;
    exit(-1);
  }

  PreCommunicate(nesterov);

  NCCL_CHECK(ncclGroupStart());
  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor != m_rank && (m_send_extrinsics_sizes[neighbor] > 0 ||
                               m_send_intrinsics_sizes[neighbor] > 0 ||
                               m_send_point_sizes[neighbor] > 0)) {
      ncclSend(m_send_data[neighbor].data().get(), m_send_data[neighbor].size(),
               sfm::traits<T>::NCCL_FLOAT_TYPE, neighbor, comm, m_stream);
    }

    if (neighbor != m_rank && (m_recv_extrinsics_sizes[neighbor] > 0 ||
                               m_recv_intrinsics_sizes[neighbor] > 0 ||
                               m_recv_point_sizes[neighbor] > 0)) {
      ncclRecv(m_recv_data[neighbor].data().get(), m_recv_data[neighbor].size(),
               sfm::traits<T>::NCCL_FLOAT_TYPE, neighbor, comm, m_stream);
    }
  }
  NCCL_CHECK(ncclGroupEnd());

  CHECK_CUDA(cudaStreamSynchronize(m_stream));

  PostCommunicate(nesterov);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Communicate(
    const std::vector<std::shared_ptr<DBASubproblem<kGPU, T, false>>> &problems,
    bool nesterov) const {
  auto recv_extrinsics =
      nesterov ? m_nesterov_extrinsics.data().get() : m_extrinsics.data().get();
  auto recv_intrinsics =
      nesterov ? m_nesterov_intrinsics.data().get() : m_intrinsics.data().get();
  auto recv_points =
      nesterov ? m_nesterov_points.data().get() : m_points.data().get();

  for (const auto &problem : problems) {
    const int_t neighbor = problem->m_rank;

    if (neighbor == m_rank ||
        (!m_recv_extrinsics_sizes[neighbor] &&
         !m_recv_intrinsics_sizes[neighbor] && !m_recv_point_sizes[neighbor])) {
      continue;
    }

    assert(m_recv_extrinsics_sizes[neighbor] ==
           problem->m_send_extrinsics_sizes[m_rank]);
    assert(m_recv_intrinsics_sizes[neighbor] ==
           problem->m_send_intrinsics_sizes[m_rank]);
    assert(m_recv_point_sizes[neighbor] == problem->m_send_point_sizes[m_rank]);

    if (m_recv_extrinsics_sizes[neighbor] !=
            problem->m_send_extrinsics_sizes[m_rank] ||
        m_recv_intrinsics_sizes[neighbor] !=
            problem->m_send_intrinsics_sizes[m_rank] ||
        m_recv_point_sizes[neighbor] != problem->m_send_point_sizes[m_rank]) {
      LOG(ERROR) << "Inconsistent received and sent data sizes between ranks "
                 << neighbor << "  and " << m_rank << "." << std::endl;
      exit(-1);
    }

    auto send_extrinsics = nesterov
                               ? problem->m_nesterov_extrinsics.data().get()
                               : problem->m_extrinsics.data().get();
    auto send_intrinsics = nesterov
                               ? problem->m_nesterov_intrinsics.data().get()
                               : problem->m_intrinsics.data().get();
    auto send_points = nesterov ? problem->m_nesterov_points.data().get()
                                : problem->m_points.data().get();

    if (m_device == problem->m_device) {
      CHECK_CUDA(cudaSetDevice(m_device));

      sfm::utils::CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
          problem->m_send_extrinsics_dicts[m_rank].data().get(),
          m_recv_extrinsics_dicts[neighbor].data().get(), EXTRINSICS_SIZE, 1,
          send_extrinsics, recv_extrinsics, problem->m_num_extrinsics,
          m_num_extrinsics, m_recv_extrinsics_sizes[neighbor],
          m_comm_streams[neighbor]);

      sfm::utils::CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
          problem->m_send_intrinsics_dicts[m_rank].data().get(),
          m_recv_intrinsics_dicts[neighbor].data().get(), INTRINSICS_SIZE, 1,
          send_intrinsics, recv_intrinsics, problem->m_num_intrinsics,
          m_num_intrinsics, m_recv_intrinsics_sizes[neighbor],
          m_comm_streams[neighbor]);

      sfm::utils::CopyFromDictedMatrixOfArrayToDictedMatrixOfArrayAsync(
          problem->m_send_point_dicts[m_rank].data().get(),
          m_recv_point_dicts[neighbor].data().get(), LANDMARK_SIZE, 1,
          send_points, recv_points, problem->m_num_points, m_num_points,
          m_recv_point_sizes[neighbor], m_comm_streams[neighbor]);
    } else {
      CHECK_CUDA(cudaSetDevice(problem->m_device));

      auto send_data = problem->m_send_data[m_rank].data().get();
      T *dst = send_data;
      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          problem->m_send_extrinsics_dicts[m_rank].data().get(),
          EXTRINSICS_SIZE, 1, send_extrinsics, dst, problem->m_num_extrinsics,
          problem->m_send_extrinsics_sizes[m_rank],
          problem->m_send_extrinsics_sizes[m_rank],
          problem->m_comm_streams[m_rank]);

      dst += EXTRINSICS_SIZE * problem->m_send_extrinsics_sizes[m_rank];
      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          problem->m_send_intrinsics_dicts[m_rank].data().get(),
          INTRINSICS_SIZE, 1, send_intrinsics, dst, problem->m_num_intrinsics,
          problem->m_send_intrinsics_sizes[m_rank],
          problem->m_send_intrinsics_sizes[m_rank],
          problem->m_comm_streams[m_rank]);

      dst += INTRINSICS_SIZE * problem->m_send_intrinsics_sizes[m_rank];
      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          problem->m_send_point_dicts[m_rank].data().get(), LANDMARK_SIZE, 1,
          send_points, dst, problem->m_num_points,
          problem->m_send_point_sizes[m_rank],
          problem->m_send_point_sizes[m_rank], problem->m_comm_streams[m_rank]);

      m_cpu_recv_data[neighbor].resize(problem->m_send_data[m_rank].size());

      cudaMemcpyAsync(m_cpu_recv_data[neighbor].data(), send_data,
                      m_cpu_recv_data[neighbor].size() * sizeof(T),
                      cudaMemcpyKind::cudaMemcpyDeviceToHost,
                      problem->m_comm_streams[m_rank]);
    }
  }

  for (const auto &problem : problems) {
    const int_t neighbor = problem->m_rank;

    if (neighbor == m_rank ||
        (!m_recv_extrinsics_sizes[neighbor] &&
         !m_recv_intrinsics_sizes[neighbor] && !m_recv_point_sizes[neighbor])) {
      continue;
    }

    if (m_device != problem->m_device) {
      CHECK_CUDA(cudaSetDevice(problem->m_device));
      CHECK_CUDA(cudaStreamSynchronize(problem->m_comm_streams[m_rank]));
      CHECK_CUDA(cudaSetDevice(m_device));

      auto recv_data = m_recv_data[neighbor].data().get();
      cudaMemcpyAsync(recv_data, m_cpu_recv_data[neighbor].data(),
                      m_cpu_recv_data[neighbor].size() * sizeof(T),
                      cudaMemcpyKind::cudaMemcpyHostToDevice,
                      m_comm_streams[neighbor]);

      T *src = recv_data;
      sfm::utils::CopyFromDictedMatrixOfArrayAsync(
          m_recv_extrinsics_dicts[neighbor].data().get(), EXTRINSICS_SIZE, 1,
          src, recv_extrinsics, m_recv_extrinsics_sizes[neighbor],
          m_num_extrinsics, m_recv_extrinsics_sizes[neighbor],
          m_comm_streams[neighbor]);

      src += EXTRINSICS_SIZE * m_recv_extrinsics_sizes[neighbor];
      sfm::utils::CopyFromDictedMatrixOfArrayAsync(
          m_recv_intrinsics_dicts[neighbor].data().get(), INTRINSICS_SIZE, 1,
          src, recv_intrinsics, m_recv_intrinsics_sizes[neighbor],
          m_num_intrinsics, m_recv_intrinsics_sizes[neighbor],
          m_comm_streams[neighbor]);

      src += INTRINSICS_SIZE * m_recv_intrinsics_sizes[neighbor];
      sfm::utils::CopyFromDictedMatrixOfArrayAsync(
          m_recv_point_dicts[neighbor].data().get(), LANDMARK_SIZE, 1, src,
          recv_points, m_recv_point_sizes[neighbor], m_num_points,
          m_recv_point_sizes[neighbor], m_comm_streams[neighbor]);
    }
  }

  CHECK_CUDA(cudaSetDevice(m_device));
  for (const auto &problem : problems) {
    const int_t rank = problem->m_rank;

    if (rank == m_rank ||
        (!m_recv_extrinsics_sizes[rank] && !m_recv_intrinsics_sizes[rank] &&
         !m_recv_point_sizes[rank])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[rank]));
  }

  return 0;
}

template <typename T>
const typename DBASubproblem<kGPU, T, false>::template Container<T> &
DBASubproblem<kGPU, T, false>::GetDeviceExtrinsics() const {
  return m_extrinsics;
}

template <typename T>
const typename DBASubproblem<kGPU, T, false>::template Container<T> &

DBASubproblem<kGPU, T, false>::GetDeviceIntrinsics() const {
  return m_intrinsics;
}
template <typename T>

const typename DBASubproblem<kGPU, T, false>::template Container<T> &
DBASubproblem<kGPU, T, false>::GetDevicePoints() const {
  return m_points;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::UpdateSurrogateFunction() {
  CHECK_CUDA(cudaSetDevice(m_device));

  auto &future_objective_f = m_future_objective_f;
  auto &future_surrogate_f = m_future_surrogate_f;

  const auto extrinsics = m_extrinsics.data().get();
  const auto intrinsics = m_intrinsics.data().get();
  const auto points = m_points.data().get();
  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  T *f_values[3] = {m_f_values.data().get() + m_n_measurement_offsets[0],
                    m_f_values.data().get() + m_n_measurement_offsets[1],
                    m_f_values.data().get() + m_n_measurement_offsets[2]};

  const T *rescaled_sqrt_weights[3] = {
      nullptr,
      m_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]),
      m_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[2] - m_n_measurement_sizes[0])};

  const T *rescaled_a_vals[3] = {
      nullptr,
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
          3 * m_n_measurement_sizes[1],
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
          3 * m_n_measurement_sizes[2]};

  const T *rescaled_g_vecs[3] = {
      nullptr,
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
          4 * m_n_measurement_sizes[1],
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
          4 * m_n_measurement_sizes[2]};

  const T *rescaled_constants[3] = {
      nullptr,
      m_rescaled_constants.data().get() +
          (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]),
      m_rescaled_constants.data().get() +
          (m_n_measurement_offsets[2] - m_n_measurement_sizes[0])};

  if (future_objective_f.size() < 2) {
    future_objective_f.resize(2);
  }

  if (future_surrogate_f.size() < 2) {
    future_surrogate_f.resize(2);
  }

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  for (int_t n = 1; n < 3; n++) {
    sfm::ba::EvaluateReprojectionLossFunctionAsync(
        measurement_indices[n], extrinsics, intrinsics, points, measurements,
        extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
        f_values[n], m_option.robust_loss, m_option.loss_radius,
        m_num_extrinsics, m_num_intrinsics, m_num_points, m_num_measurements,
        m_n_measurement_sizes[n], m_stream);

    sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values[n],
                            m_n_measurement_sizes[n],
                            future_objective_f.data().get() + n - 1, T(0.0),
                            cub::Sum(), m_stream);
  }

  if (m_option.accelerated) {
    const int_t num_neighbors = m_n_measurement_offsets_by_ranks.size();

    if (m_nesterov_reduced_f.size() < 3 * num_neighbors) {
      m_nesterov_reduced_f.resize(3 * num_neighbors);
    }

    for (int_t n = 0; n < num_neighbors; n++) {
      auto reduce_buffer_n = (void *)m_nesterov_reduce_buffer[n].data().get();
      auto reduce_buffer_size_n = m_nesterov_reduce_buffer[n].size();
      auto reduce_begin_n = sfm::utils::MakeTransformIterator<T>(
          m_n_measurement_dicts_by_ranks.data().get() +
              m_n_measurement_offsets_by_ranks[n],
          [f_values = m_f_values.data().get()] __device__(auto idx) {
            return f_values[idx];
          });

      sfm::utils::ReduceAsync(reduce_buffer_n, reduce_buffer_size_n,
                              reduce_begin_n, m_n_measurement_sizes_by_ranks[n],
                              m_nesterov_future_reduced_f.data().get() + n,
                              T(0.0), cub::Sum(), m_stream);
    }

    sfm::ba::EvaluatePointSurrogateFunctionAsync(
        measurement_indices[1], points, point_infos, rescaled_a_vals[1],
        rescaled_g_vecs[1], rescaled_constants[1], f_values[1], m_num_points,
        m_n_measurement_sizes[1], m_stream);

    sfm::ba::EvaluateCameraSurrogateFunctionAsync(
        measurement_indices[2], extrinsics, intrinsics, measurements,
        extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights[2],
        rescaled_a_vals[2], rescaled_g_vecs[2], rescaled_constants[2],
        f_values[2], m_num_extrinsics, m_num_intrinsics, m_num_measurements,
        m_n_measurement_sizes[2], m_stream);

    for (int_t n = 0; n < num_neighbors; n++) {
      auto reduce_buffer_n = (void *)m_nesterov_reduce_buffer[n].data().get();
      auto reduce_buffer_size_n = m_nesterov_reduce_buffer[n].size();
      auto reduce_begin_n = sfm::utils::MakeTransformIterator<T>(
          m_n_measurement_dicts_by_ranks.data().get() +
              m_n_measurement_offsets_by_ranks[n],
          [f_values = m_f_values.data().get()] __device__(auto idx) {
            return f_values[idx];
          });

      sfm::utils::ReduceAsync(reduce_buffer_n, reduce_buffer_size_n,
                              reduce_begin_n, m_n_measurement_sizes_by_ranks[n],
                              m_nesterov_future_reduced_f.data().get() +
                                  2 * num_neighbors + n,
                              m_nesterov_reduced_f[2 * num_neighbors + n] -
                                  0.5 * m_nesterov_reduced_f[n],
                              cub::Sum(), m_stream);
    }

    sfm::utils::ForEachAsync(
        sfm::utils::MakeCountingIterator(0), num_neighbors,
        [objective_f = m_nesterov_future_reduced_f.data().get(),
         surrogate_f_0 =
             m_nesterov_future_reduced_f.data().get() + num_neighbors,
         surrogate_f_1 = m_nesterov_future_reduced_f.data().get() +
                         2 * num_neighbors] __device__(auto n) {
          T fobj = objective_f[n];
          T fval[2] = {surrogate_f_0[n], surrogate_f_1[n]};

          if (fval[0] >= 0.5 * fobj && fval[1] >= 0.5 * fobj) {
            fval[0] = 0.5 * fobj;
            fval[1] = 0.5 * fobj;
          } else {
            T fsub = fval[0] + fval[1] - fobj;
            T diff1 = min(fsub, fabs(fval[0] - fval[1]));

            if (fval[0] > fval[1]) {
              fval[0] -= diff1;
            } else {
              fval[1] -= diff1;
            }

            T diff2 = 0.5 * (fsub - diff1);
            fval[0] -= diff2;
            fval[1] -= diff2;
          }

          surrogate_f_0[n] = fval[0];
          surrogate_f_1[n] = fval[1];
        },
        m_stream);
  } else {
    for (int_t n = 1; n < 3; n++) {
      sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values[n],
                              m_n_measurement_sizes[n],
                              future_objective_f.data().get() + n - 1, T(0.0),
                              cub::Sum(), m_stream);
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  if (m_option.accelerated) {
    const int_t num_neighbors = m_n_measurement_offsets_by_ranks.size();

    cudaMemcpy(m_nesterov_reduced_f.data(),
               m_nesterov_future_reduced_f.data().get(),
               sizeof(T) * m_nesterov_reduced_f.size(), cudaMemcpyDeviceToHost);

    T surrogate_f[2] = {0, 0};

    for (int n = 0; n < num_neighbors; n++) {
      surrogate_f[0] += m_nesterov_reduced_f[num_neighbors + n];
      surrogate_f[1] += m_nesterov_reduced_f[n];
    }

    m_surrogate_f[1] = 0.5 * surrogate_f[1];
    m_surrogate_f[2] = 0.5 * surrogate_f[1];

    m_cost = m_surrogate_f[0] + surrogate_f[0];
    m_surrogate_f_constant = surrogate_f[0] - surrogate_f[1];

    T p = m_nesterov_eta / (1 + m_nesterov_eta);
    m_nesterov_avg_objective_value =
        (1 - p) * m_nesterov_avg_objective_value + p * m_cost;

    m_nesterov_eta *= (1 + m_option.increasing_nesterov_eta_ratio);
  } else {
    sfm::PinnedHostVector<T> objective_f(2);

    cudaMemcpy(objective_f.data(), future_objective_f.data().get(),
               objective_f.size() * sizeof(T), cudaMemcpyDeviceToHost);
    m_surrogate_f[1] = 0.5 * objective_f[0];
    m_surrogate_f[2] = 0.5 * objective_f[1];

    m_cost = m_surrogate_f[0] + m_surrogate_f[1] + m_surrogate_f[2];
    m_surrogate_f_constant = 0;
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::ConstructSurrogateFunction() const {
  ConstructSurrogateFunctionAsync();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::EvaluateSurrogateFunction(
    const Container<T> &extrinsics_data, const Container<T> &intrinsics_data,
    const Container<T> &point_data, std::array<T, 3> &surrogate_f, T &cost,
    bool nesterov) const {
  cudaEvent_t event;
  CHECK_CUDA(cudaEventCreate(&event));

  EvaluateSurrogateFunctionAsync(extrinsics_data, intrinsics_data, point_data,
                                 m_future_surrogate_f, nesterov, event);

  EvaluateSurrogateFunctionSync(m_future_surrogate_f, surrogate_f, cost, event);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::EvaluateObjectiveFunction(
    const Container<T> &extrinsics_data, const Container<T> &intrinsics_data,
    const Container<T> &point_data, std::array<T, 3> &surrogate_f) const {
  const auto extrinsics = extrinsics_data.data().get();
  const auto intrinsics = intrinsics_data.data().get();
  const auto points = point_data.data().get();

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  T *f_values[3] = {m_f_values.data().get() + m_n_measurement_offsets[0],
                    m_f_values.data().get() + m_n_measurement_offsets[1],
                    m_f_values.data().get() + m_n_measurement_offsets[2]};

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices[0], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      f_values[0], m_option.robust_loss, m_option.loss_radius, m_num_extrinsics,
      m_num_intrinsics, m_num_points, m_num_measurements,
      m_n_measurement_sizes[0], m_stream);

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices[1], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      f_values[1], m_option.robust_loss, m_option.loss_radius, m_num_extrinsics,
      m_num_intrinsics, m_num_points, m_num_measurements,
      m_n_measurement_sizes[1], m_stream);

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices[2], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      f_values[2], m_option.robust_loss, m_option.loss_radius, m_num_extrinsics,
      m_num_intrinsics, m_num_points, m_num_measurements,
      m_n_measurement_sizes[2], m_stream);

  surrogate_f[0] =
      thrust::reduce(thrust::cuda::par.on(m_stream), f_values[0],
                     f_values[0] + m_n_measurement_sizes[0], T(0), cub::Sum());
  surrogate_f[1] =
      thrust::reduce(thrust::cuda::par.on(m_stream), f_values[1],
                     f_values[1] + m_n_measurement_sizes[1], T(0), cub::Sum());
  surrogate_f[2] =
      thrust::reduce(thrust::cuda::par.on(m_stream), f_values[2],
                     f_values[2] + m_n_measurement_sizes[2], T(0), cub::Sum());

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::TrustRegionMethod(
    const Container<T> &extrinsics, const Container<T> &intrinsics,
    const Container<T> &points, T cost, bool nesterov) const {
  cudaMemcpyAsync(m_trust_region_extrinsics[0].data().get(),
                  extrinsics.data().get(),
                  EXTRINSICS_SIZE * m_n_num_extrinsics * sizeof(T),
                  cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream);

  cudaMemcpyAsync(m_trust_region_intrinsics[0].data().get(),
                  intrinsics.data().get(),
                  INTRINSICS_SIZE * m_n_num_intrinsics * sizeof(T),
                  cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream);

  cudaMemcpyAsync(m_trust_region_points[0].data().get(), points.data().get(),
                  LANDMARK_SIZE * m_n_num_points * sizeof(T),
                  cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream);

  m_trust_region_cost[0] = cost;

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  Base::TrustRegionMethod(nesterov);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::LevenbergMarquardtMethod(
    bool nesterov) const {
  auto pcg_buffer = m_buffer[4].data().get();
  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  sfm::optimization::Objective<T> objective = [this](T &objective) -> void {
    objective = this->m_trust_region_cost[0];
  };

  sfm::optimization::QuadraticModel<T> quadratic_model =
      [this, nesterov]() -> void { this->Linearize(nesterov); };

  sfm::optimization::RescaleDiagonal<T> rescale_diagonal =
      [this, nesterov](T prev_ratio, T curr_ratio) -> void {
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
      [this, pcg_buffer, reduce_buffer,
       &reduce_buffer_size](T &predicted_reduction, T ratio) -> void {
    auto future_inner_product = m_future_inner_product.data().get();

    sfm::utils::PlusAsync(T(1.0), m_pcg_r_c.data().get(), T(1.0),
                          m_reduced_grad_c.data().get(), pcg_buffer,
                          m_pcg_r_c.size(), m_stream);

    sfm::utils::InnerProductAsync(reduce_buffer, reduce_buffer_size,
                                  m_pcg_x_c.data().get(), m_pcg_x_c.size(),
                                  pcg_buffer, future_inner_product, m_stream);

    sfm::utils::MatrixDiagonalWeightedSquaredNormAsync(
        reduce_buffer, reduce_buffer_size, m_pcg_x_c.data().get(),
        m_n_num_extrinsics, D_CAMERA_SIZE, m_hess_cc.data().get(),
        future_inner_product + 1, m_stream);

    sfm::utils::MatrixDiagonalWeightedSquaredNormAsync(
        reduce_buffer, reduce_buffer_size, m_pcg_x_l.data().get(),
        m_n_num_points, D_LANDMARK_SIZE, m_hess_ll.data().get(),
        future_inner_product + 2, m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));

    PinnedHostVector<T> inner_product(3);
    cudaMemcpy(inner_product.data(), m_future_inner_product.data().get(),
               3 * sizeof(T), cudaMemcpyDeviceToHost);

    predicted_reduction = 0.5 * (m_schur_reduction_l - inner_product[0]);

    predicted_reduction += 0.5 * ratio / (1 + ratio) * inner_product[1];

    predicted_reduction += 0.5 * ratio / (1 + ratio) * inner_product[2];
  };

  sfm::optimization::Update<T> update =
      [this, nesterov](T stepsize, T &new_objective) -> void {
    this->Retract(stepsize);
    EvaluateSurrogateFunction(
        this->m_trust_region_extrinsics[1], this->m_trust_region_intrinsics[1],
        this->m_trust_region_points[1], this->m_trust_region_surrogate_f[1],
        m_trust_region_cost[1], nesterov);
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
int DBASubproblem<kGPU, T, false>::Linearize(bool nesterov) const {
  LinearizeAsync(nesterov);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::BuildLinearSystem(T ratio) const {
  cudaEvent_t event;
  CHECK_CUDA(cudaEventCreate(&event));

  BuildLinearSystemAsync(ratio, m_future_inner_product, event);
  BuildLinearSystemSync(m_future_inner_product, event);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PCG(int_t &num_iters,
                                       T &update_step_norm) const {
  std::array<const int_t *, 3> measurement_indices = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  std::array<int_t, 3> measurement_sizes = {m_n_measurement_sizes[0],
                                            m_n_measurement_sizes[1],
                                            m_n_measurement_sizes[2]};

  const int_t *n_camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  const auto &m_n_num_cameras = m_n_num_extrinsics;

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  sfm::optimization::SymmetricLinearOperator<Container<T>> hessian =
      [this](const Container<T> &x, Container<T> &y) -> void {
    const T *hess_cc = m_hess_cc.data().get();
    const T *hess_cl[2] = {m_buffer[0].data().get(), m_buffer[1].data().get()};
    const T *hess_ll_inv = m_hess_ll_inv.data().get();
    const T *x_ptr = x.data().get();
    T *y_ptr = y.data().get();
    T *pcg_buffer = m_buffer[4].data().get();

    this->ComputeReducedCameraMatrixVectorMultiplicationAsync(
        hess_cc, {hess_cl[0], hess_cl[1]}, hess_ll_inv, x_ptr, y_ptr,
        pcg_buffer);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::Preconditioner<Container<T>> precondition =
      [this, m_n_num_cameras](const Container<T> &x, Container<T> &y) -> void {
    auto hess_cc_inv = m_hess_cc_inv.data().get();
    auto x_ptr = x.data().get();
    auto y_ptr = y.data().get();

    sfm::utils::ComputeMatrixVectorMultiplicationAsync(
        T(1.0), hess_cc_inv, x_ptr, T(0.0), y_ptr, D_CAMERA_SIZE,
        m_n_num_cameras, m_stream);

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
      [this, m_n_num_cameras](const Container<T> &x, Container<T> &y) -> void {
    auto x_ptr = x.data().get();
    auto y_ptr = y.data().get();

    cudaMemcpyAsync(y_ptr, x_ptr, D_CAMERA_SIZE * m_n_num_cameras * sizeof(T),
                    cudaMemcpyDeviceToDevice, m_stream);

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());
  };

  sfm::optimization::SetZero<Container<T>> set_zero =
      [this, m_n_num_cameras](Container<T> &x) -> void {
    auto x_ptr = x.data().get();

    cudaMemsetAsync(x_ptr, 0, D_CAMERA_SIZE * m_n_num_cameras * sizeof(T),
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
  T *spmv_buffer = m_buffer[2].data().get();
  T *hess_ll_inv = m_hess_ll_inv.data().get();
  T *grad_l = m_grad_l.data().get();
  T *pcg_x_c = m_pcg_x_c.data().get();
  T *pcg_x_l = m_pcg_x_l.data().get();

  cudaMemcpyAsync(pcg_x_l, grad_l, LANDMARK_SIZE * m_n_num_points * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  const auto &measurement_dicts_by_points =
      m_n_measurement_dicts_by_points[0].data().get();
  const auto &measurement_offsets_by_points =
      m_n_measurement_offsets_by_points[0].data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
      measurement_dicts_by_points, measurement_offsets_by_points,
      n_camera_indices, n_point_indices, hess_cl[0], T(1.0), pcg_x_c,
      camera_infos, point_infos, T(1.0), pcg_x_l, spmv_buffer, m_n_num_cameras,
      m_n_num_points, m_n_measurement_sizes[0],
      m_n_measurement_offsets_by_points[0].size() - 1, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(-1.0), hess_ll_inv, pcg_x_l, T(0.0), pcg_x_l, LANDMARK_SIZE,
      m_n_num_points, m_stream);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::Retract(T stepsize) const {
  RetractAsync(stepsize);

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T> int DBASubproblem<kGPU, T, false>::Update() const {
  const int_t *n_extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *n_intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *n_point_dicts = m_n_point_dicts.data().get();

  SolveProximalMethodAsync(false);

  EvaluateSurrogateFunction(m_proximal_extrinsics, m_proximal_intrinsics,
                            m_proximal_points, m_proximal_surrogate_f,
                            m_proximal_cost, false);

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    if (m_proximal_cost < m_cost) {
      m_trust_region_surrogate_f[0] = m_proximal_surrogate_f;
      TrustRegionMethod(m_proximal_extrinsics, m_proximal_intrinsics,
                        m_proximal_points, m_proximal_cost, false);
    } else {
      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          n_extrinsics_dicts, EXTRINSICS_SIZE, 1, m_extrinsics.data().get(),
          m_trust_region_extrinsics[0].data().get(), m_num_extrinsics,
          m_n_num_extrinsics, m_n_num_extrinsics, m_stream);

      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          n_intrinsics_dicts, INTRINSICS_SIZE, 1, m_intrinsics.data().get(),
          m_trust_region_intrinsics[0].data().get(), m_num_intrinsics,
          m_n_num_intrinsics, m_n_num_intrinsics, m_stream);

      sfm::utils::CopyToDictedMatrixOfArrayAsync(
          n_point_dicts, LANDMARK_SIZE, 1, m_points.data().get(),
          m_trust_region_points[0].data().get(), m_num_points, m_n_num_points,
          m_n_num_points, m_stream);

      m_trust_region_surrogate_f[0] = m_surrogate_f;
      m_trust_region_cost[0] = m_cost;

      CHECK_CUDA(cudaStreamSynchronize(m_stream));
      CHECK_CUDA(cudaGetLastError());

      Base::TrustRegionMethod(false);
    }

    Accept(m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
           m_trust_region_points[0], m_f_values, m_trust_region_surrogate_f[0],
           m_trust_region_cost[0]);
  } else {
    Accept(m_proximal_extrinsics, m_proximal_intrinsics, m_proximal_points,
           m_f_values, m_proximal_surrogate_f, m_proximal_cost);
  }

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::TrustRegionUpdate() const {
  const int_t *n_extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *n_intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *n_point_dicts = m_n_point_dicts.data().get();

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_extrinsics_dicts, EXTRINSICS_SIZE, 1, m_extrinsics.data().get(),
        m_trust_region_extrinsics[0].data().get(), m_num_extrinsics,
        m_n_num_extrinsics, m_n_num_extrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_intrinsics_dicts, INTRINSICS_SIZE, 1, m_intrinsics.data().get(),
        m_trust_region_intrinsics[0].data().get(), m_num_intrinsics,
        m_n_num_intrinsics, m_n_num_intrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_point_dicts, LANDMARK_SIZE, 1, m_points.data().get(),
        m_trust_region_points[0].data().get(), m_num_points, m_n_num_points,
        m_n_num_points, m_stream);

    m_trust_region_surrogate_f[0] = m_surrogate_f;
    m_trust_region_cost[0] = m_cost;

    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaGetLastError());

    Base::TrustRegionMethod(false);

    Accept(m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
           m_trust_region_points[0], m_f_values, m_trust_region_surrogate_f[0],
           m_trust_region_cost[0]);
  }
  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::NesterovUpdate() const {
  const int_t *n_extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *n_intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *n_point_dicts = m_n_point_dicts.data().get();

  SolveProximalMethodAsync(true);
  EvaluateSurrogateFunction(m_proximal_extrinsics, m_proximal_intrinsics,
                            m_proximal_points, m_proximal_surrogate_f,
                            m_proximal_cost, true);

  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_extrinsics_dicts, EXTRINSICS_SIZE, 1, m_extrinsics.data().get(),
        m_trust_region_extrinsics[0].data().get(), m_num_extrinsics,
        m_n_num_extrinsics, m_n_num_extrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_intrinsics_dicts, INTRINSICS_SIZE, 1, m_intrinsics.data().get(),
        m_trust_region_intrinsics[0].data().get(), m_num_intrinsics,
        m_n_num_intrinsics, m_n_num_intrinsics, m_stream);

    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        n_point_dicts, LANDMARK_SIZE, 1, m_points.data().get(),
        m_trust_region_points[0].data().get(), m_num_points, m_n_num_points,
        m_n_num_points, m_stream);

    EvaluateSurrogateFunction(
        m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
        m_trust_region_points[0], m_trust_region_surrogate_f[0],
        m_trust_region_cost[0], true);

    if (m_proximal_cost < m_trust_region_cost[0]) {
      m_trust_region_surrogate_f[0] = m_proximal_surrogate_f;
      TrustRegionMethod(m_proximal_extrinsics, m_proximal_intrinsics,
                        m_proximal_points, m_proximal_cost, true);
    } else {
      SolveProximalMethodAsync(false);
      EvaluateSurrogateFunction(m_proximal_extrinsics, m_proximal_intrinsics,
                                m_proximal_points, m_proximal_surrogate_f,
                                m_proximal_cost, true);
      if (m_proximal_cost < m_trust_region_cost[0]) {
        m_trust_region_surrogate_f[0] = m_proximal_surrogate_f;
        TrustRegionMethod(m_proximal_extrinsics, m_proximal_intrinsics,
                          m_proximal_points, m_proximal_cost, true);
      } else {
        TrustRegionMethod(
            m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
            m_trust_region_points[0], m_trust_region_cost[0], true);
      }
    }

    NesterovAdaptiveRestart(
        m_trust_region_extrinsics[0], m_trust_region_intrinsics[0],
        m_trust_region_points[0], m_trust_region_surrogate_f[0],
        m_trust_region_cost[0]);
  } else {
    NesterovAdaptiveRestart(m_proximal_extrinsics, m_proximal_intrinsics,
                            m_proximal_points, m_proximal_surrogate_f,
                            m_proximal_cost);
  }

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PreNesterovUpdate() const {
  PreNesterovUpdateAsync();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PostNesterovUpdate() const {
  PostNesterovUpdateAsync();

  CHECK_CUDA(cudaStreamSynchronize(m_stream));
  CHECK_CUDA(cudaGetLastError());
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::NesterovAdaptiveRestart(
    const Container<T> &extrinsics, const Container<T> &intrinsics,
    const Container<T> &points, std::array<T, 3> &surrogate_f, T &cost) const {
  EvaluateSurrogateFunction(extrinsics, intrinsics, points, surrogate_f, cost,
                            false);
  if (cost < m_nesterov_avg_objective_value) {
    Accept(extrinsics, intrinsics, points, m_f_values, surrogate_f, cost);
  } else {
    Update();
    // m_nesterov_s.back() = std::max(T(1.0), T(0.75 * m_nesterov_s.back()));
  }
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PreNesterovUpdateAsync() const {
  ConstructSurrogateFunctionAsync();

  const int_t *n_extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *n_intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *n_point_dicts = m_n_point_dicts.data().get();

  const auto extrinsics = m_extrinsics.data().get();
  const auto intrinsics = m_intrinsics.data().get();
  const auto points = m_points.data().get();

  auto nesterov_extrinsics = m_nesterov_extrinsics.data().get();
  auto nesterov_intrinsics = m_nesterov_intrinsics.data().get();
  auto nesterov_points = m_nesterov_points.data().get();

  T nesterov_s0 = m_nesterov_s.back();
  T nesterov_s1 = 0.5 + 0.5 * std::sqrt(4.0 * nesterov_s0 * nesterov_s0 + 1.0);
  T nesterov_beta = (nesterov_s0 - 1) / nesterov_s1;
  m_nesterov_s.push_back(nesterov_s1);
  m_nesterov_beta.push_back(nesterov_beta);

  sfm::utils::NesterovExtrapolateSE3Async(nesterov_extrinsics, extrinsics,
                                          nesterov_beta, nesterov_extrinsics,
                                          m_num_extrinsics, m_stream);

  sfm::utils::NesterovExtrapolateMatrixAsync(
      nesterov_intrinsics, intrinsics, nesterov_beta, nesterov_intrinsics,
      INTRINSICS_SIZE, 1, m_num_intrinsics, m_stream);

  sfm::utils::NesterovExtrapolateMatrixAsync(
      nesterov_points, points, nesterov_beta, nesterov_points, LANDMARK_SIZE, 1,
      m_num_points, m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PostNesterovUpdateAsync() const {
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::RetractAsync(T stepsize) const {
  const auto &m_n_num_cameras = m_n_num_extrinsics;

  auto d_extrinsics = m_pcg_x_c.data().get();
  auto extrinsics = m_trust_region_extrinsics[0].data().get();
  auto extrinsics_plus = m_trust_region_extrinsics[1].data().get();
  sfm::utils::ComputeSE3RetractionAsync(extrinsics, stepsize, d_extrinsics,
                                        extrinsics_plus, m_n_num_cameras,
                                        m_stream);

  sfm::utils::PlusAsync(
      T(1.0), m_trust_region_intrinsics[0].data().get(), stepsize,
      m_pcg_x_c.data().get() + D_EXTRINSICS_SIZE * m_n_num_cameras,
      m_trust_region_intrinsics[1].data().get(),
      INTRINSICS_SIZE * m_n_num_cameras, m_stream);

  sfm::utils::PlusAsync(T(1.0), m_trust_region_points[0].data().get(), stepsize,
                        m_pcg_x_l.data().get(),
                        m_trust_region_points[1].data().get(),
                        LANDMARK_SIZE * m_n_num_points, m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::LinearizeAsync(bool nesterov) const {
  T *extrinsics = m_trust_region_extrinsics[0].data().get();
  T *intrinsics = m_trust_region_intrinsics[0].data().get();
  T *points = m_trust_region_points[0].data().get();

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  const int_t *n_extrinsics_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_intrinsics_indices = m_n_intrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  const T *rescaled_sqrt_weights[3] = {nullptr, nullptr, nullptr};
  const T *rescaled_a_vals[3] = {nullptr, nullptr, nullptr};
  const T *rescaled_g_vecs[3] = {nullptr, nullptr, nullptr};
  const T *rescaled_constants[3] = {nullptr, nullptr, nullptr};

  if (nesterov) {
    rescaled_sqrt_weights[1] =
        m_nesterov_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_sqrt_weights[2] =
        m_nesterov_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);

    rescaled_a_vals[1] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[1];
    rescaled_a_vals[2] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[2];

    rescaled_g_vecs[1] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[1];
    rescaled_g_vecs[2] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[2];

    rescaled_constants[1] =
        m_nesterov_rescaled_constants.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_constants[2] =
        m_nesterov_rescaled_constants.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);
  } else {
    rescaled_sqrt_weights[1] =
        m_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_sqrt_weights[2] =
        m_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);

    rescaled_a_vals[1] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[1];
    rescaled_a_vals[2] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[2];

    rescaled_g_vecs[1] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[1];
    rescaled_g_vecs[2] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[2];

    rescaled_constants[1] =
        m_rescaled_constants.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_constants[2] =
        m_rescaled_constants.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);
  }

  const auto &n_num_extrinsics = m_extrinsics_sizes[m_rank];
  const auto &n_num_intrinsics = m_intrinsics_sizes[m_rank];
  const auto &n_num_cameras = n_num_extrinsics;
  const auto &n_num_points = m_point_sizes[m_rank];

  const int_t *measurement_indices_by_cameras[3] = {
      m_n_measurement_indices_by_cameras[0].data().get(),
      m_n_measurement_indices_by_cameras[1].data().get(),
      m_n_measurement_indices_by_cameras[2].data().get()};
  const int_t *measurement_indices_by_points[3] = {
      m_n_measurement_indices_by_points[0].data().get(),
      m_n_measurement_indices_by_points[1].data().get(),
      m_n_measurement_indices_by_points[2].data().get()};
  const int_t *measurement_dicts_by_cameras[3] = {
      m_n_measurement_dicts_by_cameras[0].data().get(),
      m_n_measurement_dicts_by_cameras[1].data().get(),
      m_n_measurement_dicts_by_cameras[2].data().get()};
  const int_t *measurement_dicts_by_points[3] = {
      m_n_measurement_dicts_by_points[0].data().get(),
      m_n_measurement_dicts_by_points[1].data().get(),
      m_n_measurement_dicts_by_points[2].data().get()};
  const int_t *measurement_offsets_by_cameras[3] = {
      m_n_measurement_offsets_by_cameras[0].data().get(),
      m_n_measurement_offsets_by_cameras[1].data().get(),
      m_n_measurement_offsets_by_cameras[2].data().get()};
  const int_t *measurement_offsets_by_points[3] = {
      m_n_measurement_offsets_by_points[0].data().get(),
      m_n_measurement_offsets_by_points[1].data().get(),
      m_n_measurement_offsets_by_points[2].data().get()};

#if TEST
  T *jacobians[3] = {m_jacobians[0].data().get(), m_jacobians[1].data().get(),
                     m_jacobians[2].data().get()};
  T *rescaled_errs[3] = {m_rescaled_errors[0].data().get(),
                         m_rescaled_errors[1].data().get(),
                         m_rescaled_errors[2].data().get()};
#else
  T *jacobians[3] = {m_buffer[0].data().get(), m_buffer[0].data().get(),
                     m_buffer[0].data().get()};

  T *rescaled_errs[3] = {m_buffer[2].data().get(), m_buffer[2].data().get(),
                         m_buffer[2].data().get()};
#endif

  auto hess_cc = m_hess_cc.data().get();
  auto hess_cl = m_buffer[1].data().get();
  auto hess_ll = m_hess_ll.data().get();
  auto grad_c = m_grad_c.data().get();
  auto grad_l = m_grad_l.data().get();

  auto hess_ll_n = m_buffer[3].data().get();
  auto hess_cl_n = hess_cl;
  auto grad_l_n = m_buffer[4].data().get();

  cudaMemsetAsync(hess_cc, 0,
                  D_CAMERA_SIZE * D_CAMERA_SIZE * n_num_cameras * sizeof(T),
                  m_stream);
  cudaMemsetAsync(hess_ll, 0,
                  D_LANDMARK_SIZE * D_LANDMARK_SIZE * n_num_points * sizeof(T),
                  m_stream);
  cudaMemsetAsync(grad_c, 0, D_CAMERA_SIZE * n_num_extrinsics * sizeof(T),
                  m_stream);
  cudaMemsetAsync(grad_l, 0, D_LANDMARK_SIZE * n_num_points * sizeof(T),
                  m_stream);

  sfm::ba::LinearizeCameraSurrogateFunctionAsync(
      measurement_indices[1], n_extrinsics_indices, n_intrinsics_indices,
      extrinsics, intrinsics, measurements, extrinsics_infos, intrinsics_infos,
      rescaled_sqrt_weights[1], rescaled_a_vals[1], rescaled_g_vecs[1],
      rescaled_constants[1], jacobians[1], rescaled_errs[1], n_num_extrinsics,
      n_num_intrinsics, m_num_measurements, m_n_measurement_sizes[1], m_stream);

  sfm::ba::ComputeCameraSurrogateFunctionHessianGradientAsync(
      measurement_dicts_by_cameras[1], measurement_offsets_by_cameras[1],
      n_extrinsics_indices, jacobians[1], rescaled_errs[1], extrinsics_infos,
      hess_cc, grad_c, n_num_cameras, m_n_measurement_sizes[1],
      m_n_measurement_offsets_by_cameras[1].size() - 1, m_stream);

  sfm::ba::LinearizePointSurrogateFunctionAsync(
      measurement_indices[2], n_point_indices, points, point_infos,
      rescaled_a_vals[2], rescaled_g_vecs[2], rescaled_constants[2],
      jacobians[2], rescaled_errs[2], n_num_points, m_n_measurement_sizes[2],
      m_stream);

  sfm::ba::ComputePointSurrogateFunctionHessianGradientProductAsync(
      measurement_indices_by_points[2], jacobians[2], rescaled_errs[2],
      hess_ll_n, grad_l_n, m_n_measurement_sizes[2], m_stream);

  sfm::ba::UpdateHessianSumForPointAsync(
      measurement_dicts_by_points[2], measurement_offsets_by_points[2],
      n_point_indices, hess_ll_n, point_infos, hess_ll, n_num_points,
      m_n_measurement_sizes[2], m_n_measurement_offsets_by_points[2].size() - 1,
      m_stream);

  sfm::ba::ComputePointDictedReductionAsync(
      measurement_dicts_by_points[2], measurement_offsets_by_points[2],
      n_point_indices, T(1.0), grad_l_n, point_infos, T(1.0), grad_l,
      n_num_points, m_n_measurement_sizes[2], D_LANDMARK_SIZE,
      m_n_measurement_offsets_by_points[2].size() - 1, m_stream);

  sfm::ba::LinearizeReprojectionLossFunctionAsync(
      measurement_indices[0], n_extrinsics_indices, n_intrinsics_indices,
      n_point_indices, extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      jacobians[0], rescaled_errs[0], m_option.robust_loss,
      m_option.loss_radius, n_num_extrinsics, n_num_intrinsics, n_num_points,
      m_num_measurements, m_n_measurement_sizes[0], m_stream);

  sfm::ba::ComputeReprojectionLossFunctionHessianGradientProductAsync(
      measurement_dicts_by_cameras[0], measurement_offsets_by_cameras[0],
      measurement_indices_by_points[0], n_extrinsics_indices, jacobians[0],
      rescaled_errs[0], extrinsics_infos, hess_cc, hess_cl_n, hess_ll_n, grad_c,
      grad_l_n, n_num_cameras, m_n_measurement_sizes[0],
      m_n_measurement_offsets_by_cameras[0].size() - 1, m_stream);

  sfm::ba::UpdateHessianSumForPointAsync(
      measurement_dicts_by_points[0], measurement_offsets_by_points[0],
      n_point_indices, hess_ll_n, point_infos, hess_ll, n_num_points,
      m_n_measurement_sizes[0], m_n_measurement_offsets_by_points[0].size() - 1,
      m_stream);

  sfm::ba::ComputePointDictedReductionAsync(
      measurement_dicts_by_points[0], measurement_offsets_by_points[0],
      n_point_indices, T(1.0), grad_l_n, point_infos, T(1.0), grad_l,
      n_num_points, m_n_measurement_sizes[0], D_LANDMARK_SIZE,
      m_n_measurement_offsets_by_points[0].size() - 1, m_stream);

  sfm::utils::SetSymmetricMatrixAsync(T(1.00001), T(1e-5), hess_cc,
                                      D_CAMERA_SIZE, n_num_cameras, m_stream);

  sfm::utils::SetSymmetricMatrixAsync(T(1.00001), T(1e-5), hess_ll,
                                      LANDMARK_SIZE, n_num_points, m_stream);

  sfm::utils::CopyToDictedMatrixOfArrayAsync(
      m_n_measurement_dicts_by_points[0].data().get(), D_CAMERA_SIZE,
      D_LANDMARK_SIZE, hess_cl, m_buffer[0].data().get(),
      m_n_measurement_sizes[0], m_n_measurement_sizes[0],
      m_n_measurement_sizes[0], m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::BuildLinearSystemAsync(
    T ratio, Container<T> &future_schur_reduction_l, cudaEvent_t event) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  std::array<const int_t *, 3> measurement_indices = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  std::array<int_t, 3> measurement_sizes = {m_n_measurement_sizes[0],
                                            m_n_measurement_sizes[1],
                                            m_n_measurement_sizes[2]};

  const int_t *n_camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  const auto &m_n_num_cameras = m_n_num_extrinsics;

  T *hess_cc = m_hess_cc.data().get();
  T *hess_cl[2] = {m_buffer[0].data().get(), m_buffer[1].data().get()};
  T *hess_ll = m_hess_ll.data().get();
  T *grad_c = m_grad_c.data().get();
  T *grad_l = m_grad_l.data().get();
  T *reduced_grad_c = m_reduced_grad_c.data().get();
  T *pcg_buffer = m_buffer[4].data().get();
  T *spmv_buffer = m_buffer[2].data().get();

  auto hess_cc_inv = m_hess_cc_inv.data().get();
  auto hess_ll_inv = m_hess_ll_inv.data().get();

  sfm::utils::RescaleSymmetricMatrixDiagonalAsync(ratio, hess_cc, D_CAMERA_SIZE,
                                                  m_n_num_cameras, m_stream);

  sfm::utils::RescaleSymmetricMatrixDiagonalAsync(
      ratio, hess_ll, D_LANDMARK_SIZE, m_n_num_points, m_stream);

  sfm::utils::ComputePositiveDefiniteMatrixInverseAsync(
      T(1.001), T(1e-5), hess_cc, hess_cc_inv, D_CAMERA_SIZE, m_n_num_cameras,
      m_stream);

  sfm::ba::ComputeHessianPointPointInverseAsync(hess_ll, hess_ll_inv,
                                                m_n_num_points, m_stream);

  cudaMemcpyAsync(reduced_grad_c, grad_c,
                  D_CAMERA_SIZE * m_n_num_cameras * sizeof(T),
                  cudaMemcpyDeviceToDevice, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, grad_l, T(0.0), pcg_buffer, LANDMARK_SIZE,
      m_n_num_points, m_stream);

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  if (future_schur_reduction_l.size() < 1) {
    future_schur_reduction_l.resize(1);
  }

  sfm::utils::InnerProductAsync(
      reduce_buffer, reduce_buffer_size, grad_l, m_grad_l.size(), pcg_buffer,
      future_schur_reduction_l.data().get(), m_stream);

  CHECK_CUDA(cudaEventRecord(event, m_stream));

  const auto measurement_dicts_by_cameras =
      m_n_measurement_dicts_by_cameras[0].data().get();
  const auto measurement_offsets_by_cameras =
      m_n_measurement_offsets_by_cameras[0].data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      n_camera_indices, n_point_indices, hess_cl[1], T(-1.0), pcg_buffer,
      camera_infos, point_infos, T(1.0), reduced_grad_c, spmv_buffer,
      m_n_num_cameras, m_n_num_points, m_n_measurement_sizes[0],
      m_n_measurement_offsets_by_cameras[0].size() - 1, m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::BuildLinearSystemSync(
    Container<T> &future_schur_reduction_l, cudaEvent_t event) const {
  CHECK_CUDA(cudaEventSynchronize(event));
  m_schur_reduction_l = future_schur_reduction_l[0];
  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::ConstructSurrogateFunctionAsync() const {
  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const auto extrinsics = m_extrinsics.data().get();
  const auto intrinsics = m_intrinsics.data().get();
  const auto points = m_points.data().get();

  T *rescaled_sqrt_weights[3] = {
      nullptr,
      m_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]),
      m_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[2] - m_n_measurement_sizes[0])};

  T *rescaled_constants[3] = {
      nullptr,
      m_rescaled_constants.data().get() +
          (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]),
      m_rescaled_constants.data().get() +
          (m_n_measurement_offsets[2] - m_n_measurement_sizes[0])};

  T *rescaled_a_vals[3] = {
      nullptr,
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
          3 * m_n_measurement_sizes[1],
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
          3 * m_n_measurement_sizes[2]};

  T *rescaled_g_vecs[3] = {
      nullptr,
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
          4 * m_n_measurement_sizes[1],
      m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
          4 * m_n_measurement_sizes[2]};

  T *f_values[3] = {m_f_values.data().get() + m_n_measurement_offsets[0],
                    m_f_values.data().get() + m_n_measurement_offsets[1],
                    m_f_values.data().get() + m_n_measurement_offsets[2]};

  sfm::ba::ConstructSurrogateFunctionAsync(
      measurement_indices[1], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_a_vals[1], rescaled_g_vecs[1], rescaled_sqrt_weights[1],
      rescaled_constants[1], f_values[1], m_option.robust_loss,
      m_option.loss_radius, m_num_extrinsics, m_num_intrinsics, m_num_points,
      m_num_measurements, m_n_measurement_sizes[1], m_stream);

  sfm::ba::ConstructSurrogateFunctionAsync(
      measurement_indices[2], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      rescaled_a_vals[2], rescaled_g_vecs[2], rescaled_sqrt_weights[2],
      rescaled_constants[2], f_values[2], m_option.robust_loss,
      m_option.loss_radius, m_num_extrinsics, m_num_intrinsics, m_num_points,
      m_num_measurements, m_n_measurement_sizes[2], m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::EvaluateSurrogateFunctionAsync(
    const Container<T> &extrinsics_data, const Container<T> &intrinsics_data,
    const Container<T> &point_data, Container<T> &future_surrogate_f,
    bool nesterov, cudaEvent_t event) const {
  assert(extrinsics_data.size() == EXTRINSICS_SIZE * m_n_num_extrinsics);
  assert(intrinsics_data.size() == INTRINSICS_SIZE * m_n_num_intrinsics);
  assert(point_data.size() == LANDMARK_SIZE * m_n_num_points);

  if (extrinsics_data.size() != EXTRINSICS_SIZE * m_n_num_extrinsics) {
    LOG(ERROR) << "Inconsistent data size of extrinsics" << std::endl;
    exit(-1);
  }

  if (intrinsics_data.size() != INTRINSICS_SIZE * m_n_num_intrinsics) {
    LOG(ERROR) << "Inconsistent data size of intrinsics" << std::endl;
    exit(-1);
  }

  if (point_data.size() != LANDMARK_SIZE * m_n_num_points) {
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

  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  T *f_values[3];
  const T *rescaled_sqrt_weights[3];
  const T *rescaled_a_vals[3];
  const T *rescaled_g_vecs[3];
  const T *rescaled_constants[3];

  if (nesterov) {
    f_values[0] = m_nesterov_f_values.data().get() + m_n_measurement_offsets[0];
    f_values[1] = m_nesterov_f_values.data().get() + m_n_measurement_offsets[1];
    f_values[2] = m_nesterov_f_values.data().get() + m_n_measurement_offsets[2];

    rescaled_sqrt_weights[0] = nullptr;
    rescaled_sqrt_weights[1] =
        m_nesterov_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_sqrt_weights[2] =
        m_nesterov_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);

    rescaled_a_vals[0] = nullptr;
    rescaled_a_vals[1] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[1];
    rescaled_a_vals[2] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[2];

    rescaled_g_vecs[0] = nullptr;
    rescaled_g_vecs[1] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[1];
    rescaled_g_vecs[2] =
        m_nesterov_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[2];

    rescaled_constants[0] = nullptr;
    rescaled_constants[1] =
        m_nesterov_rescaled_constants.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_constants[2] =
        m_nesterov_rescaled_constants.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);
  } else {
    f_values[0] = m_f_values.data().get() + m_n_measurement_offsets[0];
    f_values[1] = m_f_values.data().get() + m_n_measurement_offsets[1];
    f_values[2] = m_f_values.data().get() + m_n_measurement_offsets[2];

    rescaled_sqrt_weights[0] = nullptr;
    rescaled_sqrt_weights[1] =
        m_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_sqrt_weights[2] =
        m_rescaled_sqrt_weights.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);

    rescaled_a_vals[0] = nullptr;
    rescaled_a_vals[1] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[1];
    rescaled_a_vals[2] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        3 * m_n_measurement_sizes[2];

    rescaled_g_vecs[0] = nullptr;
    rescaled_g_vecs[1] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[1];
    rescaled_g_vecs[2] =
        m_rescaled_h_a_g_vecs.data().get() +
        7 * (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]) +
        4 * m_n_measurement_sizes[2];

    rescaled_constants[0] = nullptr;
    rescaled_constants[1] =
        m_rescaled_constants.data().get() +
        (m_n_measurement_offsets[1] - m_n_measurement_sizes[0]);
    rescaled_constants[2] =
        m_rescaled_constants.data().get() +
        (m_n_measurement_offsets[2] - m_n_measurement_sizes[0]);
  }

  sfm::ba::EvaluateReprojectionLossFunctionAsync(
      measurement_indices[0], extrinsics, intrinsics, points, measurements,
      extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
      f_values[0], m_option.robust_loss, m_option.loss_radius,
      m_n_num_extrinsics, m_n_num_intrinsics, m_n_num_points,
      m_num_measurements, m_n_measurement_sizes[0], m_stream);

  sfm::ba::EvaluateCameraSurrogateFunctionAsync(
      measurement_indices[1], extrinsics, intrinsics, measurements,
      extrinsics_infos, intrinsics_infos, rescaled_sqrt_weights[1],
      rescaled_a_vals[1], rescaled_g_vecs[1], rescaled_constants[1],
      f_values[1], m_n_num_extrinsics, m_n_num_intrinsics, m_num_measurements,
      m_n_measurement_sizes[1], m_stream);

  sfm::ba::EvaluatePointSurrogateFunctionAsync(
      measurement_indices[2], points, point_infos, rescaled_a_vals[2],
      rescaled_g_vecs[2], rescaled_constants[2], f_values[2], m_n_num_points,
      m_n_measurement_sizes[2], m_stream);

  if (future_surrogate_f.size() < 3) {
    future_surrogate_f.resize(3);
  }

  auto reduce_buffer = m_buffer[3].data().get();
  size_t reduce_buffer_size = m_buffer[3].size() * sizeof(T);

  for (int_t n = 0; n < 3; n++) {
    sfm::utils::ReduceAsync(reduce_buffer, reduce_buffer_size, f_values[n],
                            m_n_measurement_sizes[n],
                            future_surrogate_f.data().get() + n, T(0.0),
                            cub::Sum(), m_stream);
  }

  CHECK_CUDA(cudaEventRecord(event, m_stream));

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::EvaluateSurrogateFunctionSync(
    Container<T> &future_surrogate_f, std::array<T, 3> &surrogate_f, T &cost,
    cudaEvent_t event) const {
  CHECK_CUDA(cudaEventSynchronize(event));

  cudaMemcpy(surrogate_f.data(), future_surrogate_f.data().get(),
             surrogate_f.size() * sizeof(T), cudaMemcpyDeviceToHost);

  cost =
      surrogate_f[0] + surrogate_f[1] + surrogate_f[2] + m_surrogate_f_constant;

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::
    ComputeReducedCameraMatrixVectorMultiplicationAsync(const T *hess_cc,
                                                        const T *hess_cl,
                                                        const T *hess_ll_inv,
                                                        const T *x, T *y,
                                                        T *buffer) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  std::array<const int_t *, 3> measurement_indices = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  const int_t *n_camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  std::array<int_t, 3> measurement_sizes = {m_n_measurement_sizes[0],
                                            m_n_measurement_sizes[1],
                                            m_n_measurement_sizes[2]};

  const auto &m_n_num_cameras = m_n_num_extrinsics;

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_cc, x, T(0.0), y, D_CAMERA_SIZE, m_n_num_cameras, m_stream);

  sfm::ba::ComputeHessianCameraPointLeftMultiplicationAsync(
      measurement_indices[0], n_camera_indices, n_point_indices, hess_cl, x,
      camera_infos, point_infos, T(1.0), buffer, m_n_num_cameras,
      m_n_num_points, m_num_measurements, measurement_sizes[0], true, m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, buffer, T(0.0), buffer, LANDMARK_SIZE,
      m_n_num_points, m_stream);

  sfm::ba::ComputeHessianCameraPointRightMultiplicationAsync(
      measurement_indices[0], n_camera_indices, n_point_indices, hess_cl,
      buffer, camera_infos, point_infos, T(-1.0), y, m_n_num_cameras,
      m_n_num_points, m_num_measurements, measurement_sizes[0], false,
      m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::
    ComputeReducedCameraMatrixVectorMultiplicationAsync(
        const T *hess_cc, std::array<const T *, 2> hess_cl,
        const T *hess_ll_inv, const T *x, T *y, T *temp) const {
  const auto measurements = m_measurements.data().get();
  const auto camera_infos = m_extrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();

  std::array<const int_t *, 3> measurement_indices = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  const int_t *n_camera_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  std::array<int_t, 3> measurement_sizes = {m_n_measurement_sizes[0],
                                            m_n_measurement_sizes[1],
                                            m_n_measurement_sizes[2]};

  const auto &m_n_num_cameras = m_n_num_extrinsics;

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_cc, x, T(0.0), y, D_CAMERA_SIZE, m_n_num_cameras, m_stream);

  T *spmv_buffer = m_buffer[2].data().get();
  const auto &measurement_dicts_by_points =
      m_n_measurement_dicts_by_points[0].data().get();
  const auto &measurement_offsets_by_points =
      m_n_measurement_offsets_by_points[0].data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointLeftMultiplicationAsync(
      measurement_dicts_by_points, measurement_offsets_by_points,
      n_camera_indices, n_point_indices, hess_cl[0], T(1.0), x, camera_infos,
      point_infos, T(0.0), temp, spmv_buffer, m_n_num_cameras, m_n_num_points,
      m_n_measurement_sizes[0], m_n_measurement_offsets_by_points[0].size() - 1,
      m_stream);

  sfm::utils::ComputeMatrixVectorMultiplicationAsync(
      T(1.0), hess_ll_inv, temp, T(0.0), temp, LANDMARK_SIZE, m_n_num_points,
      m_stream);

  const auto measurement_dicts_by_cameras =
      m_n_measurement_dicts_by_cameras[0].data().get();
  const auto measurement_offsets_by_cameras =
      m_n_measurement_offsets_by_cameras[0].data().get();
  sfm::ba::ComputeBlockSparseHessianCameraPointRightMultiplicationAsync(
      measurement_dicts_by_cameras, measurement_offsets_by_cameras,
      n_camera_indices, n_point_indices, hess_cl[1], T(-1.0), temp,
      camera_infos, point_infos, T(1.0), y, spmv_buffer, m_n_num_cameras,
      m_n_num_points, m_n_measurement_sizes[0],
      m_n_measurement_offsets_by_cameras[0].size() - 1, m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::SolveProximalMethodAsync(
    bool nesterov) const {
  const int_t *measurement_indices[3] = {
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[0],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[1],
      m_n_measurement_indices.data().get() + m_n_measurement_offsets[2]};

  const auto measurements = m_measurements.data().get();
  const auto extrinsics_infos = m_extrinsics_infos.data().get();
  const auto intrinsics_infos = m_intrinsics_infos.data().get();
  const auto point_infos = m_point_infos.data().get();
  const auto sqrt_weights = m_sqrt_weights.data().get();

  const auto extrinsics =
      nesterov ? m_nesterov_extrinsics.data().get() : m_extrinsics.data().get();
  const auto intrinsics =
      nesterov ? m_nesterov_intrinsics.data().get() : m_intrinsics.data().get();
  const auto points =
      nesterov ? m_nesterov_points.data().get() : m_points.data().get();

  const int_t *n_extrinsics_indices = m_n_extrinsics_indices.data().get();
  const int_t *n_intrinsics_indices = m_n_intrinsics_indices.data().get();
  const int_t *n_point_indices = m_n_point_indices.data().get();

  const int_t *n_extrinsics_dicts = m_n_extrinsics_dicts.data().get();
  const int_t *n_intrinsics_dicts = m_n_intrinsics_dicts.data().get();
  const int_t *n_point_dicts = m_n_point_dicts.data().get();

  T *rescaled_h_a_g_vecs[3] = {nullptr, nullptr, nullptr};
  T *rescaled_f_s_vecs[3] = {nullptr, nullptr, nullptr};
  T *rescaled_sqrt_weights[3] = {nullptr, nullptr, nullptr};
  T *rescaled_constants[3] = {nullptr, nullptr, nullptr};
  T *f_values[3] = {nullptr, nullptr, nullptr};

  rescaled_h_a_g_vecs[0] = m_buffer[1].data().get();
  rescaled_f_s_vecs[0] =
      m_buffer[1].data().get() + 7 * m_n_measurement_sizes[0];
  rescaled_sqrt_weights[0] =
      m_buffer[1].data().get() + (7 + 8) * m_n_measurement_sizes[0];
  rescaled_constants[0] =
      m_buffer[1].data().get() + (7 + 8 + 1) * m_n_measurement_sizes[0];

#pragma unroll
  for (int_t n = 1; n < 3; n++) {
    if (nesterov) {
      rescaled_h_a_g_vecs[n] =
          m_nesterov_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
      rescaled_sqrt_weights[n] =
          m_nesterov_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
      rescaled_constants[n] =
          m_nesterov_rescaled_constants.data().get() +
          (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
    } else {
      rescaled_h_a_g_vecs[n] =
          m_rescaled_h_a_g_vecs.data().get() +
          7 * (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
      rescaled_sqrt_weights[n] =
          m_rescaled_sqrt_weights.data().get() +
          (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
      rescaled_constants[n] =
          m_rescaled_constants.data().get() +
          (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
    }

    rescaled_f_s_vecs[n] =
        m_rescaled_f_s_vecs.data().get() +
        8 * (m_n_measurement_offsets[n] - m_n_measurement_sizes[0]);
  }

#pragma unroll
  for (int_t n = 0; n < 3; n++) {
    f_values[n] = (nesterov ? m_nesterov_f_values : m_f_values).data().get() +
                  m_n_measurement_offsets[n];

    MajorizeReprojectionLossFunctionAsync(
        measurement_indices[n], extrinsics, intrinsics, points, measurements,
        extrinsics_infos, intrinsics_infos, point_infos, sqrt_weights,
        rescaled_h_a_g_vecs[n], rescaled_f_s_vecs[n], rescaled_sqrt_weights[n],
        rescaled_constants[n], f_values[n], m_option.robust_loss,
        m_option.loss_radius, m_num_extrinsics, m_num_intrinsics, m_num_points,
        m_num_measurements, m_n_measurement_sizes[n], m_stream);
  }

  const int_t *measurement_indices_by_cameras[3] = {
      m_n_measurement_indices_by_cameras[0].data().get(),
      m_n_measurement_indices_by_cameras[1].data().get(),
      m_n_measurement_indices_by_cameras[2].data().get()};
  const int_t *measurement_indices_by_points[3] = {
      m_n_measurement_indices_by_points[0].data().get(),
      m_n_measurement_indices_by_points[1].data().get(),
      m_n_measurement_indices_by_points[2].data().get()};
  const int_t *measurement_dicts_by_cameras[3] = {
      m_n_measurement_dicts_by_cameras[0].data().get(),
      m_n_measurement_dicts_by_cameras[1].data().get(),
      m_n_measurement_dicts_by_cameras[2].data().get()};
  const int_t *measurement_dicts_by_points[3] = {
      m_n_measurement_dicts_by_points[0].data().get(),
      m_n_measurement_dicts_by_points[1].data().get(),
      m_n_measurement_dicts_by_points[2].data().get()};
  const int_t *measurement_offsets_by_cameras[3] = {
      m_n_measurement_offsets_by_cameras[0].data().get(),
      m_n_measurement_offsets_by_cameras[1].data().get(),
      m_n_measurement_offsets_by_cameras[2].data().get()};
  const int_t *measurement_offsets_by_points[3] = {
      m_n_measurement_offsets_by_points[0].data().get(),
      m_n_measurement_offsets_by_points[1].data().get(),
      m_n_measurement_offsets_by_points[2].data().get()};

  T *extrinsics_proximal_operator = m_extrinsics_proximal_operator.data().get();

  T *intrinsics_proximal_operator = m_intrinsics_proximal_operator.data().get();

  T *point_proximal_operator = m_point_proximal_operator.data().get();

  auto extrinsics_proximal_operator_n = m_buffer[0].data().get();
  auto point_proximal_operator_n = m_buffer[3].data().get();

  cudaMemsetAsync(extrinsics_proximal_operator, 0,
                  4 * 4 * m_n_num_extrinsics * sizeof(T), m_stream);

  cudaMemsetAsync(intrinsics_proximal_operator, 0,
                  1 * 8 * m_n_num_intrinsics * sizeof(T), m_stream);

  cudaMemsetAsync(point_proximal_operator, 0,
                  1 * 4 * m_n_num_points * sizeof(T), m_stream);

  ComputeExtrinsicsAndPointProximalOperatorProductAsync(
      measurement_indices_by_cameras[0], measurement_indices_by_points[0],
      rescaled_h_a_g_vecs[0],
      rescaled_h_a_g_vecs[0] + 3 * m_n_measurement_sizes[0],
      extrinsics_proximal_operator_n, point_proximal_operator_n,
      m_n_measurement_sizes[0], m_stream);

  ComputeCameraDictedReductionAsync(
      measurement_dicts_by_cameras[0], measurement_offsets_by_cameras[0],
      n_extrinsics_indices, T(1.0), extrinsics_proximal_operator_n,
      extrinsics_infos, T(1.0), extrinsics_proximal_operator,
      m_n_num_extrinsics, m_n_measurement_sizes[0], 16,
      m_n_measurement_offsets_by_cameras[0].size() - 1, m_stream);

  ComputePointDictedReductionAsync(
      measurement_dicts_by_points[0], measurement_offsets_by_points[0],
      n_point_indices, T(1.0), point_proximal_operator_n, point_infos, T(1.0),
      point_proximal_operator, m_n_num_points, m_n_measurement_sizes[0], 4,
      m_n_measurement_offsets_by_points[0].size() - 1, m_stream);

  ComputeExtrinsicsProximalOperatorAsync(
      measurement_dicts_by_cameras[1], measurement_offsets_by_cameras[1],
      n_extrinsics_indices, rescaled_h_a_g_vecs[1],
      rescaled_h_a_g_vecs[1] + 3 * m_n_measurement_sizes[1], extrinsics_infos,
      extrinsics_proximal_operator, m_n_num_extrinsics,
      m_n_measurement_sizes[1],
      m_n_measurement_offsets_by_cameras[1].size() - 1, m_stream);

  ComputePointProximalOperatorProductAsync(
      measurement_indices_by_points[2],
      rescaled_h_a_g_vecs[2] + 3 * m_n_measurement_sizes[2],
      point_proximal_operator_n, m_n_measurement_sizes[2], m_stream);

  ComputePointDictedReductionAsync(
      measurement_dicts_by_points[2], measurement_offsets_by_points[2],
      n_point_indices, T(1.0), point_proximal_operator_n, point_infos, T(1.0),
      point_proximal_operator, m_n_num_points, m_n_measurement_sizes[2], 4,
      m_n_measurement_offsets_by_points[2].size() - 1, m_stream);

  ComputeCameraDictedReductionAsync(
      measurement_dicts_by_cameras[0], measurement_offsets_by_cameras[0],
      n_intrinsics_indices, T(1.0), rescaled_f_s_vecs[0], intrinsics_infos,
      T(1.0), intrinsics_proximal_operator, m_n_num_intrinsics,
      m_n_measurement_sizes[0], 8,
      m_n_measurement_offsets_by_cameras[0].size() - 1, m_stream);

  ComputeCameraDictedReductionAsync(
      measurement_dicts_by_cameras[1], measurement_offsets_by_cameras[1],
      n_intrinsics_indices, T(1.0), rescaled_f_s_vecs[1], intrinsics_infos,
      T(1.0), intrinsics_proximal_operator, m_n_num_intrinsics,
      m_n_measurement_sizes[1], 8,
      m_n_measurement_offsets_by_cameras[1].size() - 1, m_stream);

  const auto proximal_extrinsics = m_proximal_extrinsics.data().get();
  const auto proximal_intrinsics = m_proximal_intrinsics.data().get();
  const auto proximal_points = m_proximal_points.data().get();

  SolveExtrinsicsProximalOperatorAsync(
      extrinsics_proximal_operator, m_option.regularizer[0], n_extrinsics_dicts,
      extrinsics, m_num_extrinsics, proximal_extrinsics, m_n_num_extrinsics,
      m_n_num_extrinsics, m_stream);

  SolveIntrinsicsProximalOperatorAsync(
      intrinsics_proximal_operator, m_option.regularizer[1], n_intrinsics_dicts,
      intrinsics, m_num_intrinsics, proximal_intrinsics, m_n_num_intrinsics,
      m_n_num_intrinsics, m_stream);

  SolvePointProximalOperatorAsync(
      point_proximal_operator, m_option.regularizer[2], n_point_dicts, points,
      m_num_points, proximal_points, m_n_num_points, m_n_num_points, m_stream);

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PreCommunicateAsync(bool nesterov) const {
  if (nesterov) {
    assert(m_option.accelerated);
    if (!m_option.accelerated) {
      LOG(ERROR) << "The problems is not accelerated." << std::endl;
      exit(-1);
    }
  }

  auto &extrinsics = nesterov ? m_nesterov_extrinsics : m_extrinsics;
  auto &intrinsics = nesterov ? m_nesterov_intrinsics : m_intrinsics;
  auto &points = nesterov ? m_nesterov_points : m_points;

  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank ||
        (!m_send_extrinsics_sizes[neighbor] &&
         !m_send_intrinsics_sizes[neighbor] && !m_send_point_sizes[neighbor])) {
      continue;
    }

    T *dst = m_send_data[neighbor].data().get();
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        m_send_extrinsics_dicts[neighbor].data().get(), EXTRINSICS_SIZE, 1,
        extrinsics.data().get(), dst, m_num_extrinsics,
        m_send_extrinsics_sizes[neighbor], m_send_extrinsics_sizes[neighbor],
        m_comm_streams[neighbor]);

    dst += EXTRINSICS_SIZE * m_send_extrinsics_sizes[neighbor];
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        m_send_intrinsics_dicts[neighbor].data().get(), INTRINSICS_SIZE, 1,
        intrinsics.data().get(), dst, m_num_intrinsics,
        m_send_intrinsics_sizes[neighbor], m_send_intrinsics_sizes[neighbor],
        m_comm_streams[neighbor]);

    dst += INTRINSICS_SIZE * m_send_intrinsics_sizes[neighbor];
    sfm::utils::CopyToDictedMatrixOfArrayAsync(
        m_send_point_dicts[neighbor].data().get(), LANDMARK_SIZE, 1,
        points.data().get(), dst, m_num_points, m_send_point_sizes[neighbor],
        m_send_point_sizes[neighbor], m_comm_streams[neighbor]);
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PreCommunicateSync() const {
  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank ||
        (!m_send_extrinsics_sizes[neighbor] &&
         !m_send_intrinsics_sizes[neighbor] && !m_send_point_sizes[neighbor])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[neighbor]));
    CHECK_CUDA(cudaGetLastError());
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PostCommunicateAsync(bool nesterov) const {
  if (nesterov) {
    assert(m_option.accelerated);
    if (!m_option.accelerated) {
      LOG(ERROR) << "The problems is not accelerated." << std::endl;
      exit(-1);
    }
  }

  auto &extrinsics = nesterov ? m_nesterov_extrinsics : m_extrinsics;
  auto &intrinsics = nesterov ? m_nesterov_intrinsics : m_intrinsics;
  auto &points = nesterov ? m_nesterov_points : m_points;

  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank ||
        (!m_recv_extrinsics_sizes[neighbor] &&
         !m_recv_intrinsics_sizes[neighbor] && !m_recv_point_sizes[neighbor])) {
      continue;
    }

    T *src = m_recv_data[neighbor].data().get();
    sfm::utils::CopyFromDictedMatrixOfArrayAsync(
        m_recv_extrinsics_dicts[neighbor].data().get(), EXTRINSICS_SIZE, 1, src,
        extrinsics.data().get(), m_recv_extrinsics_sizes[neighbor],
        m_num_extrinsics, m_recv_extrinsics_sizes[neighbor],
        m_comm_streams[neighbor]);

    src += EXTRINSICS_SIZE * m_recv_extrinsics_sizes[neighbor];
    sfm::utils::CopyFromDictedMatrixOfArrayAsync(
        m_recv_intrinsics_dicts[neighbor].data().get(), INTRINSICS_SIZE, 1, src,
        intrinsics.data().get(), m_recv_intrinsics_sizes[neighbor],
        m_num_intrinsics, m_recv_intrinsics_sizes[neighbor],
        m_comm_streams[neighbor]);

    src += INTRINSICS_SIZE * m_recv_intrinsics_sizes[neighbor];
    sfm::utils::CopyFromDictedMatrixOfArrayAsync(
        m_recv_point_dicts[neighbor].data().get(), LANDMARK_SIZE, 1, src,
        points.data().get(), m_recv_point_sizes[neighbor], m_num_points,
        m_recv_point_sizes[neighbor], m_comm_streams[neighbor]);
  }

  return 0;
}

template <typename T>
int DBASubproblem<kGPU, T, false>::PostCommunicateSync() const {
  for (int_t neighbor = 0; neighbor < m_num_ranks; neighbor++) {
    if (neighbor == m_rank ||
        (!m_recv_extrinsics_sizes[neighbor] &&
         !m_recv_intrinsics_sizes[neighbor] && !m_recv_point_sizes[neighbor])) {
      continue;
    }

    CHECK_CUDA(cudaStreamSynchronize(m_comm_streams[neighbor]));
    CHECK_CUDA(cudaGetLastError());
  }

  return 0;
}

template class DBASubproblemBase<kGPU, float, false>;
template class DBASubproblemBase<kGPU, double, false>;

template class DBASubproblem<kGPU, float, false>;
template class DBASubproblem<kGPU, double, false>;
} // namespace ba
} // namespace sfm
