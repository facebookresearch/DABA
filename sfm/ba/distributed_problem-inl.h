// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdlib>
#include <limits>
#include <unordered_map>

#include <glog/logging.h>

#include <sfm/ba/distributed_problem.h>
#include <sfm/ba/macro.h>
#include <sfm/types.h>

namespace sfm {
namespace ba {
template <Memory kMemory, typename T, bool kSharedIntrinsics>
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::DBASubproblemBase(
    const Option<T> &option, int_t rank, int_t num_ranks)
    : m_option(option), m_rank(rank), m_num_ranks(num_ranks),
      m_send_extrinsics_sizes(num_ranks, 0),
      m_send_intrinsics_sizes(num_ranks, 0), m_send_point_sizes(num_ranks, 0),
      m_send_extrinsics_dicts(num_ranks), m_send_intrinsics_dicts(num_ranks),
      m_send_point_dicts(num_ranks), m_send_data(num_ranks),
      m_recv_extrinsics_sizes(num_ranks, 0),
      m_recv_intrinsics_sizes(num_ranks, 0), m_recv_point_sizes(num_ranks, 0),
      m_recv_extrinsics_dicts(num_ranks), m_recv_intrinsics_dicts(num_ranks),
      m_recv_point_dicts(num_ranks), m_recv_data(num_ranks),
      m_cpu_send_extrinsics_dicts(num_ranks),
      m_cpu_send_intrinsics_dicts(num_ranks), m_cpu_send_point_dicts(num_ranks),
      m_cpu_send_data(num_ranks), m_cpu_recv_extrinsics_dicts(num_ranks),
      m_cpu_recv_intrinsics_dicts(num_ranks), m_cpu_recv_point_dicts(num_ranks),
      m_cpu_recv_data(num_ranks), m_extrinsics_offsets(num_ranks, -1),
      m_extrinsics_sizes(num_ranks, -1), m_extrinsics_ids(num_ranks),
      m_intrinsics_offsets(num_ranks, -1), m_intrinsics_sizes(num_ranks, -1),
      m_intrinsics_ids(num_ranks), m_point_offsets(num_ranks, -1),
      m_point_sizes(num_ranks, -1), m_point_ids(num_ranks),
      m_trust_region_radius(-1.0), m_nesterov_eta(1.0) {
  if (m_num_ranks < 0) {
    LOG(ERROR) << "The number of ranks must be positive." << std::endl;
    exit(-1);
  }

  if (m_rank < 0) {
    LOG(ERROR) << "The rank index must be non-negative." << std::endl;
    exit(-1);
  }

  if (m_rank >= m_num_ranks) {
    LOG(ERROR) << "The rank index must be less than the number of ranks."
               << std::endl;
    exit(-1);
  }
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetCost(T &cost) const {
  cost = m_cost;
  return 0;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetSurrogateCost(
    std::array<T, 3 + 2 * kSharedIntrinsics> &surrogate_cost) const {
  surrogate_cost = m_surrogate_f;
  return 0;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetOption(
    Option<T> &option) const {
  option = m_option;
  return 0;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::unordered_map<int_t, int_t>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetExtrinsicsIds() const {
  return m_extrinsics_ids;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::unordered_map<int_t, int_t>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetIntrinsicsIds() const {
  return m_intrinsics_ids;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::unordered_map<int_t, int_t>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetPointIds() const {
  return m_point_ids;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T,
                      kSharedIntrinsics>::InitializeTrustRegionMethod() {
  if (m_option.trust_region_option.max_iterations > 0 &&
      m_option.trust_region_option.max_accepted_iterations > 0) {
    m_trust_region_radius = std::numeric_limits<T>::max();
  } else {
    m_trust_region_radius = -1.0;
  }

  return 0;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::array<int_t, 2>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetExtrinsicsDicts() const {
  return m_extrinsics_dicts;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::array<int_t, 2>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetIntrinsicsDicts() const {
  return m_intrinsics_dicts;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
const std::vector<std::array<int_t, 2>> &
DBASubproblemBase<kMemory, T, kSharedIntrinsics>::GetPointDicts() const {
  return m_point_dicts;
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T, kSharedIntrinsics>::TrustRegionMethod(
    const Container<T> &extrinsics, const Container<T> &intrinsics,
    const Container<T> &points, bool nesterov) const {
  T cost = 0;
  this->EvaluateSurrogateFunction(extrinsics, intrinsics, points,
                                  m_trust_region_surrogate_f[0], cost,
                                  nesterov);
  return TrustRegionMethod(extrinsics, intrinsics, points, cost, nesterov);
}

template <Memory kMemory, typename T, bool kSharedIntrinsics>
int DBASubproblemBase<kMemory, T, kSharedIntrinsics>::TrustRegionMethod(
    bool nesterov) const {
  LevenbergMarquardtMethod(nesterov);
  return 0;
}
} // namespace ba
} // namespace sfm
