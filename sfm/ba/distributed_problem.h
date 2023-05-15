// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>

#include <sfm/ba/types.h>
#include <sfm/types.h>
#include <sfm/utils/cuda_utils.h>
#include <sfm/utils/utils.h>

namespace sfm {
namespace ba {
template <Memory kMemory, typename T, bool kSharedIntrinsics>
class DBASubproblemBase {
public:
  template <typename P>
  using Container = typename std::conditional<kMemory == kCPU, std::vector<P>,
                                              thrust::device_vector<P>>::type;

  DBASubproblemBase(const Option<T> &option, int_t rank, int_t num_ranks);

  int GetCost(T &cost) const;

  int GetSurrogateCost(
      std::array<T, 3 + 2 * kSharedIntrinsics> &surrogate_cost) const;

  int GetOption(Option<T> &option) const;

  virtual T GetMemoryUsage() const = 0;

  virtual T GetCommunicationLoad() const = 0;

  virtual int
  GetExtrinsics(std::vector<Eigen::Matrix<T, 3, 4>> &extrinsics) const = 0;

  virtual int
  GetIntrinsics(std::vector<Eigen::Vector3<T>> &intrinsics) const = 0;

  virtual int GetPoints(std::vector<Eigen::Vector3<T>> &points) const = 0;

  virtual int Initialize(
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 4>>>
          &extrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &intrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &points) = 0;

  virtual int Initialize(
      const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &extrinsics,
      const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &intrinsics,
      const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &points) = 0;

  virtual int InitializeSurrogateFunction() = 0;
  virtual int InitializeTrustRegionMethod();
  virtual int InitializeNesterovAcceleration() = 0;

  virtual int Iterate() const = 0;

  virtual int MPICommunicate(const MPI_Comm &comm, bool nesterov) const = 0;
  virtual int NCCLCommunicate(const ncclComm_t &comm, bool nesterov) const = 0;

  virtual const std::vector<std::unordered_map<int_t, int_t>> &
  GetExtrinsicsIds() const;
  virtual const std::vector<std::unordered_map<int_t, int_t>> &
  GetIntrinsicsIds() const;
  virtual const std::vector<std::unordered_map<int_t, int_t>> &
  GetPointIds() const;

  virtual const std::vector<std::array<int_t, 2>> &GetExtrinsicsDicts() const;
  virtual const std::vector<std::array<int_t, 2>> &GetIntrinsicsDicts() const;
  virtual const std::vector<std::array<int_t, 2>> &GetPointDicts() const;

  virtual int Setup(const std::vector<std::array<int_t, 2>> &extrinsics_infos,
                    const std::vector<std::array<int_t, 2>> &intrinsics_infos,
                    const std::vector<std::array<int_t, 2>> &point_infos,
                    const std::vector<Eigen::Vector2<T>> &measurements,
                    const std::vector<T> &sqrt_weights) = 0;

protected:
  virtual int
  Accept(const Container<T> &extrinsics, const Container<T> &intrinsics,
         const Container<T> &points, const Container<T> &f_values,
         const std::array<T, 3 + 2 * kSharedIntrinsics> &surrogate_f,
         T cost) const = 0;

  virtual int UpdateSurrogateFunction() = 0;

  virtual int ConstructSurrogateFunction() const = 0;

  virtual int PreCommunicate(bool nesterov) const = 0;
  virtual int PostCommunicate(bool nesterov) const = 0;

  virtual int EvaluateSurrogateFunction(
      const Container<T> &extrinsics, const Container<T> &intrinsics,
      const Container<T> &points,
      std::array<T, 3 + 2 * kSharedIntrinsics> &surrogate_f, T &cost,
      bool nesterov) const = 0;

  virtual int EvaluateObjectiveFunction(
      const Container<T> &extrinsics, const Container<T> &intrinsics,
      const Container<T> &points,
      std::array<T, 3 + 2 * kSharedIntrinsics> &surrogate_f) const = 0;

  virtual int TrustRegionMethod(const Container<T> &extrinsics,
                                const Container<T> &intrinsics,
                                const Container<T> &points, T cost,
                                bool nesterov) const = 0;
  int TrustRegionMethod(const Container<T> &extrinsics,
                        const Container<T> &intrinsics,
                        const Container<T> &points, bool nesterov) const;
  int TrustRegionMethod(bool nesterov) const;

  virtual int LevenbergMarquardtMethod(bool nesterov) const = 0;

  virtual int Linearize(bool nesterov) const = 0;
  virtual int BuildLinearSystem(T ratio) const = 0;
  virtual int PCG(int_t &num_iters, T &update_step_norm) const = 0;
  virtual int Retract(T stepsize) const = 0;

  // distributed bundle adjustment subproblem infomation
  const Option<T> m_option;
  int_t m_rank;
  int_t m_num_ranks;

  // problem information
  int_t m_num_extrinsics;
  int_t m_num_intrinsics;
  int_t m_num_points;
  int_t m_num_measurements;

  // extrinsics, intrinsics and points info
  Container<int_t> m_extrinsics_infos;
  Container<int_t> m_intrinsics_infos;
  Container<int_t> m_point_infos;
  // array of matrix for kCPU, matrix of array for kGPU
  Container<T> m_measurements; // 2x1 vector
  // vector of scalars
  Container<T> m_sqrt_weights;

  // distributed bundle adjustment
  std::vector<int_t> m_extrinsics_offsets;
  std::vector<int_t> m_extrinsics_sizes;
  std::vector<std::unordered_map<int_t, int_t>> m_extrinsics_ids;
  std::vector<std::array<int_t, 2>> m_extrinsics_dicts;

  std::vector<int_t> m_intrinsics_offsets;
  std::vector<int_t> m_intrinsics_sizes;
  std::vector<std::unordered_map<int_t, int_t>> m_intrinsics_ids;
  std::vector<std::array<int_t, 2>> m_intrinsics_dicts;

  std::vector<int_t> m_point_offsets;
  std::vector<int_t> m_point_sizes;
  std::vector<std::unordered_map<int_t, int_t>> m_point_ids;
  std::vector<std::array<int_t, 2>> m_point_dicts;

  // measurement, extrinsics, intrinsics and point size for each iteration
  int_t m_n_num_measurements;
  Container<int_t> m_n_measurement_indices;

  int_t m_n_measurement_offsets[3 + 2 * kSharedIntrinsics] = {0};
  int_t m_n_measurement_sizes[3 + 2 * kSharedIntrinsics] = {0};

  int_t m_n_num_extrinsics;
  Container<int_t> m_n_extrinsics_dicts;
  Container<int_t> m_n_extrinsics_indices;

  int_t m_n_num_intrinsics;
  Container<int_t> m_n_intrinsics_dicts;
  Container<int_t> m_n_intrinsics_indices;

  int_t m_n_num_points;
  Container<int_t> m_n_point_dicts;
  Container<int_t> m_n_point_indices;

  // array of matrix for kCPU, matrix of array for kGPU
  mutable Container<T> m_extrinsics; // 3x4 matrix
  mutable Container<T> m_intrinsics; // 3x1 matrix
  mutable Container<T> m_points;     // 3x1 matrix
  mutable T m_surrogate_f_constant = 0;

  mutable std::array<T, 3 + 2 * kSharedIntrinsics> m_surrogate_f;
  mutable T m_cost = 0; // Objective values

  // array of matrix for kCPU, matrix of array for kGPU
  mutable Container<T>
      m_rescaled_h_a_g_vecs; // 1x7 matrix: [h0 h1 h2 a g0 g1 g2]
  mutable Container<T>
      m_rescaled_f_s_vecs; // 1x7 matrix: [f0 f1 f2 f3 s0 s1 s2]
  // vector of scalars
  mutable Container<T> m_rescaled_sqrt_weights;
  mutable Container<T> m_rescaled_constants;
  mutable Container<T> m_f_values; // Objective values for each factor

  // array of matrix for kCPU, matrix of array for kGPU
  mutable Container<T> m_nesterov_extrinsics; // 3x4 matrix
  mutable Container<T> m_nesterov_intrinsics; // 3x1 matrix
  mutable Container<T> m_nesterov_points;     // 3x1 matrix
  mutable T m_nesterov_eta;
  mutable std::vector<T> m_nesterov_s;
  mutable std::vector<T> m_nesterov_beta;
  mutable T m_nesterov_avg_objective_value;

  // array of matrix for kCPU, matrix of array for kGPU
  mutable Container<T>
      m_nesterov_rescaled_h_a_g_vecs; // 1x7 matrix: [h0 h1 h2 a g0 g1 g2]
  // vector of scalars
  mutable Container<T> m_nesterov_rescaled_sqrt_weights;
  mutable Container<T> m_nesterov_rescaled_constants;
  mutable Container<T> m_nesterov_f_values;

  mutable Container<T> m_proximal_extrinsics;
  mutable Container<T> m_proximal_intrinsics;
  mutable Container<T> m_proximal_points;
  mutable std::array<T, 3 + 2 * kSharedIntrinsics> m_proximal_surrogate_f;
  mutable T m_proximal_cost = 0; // Objective values

  mutable Container<T> m_trust_region_extrinsics[2]; // 3x4 matrix
  mutable Container<T> m_trust_region_intrinsics[2]; // 3x1 matrix
  mutable Container<T> m_trust_region_points[2];     // 3x1 matrix
  mutable std::array<T, 3 + 2 * kSharedIntrinsics>
      m_trust_region_surrogate_f[2];
  mutable T m_trust_region_cost[2] = {0, 0}; // Objective values
  mutable T m_trust_region_radius;

  // array of matrix for kCPU, matrix of array for kGPU
  // extrinsics hessian and gradient info to construct the surrogate function:
  // 4x4 matrix [c0 c1 c2 c3]
  // Hessian: [c0] --- note that the roation part is a skew-symmetric matrix
  // Gradient: [c1 c2 c3]^T
  mutable Container<T> m_extrinsics_proximal_operator;
  // intrinsics hessian and gradient info to construct the surrogate function:
  // 1x7 matrix [c0 c1 c2 c3 c4 c5 c6]
  // Hessian: [c0  0  0;
  //            0 c1 c2;
  //            0 c2 c3]
  // Graident: [c4 c5 c6]^T
  mutable Container<T> m_intrinsics_proximal_operator;
  // point hessian and gradient info to construct the surrogate function:
  // 1x4 matrix [c0 c1 c2 c3]
  // Hessian: [c0 0 0;
  //           0 c0 0;
  //           0 0 c0]
  // Gradient: [c1 c2 c3]^T
  mutable Container<T> m_point_proximal_operator;

  // array of matrix for kCPU, matrix of array for kGPU
  // jacobians and rescaled errors
  //   mutable Container<T> m_jacobians[3 + 2 * kSharedIntrinsics];
  //   mutable Container<T> m_rescaled_errs[3 + 2 * kSharedIntrinsics];

  // Communication
  std::vector<int_t> m_send_extrinsics_sizes;
  std::vector<int_t> m_send_intrinsics_sizes;
  std::vector<int_t> m_send_point_sizes;
  std::vector<Container<int_t>> m_send_extrinsics_dicts;
  std::vector<Container<int_t>> m_send_intrinsics_dicts;
  std::vector<Container<int_t>> m_send_point_dicts;
  mutable std::vector<Container<T>> m_send_data;

  std::vector<int_t> m_recv_extrinsics_sizes;
  std::vector<int_t> m_recv_intrinsics_sizes;
  std::vector<int_t> m_recv_point_sizes;
  std::vector<Container<int_t>> m_recv_extrinsics_dicts;
  std::vector<Container<int_t>> m_recv_intrinsics_dicts;
  std::vector<Container<int_t>> m_recv_point_dicts;
  mutable std::vector<Container<T>> m_recv_data;

  // CPU communication
  std::vector<PinnedHostVector<int_t>> m_cpu_send_extrinsics_dicts;
  std::vector<PinnedHostVector<int_t>> m_cpu_send_intrinsics_dicts;
  std::vector<PinnedHostVector<int_t>> m_cpu_send_point_dicts;
  mutable std::vector<PinnedHostVector<T>> m_cpu_send_data;

  std::vector<PinnedHostVector<int_t>> m_cpu_recv_extrinsics_dicts;
  std::vector<PinnedHostVector<int_t>> m_cpu_recv_intrinsics_dicts;
  std::vector<PinnedHostVector<int_t>> m_cpu_recv_point_dicts;
  mutable std::vector<PinnedHostVector<T>> m_cpu_recv_data;
};

template <Memory kMemory, typename T, bool kSharedIntrinsics>
class DBASubproblem;

template <Memory kMemory, typename T, bool kSharedIntrinsics> class DBAProblem;

template <typename T>
class DBASubproblem<kGPU, T, false> : public DBASubproblemBase<kGPU, T, false> {
public:
  friend class DBAProblem<kGPU, T, false>;

  template <typename P>
  using Container =
      typename DBASubproblemBase<kGPU, T, false>::template Container<P>;

  using Base = DBASubproblemBase<kGPU, T, false>;

  DBASubproblem(const Option<T> &option, int_t rank, int_t num_ranks);

  DBASubproblem(const Option<T> &option, int_t rank, int_t num_ranks,
                int_t device);

  ~DBASubproblem();

  int_t GetDevice() const;

  virtual T GetMemoryUsage() const override;

  virtual T GetCommunicationLoad() const override;

  virtual int
  GetExtrinsics(std::vector<Eigen::Matrix<T, 3, 4>> &extrinsics) const override;

  virtual int
  GetIntrinsics(std::vector<Eigen::Vector3<T>> &intrinsics) const override;

  virtual int GetPoints(std::vector<Eigen::Vector3<T>> &points) const override;

  virtual int Initialize(
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 4>>>
          &extrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &intrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &points) override;

  virtual int Initialize(
      const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &extrinsics,
      const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &intrinsics,
      const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &points) override;

  virtual int InitializeSurrogateFunction() override;
  virtual int InitializeNesterovAcceleration() override;

  virtual int Iterate() const override;

  virtual int MPICommunicate(const MPI_Comm &comm,
                             bool nesterov) const override;
  virtual int NCCLCommunicate(const ncclComm_t &comm,
                              bool nesterov) const override;
  int Communicate(
      const std::vector<std::shared_ptr<DBASubproblem<kGPU, T, false>>>
          &problems,
      bool nesterov) const;

  const Container<T> &GetDeviceMeasurements() const;

  const Container<int_t> &GetDeviceExtrinsicsInfo() const;

  const Container<int_t> &GetDeviceIntrinsicsInfo() const;

  const Container<int_t> &GetDevicePointsInfo() const;

  const Container<T> &GetDeviceSqrtWeights() const;

  const Container<T> &GetDeviceExtrinsics() const;

  const Container<T> &GetDeviceIntrinsics() const;

  const Container<T> &GetDevicePoints() const;

  virtual int Setup(const std::vector<std::array<int_t, 2>> &extrinsics_infos,
                    const std::vector<std::array<int_t, 2>> &intrinsics_infos,
                    const std::vector<std::array<int_t, 2>> &point_infos,
                    const std::vector<Eigen::Vector2<T>> &measurements,
                    const std::vector<T> &sqrt_weights) override;

  virtual int UpdateSurrogateFunction() override;

protected:
  virtual int Accept(const Container<T> &extrinsics,
                     const Container<T> &intrinsics, const Container<T> &points,
                     const Container<T> &f_values,
                     const std::array<T, 3> &surrogate_f,
                     T cost) const override;

  virtual int PreCommunicate(bool nesterov) const override;
  virtual int PostCommunicate(bool nesterov) const override;

  virtual int ConstructSurrogateFunction() const override;
  virtual int EvaluateSurrogateFunction(const Container<T> &extrinsics,
                                        const Container<T> &intrinsics,
                                        const Container<T> &point,
                                        std::array<T, 3> &surrogate_f, T &cost,
                                        bool nesterov) const override;
  virtual int EvaluateObjectiveFunction(
      const Container<T> &extrinsics, const Container<T> &intrinsics,
      const Container<T> &point, std::array<T, 3> &surrogate_f) const override;
  virtual int Linearize(bool nesterov) const override;
  virtual int BuildLinearSystem(T ratio) const override;
  virtual int Retract(T stepsize) const override;

  virtual int TrustRegionMethod(const Container<T> &extrinsics,
                                const Container<T> &intrinsics,
                                const Container<T> &points, T cost,
                                bool nesterov) const override;

  virtual int LevenbergMarquardtMethod(bool nesterov) const override;
  virtual int PCG(int_t &num_iters, T &update_step_norm) const override;
  int Update() const;
  int NesterovUpdate() const;
  int TrustRegionUpdate() const;
  int PreNesterovUpdate() const;
  int PostNesterovUpdate() const;
  int NesterovAdaptiveRestart(const Container<T> &extrinsics,
                              const Container<T> &intrinsics,
                              const Container<T> &points,
                              std::array<T, 3> &surrogate_f, T &cost) const;

  using DBASubproblemBase<kGPU, T, false>::m_option;
  using DBASubproblemBase<kGPU, T, false>::m_rank;
  using DBASubproblemBase<kGPU, T, false>::m_num_ranks;

  using DBASubproblemBase<kGPU, T, false>::m_num_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_num_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_num_points;
  using DBASubproblemBase<kGPU, T, false>::m_num_measurements;

  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_infos;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_infos;
  using DBASubproblemBase<kGPU, T, false>::m_point_infos;
  using DBASubproblemBase<kGPU, T, false>::m_measurements;
  using DBASubproblemBase<kGPU, T, false>::m_sqrt_weights;

  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_offsets;
  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_ids;
  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_dicts;

  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_offsets;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_ids;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_dicts;

  using DBASubproblemBase<kGPU, T, false>::m_point_offsets;
  using DBASubproblemBase<kGPU, T, false>::m_point_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_point_ids;
  using DBASubproblemBase<kGPU, T, false>::m_point_dicts;

  using DBASubproblemBase<kGPU, T, false>::m_surrogate_f;

  using DBASubproblemBase<kGPU, T, false>::m_surrogate_f_constant;

  using DBASubproblemBase<kGPU, T, false>::m_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_points;
  using DBASubproblemBase<kGPU, T, false>::m_cost;

  using DBASubproblemBase<kGPU, T, false>::m_rescaled_h_a_g_vecs;
  using DBASubproblemBase<kGPU, T, false>::m_rescaled_f_s_vecs;
  using DBASubproblemBase<kGPU, T, false>::m_rescaled_sqrt_weights;
  using DBASubproblemBase<kGPU, T, false>::m_rescaled_constants;
  using DBASubproblemBase<kGPU, T, false>::m_f_values;

  using DBASubproblemBase<kGPU, T, false>::m_n_num_measurements;
  using DBASubproblemBase<kGPU, T, false>::m_n_measurement_indices;
  using DBASubproblemBase<kGPU, T, false>::m_n_measurement_offsets;
  using DBASubproblemBase<kGPU, T, false>::m_n_measurement_sizes;

  using DBASubproblemBase<kGPU, T, false>::m_n_num_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_n_extrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_n_extrinsics_indices;

  using DBASubproblemBase<kGPU, T, false>::m_n_num_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_n_intrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_n_intrinsics_indices;

  using DBASubproblemBase<kGPU, T, false>::m_n_num_points;
  using DBASubproblemBase<kGPU, T, false>::m_n_point_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_n_point_indices;

  using DBASubproblemBase<kGPU, T, false>::m_nesterov_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_points;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_eta;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_s;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_beta;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_avg_objective_value;

  using DBASubproblemBase<kGPU, T, false>::m_nesterov_rescaled_h_a_g_vecs;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_rescaled_sqrt_weights;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_rescaled_constants;
  using DBASubproblemBase<kGPU, T, false>::m_nesterov_f_values;

  using DBASubproblemBase<kGPU, T, false>::m_proximal_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_proximal_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_proximal_points;
  using DBASubproblemBase<kGPU, T, false>::m_proximal_surrogate_f;
  using DBASubproblemBase<kGPU, T, false>::m_proximal_cost;

  using DBASubproblemBase<kGPU, T, false>::m_trust_region_extrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_trust_region_intrinsics;
  using DBASubproblemBase<kGPU, T, false>::m_trust_region_points;
  using DBASubproblemBase<kGPU, T, false>::m_trust_region_surrogate_f;
  using DBASubproblemBase<kGPU, T, false>::m_trust_region_cost;
  using DBASubproblemBase<kGPU, T, false>::m_trust_region_radius;

  using DBASubproblemBase<kGPU, T, false>::m_extrinsics_proximal_operator;
  using DBASubproblemBase<kGPU, T, false>::m_intrinsics_proximal_operator;
  using DBASubproblemBase<kGPU, T, false>::m_point_proximal_operator;

  using DBASubproblemBase<kGPU, T, false>::m_send_extrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_send_intrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_send_point_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_send_extrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_send_intrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_send_point_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_send_data;

  using DBASubproblemBase<kGPU, T, false>::m_recv_extrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_recv_intrinsics_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_recv_point_sizes;
  using DBASubproblemBase<kGPU, T, false>::m_recv_extrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_recv_intrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_recv_point_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_recv_data;

  using DBASubproblemBase<kGPU, T, false>::m_cpu_send_extrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_send_intrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_send_point_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_send_data;

  using DBASubproblemBase<kGPU, T, false>::m_cpu_recv_extrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_recv_intrinsics_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_recv_point_dicts;
  using DBASubproblemBase<kGPU, T, false>::m_cpu_recv_data;

  int_t m_device;
  cudaStream_t m_stream;
  std::vector<cudaStream_t> m_comm_streams;

  int_t m_num_cameras;

  Container<int_t> m_n_measurement_dicts_by_cameras[3];
  Container<int_t> m_n_measurement_indices_by_cameras[3];
  Container<int_t> m_n_measurement_offsets_by_cameras[3];

  Container<int_t> m_n_measurement_dicts_by_points[3];
  Container<int_t> m_n_measurement_indices_by_points[3];
  Container<int_t> m_n_measurement_offsets_by_points[3];

  // array of matrix for kCPU, matrix of array for kGPU
  // hessians and gradients for preconditioned conjugate method
  // assume each camera has unique extrinsics and intrinsics
  mutable Container<T> m_hess_cc;        // hessians for camera-by-camera
  mutable Container<T> m_hess_ll;        // hessians for point-by-point
  mutable Container<T> m_grad_c;         // graidents for cameras
  mutable Container<T> m_grad_l;         // graidents for points
  mutable Container<T> m_reduced_grad_c; // reduced graidents for cameras
  mutable T m_schur_reduction_l;    // reduction after eliminating points using
                                    // schur complementment
  mutable Container<T> m_buffer[5]; // buffers for jacobians, hessians for
                                    // camera-by-point and rescaled_errors

  mutable Container<T> m_future_surrogate_f;
  mutable Container<T> m_future_objective_f;
  mutable Container<T> m_future_inner_product;

  mutable Container<T> m_pcg_x_c;
  mutable Container<T> m_pcg_x_l;
  mutable Container<T> m_pcg_r_c;
  mutable Container<T> m_pcg_dx_c;
  mutable Container<T> m_pcg_dr_c;
  mutable Container<T> m_pcg_dz_c;

  // array of matrix for kCPU, matrix of array for kGPU
  // hessians and gradients for preconditioned conjugate method
  // inverse of the hessians
  mutable Container<T>
      m_hess_cc_inv; // inverse of hessians for camera-by-camera
  mutable Container<T>
      m_hess_ll_inv; // inverse of hessians for camera-by-camera

  // measurement indexed by ranks
  std::vector<int_t> m_n_measurement_offsets_by_ranks;
  std::vector<int_t> m_n_measurement_sizes_by_ranks;
  thrust::device_vector<int_t> m_n_measurement_dicts_by_ranks;

  std::vector<Container<char>> m_nesterov_reduce_buffer;
  mutable Container<T> m_nesterov_future_reduced_f;
  mutable std::vector<T> m_nesterov_reduced_f;

  // private:
  int PreNesterovUpdateAsync() const;
  int PostNesterovUpdateAsync() const;
  int RetractAsync(T stepsize) const;
  int LinearizeAsync(bool nesterov) const;
  int BuildLinearSystemAsync(T ratio, Container<T> &future_schur_reduction_l,
                             cudaEvent_t event) const;
  int BuildLinearSystemSync(Container<T> &future_schur_reduction_l,
                            cudaEvent_t event) const;
  int ConstructSurrogateFunctionAsync() const;
  int EvaluateSurrogateFunctionAsync(const Container<T> &extrinsics,
                                     const Container<T> &intrinsics,
                                     const Container<T> &point,
                                     Container<T> &future_surrogate_f,
                                     bool nesterov, cudaEvent_t event) const;
  int EvaluateSurrogateFunctionSync(Container<T> &future_surrogate_f,
                                    std::array<T, 3> &surrogate_f, T &cost,
                                    cudaEvent_t event) const;
  int ComputeReducedCameraMatrixVectorMultiplicationAsync(const T *hess_cc,
                                                          const T *hess_cl,
                                                          const T *hess_ll_inv,
                                                          const T *x, T *y,
                                                          T *temp) const;

  int ComputeReducedCameraMatrixVectorMultiplicationAsync(
      const T *hess_cc, std::array<const T *, 2> hess_cl, const T *hess_ll_inv,
      const T *x, T *y, T *temp) const;

  int SolveProximalMethodAsync(bool nesterov) const;
  int PreCommunicateAsync(bool nesterov) const;
  int PreCommunicateSync() const;
  int PostCommunicateAsync(bool nesterov) const;
  int PostCommunicateSync() const;
};
} // namespace ba
} // namespace sfm
