// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <glog/logging.h>
#include <memory>
#include <mpi.h>
#include <nccl.h>
#include <sfm/ba/types.h>
#include <sfm/types.h>
#include <sfm/utils/cuda_utils.h>
#include <sfm/utils/utils.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace sfm {
namespace ba {
template <Memory kMemory, typename T, bool kSharedIntrinsics>
class DouglasRachfordSubproblem;

template <typename T> struct DROption {
  /** Whether to print output as the algorithm runs */
  bool verbose = false;

  /** Robust loss function */
  RobustLoss robust_loss = Trivial;

  /** loss radius */
  T loss_radius = 1024;

  //------------------------------------------------------
  // Trust Region Method Option
  //------------------------------------------------------
  sfm::optimization::TrustRegionOption<T> trust_region_option{
      100, 1, 5e-3, 1e-4, 1e-6, 1e-4, 1e-2, 0.6, 0.5, 3, false};

  //------------------------------------------------------
  // Preconditioned Conjugate Graident Method Option
  //------------------------------------------------------
  sfm::optimization::PCGOption<T> pcg_option{100, 1e-2, 1e-1, 0.5, false};
};

template <typename T> class DouglasRachfordSubproblem<kGPU, T, false> {
public:
  template <typename P> using Container = thrust::device_vector<P>;

  DouglasRachfordSubproblem(const DROption<T> &option, int_t rank,
                              int_t num_ranks);

  DouglasRachfordSubproblem(const DROption<T> &option, int_t rank,
                              int_t num_ranks, int_t device);

  ~DouglasRachfordSubproblem();
  int GetCost(T &cost) const;

  int GetSurrogateCost(std::array<T, 2> &surrogate_cost) const;

  int GetOption(DROption<T> &option) const;

  virtual int
  GetExtrinsics(std::vector<Eigen::Matrix<T, 3, 4>> &extrinsics) const;

  virtual int GetIntrinsics(std::vector<Eigen::Vector3<T>> &intrinsics) const;

  virtual int GetPoints(std::vector<Eigen::Vector3<T>> &points) const;

  int_t GetDevice() const;

  T GetMemoryUsage() const;

  T GetCommunicationLoad() const;

  const Container<T> &GetDeviceMeasurements() const;

  const Container<int_t> &GetDeviceExtrinsicsInfo() const;

  const Container<int_t> &GetDeviceIntrinsicsInfo() const;

  const Container<int_t> &GetDevicePointsInfo() const;

  const Container<T> &GetDeviceSqrtWeights() const;

  const Container<T> &GetDeviceExtrinsics() const;

  const Container<T> &GetDeviceIntrinsics() const;

  const Container<T> &GetDevicePoints() const;

  virtual int Initialize(
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 4>>>
          &extrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &intrinsics,
      const std::vector<std::unordered_map<int_t, Eigen::Matrix<T, 3, 1>>>
          &points);

  virtual int
  Initialize(const std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> &extrinsics,
             const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &intrinsics,
             const std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> &points);

  virtual int Iterate(T rho) const;

  virtual int MPICommunicate(const MPI_Comm &comm, int_t round) const;

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
                    const std::vector<T> &sqrt_weights);

  // protected:
  virtual int Accept(const Container<T> &extrinsics,
                     const Container<T> &intrinsics, const Container<T> &points,
                     const std::array<T, 2> &surrogate_f, T cost) const;

  virtual int InitializeSurrogateFunction();

  virtual int PreCommunicate(int_t round) const;
  virtual int PostCommunicate(int_t round) const;

  virtual int EvaluateSurrogateFunction(const Container<T> &extrinsics,
                                        const Container<T> &intrinsics,
                                        const Container<T> &points, T rho,
                                        std::array<T, 2> &surrogate_f,
                                        T &cost) const;

  virtual int LevenbergMarquardtMethod(T rho) const;

  virtual int Linearize(T rho) const;
  virtual int BuildLinearSystem(T ratio) const;
  virtual int PCG(int_t &num_iters, T &update_step_norm) const;
  virtual int Retract(T stepsize) const;

  int Update(T rho) const;

  // distributed bundle adjustment subproblem infomation
  const DROption<T> m_option;
  int_t m_rank;
  int_t m_num_ranks;

  int_t m_device;
  cudaStream_t m_stream;
  std::vector<cudaStream_t> m_comm_streams;

  // problem information
  int_t m_num_cameras;
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
  Container<int_t> m_n_measurement_indices;
  Container<int_t> m_n_extrinsics_dicts;
  Container<int_t> m_n_extrinsics_indices;
  Container<int_t> m_n_intrinsics_dicts;
  Container<int_t> m_n_intrinsics_indices;
  Container<int_t> m_n_point_dicts;
  Container<int_t> m_n_point_indices;

  // array of matrix for kCPU, matrix of array for kGPU
  mutable Container<T> m_extrinsics; // 3x4 matrix
  mutable Container<T> m_intrinsics; // 3x1 matrix
  mutable Container<T> m_points;     // 3x1 matrix

  mutable Container<T> m_points_consensus; // 3x1 matrix
  mutable Container<T> m_points_corrected; // 3x1 matrix
  mutable Container<T> m_points_reference; // 3x1 matrix

  mutable std::array<T, 2> m_surrogate_f;
  mutable T m_cost = 0; // Objective values

  mutable Container<T> m_f_values; // Objective values for each factor

  mutable Container<T> m_trust_region_extrinsics[2]; // 3x4 matrix
  mutable Container<T> m_trust_region_intrinsics[2]; // 3x1 matrix
  mutable Container<T> m_trust_region_points[2];     // 3x1 matrix
  mutable std::array<T, 2> m_trust_region_surrogate_f[2];
  mutable T m_trust_region_cost[2] = {0, 0}; // Objective values
  mutable T m_trust_region_radius;
  mutable T m_trust_region_decreasing_ratio;

  // Shared point dicts
  Container<T> m_point_consensus_cnts;

  // Communication
  // [0]: points sent to the other ranks
  // [1]: points received from the other ranks
  std::vector<int_t> m_shared_point_sizes[2];
  std::vector<Container<int_t>> m_shared_point_dicts[2];
  mutable std::vector<Container<T>> m_shared_point_data[2];

  // CPU communication
  std::vector<int_t> m_cpu_shared_point_sizes[2];
  std::vector<PinnedHostVector<int_t>> m_cpu_shared_point_dicts[2];
  mutable std::vector<PinnedHostVector<T>> m_cpu_shared_point_data[2];

  // sort measurements according to cameras and points
  Container<int_t> m_measurement_dicts_by_cameras;
  Container<int_t> m_measurement_indices_by_cameras;
  Container<int_t> m_measurement_offsets_by_cameras;

  Container<int_t> m_measurement_dicts_by_points;
  Container<int_t> m_measurement_indices_by_points;
  Container<int_t> m_measurement_offsets_by_points;

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
  mutable Container<T> m_pcg_buffer;

  // array of matrix for kCPU, matrix of array for kGPU
  // hessians and gradients for preconditioned conjugate method
  // inverse of the hessians
  mutable Container<T>
      m_hess_cc_inv; // inverse of hessians for camera-by-camera
  mutable Container<T>
      m_hess_ll_inv; // inverse of hessians for camera-by-camera

  // private:
  int InitializeSurrogateFunctionAsync(Container<T> &future_surrogate_f,
                                       cudaEvent_t event);
  int InitializeSurrogateFunctionSync(Container<T> &future_surrogate_f,
                                      cudaEvent_t event);
  int AcceptAsync(const Container<T> &extrinsics,
                  const Container<T> &intrinsics, const Container<T> &points,
                  const std::array<T, 2> &surrogate_f, T cost) const;
  int RetractAsync(T stepsize) const;
  int LinearizeAsync(T rho) const;
  int BuildLinearSystemAsync(T ratio, Container<T> &future_schur_reduction_l,
                             cudaEvent_t event) const;
  int BuildLinearSystemSync(Container<T> &future_schur_reduction_l,
                            cudaEvent_t event) const;
  int EvaluateSurrogateFunctionAsync(const Container<T> &extrinsics,
                                     const Container<T> &intrinsics,
                                     const Container<T> &point,
                                     Container<T> &futrue_surrogate_f,
                                     cudaEvent_t event) const;
  int EvaluateSurrogateFunctionSync(Container<T> &future_surrogate_f, T rho,
                                    std::array<T, 2> &surrogate_f, T &cost,
                                    cudaEvent_t event) const;
  int ComputeReducedCameraMatrixVectorMultiplicationAsync(const T *hess_cc,
                                                          const T *hess_cl,
                                                          const T *hess_ll_inv,
                                                          const T *x, T *y,
                                                          T *temp) const;

  int ComputeReducedCameraMatrixVectorMultiplicationAsync(
      const T *hess_cc, std::array<const T *, 2> hess_cl, const T *hess_ll_inv,
      const T *x, T *y, T *temp) const;

  int PreCommunicateAsync(int_t round) const;
  int PreCommunicateSync(int_t round) const;
  int PostCommunicateAsync(int_t round) const;
  int PostCommunicateSync(int_t round) const;
};
} // namespace ba
} // namespace sfm