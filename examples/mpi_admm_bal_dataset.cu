// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sfm/ba/dataset.h>
#include <sfm/utils/timer.h>
#include <sfm/utils/utils.cuh>
#include <sfm/utils/utils.h>

#include <examples/admm/admm_problem.h>

template <typename T>
T DifferenceNorm(thrust::device_vector<T> x, thrust::device_vector<T> y,
                 cudaStream_t stream) {
  auto iter = thrust::make_zip_iterator(x.data(), y.data());
  return thrust::transform_reduce(
      thrust::cuda::par.on(stream), iter, iter + x.size(),
      [] __device__(auto inputs) -> T {
        auto input1 = thrust::get<0>(inputs);
        auto input2 = thrust::get<1>(inputs);
        return std::pow(input1 - input2, 2);
      },
      T(0.0), cub::Sum());
}

int main(int argc, char *argv[]) {
  using T = double;
  using sfm::int_t;

  MPI_Init(&argc, &argv);

  // Get the number of processes
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  google::InitGoogleLogging(argv[0]);

  int num_devices;
  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  CHECK_CUDA(cudaSetDevice(rank % num_devices));

  assert(num_devices > 0);
  if (num_devices <= 0) {
    LOG(ERROR) << "The are no GPUs available." << std::endl;
    exit(-1);
  }

  boost::program_options::options_description desc("Program options");
  desc.add_options()                   // solver options
      ("help", "produce help message") // produce help message
      ("dataset", boost::program_options::value<std::string>(),
       "path to BAL dataset") // path to BAL dataset
      ("iters", boost::program_options::value<int>()->default_value(1000),
       "number of iterations") // maximum number of iterations
      ("loss",
       boost::program_options::value<std::string>()->default_value("trivial"),
       "loss type (\"trivial\" or \"huber\")"); // loss types

  boost::program_options::variables_map program_options;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc),
      program_options);

  if (program_options.count("help")) {
    if (rank == 0) {
      std::cout << desc << "\n";
    }
    exit(1);
  }

  if (program_options.count("dataset") == false) {
    if (rank == 0) {
      LOG(ERROR) << "No dataset has been specfied." << std::endl;
    }
    exit(-1);
  }

  std::string dataset_file = program_options["dataset"].as<std::string>();
  int max_iters = program_options["iters"].as<int>();
  std::string robust_loss_info = program_options["loss"].as<std::string>();
  std::string trust_region_solver_info = "lm";

  sfm::RobustLoss robust_loss;
  if (robust_loss_info == "trivial") {
    robust_loss = sfm::RobustLoss::Trivial;
  } else if (robust_loss_info == "huber") {
    robust_loss = sfm::RobustLoss::Huber;
  } else {
    if (rank == 0) {
      LOG(ERROR) << "The loss type can only be \"trivial\" and "
                    "\"huber\"."
                 << std::endl;
    }
    exit(-1);
  }

  int width = std::ceil(std::log10(max_iters + 1));

  auto ba_dataset = std::make_shared<sfm::ba::BALDataset<T>>(dataset_file);
  T initial_resolution, refined_resolution;
  bool memory_efficient = true;

  if (ba_dataset->Extrinsics().size() < 700) {
    initial_resolution = 0.5;
    refined_resolution = 1.0;
    memory_efficient = false;
  } else if (ba_dataset->Extrinsics().size() < 1500) {
    initial_resolution = 1.0;
    refined_resolution = 2.5;
    memory_efficient = true;
  } else {
    initial_resolution = 1.0;
    refined_resolution = 2.0;
    memory_efficient = true;
  }

  std::shared_ptr<sfm::ba::DBADataset<T>> dist_ba_dataset =
      std::make_shared<sfm::ba::DBADataset<T>>(
          ba_dataset, num_ranks, initial_resolution, refined_resolution,
          memory_efficient);

  if (rank == 0) {
    std::cout << "-----------------------------------------------------------"
              << std::endl;
    std::cout << ba_dataset->Extrinsics().size() << " cameras, "
              << ba_dataset->Points().size() << " points, "
              << ba_dataset->Measurements().size() << " measurements."
              << std::endl;
    std::cout << "-----------------------------------------------------------"
              << std::endl;
    for (int_t rank = 0; rank < num_ranks; rank++) {
      std::cout << "rank " << rank << " has "
                << dist_ba_dataset->Extrinsics()[rank].size() << " cameras, "
                << dist_ba_dataset->Points()[rank].size() << " points and "
                << dist_ba_dataset->Measurements()[rank].size()
                << " measruements." << std::endl;
    }
    std::cout << "-----------------------------------------------------------"
              << std::endl;
  }

  std::vector<Eigen::Vector2<T>> measurements;
  std::vector<std::array<sfm::int_t, 2>> extrinsics_infos;
  std::vector<std::array<sfm::int_t, 2>> intrinsics_infos;
  std::vector<std::array<sfm::int_t, 2>> point_infos;
  std::vector<T> sqrt_weights;

  const auto num_measurements = dist_ba_dataset->Measurements()[rank].size();
  measurements.reserve(num_measurements);
  extrinsics_infos.reserve(num_measurements);
  intrinsics_infos.reserve(num_measurements);
  point_infos.reserve(num_measurements);
  sqrt_weights.reserve(num_measurements);

  for (const auto &measurement : dist_ba_dataset->Measurements()[rank]) {
    measurements.push_back((measurement.measurement));
    extrinsics_infos.push_back(measurement.extrinsic_index);
    intrinsics_infos.push_back(measurement.intrinsic_index);
    point_infos.push_back(measurement.point_index);
    sqrt_weights.push_back(measurement.sqrt_weight);
  }
  sfm::ba::ADMMOption<T> option;
  option.trust_region_option.max_iterations = 20;
  option.trust_region_option.max_accepted_iterations = 15;
  option.trust_region_option.verbose = false;
  option.robust_loss = robust_loss;
  option.pcg_option.max_iterations = 400;
  option.pcg_option.relative_reduction_tol = 0.1;
  option.initial_extrinsics_penalty = (2.5 * ba_dataset->Measurements().size() /
                                       ba_dataset->Extrinsics().size());
  option.initial_intrinsics_penalty = (2.5 * ba_dataset->Measurements().size() /
                                       ba_dataset->Extrinsics().size());
  option.increasing_penalty_ratio = 1.5;
  option.decreasing_penalty_ratio = 0.8;
  std::shared_ptr<sfm::ba::ADMMSubproblem<sfm::kGPU, T, false>> problem;

  MPI_Barrier(MPI_COMM_WORLD);

  problem = std::make_shared<sfm::ba::ADMMSubproblem<sfm::kGPU, T, false>>(
      option, rank, num_ranks);
  problem->Setup(extrinsics_infos, intrinsics_infos, point_infos, measurements,
                 sqrt_weights);

  const auto &initial_extrinsics = dist_ba_dataset->Extrinsics();
  const auto &initial_intrinsics = dist_ba_dataset->Intrinsics();
  const auto &initial_points = dist_ba_dataset->Points();

  problem->Initialize(initial_extrinsics, initial_intrinsics, initial_points);

  T time_comm = 0;
  T time_optim_max = 0;
  T time_optim_avg = 0;

  std::vector<Eigen::Vector<T, 5>> records;
  std::vector<std::array<T, 3>> recv_data(num_ranks);
  T time_optim_n = 0;

  T cost = 0, error = 0;
  problem->GetCost(cost);

  auto start = sfm::utils::Timer::tick();
  for (int iter = 0; iter < max_iters; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      recv_data[0][0] = cost;
      recv_data[0][1] = error;
      recv_data[0][2] = time_optim_n;

      auto start_comm = sfm::utils::Timer::tick();
      for (int_t rank = 1; rank < num_ranks; rank++) {
        MPI_Status status;
        MPI_Recv(recv_data[rank].data(), 3, sfm::traits<T>::MPI_FLOAT_TYPE,
                 rank, 1, MPI_COMM_WORLD, &status);
      }
      time_comm += sfm::utils::Timer::tock(start_comm);

      T total_cost = 0;
      T total_error = 0;
      T time_optim_max_n = recv_data[0][2];
      T time_optim_sum_n = 0;
      for (int_t rank = 0; rank < num_ranks; rank++) {
        total_cost += recv_data[rank][0];
        total_error += recv_data[rank][1];
        time_optim_max_n = std::max(time_optim_max_n, recv_data[rank][2]);
        time_optim_sum_n += recv_data[rank][2];
      }
      time_optim_max += time_optim_max_n;
      time_optim_avg += time_optim_sum_n / num_ranks;

      T time = sfm::utils::Timer::tock(start);
      if (iter % 20 == 0) {
        std::cout << std::setw(width) << iter << "/" << std::setw(width)
                  << max_iters << ", " << std::setprecision(4) << std::fixed
                  << time_optim_max << "/" << std::setprecision(4) << std::fixed
                  << time << " seconds: " << std::scientific
                  << std::setprecision(25)
                  << total_cost / ba_dataset->Measurements().size() << " "
                  << std::setprecision(25) << total_error << std::endl;
      }
      records.push_back(
          {time_optim_max, time_optim_avg, time, total_cost, total_error});
    } else {
      std::array<T, 3> send_data{cost, error, time_optim_n};
      MPI_Send(send_data.data(), 3, sfm::traits<T>::MPI_FLOAT_TYPE, 0, 1,
               MPI_COMM_WORLD);
    }

    auto start_optim = sfm::utils::Timer::tick();
    problem->Iterate();
    time_optim_n = sfm::utils::Timer::tock(start_optim);

    auto start_comm = sfm::utils::Timer::tick();
    problem->MPICommunicate(MPI_COMM_WORLD, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    problem->MPICommunicate(MPI_COMM_WORLD, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    problem->MPICommunicate(MPI_COMM_WORLD, 2);
    MPI_Barrier(MPI_COMM_WORLD);
    time_comm += sfm::utils::Timer::tock(start_comm);
    problem->EvaluateObjectiveFunction(problem->m_extrinsics_consensus,
                                       problem->m_intrinsics_consensus,
                                       problem->m_points, cost);
    error = DifferenceNorm(problem->m_extrinsics,
                           problem->m_extrinsics_consensus, problem->m_stream) +
            DifferenceNorm(problem->m_intrinsics,
                           problem->m_intrinsics_consensus, problem->m_stream);

    problem->UpdateConsensus();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    recv_data[0][0] = cost;
    recv_data[0][1] = error;
    recv_data[0][2] = time_optim_n;

    auto start_comm = sfm::utils::Timer::tick();
    for (int_t rank = 1; rank < num_ranks; rank++) {
      MPI_Status status;
      MPI_Recv(recv_data[rank].data(), 3, sfm::traits<T>::MPI_FLOAT_TYPE, rank,
               1, MPI_COMM_WORLD, &status);
    }
    time_comm += sfm::utils::Timer::tock(start_comm);

    T total_cost = 0;
    T total_error = 0;
    T time_optim_max_n = recv_data[0][2];
    T time_optim_sum_n = 0;
    for (int_t rank = 0; rank < num_ranks; rank++) {
      total_cost += recv_data[rank][0];
      total_error += recv_data[rank][1];
      time_optim_max_n = std::max(time_optim_max_n, recv_data[rank][2]);
      time_optim_sum_n += recv_data[rank][2];
    }
    time_optim_max += time_optim_max_n;
    time_optim_avg += time_optim_sum_n / num_ranks;

    T time = sfm::utils::Timer::tock(start);
    std::cout << std::setw(width) << max_iters << "/" << std::setw(width)
              << max_iters << ", " << std::setprecision(4) << std::fixed
              << time_optim_max << "/" << std::setprecision(4) << std::fixed
              << time << " seconds: " << std::scientific
              << std::setprecision(25)
              << total_cost / ba_dataset->Measurements().size() << " "
              << std::setprecision(25) << total_error << std::endl;
    records.push_back(
        {time_optim_max, time_optim_avg, time, total_cost, total_error});
    std::string outfile = dataset_file.substr(dataset_file.rfind("problem-"));
    outfile = outfile.substr(0, outfile.find(".txt"));
    outfile = "admm-" + trust_region_solver_info + "-" + robust_loss_info +
              "-" + outfile + "-" + std::to_string(num_ranks) + "-GPU-" +
              std::to_string(max_iters) + "-iters.txt";

    std::ofstream fout(outfile);
    fout << ba_dataset->Extrinsics().size() << " "
         << ba_dataset->Intrinsics().size() << " "
         << ba_dataset->Points().size() << " "
         << ba_dataset->Measurements().size() << std::endl;

    for (int iter = 0; iter <= max_iters; iter++) {
      fout << iter << " " << records[iter].transpose() << std::endl;
    }
  } else {
    std::array<T, 3> send_data{cost, error, time_optim_n};
    MPI_Send(send_data.data(), 3, sfm::traits<T>::MPI_FLOAT_TYPE, 0, 1,
             MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}