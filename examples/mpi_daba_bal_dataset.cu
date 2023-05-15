// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <sfm/ba/dataset.h>
#include <sfm/ba/distributed_problem.h>
#include <sfm/utils/timer.h>
#include <sfm/utils/utils.h>

#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
       "loss type (\"trivial\" or \"huber\")") // loss types
      ("accelerated",
       boost::program_options::value<bool>()->default_value(true),
       "whether accelerated or not") // whether accelerated or not
      ("save", boost::program_options::value<bool>()->default_value(true),
       "whether to save the optimization results or not"); // whether to save
                                                           // the optimization
                                                           // results or not

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
  bool accelerated = program_options["accelerated"].as<bool>();
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

  auto ba_dataset =
      std::make_shared<sfm::ba::BALDataset<T>>(dataset_file, rank == 0);

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
  sfm::ba::Option<T> option;
  option.accelerated = accelerated;
  option.trust_region_option.max_iterations = 10;
  option.trust_region_option.max_accepted_iterations = 1;
  option.trust_region_option.verbose = false;
  option.robust_loss = robust_loss;
  option.pcg_option.max_iterations = 400;
  option.pcg_option.relative_reduction_tol = 0.1;
  option.initial_nesterov_average_objective_value_ratio = 250;
  option.initial_nesterov_eta = 1e-3;
  option.increasing_nesterov_eta_ratio = 5e-4;
  std::shared_ptr<sfm::ba::DBASubproblem<sfm::kGPU, T, false>> problem;

  MPI_Barrier(MPI_COMM_WORLD);

  problem = std::make_shared<sfm::ba::DBASubproblem<sfm::kGPU, T, false>>(
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

  std::vector<Eigen::Vector4<T>> records;
  std::vector<std::array<T, 2>> recv_data(num_ranks);
  std::vector<T> time_optim_n(num_ranks, 0);
  auto start = sfm::utils::Timer::tick();
  for (int iter = 0; iter < max_iters; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);
    T cost;
    auto start_cost = sfm::utils::Timer::tick();
    problem->GetCost(cost);
    time_optim_n[rank] += sfm::utils::Timer::tock(start_cost);
    if (rank == 0) {
      recv_data[0] = {cost, time_optim_n[0]};

      auto start_comm = sfm::utils::Timer::tick();
      for (int_t rank = 1; rank < num_ranks; rank++) {
        MPI_Status status;
        MPI_Recv(recv_data[rank].data(), 2, sfm::traits<T>::MPI_FLOAT_TYPE,
                 rank, 1, MPI_COMM_WORLD, &status);
      }
      time_comm += sfm::utils::Timer::tock(start_comm);

      T total_cost = 0;
      T time_optim_max_n = recv_data[0][1];
      T time_optim_sum_n = 0;
      for (int_t rank = 0; rank < num_ranks; rank++) {
        total_cost += recv_data[rank][0];
        time_optim_max_n = std::max(time_optim_max_n, recv_data[rank][1]);
        time_optim_sum_n += recv_data[rank][1];
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
                  << total_cost / ba_dataset->Measurements().size()
                  << std::endl;
      }
      records.push_back({time_optim_max, time_optim_avg, time, total_cost});
    } else {
      T send_data[2] = {cost, time_optim_n[rank]};
      MPI_Send(&send_data, 2, sfm::traits<T>::MPI_FLOAT_TYPE, 0, 1,
               MPI_COMM_WORLD);
    }

    auto start_optim = sfm::utils::Timer::tick();
    problem->Iterate();
    time_optim_n[rank] = sfm::utils::Timer::tock(start_optim);

    auto start_comm = sfm::utils::Timer::tick();
    problem->MPICommunicate(MPI_COMM_WORLD, false);
    time_comm += sfm::utils::Timer::tock(start_comm);

    problem->UpdateSurrogateFunction();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  T cost;
  auto start_cost = sfm::utils::Timer::tick();
  problem->GetCost(cost);
  time_optim_n[rank] += sfm::utils::Timer::tock(start_cost);
  if (rank == 0) {
    recv_data[0] = {cost, time_optim_n[0]};

    auto start_comm = sfm::utils::Timer::tick();
    for (int_t rank = 1; rank < num_ranks; rank++) {
      MPI_Status status;
      MPI_Recv(recv_data[rank].data(), 2, sfm::traits<T>::MPI_FLOAT_TYPE, rank,
               1, MPI_COMM_WORLD, &status);
    }
    time_comm += sfm::utils::Timer::tock(start_comm);

    T total_cost = 0;
    T time_optim_max_n = 0;
    T time_optim_sum_n = 0;
    for (int_t rank = 0; rank < num_ranks; rank++) {
      total_cost += recv_data[rank][0];
      time_optim_max_n = std::max(time_optim_max_n, recv_data[rank][1]);
      time_optim_sum_n += recv_data[rank][1];
    }
    time_optim_max += time_optim_max_n;
    time_optim_avg += time_optim_sum_n / num_ranks;

    T time = sfm::utils::Timer::tock(start);
    std::cout << std::setw(width) << max_iters << "/" << std::setw(width)
              << max_iters << ", " << std::setprecision(4) << std::fixed
              << time_optim_max << "/" << std::setprecision(4) << std::fixed
              << time << " seconds: " << std::scientific
              << std::setprecision(25)
              << total_cost / ba_dataset->Measurements().size() << std::endl;
    records.push_back({time_optim_max, time_optim_avg, time, total_cost});
    std::string outfile = dataset_file.substr(dataset_file.rfind("problem-"));
    outfile = outfile.substr(0, outfile.find(".txt"));
    outfile = "mm-" + trust_region_solver_info +
              std::string(accelerated ? "-acc-" : "-unacc-") +
              robust_loss_info + "-" + outfile + "-" +
              std::to_string(num_ranks) + "-GPU-" + std::to_string(max_iters) +
              "-iters.txt";

    std::ofstream fout(outfile);
    fout << ba_dataset->Extrinsics().size() << " "
         << ba_dataset->Intrinsics().size() << " "
         << ba_dataset->Points().size() << " "
         << ba_dataset->Measurements().size() << std::endl;

    for (int iter = 0; iter <= max_iters; iter++) {
      fout << iter << " " << records[iter].transpose() << std::endl;
    }
    fout.close();
  } else {
    T send_data[2] = {cost, time_optim_n[rank]};
    MPI_Send(send_data, 2, sfm::traits<T>::MPI_FLOAT_TYPE, 0, 1,
             MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (program_options["save"].as<bool>()) {
    const auto &d_extrinsics = problem->GetDeviceExtrinsics();
    const auto &d_intrinsics = problem->GetDeviceIntrinsics();
    const auto &d_points = problem->GetDevicePoints();

    if (rank == 0) {
      std::vector<std::array<int_t, 3>> problem_info(num_ranks);

      for (int_t rank = 0; rank < num_ranks; rank++) {
        problem_info[rank][0] = dist_ba_dataset->Extrinsics()[rank].size();
        problem_info[rank][1] = dist_ba_dataset->Intrinsics()[rank].size();
        problem_info[rank][2] = dist_ba_dataset->Points()[rank].size();
      }

      std::vector<std::vector<T>> recv_data(num_ranks);
      std::vector<MPI_Request> recv_requests(num_ranks);

      for (int_t rank = 1; rank < num_ranks; rank++) {
        recv_data[rank].resize(12 * problem_info[rank][0] +
                                   3 * problem_info[rank][1] +
                                   3 * problem_info[rank][2],
                               0);
        MPI_Irecv(recv_data[rank].data(), recv_data[rank].size(),
                  sfm::traits<T>::MPI_FLOAT_TYPE, rank, rank, MPI_COMM_WORLD,
                  &recv_requests[rank]);
      }

      for (int_t rank = 1; rank < num_ranks; rank++) {
        MPI_Status status;
        MPI_Wait(&recv_requests[rank], &status);
      }

      std::vector<std::vector<Eigen::Matrix<T, 3, 4>>> extrinsics(num_ranks);
      std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> intrinsics(num_ranks);
      std::vector<std::vector<Eigen::Matrix<T, 3, 1>>> points(num_ranks);

      problem->GetExtrinsics(extrinsics[0]);
      problem->GetIntrinsics(intrinsics[0]);
      problem->GetPoints(points[0]);

      for (int_t rank = 1; rank < num_ranks; rank++) {
        extrinsics[rank].resize(problem_info[rank][0]);
        intrinsics[rank].resize(problem_info[rank][1]);
        points[rank].resize(problem_info[rank][2]);

        T *src = recv_data[rank].data();
        sfm::utils::MatrixOfArrayToArrayOfMatrix(extrinsics[rank].size(), src,
                                                 extrinsics[rank].data());

        src += 12 * extrinsics[rank].size();
        sfm::utils::MatrixOfArrayToArrayOfMatrix(intrinsics[rank].size(), src,
                                                 intrinsics[rank].data());

        src += 3 * intrinsics[rank].size();
        sfm::utils::MatrixOfArrayToArrayOfMatrix(points[rank].size(), src,
                                                 points[rank].data());
      }

      std::string outfile = dataset_file.substr(dataset_file.rfind("problem-"));
      outfile = outfile.substr(0, outfile.find(".txt"));
      outfile = "results-mm-" + trust_region_solver_info +
                std::string(accelerated ? "-acc-" : "-unacc-") +
                robust_loss_info + "-" + outfile + "-" +
                std::to_string(num_ranks) + "-GPU-" +
                std::to_string(max_iters) + "-iters.txt";

      std::cout << "-----------------------------------------------------------"
                << std::endl;
      std::cout << "Save optimization results to " << outfile << std::endl;
      std::ofstream fout(outfile);
      fout << ba_dataset->Extrinsics().size() << " "
           << ba_dataset->Intrinsics().size() << " "
           << ba_dataset->Points().size() << " " << num_ranks << std::endl;

      for (int_t rank = 0; rank < num_ranks; rank++) {
        fout << problem_info[rank][0] << " " << problem_info[rank][1] << " "
             << problem_info[rank][2] << std::endl;
      }

      int_t num_extrinsics = 0;
      int_t num_intrinsics = 0;
      int_t num_points = 0;

      for (int_t rank = 0; rank < num_ranks; rank++) {
        for (const auto &extrinsics_n : extrinsics[rank]) {
          fout << std::scientific << std::setprecision(10) << extrinsics_n
               << std::endl;
          num_extrinsics++;
        }
      }

      if (num_extrinsics != ba_dataset->Extrinsics().size()) {
        LOG(ERROR) << "Inconsistent extrinsics infomration." << std::endl;
        exit(-1);
      }

      for (int_t rank = 0; rank < num_ranks; rank++) {
        for (const auto &intrinsics_n : intrinsics[rank]) {
          fout << std::scientific << std::setprecision(10)
               << intrinsics_n.transpose() << std::endl;
          num_intrinsics++;
        }
      }

      if (num_intrinsics != ba_dataset->Intrinsics().size()) {
        LOG(ERROR) << "Inconsistent intrinsics infomration." << std::endl;
        exit(-1);
      }

      for (int_t rank = 0; rank < num_ranks; rank++) {
        for (const auto &points_n : points[rank]) {
          fout << std::scientific << std::setprecision(10)
               << points_n.transpose() << std::endl;
          num_points++;
        }
      }

      if (num_points != ba_dataset->Points().size()) {
        LOG(ERROR) << "Inconsistent points infomration." << std::endl;
        exit(-1);
      }
    } else {
      // send extrinsics, intrinsics and points
      int_t num_extrinsics = d_extrinsics.size() / 12;
      int_t num_intrinsics = d_intrinsics.size() / 3;
      int_t num_points = d_points.size() / 3;

      int_t num_extrinsics_n = dist_ba_dataset->Extrinsics()[rank].size();
      int_t num_intrinsics_n = dist_ba_dataset->Intrinsics()[rank].size();
      int_t num_points_n = dist_ba_dataset->Points()[rank].size();

      std::vector<T> send_data(
          12 * num_extrinsics_n + 3 * num_intrinsics_n + 3 * num_points_n, 0);
      auto dst = send_data.begin();

      auto extrinsics_src = d_extrinsics.begin();
      for (int_t n = 0; n < 12; n++) {
        thrust::copy(extrinsics_src, extrinsics_src + num_extrinsics_n, dst);
        dst += num_extrinsics_n;
        extrinsics_src += num_extrinsics;
      }

      auto intrinsics_src = d_intrinsics.begin();
      for (int_t n = 0; n < 3; n++) {
        thrust::copy(intrinsics_src, intrinsics_src + num_intrinsics_n, dst);
        dst += num_intrinsics_n;
        intrinsics_src += num_intrinsics;
      }

      auto points_src = d_points.begin();
      for (int_t n = 0; n < 3; n++) {
        thrust::copy(points_src, points_src + num_points_n, dst);
        dst += num_points_n;
        points_src += num_points;
      }

      MPI_Request send_request;
      MPI_Isend(send_data.data(), send_data.size(),
                sfm::traits<T>::MPI_FLOAT_TYPE, 0, rank, MPI_COMM_WORLD,
                &send_request);

      MPI_Status status;
      MPI_Wait(&send_request, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();

  return 0;
}
