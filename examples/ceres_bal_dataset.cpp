// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <examples/ceres/camera.h>
#include <examples/ceres/reprojection.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sfm/ba/dataset.h>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  boost::program_options::options_description desc("Program options");
  desc.add_options()                   // solver options
      ("help", "produce help message") // produce help message
      ("dataset", boost::program_options::value<std::string>(),
       "path to BAL dataset") // path to BAL dataset
      ("loss",
       boost::program_options::value<std::string>()->default_value("trivial"),
       "loss type (\"trivial\" or \"huber\")"); // loss types

  boost::program_options::variables_map program_options;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc),
      program_options);

  if (program_options.count("help")) {
    std::cout << desc << "\n";
    exit(1);
  }

  if (program_options.count("dataset") == false) {
    LOG(ERROR) << "No dataset has been specfied." << std::endl;
    exit(-1);
  }

  std::string filename = program_options["dataset"].as<std::string>();
  std::string robust_loss_info = program_options["loss"].as<std::string>();

  sfm::ba::BALDataset<Ceres::Scalar> ba_dataset(filename, true);
  const auto &measurements = ba_dataset.Measurements();
  const auto &extrinsics = ba_dataset.Extrinsics();
  const auto &intrinsics = ba_dataset.Intrinsics();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << ba_dataset.Extrinsics().size() << " cameras, "
            << ba_dataset.Points().size() << " points, "
            << ba_dataset.Measurements().size() << " measurements."
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;

  std::vector<Ceres::Matrix<3, 5>> cameras(extrinsics.size());
  for (int i = 0; i < extrinsics.size(); i++) {
    cameras[i].leftCols<4>() = extrinsics[i];
    cameras[i].col(4) = intrinsics[i];
  }

  std::vector<Ceres::Vector<3>> points = ba_dataset.Points();

  ceres::Problem problem;
  ceres::Manifold *manifold = new Ceres::Camera();
  ceres::LossFunction *loss = nullptr;
  if (robust_loss_info == "huber") {
    loss = new ceres::HuberLoss(32);
  } else if (robust_loss_info == "trivial") {
    loss = nullptr;
  } else {
    LOG(ERROR) << "The loss type can only be \"trivial\" and "
                  "\"huber\"."
               << std::endl;
    exit(-1);
  }

  std::vector<Ceres::ReprojectionError *> edges;

  for (const auto &measurement : measurements) {
    auto edge = new Ceres::ReprojectionError(measurement.measurement,
                                             measurement.sqrt_weight);
    edges.push_back(edge);
    problem.AddResidualBlock(edge, loss,
                             cameras[measurement.extrinsics_index].data(),
                             points[measurement.point_index].data());
  }

  ceres::Problem::EvaluateOptions eval_options;
  std::vector<double> gradients;

  for (auto &camera : cameras) {
    eval_options.parameter_blocks.push_back(camera.data());
  }

  for (auto &point : points) {
    eval_options.parameter_blocks.push_back(point.data());
  }

  for (auto &camera : cameras) {
    problem.SetManifold(camera.data(), manifold);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 40;
  options.num_threads = 64;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.gradient_tolerance = 0;
  options.max_solver_time_in_seconds = 14400;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::string outfile = filename.substr(filename.rfind("problem-"));
  outfile = outfile.substr(0, outfile.find(".txt"));
  outfile = "ceres-" + robust_loss_info + "-" + outfile + ".txt";

  std::ofstream fout(outfile);

  double min_cost = summary.iterations[0].cost;
  for (int n = 0; n < summary.iterations.size(); n++) {
    const auto &iteration = summary.iterations[n];
    min_cost = std::min(iteration.cost, min_cost);
    fout << n << " " << min_cost << " " << iteration.cost << " "
         << iteration.iteration_time_in_seconds << " "
         << iteration.cumulative_time_in_seconds << std::endl;
  }

  fout.close();

  return 0;
}