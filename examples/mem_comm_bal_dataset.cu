// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <sfm/ba/dataset.h>
#include <sfm/ba/distributed_problem.h>

#include <examples/admm/admm_problem.h>
#include <examples/douglas_rachford/douglas_rachford_problem.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [input bal dataset file]."
              << std::endl;
    exit(-1);
  }

  std::string filename = argv[1];
  using T = double;
  using int_t = sfm::int_t;

  auto ba_dataset = std::make_shared<sfm::ba::BALDataset<T>>(filename, true);

  T initial_resolution, refined_resolution;
  bool memory_efficient = true;

  if (ba_dataset->Extrinsics().size() < 700) {
    initial_resolution = 1.0;
    refined_resolution = 4.0;
    memory_efficient = false;
  } else {
    initial_resolution = 1.0;
    refined_resolution = 2.0;
    memory_efficient = true;
  }

  std::vector<Eigen::Vector<T, 6>> memory(4, Eigen::Vector<T, 6>::Zero());
  std::vector<Eigen::Vector<T, 6>> comm(4, Eigen::Vector<T, 6>::Zero());

  for (int_t num_ranks = 1, index = 0; num_ranks <= 32;
       num_ranks *= 2, index++) {
    sfm::ba::DBADataset<T> dist_bal_dataset(
        ba_dataset, num_ranks, initial_resolution, refined_resolution,
        memory_efficient);

    for (int_t rank = 0; rank < num_ranks; rank++) {
      std::vector<Eigen::Vector2<T>> measurements;
      std::vector<std::array<sfm::int_t, 2>> extrinsics_infos;
      std::vector<std::array<sfm::int_t, 2>> intrinsics_infos;
      std::vector<std::array<sfm::int_t, 2>> point_infos;
      std::vector<T> sqrt_weights;

      const auto num_measurements =
          dist_bal_dataset.Measurements()[rank].size();
      measurements.reserve(num_measurements);
      extrinsics_infos.reserve(num_measurements);
      intrinsics_infos.reserve(num_measurements);
      point_infos.reserve(num_measurements);
      sqrt_weights.reserve(num_measurements);

      for (const auto &measurement : dist_bal_dataset.Measurements()[rank]) {
        measurements.push_back((measurement.measurement));
        extrinsics_infos.push_back(measurement.extrinsic_index);
        intrinsics_infos.push_back(measurement.intrinsic_index);
        point_infos.push_back(measurement.point_index);
        sqrt_weights.push_back(measurement.sqrt_weight);
      }

      {
        int_t solver = 0;
        sfm::ba::Option<T> mm_option;
        mm_option.accelerated = false;
        mm_option.trust_region_option.max_iterations = 10;
        mm_option.trust_region_option.max_accepted_iterations = 1;
        mm_option.trust_region_option.verbose = false;
        mm_option.robust_loss = sfm::Trivial;
        mm_option.pcg_option.max_iterations = 400;

        std::shared_ptr<sfm::ba::DBASubproblem<sfm::kGPU, T, false>> mm;
        mm = std::make_shared<sfm::ba::DBASubproblem<sfm::kGPU, T, false>>(
            mm_option, rank, num_ranks);
        mm->Setup(extrinsics_infos, intrinsics_infos, point_infos, measurements,
                  sqrt_weights);

        memory[solver][index] =
            std::max(memory[solver][index], mm->GetMemoryUsage());
        comm[solver][index] += mm->GetCommunicationLoad();
      }

      {
        int_t solver = 1;
        sfm::ba::Option<T> amm_option;
        amm_option.accelerated = true;
        amm_option.trust_region_option.max_iterations = 10;
        amm_option.trust_region_option.max_accepted_iterations = 1;
        amm_option.trust_region_option.verbose = false;
        amm_option.robust_loss = sfm::Trivial;
        amm_option.pcg_option.max_iterations = 400;

        std::shared_ptr<sfm::ba::DBASubproblem<sfm::kGPU, T, false>> amm;
        amm = std::make_shared<sfm::ba::DBASubproblem<sfm::kGPU, T, false>>(
            amm_option, rank, num_ranks);
        amm->Setup(extrinsics_infos, intrinsics_infos, point_infos,
                   measurements, sqrt_weights);

        memory[solver][index] =
            std::max(memory[solver][index], amm->GetMemoryUsage());
        comm[solver][index] += amm->GetCommunicationLoad();
      }

      {
        int_t solver = 2;
        sfm::ba::DROption<T> dr_option;
        dr_option.trust_region_option.max_iterations = 1000;
        dr_option.trust_region_option.max_accepted_iterations = 20;
        dr_option.trust_region_option.verbose = false;
        dr_option.robust_loss = sfm::Trivial;
        dr_option.trust_region_option.relative_function_decrease_tolerance =
            1e-6;
        dr_option.pcg_option.max_iterations = 400;

        std::shared_ptr<sfm::ba::DouglasRachfordSubproblem<sfm::kGPU, T, false>>
            dr;
        dr = std::make_shared<
            sfm::ba::DouglasRachfordSubproblem<sfm::kGPU, T, false>>(
            dr_option, rank, num_ranks);
        dr->Setup(extrinsics_infos, intrinsics_infos, point_infos, measurements,
                  sqrt_weights);

        memory[solver][index] =
            std::max(memory[solver][index], dr->GetMemoryUsage());
        comm[solver][index] += dr->GetCommunicationLoad();
      }

      {
        int_t solver = 3;
        sfm::ba::ADMMOption<T> admm_option;
        admm_option.trust_region_option.max_iterations = 1000;
        admm_option.trust_region_option.max_accepted_iterations = 20;
        admm_option.trust_region_option.verbose = false;
        admm_option.robust_loss = sfm::Trivial;
        admm_option.trust_region_option.relative_function_decrease_tolerance =
            1e-6;
        admm_option.pcg_option.max_iterations = 400;

        std::shared_ptr<sfm::ba::ADMMSubproblem<sfm::kGPU, T, false>> admm;
        admm = std::make_shared<sfm::ba::ADMMSubproblem<sfm::kGPU, T, false>>(
            admm_option, rank, num_ranks);
        admm->Setup(extrinsics_infos, intrinsics_infos, point_infos,
                    measurements, sqrt_weights);

        memory[solver][index] =
            std::max(memory[solver][index], admm->GetMemoryUsage());
        comm[solver][index] += admm->GetCommunicationLoad();
      }
    }
  }

  std::string outfile = filename.substr(filename.rfind("problem-"));
  outfile = outfile.substr(0, outfile.find(".txt"));

  std::string mem_outfile = "mem-" + outfile + ".txt";
  std::string comm_outfile = "comm-" + outfile + ".txt";
  std::ofstream fout_mem(mem_outfile);
  std::ofstream fout_comm(comm_outfile);

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "Memory Usage" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  for (int_t solver = 0; solver < 4; solver++) {
    for (int n = 0; n < memory[solver].size(); n++) {
      std::cout << memory[solver][n] << " ";
      fout_mem << memory[solver][n] << " ";
    }
    std::cout << std::endl;
    fout_mem << std::endl;
  }
  std::cout << std::endl;

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "Communication Load" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  for (int_t solver = 0; solver < 4; solver++) {
    for (int n = 0; n < comm[solver].size(); n++) {
      std::cout << comm[solver][n] << " ";
      fout_comm << comm[solver][n] << " ";
    }
    std::cout << std::endl;
    fout_comm << std::endl;
  }

  return 0;
}