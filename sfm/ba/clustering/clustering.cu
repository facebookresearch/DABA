// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <sfm/ba/clustering/clustering.cuh>
#include <sfm/ba/clustering/common_methods.cuh>
#include <sfm/ba/clustering/merger.cuh>
#include <sfm/graph/converters/COOtoCSR.cuh>
#include <sfm/types.h>
#include <sfm/utils/utils.cuh>
#include <thrust/logical.h>

#define DEBUG 1

namespace sfm {
namespace ba {
namespace clustering {
template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Cluster(const sfm::graph::Handle &handle, Size number_of_cameras,
             Size number_of_points, const Vertex *src_indices,
             const Vertex *dst_indices, const Weight *weights,
             Size number_of_edges, Size targeted_number_of_clusters,
             Vertex *camera_clustering, Vertex *point_clustering,
             Float &modularity, Size &number_of_clusters,
             Float initial_resolution, Float refined_resolution,
             bool memory_efficient, Size max_iters) {
  bool verbose = false;

  Size number_of_vertices = number_of_cameras + number_of_points;
  thrust::device_vector<Score> tmp_vertex_scores_v(number_of_vertices, 0);
  Score max_cluster_score = 0;

  Size tmp_number_of_vertices = number_of_vertices;
  Size tmp_number_of_edges = number_of_edges;
  thrust::device_vector<Vertex> tmp_src_indices_v(tmp_number_of_edges);
  thrust::device_vector<Vertex> tmp_dst_indices_v(tmp_number_of_edges);
  thrust::device_vector<Weight> tmp_weights_v(tmp_number_of_edges, 1);
  thrust::device_vector<Size> tmp_src_offsets_v(number_of_vertices + 1, 0);

  thrust::copy(handle.GetThrustPolicy(), src_indices,
               src_indices + number_of_edges, tmp_src_indices_v.data());
  thrust::copy(handle.GetThrustPolicy(), dst_indices,
               dst_indices + number_of_edges, tmp_dst_indices_v.data());
  thrust::copy(handle.GetThrustPolicy(), weights, weights + number_of_edges,
               tmp_weights_v.data());

  sfm::graph::detail::COO2CSR(
      handle, tmp_number_of_vertices, tmp_src_indices_v.data().get(),
      tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
      tmp_number_of_edges, tmp_src_offsets_v.data().get());

  if (memory_efficient) {
    max_cluster_score =
        std::ceil(0.5 * number_of_edges *
                  (1.0 + std::min(0.15, 0.75 / targeted_number_of_clusters)) /
                  double(targeted_number_of_clusters));
    thrust::for_each_n(
        handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
        number_of_cameras,
        [vertex_scores = tmp_vertex_scores_v.data().get(),
         offsets = tmp_src_offsets_v.data().get()] __device__(int n) {
          vertex_scores[n] = offsets[n + 1] - offsets[n];
        });
  } else {
    max_cluster_score =
        std::ceil(number_of_cameras / double(targeted_number_of_clusters));
    thrust::fill_n(handle.GetThrustPolicy(), tmp_vertex_scores_v.data(),
                   number_of_cameras, 1);
  }

  thrust::device_vector<Score> tmp_cluster_scores_v;
  thrust::device_vector<Weight> tmp_cluster_self_weights_v;
  Size tmp_number_of_clusters;

  sfm::graph::Dendrogram<Vertex> dendrogram;
  dendrogram.AddLevel(0, tmp_number_of_vertices);

  sfm::graph::Louvain<sfm::graph::GraphCSRView<Vertex, Size, Weight>, Float,
                      Score>
      louvain(handle);

  number_of_clusters = tmp_number_of_vertices;
  modularity = 0;

  if (verbose) {
    std::cout << "-----------------------------------------------------"
              << std::endl;
  }

  EnhancedLouvain(handle, tmp_vertex_scores_v.data().get(),
                  tmp_number_of_vertices, tmp_src_offsets_v.data().get(),
                  tmp_src_indices_v.data().get(),
                  tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                  tmp_number_of_edges, targeted_number_of_clusters,
                  initial_resolution, max_cluster_score, modularity,
                  dendrogram.CurrentLevelBegin(), tmp_number_of_clusters,
                  tmp_cluster_scores_v, tmp_cluster_self_weights_v, verbose);

  ReclusterGraph(handle, targeted_number_of_clusters, dendrogram,
                 tmp_number_of_vertices, tmp_vertex_scores_v, tmp_src_offsets_v,
                 tmp_src_indices_v, tmp_dst_indices_v, tmp_weights_v,
                 tmp_number_of_edges);

  number_of_clusters = tmp_number_of_vertices;

  Float merged_resolution = std::min(0.25, 0.25 * refined_resolution);

  for (int_t iter = 0;
       iter < max_iters && number_of_clusters > targeted_number_of_clusters;
       iter++) {
    if (verbose) {
      std::cout << "-----------------------------------------------------"
                << std::endl;
    }

    EnhancedLouvain(handle, tmp_vertex_scores_v.data().get(),
                    tmp_number_of_vertices, tmp_src_offsets_v.data().get(),
                    tmp_src_indices_v.data().get(),
                    tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                    tmp_number_of_edges, targeted_number_of_clusters,
                    refined_resolution, max_cluster_score, modularity,
                    dendrogram.CurrentLevelBegin(), tmp_number_of_clusters,
                    tmp_cluster_scores_v, tmp_cluster_self_weights_v, verbose);

    number_of_clusters = tmp_number_of_clusters;

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    }

    // merge clusters greater to targeted_number_of_clusters
    Vertex selected_cluster_out = thrust::max_element(
        handle.GetThrustPolicy(),
        sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
        sfm::utils::MakeCountingIterator<Vertex>(number_of_clusters),
        [cluster_self_weights = tmp_cluster_self_weights_v.data().get(),
         cluster_scores =
             tmp_cluster_scores_v.data().get()] __device__(auto cluster1,
                                                           auto cluster2) {
          auto cluster_self_weight1 = cluster_self_weights[cluster1];
          auto cluster_self_weight2 = cluster_self_weights[cluster2];
          return cluster_self_weight1 != cluster_self_weight2
                     ? cluster_self_weight1 < cluster_self_weight2
                     : cluster_scores[cluster1] > cluster_scores[cluster2];
        })[0];

    bool mode = thrust::any_of(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
        sfm::utils::MakeCountingIterator(targeted_number_of_clusters),
        [cluster_scores = tmp_cluster_scores_v.data().get(),
         selected_cluster_out, max_cluster_score] __device__(auto cluster) {
          return cluster_scores[cluster] +
                     cluster_scores[selected_cluster_out] <=
                 max_cluster_score;
        });

    Vertex selected_cluster_in = -1;
    Score cluster_score_change = 0;

    if (mode == false) {
      thrust::device_vector<Vertex> selected_vertices_out_v(
          tmp_number_of_vertices);
      Size number_of_selected_vertices_out = 0;
      DecomposeAndMergeLargeCluster(
          handle, selected_cluster_out, dendrogram.CurrentLevelBegin(),
          tmp_vertex_scores_v.data().get(), tmp_number_of_vertices,
          tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
          tmp_weights_v.data().get(), tmp_number_of_edges,
          tmp_cluster_self_weights_v.data().get(),
          tmp_cluster_scores_v.data().get(), number_of_clusters,
          targeted_number_of_clusters, refined_resolution, max_cluster_score,
          selected_vertices_out_v.data().get(), selected_cluster_in,
          number_of_selected_vertices_out);
      selected_vertices_out_v.resize(number_of_selected_vertices_out);
      thrust::for_each(handle.GetThrustPolicy(),
                       selected_vertices_out_v.begin(),
                       selected_vertices_out_v.end(),
                       [clusters = dendrogram.CurrentLevelBegin(),
                        selected_cluster_in] __device__(auto vertex) {
                         clusters[vertex] = selected_cluster_in;
                       });

      cluster_score_change = thrust::transform_reduce(
          handle.GetThrustPolicy(), selected_vertices_out_v.data(),
          selected_vertices_out_v.data() + number_of_selected_vertices_out,
          [vertex_scores = tmp_vertex_scores_v.data().get()] __device__(
              auto vertex) { return vertex_scores[vertex]; },
          Score(0), cub::Sum());

      tmp_cluster_scores_v[selected_cluster_in] += cluster_score_change;
      tmp_cluster_scores_v[selected_cluster_out] -= cluster_score_change;

      UpdateClusterSelfWeights(
          handle, dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
          tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
          tmp_weights_v.data().get(), tmp_number_of_edges,
          tmp_cluster_self_weights_v.data().get(),
          {selected_cluster_in, selected_cluster_out}, tmp_number_of_clusters);
    } else {
      MergeLargeCluster(
          handle, selected_cluster_out, dendrogram.CurrentLevelBegin(),
          tmp_vertex_scores_v.data().get(), tmp_number_of_vertices,
          tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
          tmp_weights_v.data().get(), tmp_number_of_edges,
          tmp_cluster_scores_v.data().get(), tmp_number_of_clusters,
          targeted_number_of_clusters, max_cluster_score, selected_cluster_in);

      thrust::transform(
          handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
          dendrogram.CurrentLevelBegin() + tmp_number_of_vertices,
          dendrogram.CurrentLevelBegin(),
          [selected_cluster_out, selected_cluster_in] __device__(auto cluster) {
            return cluster == selected_cluster_out ? selected_cluster_in
                                                   : cluster;
          });

      cluster_score_change = tmp_cluster_scores_v[selected_cluster_out];

      tmp_cluster_scores_v[selected_cluster_in] += cluster_score_change;
      tmp_cluster_scores_v[selected_cluster_out] -= cluster_score_change;

      UpdateClusterSelfWeights(
          handle, dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
          tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
          tmp_weights_v.data().get(), tmp_number_of_edges,
          tmp_cluster_self_weights_v.data().get(),
          {selected_cluster_in, selected_cluster_out}, tmp_number_of_clusters);
    }

    ReclusterGraph(handle, targeted_number_of_clusters, dendrogram,
                   tmp_number_of_vertices, tmp_vertex_scores_v,
                   tmp_src_offsets_v, tmp_src_indices_v, tmp_dst_indices_v,
                   tmp_weights_v, tmp_number_of_edges);

    EnhancedLouvain(handle, tmp_vertex_scores_v.data().get(),
                    tmp_number_of_vertices, tmp_src_offsets_v.data().get(),
                    tmp_src_indices_v.data().get(),
                    tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                    tmp_number_of_edges, targeted_number_of_clusters,
                    refined_resolution, max_cluster_score, modularity,
                    dendrogram.CurrentLevelBegin(), tmp_number_of_clusters,
                    tmp_cluster_scores_v, tmp_cluster_self_weights_v, verbose);

    number_of_clusters = tmp_number_of_clusters;

    if (verbose) {
      std::cout << "number of clusters: " << tmp_number_of_clusters << std::endl
                << std::endl;
      std::cout << "cluster self-weights: ";
      for (const auto &weight : tmp_cluster_self_weights_v) {
        std::cout << weight << " ";
      }
      std::cout << std::endl << std::endl;

      std::cout << "cluster scores: ";
      for (const auto &score : tmp_cluster_scores_v) {
        std::cout << score << " ";
      }
      std::cout << std::endl << std::endl;
    }

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    }

    thrust::device_vector<Vertex> tmp_cluster_src_offsets_v;
    thrust::device_vector<Vertex> tmp_cluster_src_indices_v;
    thrust::device_vector<Vertex> tmp_cluster_dst_indices_v;
    thrust::device_vector<Weight> tmp_cluster_weights_v;
    Size tmp_number_of_cluster_edges;

    GetClusterInfo(handle, tmp_number_of_clusters,
                   dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
                   tmp_src_indices_v.data().get(),
                   tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                   tmp_number_of_edges, tmp_cluster_src_offsets_v,
                   tmp_cluster_src_indices_v, tmp_cluster_dst_indices_v,
                   tmp_cluster_weights_v, tmp_number_of_cluster_edges);

    thrust::device_vector<Vertex> merged_cluster_indices_v(
        tmp_number_of_clusters);
    thrust::device_vector<Score> merged_cluster_scores_v(
        tmp_number_of_clusters);

    MergeClusters(handle, Size(100), Float(0.0), targeted_number_of_clusters,
                  max_cluster_score, tmp_cluster_scores_v.data().get(),
                  tmp_cluster_src_offsets_v.data().get(),
                  tmp_cluster_dst_indices_v.data().get(),
                  tmp_cluster_weights_v.data().get(), tmp_number_of_clusters,
                  tmp_number_of_cluster_edges,
                  merged_cluster_indices_v.data().get(),
                  merged_cluster_scores_v.data().get(), tmp_number_of_clusters);

    thrust::transform(
        handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
        dendrogram.CurrentLevelEnd(), dendrogram.CurrentLevelBegin(),
        [merged_cluster_indices =
             merged_cluster_indices_v.data().get()] __device__(auto cluster) {
          return merged_cluster_indices[cluster];
        });

    tmp_cluster_scores_v.resize(tmp_number_of_clusters);
    tmp_cluster_self_weights_v.resize(tmp_number_of_clusters);
    thrust::copy(handle.GetThrustPolicy(), merged_cluster_scores_v.begin(),
                 merged_cluster_scores_v.end(), tmp_cluster_scores_v.begin());
    GetClusterSelfWeights(
        handle, dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
        tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
        tmp_weights_v.data().get(), tmp_number_of_edges,
        tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);

    number_of_clusters = tmp_number_of_clusters;

    SortClusters(
        handle, targeted_number_of_clusters, dendrogram.CurrentLevelBegin(),
        tmp_number_of_vertices, tmp_cluster_scores_v.data().get(),
        tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);

    if (verbose) {
      std::cout << "number of clusters: " << tmp_number_of_clusters << std::endl
                << std::endl;
      std::cout << "cluster self-weights: ";
      for (const auto &weight : tmp_cluster_self_weights_v) {
        std::cout << weight << " ";
      }
      std::cout << std::endl << std::endl;

      std::cout << "cluster scores: ";
      for (const auto &score : tmp_cluster_scores_v) {
        std::cout << score << " ";
      }
      std::cout << std::endl << std::endl;
    }

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    }

    GetClusterInfo(handle, tmp_number_of_clusters,
                   dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
                   tmp_src_indices_v.data().get(),
                   tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                   tmp_number_of_edges, tmp_cluster_src_offsets_v,
                   tmp_cluster_src_indices_v, tmp_cluster_dst_indices_v,
                   tmp_cluster_weights_v, tmp_number_of_cluster_edges);

    merged_cluster_indices_v.resize(tmp_number_of_clusters);

    if (MergeZeroClusterSelfWeightClusters(
            handle, tmp_cluster_scores_v.data().get(),
            tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters,
            tmp_cluster_src_offsets_v.data().get(),
            tmp_cluster_src_indices_v.data().get(),
            tmp_cluster_dst_indices_v.data().get(),
            tmp_cluster_weights_v.data().get(), tmp_number_of_cluster_edges,
            targeted_number_of_clusters, max_cluster_score,
            merged_cluster_indices_v.data().get(),
            tmp_cluster_scores_v.data().get(),
            tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters)) {
      thrust::transform(
          handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
          dendrogram.CurrentLevelBegin() + tmp_number_of_vertices,
          dendrogram.CurrentLevelBegin(),
          [merged_cluster_indices =
               merged_cluster_indices_v.data().get()] __device__(auto cluster) {
            return merged_cluster_indices[cluster];
          });

      number_of_clusters = tmp_number_of_clusters;

      tmp_cluster_scores_v.resize(tmp_number_of_clusters);
      tmp_cluster_self_weights_v.resize(tmp_number_of_clusters);
    }

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    }

    ReclusterGraph(handle, targeted_number_of_clusters, dendrogram,
                   tmp_number_of_vertices, tmp_vertex_scores_v,
                   tmp_src_offsets_v, tmp_src_indices_v, tmp_dst_indices_v,
                   tmp_weights_v, tmp_number_of_edges);

    EnhancedLouvain(handle, tmp_vertex_scores_v.data().get(),
                    tmp_number_of_vertices, tmp_src_offsets_v.data().get(),
                    tmp_src_indices_v.data().get(),
                    tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                    tmp_number_of_edges, targeted_number_of_clusters,
                    merged_resolution, max_cluster_score, modularity,
                    dendrogram.CurrentLevelBegin(), tmp_number_of_clusters,
                    tmp_cluster_scores_v, tmp_cluster_self_weights_v, verbose);

    merged_resolution *= 0.25;
    number_of_clusters = tmp_number_of_clusters;

    if (verbose) {
      std::cout << "number of clusters: " << tmp_number_of_clusters << std::endl
                << std::endl;
      std::cout << "cluster self-weights: ";
      for (const auto &weight : tmp_cluster_self_weights_v) {
        std::cout << weight << " ";
      }
      std::cout << std::endl << std::endl;

      std::cout << "cluster scores: ";
      for (const auto &score : tmp_cluster_scores_v) {
        std::cout << score << " ";
      }
      std::cout << std::endl << std::endl;
    }

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    } else {
      merged_cluster_indices_v.resize(tmp_number_of_clusters);
      merged_cluster_scores_v.resize(tmp_number_of_clusters);
      GetClusterInfo(handle, tmp_number_of_clusters,
                     dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
                     tmp_src_indices_v.data().get(),
                     tmp_dst_indices_v.data().get(), tmp_weights_v.data().get(),
                     tmp_number_of_edges, tmp_cluster_src_offsets_v,
                     tmp_cluster_src_indices_v, tmp_cluster_dst_indices_v,
                     tmp_cluster_weights_v, tmp_number_of_cluster_edges);

      MergeClusters(
          handle, Size(100), Float(0.0), targeted_number_of_clusters,
          max_cluster_score, tmp_cluster_scores_v.data().get(),
          tmp_cluster_src_offsets_v.data().get(),
          tmp_cluster_dst_indices_v.data().get(),
          tmp_cluster_weights_v.data().get(), tmp_number_of_clusters,
          tmp_number_of_cluster_edges, merged_cluster_indices_v.data().get(),
          merged_cluster_scores_v.data().get(), tmp_number_of_clusters);

      thrust::transform(
          handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
          dendrogram.CurrentLevelEnd(), dendrogram.CurrentLevelBegin(),
          [merged_cluster_indices =
               merged_cluster_indices_v.data().get()] __device__(auto cluster) {
            return merged_cluster_indices[cluster];
          });

      tmp_cluster_scores_v.resize(tmp_number_of_clusters);
      tmp_cluster_self_weights_v.resize(tmp_number_of_clusters);
      thrust::copy(handle.GetThrustPolicy(), merged_cluster_scores_v.begin(),
                   merged_cluster_scores_v.end(), tmp_cluster_scores_v.begin());
      GetClusterSelfWeights(
          handle, dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
          tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
          tmp_weights_v.data().get(), tmp_number_of_edges,
          tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);

      number_of_clusters = tmp_number_of_clusters;

      SortClusters(
          handle, targeted_number_of_clusters, dendrogram.CurrentLevelBegin(),
          tmp_number_of_vertices, tmp_cluster_scores_v.data().get(),
          tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);

      if (verbose) {
        std::cout << "number of clusters: " << tmp_number_of_clusters
                  << std::endl
                  << std::endl;
        std::cout << "cluster self-weights: ";
        for (const auto &weight : tmp_cluster_self_weights_v) {
          std::cout << weight << " ";
        }
        std::cout << std::endl << std::endl;

        std::cout << "cluster scores: ";
        for (const auto &score : tmp_cluster_scores_v) {
          std::cout << score << " ";
        }
        std::cout << std::endl << std::endl;
      }
    }

    // GetClusterInfo(handle, tmp_number_of_clusters,
    //                dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
    //                tmp_src_indices_v.data().get(),
    //                tmp_dst_indices_v.data().get(),
    //                tmp_weights_v.data().get(), tmp_number_of_edges,
    //                tmp_cluster_src_offsets_v, tmp_cluster_src_indices_v,
    //                tmp_cluster_dst_indices_v, tmp_cluster_weights_v,
    //                tmp_number_of_cluster_edges);

    if (number_of_clusters <= targeted_number_of_clusters) {
      break;
    } else {
      merged_cluster_indices_v.resize(tmp_number_of_clusters);
      selected_cluster_out = targeted_number_of_clusters;
      auto merged_score = tmp_cluster_scores_v[targeted_number_of_clusters] +
                          tmp_cluster_scores_v[targeted_number_of_clusters - 1];
      if (merged_score <= max_cluster_score) {
        merged_cluster_indices_v.resize(tmp_number_of_clusters);
        merged_cluster_scores_v.resize(tmp_number_of_clusters);

        MergeSortedLargeClusters(
            handle, targeted_number_of_clusters, max_cluster_score,
            tmp_cluster_scores_v.data().get(), tmp_number_of_clusters,
            merged_cluster_indices_v.data().get(),
            merged_cluster_scores_v.data().get(), tmp_number_of_clusters);

        merged_cluster_indices_v.resize(tmp_number_of_clusters);
        merged_cluster_scores_v.resize(tmp_number_of_clusters);
        if (verbose) {
          std::cout << "cluster scores: ";
          for (const auto &score : merged_cluster_scores_v) {
            std::cout << score << " ";
          }
          std::cout << std::endl << std::endl;
        }

        thrust::transform(
            handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
            dendrogram.CurrentLevelEnd(), dendrogram.CurrentLevelBegin(),
            [merged_cluster_indices = merged_cluster_indices_v.data()
                                          .get()] __device__(auto cluster) {
              return merged_cluster_indices[cluster];
            });

        tmp_cluster_scores_v.resize(tmp_number_of_clusters);
        tmp_cluster_self_weights_v.resize(tmp_number_of_clusters);
        thrust::copy(handle.GetThrustPolicy(), merged_cluster_scores_v.begin(),
                     merged_cluster_scores_v.end(),
                     tmp_cluster_scores_v.begin());
        GetClusterSelfWeights(
            handle, dendrogram.CurrentLevelBegin(), tmp_number_of_vertices,
            tmp_src_indices_v.data().get(), tmp_dst_indices_v.data().get(),
            tmp_weights_v.data().get(), tmp_number_of_edges,
            tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);

        number_of_clusters = tmp_number_of_clusters;

        SortClusters(
            handle, targeted_number_of_clusters, dendrogram.CurrentLevelBegin(),
            tmp_number_of_vertices, tmp_cluster_scores_v.data().get(),
            tmp_cluster_self_weights_v.data().get(), tmp_number_of_clusters);
      }
    }

    ReclusterGraph(handle, targeted_number_of_clusters, dendrogram,
                   tmp_number_of_vertices, tmp_vertex_scores_v,
                   tmp_src_offsets_v, tmp_src_indices_v, tmp_dst_indices_v,
                   tmp_weights_v, tmp_number_of_edges);

    number_of_clusters = tmp_number_of_vertices;

    if (verbose) {
      std::cout << "number of clusters: " << tmp_number_of_clusters << std::endl
                << std::endl;
      std::cout << "cluster self-weights: ";
      for (const auto &weight : tmp_cluster_self_weights_v) {
        std::cout << weight << " ";
      }
      std::cout << std::endl << std::endl;

      std::cout << "cluster scores: ";
      for (const auto &score : tmp_cluster_scores_v) {
        std::cout << score << " ";
      }
      std::cout << std::endl << std::endl;
    }
  }

  sfm::graph::detail::FlattenDendrogram(handle, dendrogram);

  thrust::copy(handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
               dendrogram.CurrentLevelBegin() + number_of_cameras,
               camera_clustering);
  thrust::copy(handle.GetThrustPolicy(),
               dendrogram.CurrentLevelBegin() + number_of_cameras,
               dendrogram.CurrentLevelBegin() + number_of_cameras +
                   number_of_points,
               point_clustering);
}

template void Cluster<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t num_cameras, int_t num_points,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t num_edges, int_t targeted_number_of_clusters,
    int_t *camera_clustering, int_t *point_clustering, float &modularity,
    int_t &number_of_clusters, float initial_resolution, float final_resolution,
    bool memory_efficient, int_t max_iters);

template void Cluster<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t num_cameras, int_t num_points,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t num_edges, int_t targeted_number_of_clusters,
    int_t *camera_clustering, int_t *point_clustering, double &modularity,
    int_t &number_of_clusters, double initial_resolution,
    double final_resolution, bool memory_efficient, int_t max_iters);
} // namespace clustering
} // namespace ba
} // namespace sfm