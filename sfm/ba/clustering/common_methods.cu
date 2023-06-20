// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <type_traits>

#include <thrust/logical.h>

#include <sfm/ba/clustering/common_methods.cuh>
#include <sfm/ba/clustering/merger.cuh>
#include <sfm/graph/converters/COOtoCSR.cuh>
#include <sfm/graph/types.hpp>
#include <sfm/types.h>
#include <sfm/utils/utils.cuh>

namespace sfm {
namespace ba {
namespace clustering {
template <typename Size, typename Vertex>
void SimplifyClusters(const sfm::graph::Handle &handle, Vertex *clusters,
                      Size number_of_vertices, Size &number_of_clusters) {
  thrust::device_vector<unsigned short> cluster_counts_v(number_of_clusters, 0);
  thrust::for_each(handle.GetThrustPolicy(), clusters,
                   clusters + number_of_vertices,
                   [cluster_counts = cluster_counts_v.data().get()] __device__(
                       auto cluster) {
                     atomicCAS(cluster_counts + cluster, (unsigned short)0,
                               (unsigned short)1);
                   });

  Size prev_number_of_clusters = number_of_clusters;
  thrust::device_vector<Vertex> selected_clusters_v(number_of_clusters);
  number_of_clusters =
      thrust::copy_if(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
          sfm::utils::MakeCountingIterator<Vertex>(prev_number_of_clusters),
          selected_clusters_v.data(),
          [cluster_counts = cluster_counts_v.data().get()] __device__(
              auto cluster) { return cluster_counts[cluster] != 0; }) -
      selected_clusters_v.data();
  selected_clusters_v.resize(number_of_clusters);

  thrust::device_vector<Vertex> simplified_cluster_indices_v(
      prev_number_of_clusters, -1);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(prev_number_of_clusters),
      [selected_clusters = selected_clusters_v.data().get(),
       simplified_cluster_indices =
           simplified_cluster_indices_v.data().get()] __device__(auto idx) {
        simplified_cluster_indices[selected_clusters[idx]] = idx;
      });

  thrust::device_vector<Vertex> prev_clusters_v(number_of_vertices, 0);
  thrust::copy(handle.GetThrustPolicy(), clusters,
               clusters + number_of_vertices, prev_clusters_v.begin());
  thrust::transform(
      handle.GetThrustPolicy(), prev_clusters_v.begin(), prev_clusters_v.end(),
      clusters,
      [simplified_cluster_indices =
           simplified_cluster_indices_v.data().get()] __device__(auto cluster) {
        return simplified_cluster_indices[cluster];
      });
}

template <typename Size, typename Vertex, typename Score>
void SimplifyClusters(const sfm::graph::Handle &handle, Vertex *clusters,
                      Size number_of_vertices, Score *cluster_scores,
                      Size &number_of_clusters) {
  thrust::device_vector<unsigned short> cluster_counts_v(number_of_clusters, 0);
  thrust::for_each(handle.GetThrustPolicy(), clusters,
                   clusters + number_of_vertices,
                   [cluster_counts = cluster_counts_v.data().get()] __device__(
                       auto cluster) {
                     atomicCAS(cluster_counts + cluster, (unsigned short)0,
                               (unsigned short)1);
                   });

  Size prev_number_of_clusters = number_of_clusters;
  thrust::device_vector<Vertex> selected_clusters_v(number_of_clusters);
  number_of_clusters =
      thrust::copy_if(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
          sfm::utils::MakeCountingIterator<Vertex>(prev_number_of_clusters),
          selected_clusters_v.data(),
          [cluster_counts = cluster_counts_v.data().get()] __device__(
              auto cluster) { return cluster_counts[cluster] != 0; }) -
      selected_clusters_v.data();
  selected_clusters_v.resize(number_of_clusters);

  thrust::device_vector<Vertex> simplified_cluster_indices_v(
      prev_number_of_clusters, -1);
  thrust::device_vector<Score> simplified_cluster_scores_v(number_of_clusters);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_clusters),
      [selected_clusters = selected_clusters_v.data().get(),
       simplified_cluster_indices = simplified_cluster_indices_v.data().get(),
       cluster_scores,
       simplified_cluster_scores =
           simplified_cluster_scores_v.data().get()] __device__(auto idx) {
        auto cluster = selected_clusters[idx];
        simplified_cluster_indices[cluster] = idx;
        simplified_cluster_scores[idx] = cluster_scores[cluster];
      });
  thrust::copy(handle.GetThrustPolicy(), simplified_cluster_scores_v.begin(),
               simplified_cluster_scores_v.end(), cluster_scores);

  thrust::device_vector<Vertex> prev_clusters_v(number_of_vertices, 0);
  thrust::copy(handle.GetThrustPolicy(), clusters,
               clusters + number_of_vertices, prev_clusters_v.begin());
  thrust::transform(
      handle.GetThrustPolicy(), prev_clusters_v.begin(), prev_clusters_v.end(),
      clusters,
      [simplified_cluster_indices =
           simplified_cluster_indices_v.data().get()] __device__(auto cluster) {
        return simplified_cluster_indices[cluster];
      });
}

template <typename Size, typename Vertex>
void SimplifyClusterIndices(const sfm::graph::Handle &handle,
                            const Vertex *cluster_indices,
                            Size number_of_clusters,
                            Vertex *simplified_cluster_indices,
                            Size &number_of_simplified_clusters) {
  thrust::copy(handle.GetThrustPolicy(), cluster_indices,
               cluster_indices + number_of_clusters,
               simplified_cluster_indices);
  thrust::device_vector<Vertex> selected_clusters_v(number_of_clusters);
  number_of_simplified_clusters =
      thrust::copy_if(handle.GetThrustPolicy(),
                      sfm::utils::MakeCountingIterator(0),
                      sfm::utils::MakeCountingIterator(number_of_clusters),
                      selected_clusters_v.data(),
                      [simplified_cluster_indices] __device__(auto cluster) {
                        return simplified_cluster_indices[cluster] == cluster;
                      }) -
      selected_clusters_v.data();
  selected_clusters_v.resize(number_of_simplified_clusters);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_simplified_clusters),
      [selected_clusters = selected_clusters_v.data().get(),
       simplified_cluster_indices] __device__(auto index) {
        simplified_cluster_indices[selected_clusters[index]] = index;
      });
}

template <typename Size, typename Vertex, typename Score>
void SimplifyClusterIndicesAndScores(const sfm::graph::Handle &handle,
                                     const Vertex *cluster_indices,
                                     const Score *cluster_scores,
                                     Size number_of_clusters,
                                     Vertex *simplified_cluster_indices,
                                     Score *simplified_cluster_scores,
                                     Size &number_of_simplified_clusters) {
  thrust::copy(handle.GetThrustPolicy(), cluster_indices,
               cluster_indices + number_of_clusters,
               simplified_cluster_indices);
  thrust::device_vector<Vertex> unmerged_clusters_v(number_of_clusters);
  number_of_simplified_clusters =
      thrust::copy_if(handle.GetThrustPolicy(),
                      sfm::utils::MakeCountingIterator(0),
                      sfm::utils::MakeCountingIterator(number_of_clusters),
                      unmerged_clusters_v.data(),
                      [simplified_cluster_indices] __device__(auto cluster) {
                        return simplified_cluster_indices[cluster] == cluster;
                      }) -
      unmerged_clusters_v.data();
  unmerged_clusters_v.resize(number_of_simplified_clusters);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_simplified_clusters),
      [unmerged_clusters = unmerged_clusters_v.data().get(),
       simplified_cluster_indices, simplified_cluster_scores,
       cluster_scores] __device__(auto index) {
        Vertex cluster = unmerged_clusters[index];
        simplified_cluster_indices[cluster] = index;
        simplified_cluster_scores[index] = cluster_scores[cluster];
      });
}

template <typename Size, typename Vertex, typename Weight, typename Score>
void SortClusters(const sfm::graph::Handle &handle,
                  Size targeted_number_of_clusters, Vertex *clusters,
                  Size number_of_vertices, Score *cluster_scores,
                  Weight *cluster_self_weights, Size number_of_clusters) {
  thrust::device_vector<Vertex> sorted_cluster_indices(number_of_clusters, 0);
  thrust::device_vector<Vertex> sorted_cluster_indices_inverse(
      number_of_clusters, 0);
  auto policy = handle.GetThrustPolicy();
  thrust::sequence(policy, sorted_cluster_indices_inverse.begin(),
                   sorted_cluster_indices_inverse.end(), 0);

  thrust::stable_sort_by_key(
      policy, cluster_self_weights, cluster_self_weights + number_of_clusters,
      sfm::utils::MakeZipIterator(sorted_cluster_indices_inverse.begin(),
                                  cluster_scores),
      thrust::greater<Weight>());

  thrust::stable_sort_by_key(
      policy, cluster_scores, cluster_scores + targeted_number_of_clusters,
      sfm::utils::MakeZipIterator(sorted_cluster_indices_inverse.begin(),
                                  cluster_self_weights),
      thrust::greater<Score>());

  thrust::for_each(
      policy, sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_clusters),
      [cluster_indices = sorted_cluster_indices.data().get(),
       cluster_indices_inverse =
           sorted_cluster_indices_inverse.data().get()] __device__(Vertex id) {
        cluster_indices[cluster_indices_inverse[id]] = id;
      });

  thrust::for_each(
      policy, sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_vertices),
      [clusters,
       cluster_indices =
           sorted_cluster_indices.data().get()] __device__(Vertex id) {
        clusters[id] = cluster_indices[clusters[id]];
      });
}

template <typename Size, typename Vertex, typename Weight, typename Score>
void MergeZeroScoreVertices(const sfm::graph::Handle &handle, Vertex *clusters,
                            const Score *vertex_scores, Size number_of_vertices,
                            const Vertex *src_indices,
                            const Vertex *dst_indices, const Weight *weights,
                            Size number_of_edges) {
  thrust::device_vector<Vertex> selected_src_indices_v(number_of_edges);
  thrust::device_vector<Vertex> selected_dst_indices_v(number_of_edges);
  thrust::device_vector<Weight> selected_weights_v(number_of_edges);

  auto edge_begin =
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights);

  auto selected_edge_begin = sfm::utils::MakeZipIterator(
      selected_src_indices_v.data().get(), selected_dst_indices_v.data().get(),
      selected_weights_v.data().get());

  Size number_of_selected_edges =
      thrust::copy_if(handle.GetThrustPolicy(), edge_begin,
                      edge_begin + number_of_edges, selected_edge_begin,
                      [vertex_scores] __device__(auto edge) {
                        auto src = thrust::get<0>(edge);
                        auto dst = thrust::get<1>(edge);

                        return vertex_scores[src] == 0 &&
                               vertex_scores[dst] > 0;
                      }) -
      selected_edge_begin;

  if (number_of_selected_edges == 0) {
    return;
  }

  selected_src_indices_v.resize(number_of_selected_edges);
  selected_dst_indices_v.resize(number_of_selected_edges);
  selected_weights_v.resize(number_of_selected_edges);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_selected_edges),
      [selected_dst_indices = selected_dst_indices_v.data().get(),
       clusters] __device__(auto idx) {
        selected_dst_indices[idx] = clusters[selected_dst_indices[idx]];
      });

  thrust::sort_by_key(
      handle.GetThrustPolicy(), selected_dst_indices_v.begin(),
      selected_dst_indices_v.end(),
      sfm::utils::MakeZipIterator(selected_src_indices_v.begin(),
                                  selected_weights_v.begin()));

  thrust::sort_by_key(
      handle.GetThrustPolicy(), selected_src_indices_v.begin(),
      selected_src_indices_v.end(),
      sfm::utils::MakeZipIterator(selected_dst_indices_v.begin(),
                                  selected_weights_v.begin()));

  thrust::device_vector<Vertex> selected_reduced_src_indices_v(
      number_of_selected_edges);
  thrust::device_vector<Vertex> selected_reduced_dst_indices_v(
      number_of_selected_edges);
  thrust::device_vector<Weight> selected_reduced_weights_v(
      number_of_selected_edges);

  auto selected_src_dst_begin = sfm::utils::MakeZipIterator(
      selected_src_indices_v.begin(), selected_dst_indices_v.begin());
  auto selected_reduced_src_dst_begin =
      sfm::utils::MakeZipIterator(selected_reduced_src_indices_v.begin(),
                                  selected_reduced_dst_indices_v.begin());
  Size number_of_selected_reduced_edges =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(), selected_src_dst_begin,
          selected_src_dst_begin + number_of_selected_edges,
          selected_weights_v.begin(), selected_reduced_src_dst_begin,
          selected_reduced_weights_v.begin(),
          thrust::equal_to<thrust::tuple<Vertex, Vertex>>(), cub::Sum())
          .second -
      selected_reduced_weights_v.begin();

  selected_reduced_src_indices_v.resize(number_of_selected_reduced_edges);
  selected_reduced_dst_indices_v.resize(number_of_selected_reduced_edges);
  selected_reduced_weights_v.resize(number_of_selected_reduced_edges);

  thrust::device_vector<Vertex> selected_vertices_v(
      number_of_selected_reduced_edges);
  thrust::device_vector<Vertex> selected_cluster_indices_v(
      number_of_selected_reduced_edges);
  thrust::device_vector<Vertex> selected_cluster_weights_v(
      number_of_selected_reduced_edges);

  Size number_of_selected_vertices =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(), selected_reduced_src_indices_v.begin(),
          selected_reduced_src_indices_v.begin() +
              number_of_selected_reduced_edges,
          sfm::utils::MakeZipIterator(selected_reduced_dst_indices_v.begin(),
                                      selected_reduced_weights_v.begin()),
          selected_vertices_v.begin(),
          sfm::utils::MakeZipIterator(selected_cluster_indices_v.begin(),
                                      selected_cluster_weights_v.begin()),
          thrust::equal_to<Vertex>(),
          [] __device__(auto pair1, auto pair2) {
            auto cluster1 = thrust::get<0>(pair1);
            auto cluster2 = thrust::get<0>(pair2);
            auto weight1 = thrust::get<1>(pair1);
            auto weight2 = thrust::get<1>(pair2);

            if (weight1 > weight2) {
              return pair1;
            } else if (weight1 == weight2 && cluster1 > cluster2) {
              return pair1;
            } else {
              return pair2;
            }
          })
          .first -
      selected_vertices_v.begin();

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_selected_vertices),
      [clusters, selected_vertices = selected_vertices_v.data().get(),
       selected_cluster_indices =
           selected_cluster_indices_v.data().get()] __device__(auto idx) {
        auto vertex = selected_vertices[idx];
        clusters[vertex] = selected_cluster_indices[idx];
      });
}

template <typename Size, typename Vertex, typename Weight>
void GetSelfWeights(const sfm::graph::Handle &handle, const Vertex *src_indices,
                    const Vertex *dst_indices, const Weight *weights,
                    Size number_of_edges, Weight *self_weights,
                    Size number_of_vertices) {
  cudaMemset(self_weights, 0, sizeof(Weight) * number_of_vertices);
  thrust::for_each_n(
      handle.GetThrustPolicy(),
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights),
      number_of_edges, [self_weights] __device__(auto input) {
        auto src = thrust::get<0>(input);
        auto dst = thrust::get<1>(input);
        auto weight = thrust::get<2>(input);
        if (src == dst) {
          atomicAdd(self_weights + src, weight);
        }
      });
}

template <typename Size, typename Vertex, typename Weight>
void GetClusterSelfWeights(const sfm::graph::Handle &handle,
                           const Vertex *clusters, Size number_of_vertices,
                           const Vertex *src_indices, const Vertex *dst_indices,
                           const Weight *weights, Size number_of_edges,
                           Weight *cluster_self_weights,
                           Size number_of_clusters) {
  cudaMemset(cluster_self_weights, 0, sizeof(Weight) * number_of_clusters);
  thrust::for_each_n(
      handle.GetThrustPolicy(),
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights),
      number_of_edges, [clusters, cluster_self_weights] __device__(auto input) {
        auto cluster_src = clusters[thrust::get<0>(input)];
        auto cluster_dst = clusters[thrust::get<1>(input)];
        auto weight = thrust::get<2>(input);
        if (cluster_src == cluster_dst) {
          atomicAdd(cluster_self_weights + cluster_src, weight);
        }
      });
}

template <typename Size, typename Vertex, typename Weight>
void UpdateClusterSelfWeights(const sfm::graph::Handle &handle,
                              const Vertex *clusters, Size number_of_vertices,
                              const Vertex *src_indices,
                              const Vertex *dst_indices, const Weight *weights,
                              Size number_of_edges,
                              Weight *cluster_self_weights,
                              std::array<Vertex, 2> updated_clusters,
                              Size number_of_clusters) {
  cudaMemset(cluster_self_weights + updated_clusters[0], 0, sizeof(Weight));
  cudaMemset(cluster_self_weights + updated_clusters[1], 0, sizeof(Weight));
  thrust::for_each_n(
      handle.GetThrustPolicy(),
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights),
      number_of_edges,
      [clusters, cluster_self_weights, cluster1 = updated_clusters[0],
       cluster2 = updated_clusters[1]] __device__(auto input) {
        auto cluster_src = clusters[thrust::get<0>(input)];
        auto cluster_dst = clusters[thrust::get<1>(input)];
        auto weight = thrust::get<2>(input);
        if (cluster_src == cluster_dst &&
            (cluster_src == cluster1 || cluster_src == cluster2)) {
          atomicAdd(cluster_self_weights + cluster_src, weight);
        }
      });
}

template <typename Size, typename Vertex, typename Weight>
void GetClusterInfo(const sfm::graph::Handle &handle, Size number_of_clusters,
                    const Vertex *clusters, Size number_of_vertices,
                    const Vertex *src_indices, const Vertex *dst_indices,
                    const Weight *weights, Size number_of_edges,
                    thrust::device_vector<Vertex> &cluster_src_offsets_v,
                    thrust::device_vector<Vertex> &cluster_src_indices_v,
                    thrust::device_vector<Vertex> &cluster_dst_indices_v,
                    thrust::device_vector<Weight> &cluster_weights_v,
                    Size &number_of_cluster_edges) {
  thrust::device_vector<Vertex> tmp_cluster_src_indices_v(number_of_edges);
  thrust::device_vector<Vertex> tmp_cluster_dst_indices_v(number_of_edges);
  thrust::device_vector<Weight> tmp_cluster_weights_v(number_of_edges);

  thrust::transform(handle.GetThrustPolicy(), src_indices,
                    src_indices + number_of_edges,
                    tmp_cluster_src_indices_v.begin(),
                    [clusters] __device__(auto src) { return clusters[src]; });
  thrust::transform(handle.GetThrustPolicy(), dst_indices,
                    dst_indices + number_of_edges,
                    tmp_cluster_dst_indices_v.begin(),
                    [clusters] __device__(auto dst) { return clusters[dst]; });
  thrust::copy(handle.GetThrustPolicy(), weights, weights + number_of_edges,
               tmp_cluster_weights_v.begin());

  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), tmp_cluster_dst_indices_v.begin(),
      tmp_cluster_dst_indices_v.end(),
      sfm::utils::MakeZipIterator(tmp_cluster_src_indices_v.begin(),
                                  tmp_cluster_weights_v.begin()));
  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), tmp_cluster_src_indices_v.begin(),
      tmp_cluster_src_indices_v.end(),
      sfm::utils::MakeZipIterator(tmp_cluster_dst_indices_v.begin(),
                                  tmp_cluster_weights_v.begin()));

  auto tmp_cluster_edge_begin = sfm::utils::MakeZipIterator(
      tmp_cluster_src_indices_v.begin(), tmp_cluster_dst_indices_v.begin());
  auto tmp_cluster_edge_end = tmp_cluster_edge_begin + number_of_edges;

  cluster_src_indices_v.resize(number_of_edges);
  cluster_dst_indices_v.resize(number_of_edges);
  cluster_weights_v.resize(number_of_edges);
  number_of_cluster_edges =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(), tmp_cluster_edge_begin,
          tmp_cluster_edge_end, tmp_cluster_weights_v.begin(),
          sfm::utils::MakeZipIterator(cluster_src_indices_v.begin(),
                                      cluster_dst_indices_v.begin()),
          cluster_weights_v.begin(),
          thrust::equal_to<thrust::tuple<Vertex, Vertex>>(), cub::Sum())
          .second -
      cluster_weights_v.begin();
  cluster_src_indices_v.resize(number_of_cluster_edges);
  cluster_dst_indices_v.resize(number_of_cluster_edges);
  cluster_weights_v.resize(number_of_cluster_edges);
  cluster_src_indices_v.shrink_to_fit();
  cluster_dst_indices_v.shrink_to_fit();
  cluster_weights_v.shrink_to_fit();

  tmp_cluster_src_indices_v.clear();
  tmp_cluster_dst_indices_v.clear();
  tmp_cluster_weights_v.clear();
  tmp_cluster_src_indices_v.shrink_to_fit();
  tmp_cluster_dst_indices_v.shrink_to_fit();
  tmp_cluster_weights_v.shrink_to_fit();

  cluster_src_offsets_v.resize(number_of_clusters + 1);
  cluster_src_offsets_v.shrink_to_fit();
  sfm::graph::detail::FillOffset(
      cluster_src_indices_v.data().get(), cluster_src_offsets_v.data().get(),
      number_of_clusters, number_of_cluster_edges, handle.GetStream());
}

template <typename Size, typename Vertex, typename Weight, typename Score>
void ReclusterGraph(const sfm::graph::Handle &handle,
                    Size targeted_number_of_clusters,
                    sfm::graph::Dendrogram<Vertex> &dendrogram,
                    Size &number_of_vertices,
                    thrust::device_vector<Score> &vertex_scores_v,
                    thrust::device_vector<Vertex> &src_offsets_v,
                    thrust::device_vector<Vertex> &src_indices_v,
                    thrust::device_vector<Vertex> &dst_indices_v,
                    thrust::device_vector<Weight> &weights_v,
                    Size &number_of_edges) {
  number_of_vertices = dendrogram.CurrentLevelSize();
  number_of_edges = src_indices_v.size();

  if (dst_indices_v.size() != number_of_edges ||
      weights_v.size() != number_of_edges) {
    LOG(ERROR) << "Inconsistent edge info." << std::endl;
    exit(-1);
  }

  auto prev_clusters = dendrogram.CurrentLevelBegin();
  thrust::device_vector<Vertex> recluster_vertices_v(number_of_vertices);

  auto reclustered_vertices_begin = recluster_vertices_v.data().get();
  auto reclustered_vertices_end = thrust::copy_if(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_vertices),
      reclustered_vertices_begin,
      [prev_clusters, targeted_number_of_clusters] __device__(auto vertex) {
        return prev_clusters[vertex] >= targeted_number_of_clusters;
      });
  auto number_of_recluster_vertices =
      reclustered_vertices_end - reclustered_vertices_begin;
  recluster_vertices_v.resize(number_of_recluster_vertices);

  Size number_of_clusters =
      number_of_recluster_vertices + targeted_number_of_clusters;

  dendrogram.AddLevel(0, number_of_clusters);
  thrust::sequence(handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
                   dendrogram.CurrentLevelBegin() + targeted_number_of_clusters,
                   Vertex(0));

  thrust::for_each_n(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
      number_of_recluster_vertices,
      [prev_clusters, recluster_vertices = recluster_vertices_v.data().get(),
       curr_clusters =
           dendrogram.CurrentLevelBegin() + targeted_number_of_clusters,
       targeted_number_of_clusters] __device__(Vertex id) {
        Vertex vertex = recluster_vertices[id];
        Vertex cluster = prev_clusters[vertex];
        curr_clusters[id] = cluster;
        prev_clusters[vertex] = id + targeted_number_of_clusters;
      });

  thrust::device_vector<Score> cluster_scores_v(number_of_clusters);
  thrust::device_vector<Vertex> tmp_arr_v(number_of_vertices);
  sfm::graph::detail::GenerateSuperverticesGraph(
      handle, prev_clusters, number_of_vertices, number_of_edges, src_offsets_v,
      src_indices_v, dst_indices_v, weights_v, vertex_scores_v,
      cluster_scores_v, tmp_arr_v);

  if (number_of_clusters != number_of_vertices) {
    LOG(ERROR) << "Incosistent number of clusuers and number of vertices."
               << std::endl;
    exit(-1);
  }
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void MergeClusters(const sfm::graph::Handle &handle, Size max_level,
                   Float resolution, Size targeted_number_of_clusters,
                   Score max_cluster_score, const Score *cluster_scores,
                   const Vertex *cluster_src_offsets,
                   const Vertex *cluster_dst_indices,
                   const Weight *cluster_weights, Size number_of_clusters,
                   Size number_of_cluster_edges, Vertex *merged_cluster_indices,
                   Score *merged_cluster_scores,
                   Size &number_of_merged_clusters) {
  Merger<float, int_t, int_t, uint64_t, uint64_t> merger(handle);
  merger.Merge(cluster_src_offsets, cluster_dst_indices, cluster_weights,
               number_of_clusters, number_of_cluster_edges, max_level,
               resolution, merged_cluster_indices, cluster_scores,
               targeted_number_of_clusters, max_cluster_score, false);
  number_of_merged_clusters = merger.GetNumberOfClusters();
  thrust::copy(handle.GetThrustPolicy(), merger.GetClusterScores().begin(),
               merger.GetClusterScores().end(), merged_cluster_scores);
}

#if 0
template <typename Size, typename Vertex, typename Weight, typename Score>
void MergeSortedLargeClusters(
    const sfm::graph::Handle &handle, Size targeted_number_of_clusters,
    Score max_cluster_score, const Score *cluster_scores,
    Size number_of_clusters, const Vertex *cluster_src_indices,
    const Vertex *cluster_dst_indices, const Weight *cluster_weights,
    Size number_of_cluster_edges, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Size &number_of_merged_clusters) {
  thrust::sequence(handle.GetThrustPolicy(), merged_cluster_indices,
                   merged_cluster_indices + number_of_clusters, 0);

  if (number_of_clusters <= targeted_number_of_clusters) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  thrust::device_vector<Vertex> targeted_cluster_src_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Vertex> targeted_cluster_dst_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Weight> targeted_cluster_weights_v(
      number_of_cluster_edges);

  auto cluster_edge_begin = sfm::utils::MakeZipIterator(
      cluster_src_indices, cluster_dst_indices, cluster_weights);
  auto cluster_edge_end = cluster_edge_begin + number_of_cluster_edges;
  auto targeted_cluster_edge_begin =
      sfm::utils::MakeZipIterator(targeted_cluster_src_indices_v.begin(),
                                  targeted_cluster_dst_indices_v.begin(),
                                  targeted_cluster_weights_v.begin());

  Size number_of_targeted_clusters_in =
      thrust::copy_if(handle.GetThrustPolicy(), cluster_edge_begin,
                      cluster_edge_end, targeted_cluster_edge_begin,
                      [cluster_scores, targeted_number_of_clusters,
                       max_cluster_score] __device__(auto edge) {
                        auto src = thrust::get<0>(edge);
                        auto dst = thrust::get<1>(edge);

                        return src == targeted_number_of_clusters &&
                               dst < targeted_number_of_clusters &&
                               cluster_scores[src] + cluster_scores[dst] <=
                                   max_cluster_score;
                      }) -
      targeted_cluster_edge_begin;

  targeted_cluster_src_indices_v.resize(number_of_targeted_clusters_in);
  targeted_cluster_dst_indices_v.resize(number_of_targeted_clusters_in);
  targeted_cluster_weights_v.resize(number_of_targeted_clusters_in);

  Vertex cluster_in = -1;
  if (number_of_targeted_clusters_in != 0) {
    cluster_in = targeted_cluster_dst_indices_v[thrust::reduce(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
        sfm::utils::MakeCountingIterator<Vertex>(
            number_of_targeted_clusters_in),
        Vertex(0),
        [targeted_clusters_in = targeted_cluster_dst_indices_v.data().get(),
         targeted_cluster_weights = targeted_cluster_weights_v.data().get(),
         cluster_scores] __device__(auto index1, auto index2) {
          Weight weight1 = targeted_cluster_weights[index1];
          Weight weight2 = targeted_cluster_weights[index2];

          if (weight1 != weight2) {
            return weight1 > weight2 ? index1 : index2;
          } else {
            Vertex cluster1 = targeted_clusters_in[index1];
            Vertex cluster2 = targeted_clusters_in[index2];
            Score score1 = cluster_scores[cluster1];
            Score score2 = cluster_scores[cluster2];

            if (score1 > score2) {
              return index1;
            } else if (score1 == score2 && cluster1 < cluster2) {
              return index1;
            } else {
              return index2;
            }
          }
        })];
  } else {
    cluster_in = thrust::find_if(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
        sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
        [cluster_scores, targeted_number_of_clusters,
         max_cluster_score] __device__(auto cluster) {
          return cluster_scores[cluster] +
                     cluster_scores[targeted_number_of_clusters] <=
                 max_cluster_score;
        })[0];
  }

  if (cluster_in == targeted_number_of_clusters) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  thrust::device_vector<Vertex> selected_cluster_src_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Vertex> selected_cluster_dst_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Weight> selected_cluster_weights_v(
      number_of_cluster_edges);
  auto selected_cluster_edge_begin =
      sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.begin(),
                                  selected_cluster_dst_indices_v.begin(),
                                  selected_cluster_weights_v.begin());
  Size number_of_selected_cluster_edges =
      thrust::copy_if(handle.GetThrustPolicy(), cluster_edge_begin,
                      cluster_edge_end, selected_cluster_edge_begin,
                      [cluster_scores, targeted_number_of_clusters,
                       max_cluster_score] __device__(auto edge) {
                        auto src = thrust::get<0>(edge);
                        auto dst = thrust::get<1>(edge);

                        return src > targeted_number_of_clusters &&
                               dst <= targeted_number_of_clusters &&
                               cluster_scores[src] + cluster_scores[dst] <=
                                   max_cluster_score;
                      }) -
      selected_cluster_edge_begin;
  selected_cluster_src_indices_v.resize(number_of_selected_cluster_edges);
  selected_cluster_dst_indices_v.resize(number_of_selected_cluster_edges);
  selected_cluster_weights_v.resize(number_of_selected_cluster_edges);

  thrust::transform(
      handle.GetThrustPolicy(), selected_cluster_dst_indices_v.begin(),
      selected_cluster_dst_indices_v.end(),
      selected_cluster_dst_indices_v.begin(),
      [targeted_number_of_clusters, cluster_in] __device__(auto cluster) {
        return cluster == targeted_number_of_clusters ? cluster_in : cluster;
      });

  thrust::device_pointer_cast(
      merged_cluster_indices)[targeted_number_of_clusters] = cluster_in;

  thrust::fill(handle.GetThrustPolicy(),
               merged_cluster_indices + targeted_number_of_clusters,
               merged_cluster_indices + number_of_clusters, cluster_in);

  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), selected_cluster_dst_indices_v.begin(),
      selected_cluster_dst_indices_v.end(),
      sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.begin(),
                                  selected_cluster_weights_v.begin()));

  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), selected_cluster_src_indices_v.begin(),
      selected_cluster_src_indices_v.end(),
      sfm::utils::MakeZipIterator(selected_cluster_dst_indices_v.begin(),
                                  selected_cluster_weights_v.begin()));

  number_of_targeted_clusters_in =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(),
          sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.begin(),
                                      selected_cluster_dst_indices_v.begin()),
          sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.end(),
                                      selected_cluster_dst_indices_v.end()),
          selected_cluster_weights_v.begin(),
          sfm::utils::MakeZipIterator(targeted_cluster_src_indices_v.begin(),
                                      targeted_cluster_dst_indices_v.begin()),
          targeted_cluster_weights_v.begin())
          .second -
      targeted_cluster_weights_v.begin();
  targeted_cluster_src_indices_v.resize(number_of_targeted_clusters_in);
  targeted_cluster_dst_indices_v.resize(number_of_targeted_clusters_in);
  targeted_cluster_weights_v.resize(number_of_targeted_clusters_in);

  thrust::device_vector<Vertex> selected_merged_clusters_out_v(
      number_of_targeted_clusters_in);
  thrust::device_vector<Vertex> selected_merged_clusters_in_v(
      number_of_targeted_clusters_in);
  thrust::device_vector<Weight> selected_merged_cluster_weights_v(
      number_of_targeted_clusters_in);

  Size number_of_selected_merged_clusters =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(), targeted_cluster_src_indices_v.begin(),
          targeted_cluster_src_indices_v.end(),
          sfm::utils::MakeZipIterator(targeted_cluster_dst_indices_v.begin(),
                                      targeted_cluster_weights_v.begin()),
          selected_merged_clusters_out_v.begin(),
          sfm::utils::MakeZipIterator(
              selected_merged_clusters_in_v.begin(),
              selected_merged_cluster_weights_v.begin()),
          thrust::equal_to<Vertex>(),
          [cluster_in] __device__(auto pair1, auto pair2) {
            auto weight1 = thrust::get<1>(pair1);
            auto weight2 = thrust::get<1>(pair2);
            auto cluster1 = thrust::get<0>(pair1);
            auto cluster2 = thrust::get<0>(pair2);

            if (weight1 > weight2) {
              return pair1;
            } else if (weight1 == weight2 &&
                       (cluster1 == cluster_in || cluster1 < cluster2)) {

              return pair1;
            } else {
              return pair2;
            }
          })
          .firsr -
      selected_merged_clusters_out_v.begin();
  selected_merged_clusters_out_v.resize(number_of_selected_merged_clusters);
  selected_merged_clusters_in_v.resize(number_of_selected_merged_clusters);
  selected_merged_cluster_weights_v.resize(number_of_selected_merged_clusters);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(
          number_of_selected_merged_clusters),
      [merged_cluster_indices, cluster_in,
       selected_merged_clusters_in = selected_merged_clusters_in_v.data().get(),
       selected_merged_clusters_out =
           selected_merged_clusters_out_v.data().get()] __device__(auto index) {
        auto merged_cluster_in = selected_merged_clusters_in[index];
        auto merged_cluster_out = selected_merged_clusters_out[index];
        merged_cluster_indices[merged_cluster_out] =
            merged_cluster_in != cluster_in ? merged_cluster_out : cluster_in;
      });
}
#endif

template <typename Size, typename Vertex, typename Score>
void MergeSortedLargeClusters(
    const sfm::graph::Handle &handle, Size targeted_number_of_clusters,
    Score max_cluster_score, const Score *cluster_scores,
    Size number_of_clusters, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Size &number_of_merged_clusters) {
  thrust::sequence(handle.GetThrustPolicy(), merged_cluster_indices,
                   merged_cluster_indices + number_of_clusters, 0);

  if (number_of_clusters <= targeted_number_of_clusters) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  thrust::device_vector<Vertex> selected_clusters_out_v(
      number_of_clusters - targeted_number_of_clusters);
  Size number_of_selected_clusters_out =
      number_of_clusters - targeted_number_of_clusters;
  thrust::sequence(handle.GetThrustPolicy(), selected_clusters_out_v.begin(),
                   selected_clusters_out_v.end(), targeted_number_of_clusters);
  thrust::device_vector<Score> cluster_out_scores_scan_v(
      number_of_selected_clusters_out);
  thrust::inclusive_scan(
      handle.GetThrustPolicy(), cluster_scores + targeted_number_of_clusters,
      cluster_scores + number_of_clusters, cluster_out_scores_scan_v.data());

  Vertex cluster_in = thrust::find_if(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
      [cluster_scores, targeted_number_of_clusters,
       max_cluster_score] __device__(auto cluster) {
        return cluster_scores[cluster] +
                   cluster_scores[targeted_number_of_clusters] <=
               max_cluster_score;
      })[0];

  if (cluster_in == targeted_number_of_clusters) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  Size number_of_clusters_out =
      number_of_selected_clusters_out -
      (thrust::find_if(
           handle.GetThrustPolicy(), cluster_out_scores_scan_v.rbegin(),
           cluster_out_scores_scan_v.rend(),
           [cluster_scores, cluster_in,
            max_cluster_score] __device__(auto cluster_score_scan) {
             return cluster_score_scan + cluster_scores[cluster_in] <=
                    max_cluster_score;
           }) -
       cluster_out_scores_scan_v.rbegin());

  thrust::device_vector<Score> tmp_cluster_scores_v(number_of_clusters);
  thrust::copy(handle.GetThrustPolicy(), cluster_scores,
               cluster_scores + number_of_clusters,
               tmp_cluster_scores_v.begin());
  thrust::for_each(
      handle.GetThrustPolicy(),
      sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
      sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters +
                                               number_of_clusters_out),
      [tmp_cluster_scores = tmp_cluster_scores_v.data().get(),
       merged_cluster_indices, cluster_in] __device__(auto cluster) {
        merged_cluster_indices[cluster] = cluster_in;
        tmp_cluster_scores[cluster] = 0;
      });
  tmp_cluster_scores_v[cluster_in] +=
      cluster_out_scores_scan_v[number_of_clusters_out - 1];

  SimplifyClusterIndicesAndScores(
      handle, merged_cluster_indices, tmp_cluster_scores_v.data().get(),
      number_of_clusters, merged_cluster_indices, merged_cluster_scores,
      number_of_merged_clusters);
}

template <typename Size, typename Vertex, typename Score>
void MergeSortedSmallClusters(
    const sfm::graph::Handle &handle, Size targeted_number_of_clusters,
    Score max_cluster_score, const Score *cluster_scores,
    Size number_of_clusters, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Size &number_of_merged_clusters) {
  thrust::sequence(handle.GetThrustPolicy(), merged_cluster_indices,
                   merged_cluster_indices + number_of_clusters, 0);

  if (number_of_clusters <= targeted_number_of_clusters) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  thrust::device_vector<Vertex> selected_clusters_in_v(
      targeted_number_of_clusters);
  Size number_of_selected_clusters_in = targeted_number_of_clusters;
  thrust::device_vector<Vertex> selected_clusters_out_v(
      number_of_clusters - targeted_number_of_clusters);
  Size number_of_selected_clusters_out =
      number_of_clusters - targeted_number_of_clusters;

  number_of_selected_clusters_in =
      thrust::copy_if(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
          sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
          selected_clusters_in_v.data(),
          [cluster_scores, max_cluster_score] __device__(Vertex cluster) {
            return cluster_scores[cluster] < max_cluster_score;
          }) -
      selected_clusters_in_v.data();
  selected_clusters_in_v.resize(number_of_selected_clusters_in);

  Vertex cluster_in = thrust::max_element(
      handle.GetThrustPolicy(), selected_clusters_in_v.begin(),
      selected_clusters_in_v.end(),
      [cluster_scores] __device__(auto cluster1, auto cluster2) {
        return cluster_scores[cluster1] < cluster_scores[cluster2];
      })[0];
  Score cluster_in_score =
      thrust::device_ptr<const Score>(cluster_scores)[cluster_in];

  number_of_selected_clusters_out =
      thrust::copy_if(
          handle.GetThrustPolicy(),
          sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
          sfm::utils::MakeCountingIterator<Vertex>(number_of_clusters),
          selected_clusters_out_v.data(),
          [merged_cluster_indices, cluster_scores, cluster_in_score,
           max_cluster_score] __device__(Vertex cluster) {
            return merged_cluster_indices[cluster] == cluster &&
                   cluster_scores[cluster] + cluster_in_score <=
                       max_cluster_score;
          }) -
      selected_clusters_out_v.data();
  selected_clusters_out_v.resize(number_of_selected_clusters_out);

  if (number_of_selected_clusters_out == 0) {
    thrust::copy(handle.GetThrustPolicy(), cluster_scores,
                 cluster_scores + number_of_clusters, merged_cluster_scores);

    number_of_merged_clusters = number_of_clusters;

    return;
  }

  thrust::device_vector<Score> tmp_cluster_scores_v(number_of_clusters);
  thrust::copy(handle.GetThrustPolicy(), cluster_scores,
               cluster_scores + number_of_clusters,
               tmp_cluster_scores_v.begin());

  thrust::stable_sort(
      handle.GetThrustPolicy(), selected_clusters_out_v.begin(),
      selected_clusters_out_v.end(),
      [tmp_cluster_scores = tmp_cluster_scores_v.data().get()] __device__(
          auto cluster1, auto cluster2) {
        return tmp_cluster_scores[cluster1] < tmp_cluster_scores[cluster2];
      });

  thrust::device_vector<Score> cluster_out_scores_scan_v(
      number_of_selected_clusters_out);

  auto sorted_cluster_out_score_begin =
      sfm::utils::MakeTransformIterator<Score>(
          selected_clusters_out_v.data().get(),
          [tmp_cluster_scores = tmp_cluster_scores_v.data().get()] __device__(
              auto vertex) { return tmp_cluster_scores[vertex]; });
  auto sorted_cluster_out_score_end =
      sorted_cluster_out_score_begin + number_of_selected_clusters_out;

  thrust::inclusive_scan(handle.GetThrustPolicy(),
                         sorted_cluster_out_score_begin,
                         sorted_cluster_out_score_end,
                         cluster_out_scores_scan_v.data().get(), cub::Sum());

  Size number_of_clusters_out =
      number_of_selected_clusters_out -
      (thrust::find_if(handle.GetThrustPolicy(),
                       cluster_out_scores_scan_v.rbegin(),
                       cluster_out_scores_scan_v.rend(),
                       [max_cluster_score, cluster_in_score] __device__(
                           auto inclusive_cluser_score) {
                         return inclusive_cluser_score + cluster_in_score <=
                                max_cluster_score;
                       }) -
       cluster_out_scores_scan_v.rbegin());

  thrust::for_each(
      handle.GetThrustPolicy(), selected_clusters_out_v.data().get(),
      selected_clusters_out_v.data().get() + number_of_clusters_out,
      [cluster_in, merged_cluster_indices,
       tmp_cluster_scores =
           tmp_cluster_scores_v.data().get()] __device__(auto cluster) {
        merged_cluster_indices[cluster] = cluster_in;
        tmp_cluster_scores[cluster] = 0;
      });
  tmp_cluster_scores_v[cluster_in] +=
      cluster_out_scores_scan_v[number_of_clusters_out - 1];

  SimplifyClusterIndicesAndScores(
      handle, merged_cluster_indices, tmp_cluster_scores_v.data().get(),
      number_of_clusters, merged_cluster_indices, merged_cluster_scores,
      number_of_merged_clusters);
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void DecomposeAndMergeLargeCluster(
    const sfm::graph::Handle &handle, Vertex cluster_out,
    const Vertex *clusters, const Score *vertex_scores, Size number_of_vertices,
    const Vertex *src_indices, const Vertex *dst_indices, const Weight *weights,
    Size number_of_edges, const Score *cluster_self_weights,
    const Score *cluster_scores, Size number_of_clusters,
    Size targeted_number_of_clusters, Float resolution, Score max_cluster_score,
    Vertex *merged_vertices, Vertex &cluster_in,
    Size &number_of_merged_vertices) {
  Vertex min_cluster = thrust::min_element(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
      [cluster_self_weights, cluster_scores] __device__(auto cluster1,
                                                        auto cluster2) {
        auto cluster_score1 = cluster_scores[cluster1];
        auto cluster_score2 = cluster_scores[cluster2];
        return cluster_score1 != cluster_score2
                   ? cluster_score1 < cluster_score2
                   : cluster_self_weights[cluster1] <
                         cluster_self_weights[cluster2];
      })[0];
  Score min_cluster_score =
      thrust::device_ptr<const Score>(cluster_scores)[min_cluster];

  thrust::device_vector<Vertex> selected_vertices_v(number_of_vertices);
  thrust::device_vector<Vertex> selected_src_indices_v(number_of_edges);
  thrust::device_vector<Vertex> selected_dst_indices_v(number_of_edges);
  thrust::device_vector<Weight> selected_weights_v(number_of_edges);
  Size number_of_selected_vertices =
      thrust::copy_if(handle.GetThrustPolicy(),
                      sfm::utils::MakeCountingIterator(0),
                      sfm::utils::MakeCountingIterator(number_of_vertices),
                      selected_vertices_v.data(),
                      [clusters, cluster_out] __device__(Vertex vertex) {
                        return clusters[vertex] == cluster_out;
                      }) -
      selected_vertices_v.data();
  selected_vertices_v.resize(number_of_selected_vertices);

  thrust::device_vector<Vertex> selected_vertex_indices_v(number_of_vertices,
                                                          -1);
  thrust::device_vector<Score> selected_vertex_scores_v(
      number_of_selected_vertices);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_selected_vertices),
      [selected_vertices = selected_vertices_v.data().get(),
       selected_vertex_indices = selected_vertex_indices_v.data().get(),
       selected_vertex_scores = selected_vertex_scores_v.data().get(),
       vertex_scores] __device__(auto idx) {
        auto vertex = selected_vertices[idx];
        selected_vertex_scores[idx] = vertex_scores[vertex];
        selected_vertex_indices[vertex] = idx;
      });

  auto edge_begin =
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights);
  auto edge_end = edge_begin + number_of_edges;
  auto selected_edge_begin = sfm::utils::MakeZipIterator(
      selected_src_indices_v.data(), selected_dst_indices_v.data(),
      selected_weights_v.data());
  Size number_of_selected_edges =
      thrust::copy_if(handle.GetThrustPolicy(), edge_begin, edge_end,
                      selected_edge_begin,
                      [cluster_out, clusters] __device__(auto edge) {
                        return clusters[thrust::get<0>(edge)] == cluster_out &&
                               clusters[thrust::get<1>(edge)] == cluster_out;
                      }) -
      selected_edge_begin;
  selected_src_indices_v.resize(number_of_selected_edges);
  selected_dst_indices_v.resize(number_of_selected_edges);
  selected_weights_v.resize(number_of_selected_edges);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
      sfm::utils::MakeCountingIterator(number_of_selected_edges),
      [selected_src_indices = selected_src_indices_v.data().get(),
       selected_dst_indices = selected_dst_indices_v.data().get(),
       selected_vertex_indices =
           selected_vertex_indices_v.data().get()] __device__(auto idx) {
        auto src = selected_src_indices[idx];
        auto dst = selected_dst_indices[idx];
        selected_src_indices[idx] = selected_vertex_indices[src];
        selected_dst_indices[idx] = selected_vertex_indices[dst];
      });

  thrust::device_vector<Vertex> selected_src_offsets_v(
      number_of_selected_vertices + 1);

  sfm::graph::detail::COO2CSR(
      handle, number_of_selected_vertices, selected_src_indices_v.data().get(),
      selected_dst_indices_v.data().get(), selected_weights_v.data().get(),
      number_of_selected_edges, selected_src_offsets_v.data().get());

  Float total_edge_weight =
      thrust::reduce(handle.GetThrustPolicy(), weights,
                     weights + number_of_edges, Weight(0), cub::Sum());
  Float total_selected_edge_weight = thrust::transform_reduce(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_edges),
      [src_indices, dst_indices, weights, clusters,
       cluster_out] __device__(auto edge) -> Weight {
        return clusters[src_indices[edge]] == cluster_out ||
                       clusters[dst_indices[edge]] >= cluster_out
                   ? weights[edge]
                   : 0;
      },
      Weight(0), cub::Sum());

  // Correct the resolution to be consistent with the original graph when
  // computing the modularity
  Float corrected_resolution =
      resolution * total_selected_edge_weight / total_edge_weight;

  thrust::device_vector<Vertex> selected_vertex_clusters_v(
      number_of_selected_vertices, 0);

  Float modularity = 0;
  Size number_of_selected_vertex_clusters = 0;
  thrust::device_vector<Weight> selected_vertex_cluster_self_weights_v;
  thrust::device_vector<Weight> selected_vertex_cluster_scores_v;
  Louvain(handle, selected_vertex_scores_v.data().get(),
          number_of_selected_vertices, selected_src_offsets_v.data().get(),
          selected_src_indices_v.data().get(),
          selected_dst_indices_v.data().get(), selected_weights_v.data().get(),
          number_of_selected_edges, 0, corrected_resolution,
          max_cluster_score - min_cluster_score, modularity,
          selected_vertex_clusters_v.data().get(),
          number_of_selected_vertex_clusters, selected_vertex_cluster_scores_v,
          selected_vertex_cluster_self_weights_v);

  Vertex selected_vertex_cluster_out = thrust::max_element(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(
          number_of_selected_vertex_clusters),
      [self_weights = selected_vertex_cluster_self_weights_v.data().get(),
       scores = selected_vertex_cluster_scores_v.data()
                    .get()] __device__(auto cluster1, auto cluster2) {
        auto weight1 = self_weights[cluster1];
        auto weight2 = self_weights[cluster2];
        auto score1 = scores[cluster1];
        auto score2 = scores[cluster2];
        return weight1 != weight2 ? weight1 < weight2 : score1 > score2;
      })[0];
  Score selected_vertex_cluster_score_out =
      selected_vertex_cluster_scores_v[selected_vertex_cluster_out];
  cluster_in = -1;
  if (selected_vertex_cluster_score_out + min_cluster_score >
      max_cluster_score) {
    cluster_in = min_cluster;
  } else {
    auto &targeted_cluster_indices_v = selected_src_indices_v;
    auto &targeted_vertex_indices_v = selected_dst_indices_v;
    auto &targeted_weights_v = selected_weights_v;
    targeted_cluster_indices_v.resize(number_of_edges);
    targeted_vertex_indices_v.resize(number_of_edges);
    targeted_weights_v.resize(number_of_edges);

    auto targeted_edge_begin = sfm::utils::MakeZipIterator(
        targeted_cluster_indices_v.data(), targeted_vertex_indices_v.data(),
        targeted_weights_v.data());
    Size number_of_targeted_edges =
        thrust::copy_if(
            handle.GetThrustPolicy(), edge_begin, edge_end, targeted_edge_begin,
            [selected_vertex_indices = selected_vertex_indices_v.data().get(),
             selected_vertex_clusters = selected_vertex_clusters_v.data().get(),
             selected_vertex_cluster_out, selected_vertex_cluster_score_out,
             max_cluster_score, cluster_scores,
             targeted_number_of_clusters] __device__(auto edge) {
              Vertex cluster = thrust::get<0>(edge);
              Vertex vertex = thrust::get<1>(edge);

              return cluster < targeted_number_of_clusters &&
                     selected_vertex_indices[vertex] != -1 &&
                     cluster_scores[cluster] +
                             selected_vertex_cluster_score_out <=
                         max_cluster_score &&
                     selected_vertex_clusters
                             [selected_vertex_indices[vertex]] ==
                         selected_vertex_cluster_out;
            }) -
        targeted_edge_begin;

    targeted_cluster_indices_v.resize(number_of_targeted_edges);
    targeted_vertex_indices_v.resize(number_of_targeted_edges);
    targeted_weights_v.resize(number_of_targeted_edges);

    thrust::device_vector<Vertex> targeted_clusters_in_v(
        targeted_number_of_clusters);
    if (number_of_targeted_edges != 0) {
      thrust::stable_sort_by_key(
          handle.GetThrustPolicy(), targeted_cluster_indices_v.begin(),
          targeted_cluster_indices_v.end(), targeted_weights_v.begin());
      thrust::device_vector<Weight> targeted_clusters_in_v(
          targeted_number_of_clusters);
      thrust::device_vector<Weight> targeted_cluster_weights_v(
          targeted_number_of_clusters);
      Size number_of_targeted_clusters_in =
          thrust::reduce_by_key(
              handle.GetThrustPolicy(), targeted_cluster_indices_v.begin(),
              targeted_cluster_indices_v.end(), targeted_weights_v.begin(),
              targeted_clusters_in_v.begin(),
              targeted_cluster_weights_v.begin())
              .first -
          targeted_clusters_in_v.begin();
      targeted_clusters_in_v.resize(number_of_targeted_clusters_in);
      targeted_cluster_weights_v.resize(number_of_targeted_clusters_in);
      cluster_in = targeted_clusters_in_v[thrust::reduce(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
          sfm::utils::MakeCountingIterator<Vertex>(
              number_of_targeted_clusters_in),
          Vertex(0),
          [targeted_clusters_in = targeted_clusters_in_v.data().get(),
           targeted_cluster_weights = targeted_cluster_weights_v.data().get(),
           cluster_scores] __device__(auto index1, auto index2) {
            Weight weight1 = targeted_cluster_weights[index1];
            Weight weight2 = targeted_cluster_weights[index2];

            if (weight1 != weight2) {
              return weight1 > weight2 ? index1 : index2;
            } else {
              Vertex cluster1 = targeted_clusters_in[index1];
              Vertex cluster2 = targeted_clusters_in[index2];
              Score score1 = cluster_scores[cluster1];
              Score score2 = cluster_scores[cluster2];

              if (score1 > score2) {
                return index1;
              } else if (score1 == score2 && cluster1 < cluster2) {
                return index1;
              } else {
                return index2;
              }
            }
          })];
    } else {
      Size number_of_targeted_clusters_in =
          thrust::copy_if(handle.GetThrustPolicy(),
                          sfm::utils::MakeCountingIterator<Vertex>(0),
                          sfm::utils::MakeCountingIterator<Vertex>(
                              targeted_number_of_clusters),
                          targeted_clusters_in_v.data(),
                          [selected_vertex_cluster_score_out, max_cluster_score,
                           cluster_scores] __device__(auto cluster) {
                            return cluster_scores[cluster] +
                                       selected_vertex_cluster_score_out <=
                                   max_cluster_score;
                          }) -
          targeted_clusters_in_v.data();
      targeted_clusters_in_v.resize(number_of_targeted_clusters_in);
      cluster_in = thrust::max_element(
          handle.GetThrustPolicy(), targeted_clusters_in_v.begin(),
          targeted_clusters_in_v.end(),
          [cluster_scores, cluster_self_weights] __device__(auto cluster1,
                                                            auto cluster2) {
            auto cluster_score1 = cluster_scores[cluster1];
            auto cluster_score2 = cluster_scores[cluster2];
            return cluster_score1 != cluster_score2
                       ? cluster_score1 < cluster_score2
                       : cluster_self_weights[cluster1] <
                             cluster_self_weights[cluster2];
          })[0];
    }
  }

  number_of_merged_vertices =
      thrust::copy_if(
          handle.GetThrustPolicy(), selected_vertices_v.begin(),
          selected_vertices_v.end(), merged_vertices,
          [selected_vertex_indices = selected_vertex_indices_v.data().get(),
           selected_vertex_clusters = selected_vertex_clusters_v.data().get(),
           selected_vertex_cluster_out] __device__(auto vertex) {
            return selected_vertex_clusters[selected_vertex_indices[vertex]] ==
                   selected_vertex_cluster_out;
          }) -
      merged_vertices;
}

template <typename Size, typename Vertex, typename Weight, typename Score>
void MergeLargeCluster(const sfm::graph::Handle &handle, Vertex cluster_out,
                       const Vertex *clusters, const Score *vertex_scores,
                       Size number_of_vertices, const Vertex *src_indices,
                       const Vertex *dst_indices, const Weight *weights,
                       Size number_of_edges, const Score *cluster_scores,
                       Size number_of_clusters,
                       Size targeted_number_of_clusters,
                       Score max_cluster_score, Vertex &cluster_in) {
  cluster_in = cluster_out;

  if (number_of_clusters <= targeted_number_of_clusters ||
      cluster_out >= number_of_clusters) {
    return;
  }

  thrust::device_vector<Vertex> cluster_src_offsets_v;
  thrust::device_vector<Vertex> cluster_src_indices_v;
  thrust::device_vector<Vertex> cluster_dst_indices_v;
  thrust::device_vector<Weight> cluster_weights_v;
  Size number_of_cluster_edges;

  GetClusterInfo(handle, number_of_clusters, clusters, number_of_vertices,
                 src_indices, dst_indices, weights, number_of_edges,
                 cluster_src_offsets_v, cluster_src_indices_v,
                 cluster_dst_indices_v, cluster_weights_v,
                 number_of_cluster_edges);

  thrust::device_vector<Vertex> selected_clusters_in_v(
      targeted_number_of_clusters);
  Size number_of_selected_clusters_in =
      thrust::copy_if(
          handle.GetThrustPolicy(),
          sfm::utils::MakeCountingIterator<Vertex>(
              cluster_src_offsets_v[cluster_out]),
          sfm::utils::MakeCountingIterator<Vertex>(
              cluster_src_offsets_v[cluster_out + 1]),
          selected_clusters_in_v.data(),
          [cluster_scores, targeted_number_of_clusters, cluster_out,
           max_cluster_score,
           cluster_dst_indices =
               cluster_dst_indices_v.data().get()] __device__(auto edge) {
            Vertex cluster = cluster_dst_indices[edge];
            return cluster < targeted_number_of_clusters &&
                   cluster_scores[cluster] + cluster_scores[cluster_out] <=
                       max_cluster_score;
          }) -
      selected_clusters_in_v.data();
  selected_clusters_in_v.resize(number_of_selected_clusters_in);

  if (number_of_selected_clusters_in == 0) {
    cluster_in = thrust::find_if(
        handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
        sfm::utils::MakeCountingIterator<Vertex>(targeted_number_of_clusters),
        [cluster_scores, cluster_out,
         max_cluster_score] __device__(auto cluster) {
          return cluster_scores[cluster] + cluster_scores[cluster_out] <=
                 max_cluster_score;
        })[0];

    if (cluster_in == targeted_number_of_clusters) {
      cluster_in = cluster_out;
    }

    return;
  }

  cluster_in = cluster_dst_indices_v[thrust::max_element(
      handle.GetThrustPolicy(), selected_clusters_in_v.begin(),
      selected_clusters_in_v.end(),
      [cluster_scores, cluster_dst_indices = cluster_src_indices_v.data().get(),
       cluster_weights =
           cluster_weights_v.data().get()] __device__(auto edge1, auto edge2) {
        Vertex cluster1 = cluster_dst_indices[edge1];
        Vertex cluster2 = cluster_dst_indices[edge2];
        Weight cluster_weight1 = cluster_weights[edge1];
        Weight cluster_weight2 = cluster_weights[edge2];
        return cluster_weight1 != cluster_weight2
                   ? cluster_weight1 < cluster_weight2
                   : cluster_scores[cluster1] < cluster_scores[cluster2];
      })[0]];
}

template <typename Size, typename Vertex, typename Weight, typename Score>
bool MergeZeroClusterSelfWeightClusters(
    const sfm::graph::Handle &handle, const Score *cluster_scores,
    const Weight *cluster_self_weights, Size number_of_clusters,
    const Vertex *cluster_src_offsets, const Vertex *cluster_src_indices,
    const Vertex *cluster_dst_indices, const Weight *cluster_weights,
    Size number_of_cluster_edges, Size targeted_number_of_clusters,
    Score max_cluster_score, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Weight *merged_cluster_self_weights,
    Size &merged_number_of_clusters) {
  thrust::sequence(handle.GetThrustPolicy(), merged_cluster_indices,
                   merged_cluster_indices + number_of_clusters, 0);
  thrust::copy(handle.GetThrustPolicy(), cluster_scores,
               cluster_scores + number_of_clusters, merged_cluster_scores);
  thrust::copy(handle.GetThrustPolicy(), cluster_self_weights,
               cluster_self_weights + number_of_clusters,
               merged_cluster_self_weights);
  merged_number_of_clusters = number_of_clusters;

  if (number_of_clusters <= targeted_number_of_clusters) {
    return false;
  }

  thrust::transform(handle.GetThrustPolicy(),
                    merged_cluster_indices + targeted_number_of_clusters,
                    merged_cluster_indices + number_of_clusters,
                    merged_cluster_indices + targeted_number_of_clusters,
                    [cluster_self_weights, cluster_src_offsets,
                     cluster_src_indices, cluster_dst_indices,
                     targeted_number_of_clusters] __device__(auto cluster) {
                      auto edge = cluster_src_offsets[cluster + 1] - 1;
                      return cluster_self_weights[cluster] > 0 ||
                                     (edge >= 0 &&
                                      cluster_src_indices[edge] == cluster &&
                                      cluster_dst_indices[edge] >=
                                          targeted_number_of_clusters)
                                 ? cluster
                                 : -1;
                    });

  auto cluster_edge_begin = sfm::utils::MakeZipIterator(
      cluster_src_indices, cluster_dst_indices, cluster_weights);
  auto cluster_edge_end = cluster_edge_begin + number_of_cluster_edges;

  thrust::for_each(
      handle.GetThrustPolicy(), cluster_edge_begin, cluster_edge_end,
      [merged_cluster_indices, cluster_scores, cluster_self_weights,
       max_cluster_score, targeted_number_of_clusters] __device__(auto edge) {
        auto src = thrust::get<0>(edge);
        auto dst = thrust::get<1>(edge);

        if (src >= targeted_number_of_clusters &&
            dst < targeted_number_of_clusters &&
            cluster_self_weights[src] == 0 &&
            cluster_scores[src] + cluster_scores[dst] <= max_cluster_score) {
          atomicCAS(merged_cluster_indices, -1, src);
        }
      });

  if (thrust::all_of(handle.GetThrustPolicy(),
                     merged_cluster_indices + targeted_number_of_clusters,
                     merged_cluster_indices + number_of_clusters,
                     [] __device__(auto cluster) { return cluster != -1; })) {
    return false;
  }

  auto unary_func =
      [merged_cluster_indices,
       targeted_number_of_clusters] __device__(auto edge) -> Size {
    auto src = thrust::get<0>(edge);
    auto dst = thrust::get<1>(edge);
    return src >= targeted_number_of_clusters &&
           dst < targeted_number_of_clusters &&
           merged_cluster_indices[src] == -1;
  };

  Size number_of_selected_cluster_edges = thrust::transform_reduce(
      handle.GetThrustPolicy(), cluster_edge_begin, cluster_edge_end,
      unary_func, Size(0), cub::Sum());

  if (number_of_selected_cluster_edges == 0) {
    return false;
  }

  thrust::device_vector<Vertex> selected_cluster_src_indices_v(
      number_of_selected_cluster_edges);
  thrust::device_vector<Vertex> selected_cluster_dst_indices_v(
      number_of_selected_cluster_edges);
  thrust::device_vector<Weight> selected_cluster_weights_v(
      number_of_selected_cluster_edges);

  thrust::copy_if(
      handle.GetThrustPolicy(), cluster_edge_begin, cluster_edge_end,
      sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.data(),
                                  selected_cluster_dst_indices_v.data(),
                                  selected_cluster_weights_v.data()),
      unary_func);

  thrust::device_vector<Vertex> selected_clusters_v(
      number_of_clusters - targeted_number_of_clusters);
  thrust::device_vector<Vertex> selected_merged_cluster_indices_v(
      number_of_clusters - targeted_number_of_clusters);
  thrust::device_vector<Weight> selected_merged_cluster_weights_v(
      number_of_clusters - targeted_number_of_clusters);

  Size number_of_selected_clusters =
      thrust::reduce_by_key(
          handle.GetThrustPolicy(), selected_cluster_src_indices_v.begin(),
          selected_cluster_src_indices_v.end(),
          sfm::utils::MakeZipIterator(selected_cluster_dst_indices_v.begin(),
                                      selected_cluster_weights_v.begin()),
          selected_clusters_v.begin(),
          sfm::utils::MakeZipIterator(
              selected_merged_cluster_indices_v.begin(),
              selected_merged_cluster_weights_v.begin()),
          thrust::equal_to<Vertex>(),
          [] __device__(auto pair1, auto pair2) {
            auto dst1 = thrust::get<0>(pair1);
            auto dst2 = thrust::get<0>(pair2);
            auto weight1 = thrust::get<1>(pair1);
            auto weight2 = thrust::get<1>(pair2);

            if (weight1 > weight2) {
              return pair1;
            } else if (weight1 == weight2 && dst1 > dst2) {
              return pair1;
            } else {
              return pair2;
            }
          })
          .first -
      selected_clusters_v.begin();

  selected_clusters_v.resize(number_of_selected_clusters);
  selected_merged_cluster_indices_v.resize(number_of_selected_clusters);
  selected_merged_cluster_weights_v.resize(number_of_selected_clusters);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_selected_clusters),
      [selected_clusters = selected_clusters_v.data().get(),
       selected_merged_cluster_indices =
           selected_merged_cluster_indices_v.data().get(),
       selected_merged_cluster_weights =
           selected_merged_cluster_weights_v.data().get(),
       merged_cluster_indices, merged_cluster_scores,
       merged_cluster_self_weights] __device__(auto index) {
        auto src = selected_clusters[index];
        auto dst = selected_merged_cluster_indices[index];
        merged_cluster_indices[src] = dst;

        atomicAdd(merged_cluster_scores + dst, merged_cluster_scores[src]);
        atomicAdd(merged_cluster_self_weights + dst,
                  2 * selected_merged_cluster_weights[index]);
        merged_cluster_scores[src] = 0;
        merged_cluster_self_weights[src] = 0;
      });

  merged_number_of_clusters = number_of_clusters - number_of_selected_clusters;

  thrust::device_vector<Vertex> sorted_merged_clusters_v(number_of_clusters);
  thrust::sequence(handle.GetThrustPolicy(), sorted_merged_clusters_v.begin(),
                   sorted_merged_clusters_v.end(), Vertex(0));

  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), merged_cluster_self_weights,
      merged_cluster_self_weights + number_of_clusters,
      sfm::utils::MakeZipIterator(merged_cluster_scores,
                                  sorted_merged_clusters_v.data()),
      thrust::greater<Weight>());

  thrust::stable_sort_by_key(
      handle.GetThrustPolicy(), merged_cluster_scores,
      merged_cluster_scores + targeted_number_of_clusters,
      sfm::utils::MakeZipIterator(merged_cluster_self_weights,
                                  sorted_merged_clusters_v.data()),
      thrust::greater<Weight>());

  thrust::device_vector<Vertex> sorted_merged_clusters_indices_v(
      number_of_clusters);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_clusters),
      [sorted_merged_clusters = sorted_merged_clusters_v.data().get(),
       sorted_merged_clusters_indices = sorted_merged_clusters_indices_v.data()
                                            .get()] __device__(auto index) {
        sorted_merged_clusters_indices[sorted_merged_clusters[index]] = index;
      });

  thrust::transform(
      handle.GetThrustPolicy(), merged_cluster_indices,
      merged_cluster_indices + number_of_clusters, merged_cluster_indices,
      [sorted_merged_clusters_indices = sorted_merged_clusters_indices_v.data()
                                            .get()] __device__(auto cluster) {
        return sorted_merged_clusters_indices[cluster];
      });

  return true;
}

#if 1
template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
bool RemergeCluster(
    const sfm::graph::Handle &handle, const Score *cluster_scores,
    const Weight *cluster_self_weights, Size number_of_clusters,
    const Vertex *cluster_src_offsets, const Vertex *cluster_src_indices,
    const Vertex *cluster_dst_indices, const Weight *cluster_weights,
    Size number_of_cluster_edges, Float resolution,
    Size targeted_number_of_clusters, Score max_cluster_score,
    Vertex *merged_cluster_indices,
    thrust::device_vector<Score> &merged_cluster_scores_v,
    thrust::device_vector<Weight> &merged_cluster_self_weights_v,
    Size &number_of_merged_clusters) {
  number_of_merged_clusters = number_of_clusters;
  merged_cluster_scores_v.resize(number_of_merged_clusters);
  merged_cluster_self_weights_v.resize(number_of_merged_clusters);
  thrust::sequence(handle.GetThrustPolicy(), merged_cluster_indices,
                   merged_cluster_indices + number_of_clusters, 0);
  thrust::copy(handle.GetThrustPolicy(), cluster_scores,
               cluster_scores + number_of_clusters,
               merged_cluster_scores_v.data());
  thrust::copy(handle.GetThrustPolicy(), cluster_self_weights,
               cluster_self_weights + number_of_clusters,
               merged_cluster_self_weights_v.data());

  if (number_of_clusters <= targeted_number_of_clusters) {
    return false;
  }

  auto cluster_edge_begin = sfm::utils::MakeZipIterator(
      cluster_src_indices, cluster_dst_indices, cluster_weights);
  auto cluster_edge_end = cluster_edge_begin + number_of_cluster_edges;

  Size number_of_selected_clusters =
      number_of_clusters - targeted_number_of_clusters;
  thrust::device_vector<Vertex> selected_cluster_src_offsets_v(
      number_of_selected_clusters + 1);
  thrust::device_vector<Vertex> selected_cluster_src_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Vertex> selected_cluster_dst_indices_v(
      number_of_cluster_edges);
  thrust::device_vector<Weight> selected_cluster_weights_v(
      number_of_cluster_edges);

  auto selected_cluster_edge_begin =
      sfm::utils::MakeZipIterator(selected_cluster_src_indices_v.begin(),
                                  selected_cluster_dst_indices_v.begin(),
                                  selected_cluster_weights_v.begin());

  Size number_of_selected_cluster_edges =
      thrust::copy_if(
          handle.GetThrustPolicy(), cluster_edge_begin, cluster_edge_end,
          selected_cluster_edge_begin,
          [targeted_number_of_clusters] __device__(auto edge) {
            return thrust::get<0>(edge) >= targeted_number_of_clusters &&
                   thrust::get<1>(edge) >= targeted_number_of_clusters;
          }) -
      selected_cluster_edge_begin;

  selected_cluster_src_indices_v.resize(number_of_selected_cluster_edges);
  selected_cluster_dst_indices_v.resize(number_of_selected_cluster_edges);
  selected_cluster_weights_v.resize(number_of_selected_cluster_edges);

  thrust::transform(handle.GetThrustPolicy(),
                    selected_cluster_src_indices_v.begin(),
                    selected_cluster_src_indices_v.end(),
                    selected_cluster_src_indices_v.begin(),
                    [targeted_number_of_clusters] __device__(auto cluster) {
                      return cluster - targeted_number_of_clusters;
                    });

  thrust::transform(handle.GetThrustPolicy(),
                    selected_cluster_dst_indices_v.begin(),
                    selected_cluster_dst_indices_v.end(),
                    selected_cluster_dst_indices_v.begin(),
                    [targeted_number_of_clusters] __device__(auto cluster) {
                      return cluster - targeted_number_of_clusters;
                    });

  sfm::graph::detail::COO2CSR(handle, number_of_selected_clusters,
                              selected_cluster_src_indices_v.data().get(),
                              selected_cluster_dst_indices_v.data().get(),
                              selected_cluster_weights_v.data().get(),
                              number_of_selected_cluster_edges,
                              selected_cluster_src_offsets_v.data().get());

  thrust::device_vector<Score> selected_cluster_scores_v(
      number_of_selected_clusters);

  thrust::copy(handle.GetThrustPolicy(),
               cluster_scores + targeted_number_of_clusters,
               cluster_scores + number_of_clusters,
               selected_cluster_scores_v.data().get());

  Float selected_modularity = 0;
  Size number_of_merged_selected_clusters;
  Score max_remerged_cluster_score =
      max_cluster_score -
      thrust::reduce(handle.GetThrustPolicy(), cluster_scores,
                     cluster_scores + targeted_number_of_clusters,
                     max_cluster_score, cub::Min());
  thrust::device_vector<Vertex> merged_selected_clusters_v(
      number_of_selected_clusters);
  thrust::device_vector<Score> merged_selected_cluster_scores_v;
  thrust::device_vector<Weight> merged_selected_cluster_self_weights_v;
  Louvain(
      handle, selected_cluster_scores_v.data().get(),
      number_of_selected_clusters, selected_cluster_src_offsets_v.data().get(),
      selected_cluster_src_indices_v.data().get(),
      selected_cluster_dst_indices_v.data().get(),
      selected_cluster_weights_v.data().get(), number_of_selected_cluster_edges,
      0, resolution, max_remerged_cluster_score, selected_modularity,
      merged_selected_clusters_v.data().get(),
      number_of_merged_selected_clusters, merged_selected_cluster_scores_v,
      merged_selected_cluster_self_weights_v, false);

  SortClusters(handle, 0, merged_selected_clusters_v.data().get(),
               number_of_selected_clusters,
               merged_selected_cluster_scores_v.data().get(),
               merged_selected_cluster_self_weights_v.data().get(),
               number_of_merged_selected_clusters);

  number_of_merged_clusters =
      targeted_number_of_clusters + number_of_merged_selected_clusters;
  merged_cluster_scores_v.resize(number_of_merged_clusters);
  merged_cluster_self_weights_v.resize(number_of_merged_clusters);
  thrust::transform(handle.GetThrustPolicy(),
                    merged_selected_clusters_v.begin(),
                    merged_selected_clusters_v.end(),
                    merged_cluster_indices + targeted_number_of_clusters,
                    [targeted_number_of_clusters] __device__(auto cluster) {
                      return cluster + targeted_number_of_clusters;
                    });
  thrust::copy(handle.GetThrustPolicy(),
               merged_selected_cluster_scores_v.begin(),
               merged_selected_cluster_scores_v.end(),
               merged_cluster_scores_v.begin() + targeted_number_of_clusters);
  thrust::copy(
      handle.GetThrustPolicy(), merged_selected_cluster_self_weights_v.begin(),
      merged_selected_cluster_self_weights_v.end(),
      merged_cluster_self_weights_v.begin() + targeted_number_of_clusters);

  return true;
}
#endif

template <typename Size, typename Float, typename Vertex, typename Weight,
          typename Score>
void ComputeDeltaModularity(
    sfm::graph::Handle const &handle, Float total_edge_weight, Float resolution,
    Size number_of_vertices, Size number_of_edges,
    thrust::device_vector<Vertex> const &offsets_v,
    thrust::device_vector<Vertex> const &src_indices_v,
    thrust::device_vector<Vertex> const &dst_indices_v,
    thrust::device_vector<Weight> const &weights_v,
    thrust::device_vector<Weight> const &vertex_weights_v,
    thrust::device_vector<Weight> const &cluster_weights_v,
    thrust::device_vector<Score> const &vertex_scores_v,
    thrust::device_vector<Score> const &cluster_scores_v,
    sfm::graph::Dendrogram<Vertex> const &dendrogram,
    thrust::device_vector<Vertex> &cluster_hash_v,
    thrust::device_vector<Weight> &old_cluster_sum_v,
    thrust::device_vector<Weight> &new_cluster_sum_v,
    thrust::device_vector<Float> &delta_Q_v, Size targeted_number_of_clusters,
    Score max_cluster_socre) {
  Vertex const *d_offsets = offsets_v.data().get();
  Weight const *d_weights = weights_v.data().get();
  Vertex const *d_cluster = dendrogram.CurrentLevelBegin();
  Weight const *d_vertex_weights = vertex_weights_v.data().get();
  Weight const *d_cluster_weights = cluster_weights_v.data().get();
  Score const *d_vertex_scores = vertex_scores_v.data().get();
  Score const *d_cluster_scores = cluster_scores_v.data().get();

  Vertex *d_cluster_hash = cluster_hash_v.data().get();
  Float *d_delta_Q = delta_Q_v.data().get();
  Weight *d_old_cluster_sum = old_cluster_sum_v.data().get();
  Weight *d_new_cluster_sum = new_cluster_sum_v.data().get();

  thrust::fill(handle.GetThrustPolicy(), cluster_hash_v.begin(),
               cluster_hash_v.end(), Vertex{-1});
  thrust::fill(handle.GetThrustPolicy(), old_cluster_sum_v.begin(),
               old_cluster_sum_v.end(), Weight{0});
  thrust::fill(handle.GetThrustPolicy(), new_cluster_sum_v.begin(),
               new_cluster_sum_v.end(), Weight{0});

  thrust::for_each(handle.GetThrustPolicy(),
                   thrust::make_counting_iterator<Vertex>(0),
                   thrust::make_counting_iterator<Vertex>(number_of_edges),
                   [d_src_indices = src_indices_v.data().get(),
                    d_dst_indices = dst_indices_v.begin(), d_cluster, d_offsets,
                    d_cluster_hash, d_new_cluster_sum, d_weights,
                    d_old_cluster_sum] __device__(Vertex loc) {
                     Vertex src = d_src_indices[loc];
                     Vertex dst = d_dst_indices[loc];

                     if (src != dst) {
                       Vertex old_cluster = d_cluster[src];
                       Vertex new_cluster = d_cluster[dst];

                       Vertex hash_base = d_offsets[src];
                       Vertex n_edges = d_offsets[src + 1] - hash_base;

                       int h = (new_cluster % n_edges);
                       Vertex offset = hash_base + h;
                       while (d_cluster_hash[offset] != new_cluster) {
                         if (d_cluster_hash[offset] == -1) {
                           atomicCAS(d_cluster_hash + offset, -1, new_cluster);
                         } else {
                           h = (h + 1) % n_edges;
                           offset = hash_base + h;
                         }
                       }

                       atomicAdd(d_new_cluster_sum + offset, d_weights[loc]);

                       if (old_cluster == new_cluster)
                         atomicAdd(d_old_cluster_sum + src, d_weights[loc]);
                     }
                   });

  thrust::for_each(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
      thrust::make_counting_iterator<Vertex>(number_of_edges),
      [d_cluster_hash, d_src_indices = src_indices_v.data().get(), d_cluster,
       d_vertex_weights, d_delta_Q, d_new_cluster_sum, d_old_cluster_sum,
       d_cluster_weights, max_cluster_socre, d_vertex_scores, d_cluster_scores,
       targeted_number_of_clusters, total_edge_weight,
       resolution] __device__(Vertex loc) {
        Vertex new_cluster = d_cluster_hash[loc];
        if (new_cluster >= 0) {
          Vertex src = d_src_indices[loc];
          Vertex old_cluster = d_cluster[src];
          // NOTE: Using floating numbers to avoid overflowing
          Float k_k = d_vertex_weights[src];
          Float a_old = d_cluster_weights[old_cluster];
          Float a_new = d_cluster_weights[new_cluster];
          Float old_cluster_sum = d_old_cluster_sum[src];
          Float new_cluster_sum = d_new_cluster_sum[loc];
          Float vertex_score = d_vertex_scores[src];
          Float cluster_score = d_cluster_scores[new_cluster];

          bool score_mode =
              ((vertex_score + cluster_score) <= max_cluster_socre ||
               vertex_score == 0 || cluster_score == 0);
          bool cluster_mode = src >= targeted_number_of_clusters &&
                              new_cluster < targeted_number_of_clusters;
          d_delta_Q[loc] =
              cluster_mode && score_mode
                  ? 2 * (((new_cluster_sum - old_cluster_sum) /
                          total_edge_weight) -
                         resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                             (total_edge_weight * total_edge_weight))
                  : Float{0};
        } else {
          d_delta_Q[loc] = Float{0};
        }
      });
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Louvain(const sfm::graph::Handle &handle, const Score *vertex_scores,
             Size number_of_vertices, Vertex *src_offsets, Vertex *src_indices,
             Vertex *dst_indices, Weight *weights, Size number_of_edges,
             Size targeted_number_of_clusters, Float resolution,
             Score max_cluster_score, Float &modularity, Vertex *clusters,
             Size &number_of_clusters,
             thrust::device_vector<Score> &cluster_scores_v,
             thrust::device_vector<Weight> &cluster_self_weights_v,
             bool verbose) {
  sfm::graph::GraphCSRView<Vertex, Size, Weight> csr_graph(
      src_offsets, dst_indices, weights, number_of_vertices, number_of_edges);

  sfm::graph::Louvain<sfm::graph::GraphCSRView<Vertex, Size, Weight>, Float,
                      Score>
      louvain(handle);

  Float total_edge_weight = thrust::reduce(handle.GetThrustPolicy(), weights,
                                           weights + number_of_edges);

  modularity = louvain.Cluster(
      csr_graph, 100, resolution, clusters,
      [vertex_scores] __device__(Vertex idx) -> Score {
        return vertex_scores[idx];
      },
      max_cluster_score, false);

  number_of_clusters = louvain.GetNumberOfClusters();
  cluster_scores_v.resize(number_of_clusters);
  cluster_self_weights_v.resize(number_of_clusters);

  thrust::device_vector<Vertex> cluster_src_indices_v;
  thrust::device_vector<Vertex> cluster_dst_indices_v;
  thrust::device_vector<Weight> cluster_weights_v;

  Size number_of_cluster_edges = louvain.GetNumberOfClusterEdges();
  cluster_dst_indices_v.resize(number_of_cluster_edges);
  cluster_src_indices_v.resize(number_of_cluster_edges);
  cluster_dst_indices_v.resize(number_of_cluster_edges);
  cluster_weights_v.resize(number_of_cluster_edges);

  thrust::copy(handle.GetThrustPolicy(), louvain.GetClusterScores().begin(),
               louvain.GetClusterScores().end(), cluster_scores_v.begin());
  louvain.GetClusterEdges(cluster_src_indices_v, cluster_dst_indices_v,
                          cluster_weights_v);
  GetSelfWeights(handle, cluster_src_indices_v.data().get(),
                 cluster_dst_indices_v.data().get(),
                 cluster_weights_v.data().get(), number_of_cluster_edges,
                 cluster_self_weights_v.data().get(), number_of_clusters);
  // sort clusters according cluster scores before target number of clusters
  // and cluster self-weights after
  SortClusters(handle, targeted_number_of_clusters, clusters,
               number_of_vertices, cluster_scores_v.data().get(),
               cluster_self_weights_v.data().get(), number_of_clusters);

  modularity = sfm::graph::detail::Modularity(handle, total_edge_weight,
                                              resolution, csr_graph, clusters);

  if (verbose) {
    std::cout << "cluster self-weights: ";
    for (const auto &weight : cluster_self_weights_v) {
      std::cout << weight << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "cluster scores: ";
    for (const auto &score : cluster_scores_v) {
      std::cout << score << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "modularity: " << modularity << std::endl << std::endl;
  }

  MergeZeroScoreVertices(handle, clusters, vertex_scores, number_of_vertices,
                         src_indices, dst_indices, weights, number_of_edges);

  SimplifyClusters(handle, clusters, number_of_vertices,
                   cluster_scores_v.data().get(), number_of_clusters);

  cluster_scores_v.resize(number_of_clusters);
  cluster_self_weights_v.resize(number_of_clusters);
  GetClusterSelfWeights(
      handle, clusters, number_of_vertices, src_indices, dst_indices, weights,
      number_of_edges, cluster_self_weights_v.data().get(), number_of_clusters);
  SortClusters(handle, targeted_number_of_clusters, clusters,
               number_of_vertices, cluster_scores_v.data().get(),
               cluster_self_weights_v.data().get(), number_of_clusters);

  modularity = sfm::graph::detail::Modularity(handle, total_edge_weight,
                                              resolution, csr_graph, clusters);

  if (verbose) {
    std::cout << "cluster self-weights: ";
    for (const auto &weight : cluster_self_weights_v) {
      std::cout << weight << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "cluster scores: ";
    for (const auto &score : cluster_scores_v) {
      std::cout << score << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "modularity: " << modularity << std::endl << std::endl;
  }
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void EnhancedLouvain(const sfm::graph::Handle &handle,
                     const Score *vertex_scores, Size number_of_vertices,
                     Vertex *src_offsets, Vertex *src_indices,
                     Vertex *dst_indices, Weight *weights, Size number_of_edges,
                     Size targeted_number_of_clusters, Float resolution,
                     Score max_cluster_score, Float &modularity,
                     Vertex *clusters, Size &number_of_clusters,
                     thrust::device_vector<Score> &cluster_scores_v,
                     thrust::device_vector<Weight> &cluster_self_weights_v,
                     bool verbose) {
  Louvain(handle, vertex_scores, number_of_vertices, src_offsets, src_indices,
          dst_indices, weights, number_of_edges, targeted_number_of_clusters,
          resolution, max_cluster_score, modularity, clusters,
          number_of_clusters, cluster_scores_v, cluster_self_weights_v,
          verbose);

  if (number_of_clusters <= targeted_number_of_clusters) {
    return;
  }

  thrust::device_vector<Vertex> selected_vertices_v(number_of_vertices);
  thrust::device_vector<Vertex> selected_src_indices_v(number_of_edges);
  thrust::device_vector<Vertex> selected_dst_indices_v(number_of_edges);
  thrust::device_vector<Weight> selected_weights_v(number_of_edges);
  Size number_of_selected_vertices =
      thrust::copy_if(
          handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
          sfm::utils::MakeCountingIterator(number_of_vertices),
          selected_vertices_v.data(),
          [clusters, targeted_number_of_clusters] __device__(Vertex vertex) {
            return clusters[vertex] >= targeted_number_of_clusters;
          }) -
      selected_vertices_v.data();
  selected_vertices_v.resize(number_of_selected_vertices);

  thrust::device_vector<Vertex> selected_vertex_indices_v(number_of_vertices,
                                                          -1);

  thrust::device_vector<Score> selected_vertex_scores_v(
      number_of_selected_vertices);
  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_selected_vertices),
      [selected_vertices = selected_vertices_v.data().get(),
       selected_vertex_indices = selected_vertex_indices_v.data().get(),
       selected_vertex_scores = selected_vertex_scores_v.data().get(),
       vertex_scores] __device__(auto idx) {
        auto vertex = selected_vertices[idx];
        selected_vertex_scores[idx] = vertex_scores[vertex];
        selected_vertex_indices[vertex] = idx;
      });

  auto edge_begin =
      sfm::utils::MakeZipIterator(src_indices, dst_indices, weights);
  auto edge_end = edge_begin + number_of_edges;
  auto selected_edge_begin = sfm::utils::MakeZipIterator(
      selected_src_indices_v.data(), selected_dst_indices_v.data(),
      selected_weights_v.data());
  Size number_of_selected_edges =
      thrust::copy_if(
          handle.GetThrustPolicy(), edge_begin, edge_end, selected_edge_begin,
          [clusters, targeted_number_of_clusters] __device__(auto edge) {
            return clusters[thrust::get<0>(edge)] >=
                       targeted_number_of_clusters &&
                   clusters[thrust::get<1>(edge)] >=
                       targeted_number_of_clusters;
          }) -
      selected_edge_begin;
  selected_src_indices_v.resize(number_of_selected_edges);
  selected_dst_indices_v.resize(number_of_selected_edges);
  selected_weights_v.resize(number_of_selected_edges);

  thrust::for_each(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator(0),
      sfm::utils::MakeCountingIterator(number_of_selected_edges),
      [selected_src_indices = selected_src_indices_v.data().get(),
       selected_dst_indices = selected_dst_indices_v.data().get(),
       selected_vertex_indices =
           selected_vertex_indices_v.data().get()] __device__(auto idx) {
        auto src = selected_src_indices[idx];
        auto dst = selected_dst_indices[idx];
        selected_src_indices[idx] = selected_vertex_indices[src];
        selected_dst_indices[idx] = selected_vertex_indices[dst];
      });

  thrust::device_vector<Vertex> selected_src_offsets_v(
      number_of_selected_vertices + 1);

  sfm::graph::detail::COO2CSR(
      handle, number_of_selected_vertices, selected_src_indices_v.data().get(),
      selected_dst_indices_v.data().get(), selected_weights_v.data().get(),
      number_of_selected_edges, selected_src_offsets_v.data().get());

  Float total_edge_weight =
      thrust::reduce(handle.GetThrustPolicy(), weights,
                     weights + number_of_edges, Weight(0), cub::Sum());
  Float total_selected_edge_weight = thrust::transform_reduce(
      handle.GetThrustPolicy(), sfm::utils::MakeCountingIterator<Vertex>(0),
      sfm::utils::MakeCountingIterator<Vertex>(number_of_edges),
      [src_indices, dst_indices, weights, clusters,
       targeted_number_of_clusters] __device__(auto edge) -> Weight {
        return clusters[src_indices[edge]] >= targeted_number_of_clusters ||
                       clusters[dst_indices[edge]] >=
                           targeted_number_of_clusters
                   ? weights[edge]
                   : 0;
      },
      Weight(0), cub::Sum());

  // Correct the resolution to be consistent with the original graph when
  // computing the modularity
  Float corrected_resolution =
      resolution * (total_selected_edge_weight / total_edge_weight);

  thrust::device_vector<Vertex> selected_vertex_clusters_v(
      number_of_selected_vertices, 0);

  Float selected_modularity = 0;
  Size number_of_selected_vertex_clusters = 0;
  thrust::device_vector<Weight> selected_vertex_cluster_self_weights_v;
  thrust::device_vector<Score> selected_vertex_cluster_scores_v;
  Louvain(handle, selected_vertex_scores_v.data().get(),
          number_of_selected_vertices, selected_src_offsets_v.data().get(),
          selected_src_indices_v.data().get(),
          selected_dst_indices_v.data().get(), selected_weights_v.data().get(),
          number_of_selected_edges, 0, corrected_resolution, max_cluster_score,
          selected_modularity, selected_vertex_clusters_v.data().get(),
          number_of_selected_vertex_clusters, selected_vertex_cluster_scores_v,
          selected_vertex_cluster_self_weights_v, verbose);

  number_of_clusters =
      targeted_number_of_clusters + number_of_selected_vertex_clusters;

  thrust::for_each(
      handle.GetThrustPolicy(), selected_vertices_v.begin(),
      selected_vertices_v.end(),
      [selected_vertex_clusters = selected_vertex_clusters_v.data().get(),
       selected_vertex_indices = selected_vertex_indices_v.data().get(),
       clusters, targeted_number_of_clusters] __device__(auto vertex) {
        clusters[vertex] =
            selected_vertex_clusters[selected_vertex_indices[vertex]] +
            targeted_number_of_clusters;
      });

  cluster_scores_v.resize(number_of_clusters);
  cluster_self_weights_v.resize(number_of_clusters);
  thrust::copy(handle.GetThrustPolicy(),
               selected_vertex_cluster_scores_v.begin(),
               selected_vertex_cluster_scores_v.end(),
               cluster_scores_v.begin() + targeted_number_of_clusters);
  thrust::copy(handle.GetThrustPolicy(),
               selected_vertex_cluster_self_weights_v.begin(),
               selected_vertex_cluster_self_weights_v.end(),
               cluster_self_weights_v.begin() + targeted_number_of_clusters);

  sfm::graph::GraphCSRView<Vertex, Size, Weight> csr_graph(
      src_offsets, dst_indices, weights, number_of_vertices, number_of_edges);

  modularity = sfm::graph::detail::Modularity(handle, total_edge_weight,
                                              resolution, csr_graph, clusters);

  if (verbose) {
    std::cout << "cluster self-weights: ";
    for (const auto &weight : cluster_self_weights_v) {
      std::cout << weight << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "cluster scores: ";
    for (const auto &score : cluster_scores_v) {
      std::cout << score << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "modularity: " << modularity << std::endl << std::endl;
  }
}

template <typename T, typename Vertex, typename Weight>
void LoadGraph(const sfm::graph::Handle &handle, const BADataset<T> &dataset,
               thrust::device_vector<Vertex> &src_indices_v,
               thrust::device_vector<Vertex> &dst_indices_v,
               thrust::device_vector<Weight> &weights_v) {
  int number_of_cameras = dataset.NumberOfExtrinsics();
  int number_of_points = dataset.NumberOfPoints();
  int number_of_measurements = dataset.Measurements().size();
  int number_of_edges = 2 * number_of_measurements;

  std::vector<Vertex> h_src_indices_v;
  std::vector<Vertex> h_dst_indices_v;
  std::vector<Weight> h_weights_v;
  h_src_indices_v.reserve(number_of_edges);
  h_dst_indices_v.reserve(number_of_edges);
  h_weights_v.reserve(number_of_edges);
  for (const auto &measurement : dataset.Measurements()) {
    if (measurement.extrinsics_index != measurement.intrinsics_index) {
      LOG(ERROR) << "The extrinsics and intrinsics indices must be the same "
                    "for all the measurements."
                 << std::endl;
      exit(-1);
    }

    h_src_indices_v.push_back(measurement.extrinsics_index);
    h_dst_indices_v.push_back(measurement.point_index + number_of_cameras);
    h_weights_v.push_back(measurement.sqrt_weight);
  }

  h_src_indices_v.insert(h_src_indices_v.end(), h_dst_indices_v.begin(),
                         h_dst_indices_v.begin() + number_of_measurements);
  h_dst_indices_v.insert(h_dst_indices_v.end(), h_src_indices_v.begin(),
                         h_src_indices_v.begin() + number_of_measurements);
  h_weights_v.insert(h_weights_v.end(), h_weights_v.begin(),
                     h_weights_v.begin() + number_of_measurements);

  src_indices_v.resize(number_of_edges);
  dst_indices_v.resize(number_of_edges);
  weights_v.resize(number_of_edges);

  cudaMemcpy(src_indices_v.data().get(), h_src_indices_v.data(),
             sizeof(Vertex) * number_of_edges, cudaMemcpyHostToDevice);
  cudaMemcpy(dst_indices_v.data().get(), h_dst_indices_v.data(),
             sizeof(Vertex) * number_of_edges, cudaMemcpyHostToDevice);
  cudaMemcpy(weights_v.data().get(), h_weights_v.data(),
             sizeof(Weight) * number_of_edges, cudaMemcpyHostToDevice);

#if 0
  thrust::sort(handle.GetThrustPolicy(), weights_v.data(),
               weights_v.data() + number_of_measurements);
  T l_weight = weights_v[int(0.1 * number_of_measurements)];
  T u_weight = weights_v[int(0.9 * number_of_measurements)];
  thrust::copy(handle.GetThrustPolicy(),
               weights_v.data() + number_of_measurements,
               weights_v.data() + number_of_edges, weights_v.data());
  thrust::for_each_n(handle.GetThrustPolicy(),
                     sfm::utils::MakeCountingIterator(0), number_of_edges,
                     [weights = weights_v.data().get(), l_weight,
                      u_weight] __device__(auto idx) {
                       T weight = max(l_weight, min((T)weights[idx], u_weight));
                       weights[idx] = 100.0 * weight / l_weight + 0.5;
                     });
#else
  thrust::fill(handle.GetThrustPolicy(), weights_v.begin(), weights_v.end(), 1);
#endif
}

template void SimplifyClusters<int_t, int_t>(const sfm::graph::Handle &handle,
                                             int_t *clusters,
                                             int_t number_of_vertices,
                                             int_t &number_of_clusters);

template void SimplifyClusters<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t *clusters, int_t number_of_vertices,
    uint64_t *cluster_scores, int_t &number_of_clusters);

template void SimplifyClusterIndices<int_t, int_t>(
    const sfm::graph::Handle &handle, const int_t *cluster_indices,
    int_t number_of_clusters, int_t *simplified_cluster_indices,
    int_t &number_of_simplified_clusters);

template void SimplifyClusterIndicesAndScores<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, const int_t *cluster_indices,
    const uint64_t *cluster_scores, int_t number_of_clusters,
    int_t *simplified_cluster_indices, uint64_t *simplified_cluster_scores,
    int_t &number_of_simplified_clusters);

template void SortClusters<int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t targeted_number_of_clusters,
    int_t *clusters, int_t number_of_vertices, uint64_t *cluster_scores,
    uint64_t *cluster_self_weights, int_t number_of_clusters);

template void MergeZeroScoreVertices<int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t *clusters,
    const uint64_t *vertex_scores, int_t number_of_vertices,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t number_of_edges);

template void GetSelfWeights<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, const int_t *src_indices,
    const int_t *dst_indices, const uint64_t *weights, int_t number_of_edges,
    uint64_t *self_weights, int_t number_of_vertices);

template void GetClusterSelfWeights<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, const int_t *clusters,
    int_t number_of_vertices, const int_t *src_indices,
    const int_t *dst_indices, const uint64_t *weights, int_t number_of_edges,
    uint64_t *cluster_self_weights, int_t number_of_clusters);

template void UpdateClusterSelfWeights<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, const int_t *clusters,
    int_t number_of_vertices, const int_t *src_indices,
    const int_t *dst_indices, const uint64_t *weights, int_t number_of_edges,
    uint64_t *cluster_self_weights, std::array<int_t, 2> updated_clusters,
    int_t number_of_clusters);

template void MergeClusters<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t max_level, float resolution,
    int_t targeted_number_of_clusters, uint64_t max_cluster_score,
    const uint64_t *cluster_scores, const int_t *cluster_src_offsets,
    const int_t *cluster_dst_indices, const uint64_t *cluster_weights,
    int_t number_of_clusters, int_t number_of_cluster_edges,
    int_t *merged_cluster_indices, uint64_t *merged_cluster_scores,
    int_t &number_of_merged_clusters);

template void MergeClusters<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t max_level, double resolution,
    int_t targeted_number_of_clusters, uint64_t max_cluster_score,
    const uint64_t *cluster_scores, const int_t *cluster_src_offsets,
    const int_t *cluster_dst_indices, const uint64_t *cluster_weights,
    int_t number_of_clusters, int_t number_of_cluster_edges,
    int_t *merged_cluster_indices, uint64_t *merged_cluster_scores,
    int_t &number_of_merged_clusters);

template void MergeSortedLargeClusters<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t targeted_number_of_clusters,
    uint64_t max_cluster_score, const uint64_t *cluster_scores,
    int_t number_of_clusters, int_t *merged_cluster_indices,
    uint64_t *merged_cluster_scores, int_t &number_of_merged_clusters);

template void MergeSortedSmallClusters<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t targeted_number_of_clusters,
    uint64_t max_cluster_score, const uint64_t *cluster_scores,
    int_t number_of_clusters, int_t *merged_cluster_indices,
    uint64_t *merged_cluster_scores, int_t &number_of_merged_clusters);

template void
DecomposeAndMergeLargeCluster<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t cluster_out, const int_t *clusters,
    const uint64_t *vertex_uint64_ts, int_t number_of_vertices,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t number_of_edges, const uint64_t *cluster_self_weights,
    const uint64_t *cluster_uint64_ts, int_t number_of_clusters,
    int_t targeted_number_of_clusters, float resolution,
    uint64_t max_cluster_score, int_t *merged_vertices, int_t &cluster_in,
    int_t &number_of_merged_vertices);

template void
DecomposeAndMergeLargeCluster<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t cluster_out, const int_t *clusters,
    const uint64_t *vertex_uint64_ts, int_t number_of_vertices,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t number_of_edges, const uint64_t *cluster_self_weights,
    const uint64_t *cluster_uint64_ts, int_t number_of_clusters,
    int_t targeted_number_of_clusters, double resolution,
    uint64_t max_cluster_score, int_t *merged_vertices, int_t &cluster_in,
    int_t &number_of_merged_vertices);

template void MergeLargeCluster<int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t cluster_out, const int_t *clusters,
    const uint64_t *vertex_scores, int_t number_of_vertices,
    const int_t *src_indices, const int_t *dst_indices, const uint64_t *weights,
    int_t number_of_edges, const uint64_t *cluster_scores,
    int_t number_of_clusters, int_t targeted_number_of_clusters,
    uint64_t max_cluster_score, int_t &cluster_in);

template bool
MergeZeroClusterSelfWeightClusters<int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *cluster_scores,
    const uint64_t *cluster_self_weights, int_t number_of_clusters,
    const int_t *cluster_src_offsets, const int_t *cluster_src_indices,
    const int_t *cluster_dst_indices, const uint64_t *cluster_weights,
    int_t number_of_cluster_edges, int_t targeted_number_of_clusters,
    uint64_t max_cluster_score, int_t *merged_cluster_indices,
    uint64_t *merged_cluster_scores, uint64_t *merged_cluster_self_weights,
    int_t &merged_number_of_clusters);

template bool RemergeCluster<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *cluster_scores,
    const uint64_t *cluster_self_weights, int_t number_of_clusters,
    const int_t *cluster_src_offsets, const int_t *cluster_src_indices,
    const int_t *cluster_dst_indices, const uint64_t *cluster_weights,
    int_t number_of_cluster_edges, float resolution,
    int_t targeted_number_of_clusters, uint64_t max_cluster_score,
    int_t *merged_cluster_indices,
    thrust::device_vector<uint64_t> &merged_cluster_scores_v,
    thrust::device_vector<uint64_t> &merged_cluster_self_weights_v,
    int_t &number_of_merged_clusters);

template bool RemergeCluster<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *cluster_scores,
    const uint64_t *cluster_self_weights, int_t number_of_clusters,
    const int_t *cluster_src_offsets, const int_t *cluster_src_indices,
    const int_t *cluster_dst_indices, const uint64_t *cluster_weights,
    int_t number_of_cluster_edges, double resolution,
    int_t targeted_number_of_clusters, uint64_t max_cluster_score,
    int_t *merged_cluster_indices,
    thrust::device_vector<uint64_t> &merged_cluster_scores_v,
    thrust::device_vector<uint64_t> &merged_cluster_self_weights_v,
    int_t &number_of_merged_clusters);

template void ComputeDeltaModularity<int_t, float, int_t, uint64_t, uint64_t>(
    sfm::graph::Handle const &handle, float total_edge_weight, float resolution,
    int_t number_of_vertices, int_t number_of_edges,
    thrust::device_vector<int_t> const &offsets_v,
    thrust::device_vector<int_t> const &src_indices_v,
    thrust::device_vector<int_t> const &dst_indices_v,
    thrust::device_vector<uint64_t> const &weights_v,
    thrust::device_vector<uint64_t> const &vertex_weights_v,
    thrust::device_vector<uint64_t> const &cluster_weights_v,
    thrust::device_vector<uint64_t> const &vertex_scores_v,
    thrust::device_vector<uint64_t> const &cluster_scores_v,
    sfm::graph::Dendrogram<int_t> const &dendrogram,
    thrust::device_vector<int_t> &cluster_hash_v,
    thrust::device_vector<uint64_t> &old_cluster_sum_v,
    thrust::device_vector<uint64_t> &new_cluster_sum_v,
    thrust::device_vector<float> &delta_weight_v,
    int_t targeted_number_of_clusters, uint64_t max_cluster_socre);

template void ComputeDeltaModularity<int_t, double, int_t, uint64_t, uint64_t>(
    sfm::graph::Handle const &handle, double total_edge_weight,
    double resolution, int_t number_of_vertices, int_t number_of_edges,
    thrust::device_vector<int_t> const &offsets_v,
    thrust::device_vector<int_t> const &src_indices_v,
    thrust::device_vector<int_t> const &dst_indices_v,
    thrust::device_vector<uint64_t> const &weights_v,
    thrust::device_vector<uint64_t> const &vertex_weights_v,
    thrust::device_vector<uint64_t> const &cluster_weights_v,
    thrust::device_vector<uint64_t> const &vertex_scores_v,
    thrust::device_vector<uint64_t> const &cluster_scores_v,
    sfm::graph::Dendrogram<int_t> const &dendrogram,
    thrust::device_vector<int_t> &cluster_hash_v,
    thrust::device_vector<uint64_t> &old_cluster_sum_v,
    thrust::device_vector<uint64_t> &new_cluster_sum_v,
    thrust::device_vector<double> &delta_weight_v,
    int_t targeted_number_of_clusters, uint64_t max_cluster_socre);

template void Louvain<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *vertex_scores,
    int_t number_of_vertices, int_t *src_offsets, int_t *src_indices,
    int_t *dst_indices, uint64_t *weights, int_t number_of_edges,
    int_t targeted_number_of_clusters, float resolution,
    uint64_t max_cluster_score, float &modularity, int_t *clusters,
    int_t &number_of_clusters,
    thrust::device_vector<uint64_t> &cluster_scores_v,
    thrust::device_vector<uint64_t> &cluster_self_weights_v, bool verbose);

template void Louvain<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *vertex_scores,
    int_t number_of_vertices, int_t *src_offsets, int_t *src_indices,
    int_t *dst_indices, uint64_t *weights, int_t number_of_edges,
    int_t targeted_number_of_clusters, double resolution,
    uint64_t max_cluster_score, double &modularity, int_t *clusters,
    int_t &number_of_clusters,
    thrust::device_vector<uint64_t> &cluster_scores_v,
    thrust::device_vector<uint64_t> &cluster_self_weights_v, bool verbose);

template void EnhancedLouvain<float, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *vertex_scores,
    int_t number_of_vertices, int_t *src_offsets, int_t *src_indices,
    int_t *dst_indices, uint64_t *weights, int_t number_of_edges,
    int_t targeted_number_of_clusters, float resolution,
    uint64_t max_cluster_score, float &modularity, int_t *clusters,
    int_t &number_of_clusters,
    thrust::device_vector<uint64_t> &cluster_scores_v,
    thrust::device_vector<uint64_t> &cluster_self_weights_v, bool verbose);

template void EnhancedLouvain<double, int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, const uint64_t *vertex_scores,
    int_t number_of_vertices, int_t *src_offsets, int_t *src_indices,
    int_t *dst_indices, uint64_t *weights, int_t number_of_edges,
    int_t targeted_number_of_clusters, double resolution,
    uint64_t max_cluster_score, double &modularity, int_t *clusters,
    int_t &number_of_clusters,
    thrust::device_vector<uint64_t> &cluster_scores_v,
    thrust::device_vector<uint64_t> &cluster_self_weights_v, bool verbose);

template void GetClusterInfo<int_t, int_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t number_of_clusters,
    const int_t *clusters, int_t number_of_vertices, const int_t *src_indices,
    const int_t *dst_indices, const uint64_t *weights, int_t number_of_edges,
    thrust::device_vector<int_t> &cluster_src_offsets_v,
    thrust::device_vector<int_t> &cluster_src_indices_v,
    thrust::device_vector<int_t> &cluster_dst_indices_v,
    thrust::device_vector<uint64_t> &cluster_weights_v,
    int_t &number_of_cluster_edges);

template void ReclusterGraph<int_t, int_t, uint64_t, uint64_t>(
    const sfm::graph::Handle &handle, int_t targeted_number_of_clusters,
    sfm::graph::Dendrogram<int_t> &dendrogram, int_t &number_of_vertices,
    thrust::device_vector<uint64_t> &vertex_scores_v,
    thrust::device_vector<int_t> &src_offsets_v,
    thrust::device_vector<int_t> &src_indices_v,
    thrust::device_vector<int_t> &dst_indices_v,
    thrust::device_vector<uint64_t> &weights_v, int_t &number_of_edges);

template void
LoadGraph<float, int_t, uint64_t>(const sfm::graph::Handle &handle,
                                  const BADataset<float> &dataset,
                                  thrust::device_vector<int_t> &src_indices_v,
                                  thrust::device_vector<int_t> &dst_indices_v,
                                  thrust::device_vector<uint64_t> &weights_v);

template void
LoadGraph<double, int_t, uint64_t>(const sfm::graph::Handle &handle,
                                   const BADataset<double> &dataset,
                                   thrust::device_vector<int_t> &src_indices_v,
                                   thrust::device_vector<int_t> &dst_indices_v,
                                   thrust::device_vector<uint64_t> &weights_v);
} // namespace clustering
} // namespace ba
} // namespace sfm