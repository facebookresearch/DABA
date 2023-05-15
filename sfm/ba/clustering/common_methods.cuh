// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdlib>
#include <limits>
#include <sfm/ba/dataset.h>
#include <sfm/graph/graph.cuh>
#include <sfm/utils/utils.cuh>

namespace sfm {
namespace ba {
namespace clustering {
template <typename Size, typename Vertex>
void SimplifyClusters(const sfm::graph::Handle &handle, Vertex *clusters,
                      Size number_of_vertices, Size &number_of_clusters);

template <typename Size, typename Vertex, typename Score>
void SimplifyClusters(const sfm::graph::Handle &handle, Vertex *clusters,
                      Size number_of_vertices, Score *cluster_scores,
                      Size &number_of_clusters);

template <typename Size, typename Vertex>
void SimplifyClusterIndices(const sfm::graph::Handle &handle,
                            const Vertex *cluster_indices,
                            Size number_of_clusters,
                            Vertex *simplified_cluster_indices,
                            Size &number_of_simplified_clusters);

template <typename Size, typename Vertex, typename Score>
void SimplifyClusterIndicesAndScores(const sfm::graph::Handle &handle,
                                     const Vertex *cluster_indices,
                                     const Score *cluster_scores,
                                     Size number_of_clusters,
                                     Vertex *simplified_cluster_indices,
                                     Score *simplified_cluster_scores,
                                     Size &number_of_simplified_clusters);

template <typename Size, typename Vertex, typename Weight, typename Score>
void SortClusters(const sfm::graph::Handle &handle,
                  Size targeted_number_of_clusters, Vertex *clusters,
                  Size number_of_vertices, Score *cluster_scores,
                  Weight *self_cluster_weights, Size number_of_clusters);

template <typename Size, typename Vertex, typename Weight, typename Score>
void MergeZeroScoreVertices(const sfm::graph::Handle &handle, Vertex *clusters,
                            const Score *vertex_scores, Size number_of_vertices,
                            const Vertex *src_indices,
                            const Vertex *dst_indices, const Weight *weights,
                            Size number_of_edges);

template <typename Size, typename Vertex, typename Weight>
void GetSelfWeights(const sfm::graph::Handle &handle, const Vertex *src_indices,
                    const Vertex *dst_indices, const Weight *weights,
                    Size number_of_edges, Weight *self_weights,
                    Size number_of_vertices);

template <typename Size, typename Vertex, typename Weight>
void GetClusterSelfWeights(const sfm::graph::Handle &handle,
                           const Vertex *clusters, Size number_of_vertices,
                           const Vertex *src_indices, const Vertex *dst_indices,
                           const Weight *weights, Size number_of_edges,
                           Weight *self_cluster_weights,
                           Size number_of_clusters);

template <typename Size, typename Vertex, typename Weight>
void UpdateClusterSelfWeights(const sfm::graph::Handle &handle,
                              const Vertex *clusters, Size number_of_vertices,
                              const Vertex *src_indices,
                              const Vertex *dst_indices, const Weight *weights,
                              Size number_of_edges,
                              Weight *cluster_self_weights,
                              std::array<Vertex, 2> updated_clusters,
                              Size number_of_clusters);

template <typename Size, typename Vertex, typename Weight>
void GetClusterInfo(const sfm::graph::Handle &handle, Size number_of_clusters,
                    const Vertex *clusters, Size number_of_vertices,
                    const Vertex *src_indices, const Vertex *dst_indices,
                    const Weight *weights, Size number_of_edges,
                    thrust::device_vector<Vertex> &cluster_src_offsets,
                    thrust::device_vector<Vertex> &cluster_src_indices,
                    thrust::device_vector<Vertex> &cluster_dst_indices,
                    thrust::device_vector<Weight> &cluster_weights,
                    Size &number_of_cluster_edges);

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
                   Size &number_of_merged_clusters);

template <typename Size, typename Vertex, typename Score>
void MergeSortedLargeClusters(
    const sfm::graph::Handle &handle, Size targeted_number_of_clusters,
    Score max_cluster_score, const Score *cluster_scores,
    Size number_of_clusters, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Size &number_of_merged_clusters);

template <typename Size, typename Vertex, typename Score>
void MergeSortedSmallClusters(
    const sfm::graph::Handle &handle, Size targeted_number_of_clusters,
    Score max_cluster_score, const Score *cluster_scores,
    Size number_of_clusters, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Size &number_of_merged_clusters);

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
    Size &number_of_merged_vertices);

template <typename Size, typename Vertex, typename Weight, typename Score>
void MergeLargeCluster(const sfm::graph::Handle &handle, Vertex cluster_out,
                       const Vertex *clusters, const Score *vertex_scores,
                       Size number_of_vertices, const Vertex *src_indices,
                       const Vertex *dst_indices, const Weight *weights,
                       Size number_of_edges, const Score *cluster_scores,
                       Size number_of_clusters,
                       Size targeted_number_of_clusters,
                       Score max_cluster_score, Vertex &cluster_in);

template <typename Size, typename Vertex, typename Weight, typename Score>
bool MergeZeroClusterSelfWeightClusters(
    const sfm::graph::Handle &handle, const Score *cluster_scores,
    const Weight *cluster_self_weights, Size number_of_clusters,
    const Vertex *cluster_src_offsets, const Vertex *cluster_src_indices,
    const Vertex *cluster_dst_indices, const Weight *cluster_weights,
    Size number_of_cluster_edges, Size targeted_number_of_clusters,
    Score max_cluster_score, Vertex *merged_cluster_indices,
    Score *merged_cluster_scores, Weight *merged_cluster_self_weights,
    Size &merged_number_of_clusters);

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
    Size &number_of_merged_clusters);

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
    Score max_cluster_socre);

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
             bool verbose = false);

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
                     bool verbose);

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
                    Size &number_of_edges);

template <typename T, typename Vertex, typename Weight>
void LoadGraph(const sfm::graph::Handle &handle, const BADataset<T> &dataset,
               thrust::device_vector<Vertex> &src_indices_v,
               thrust::device_vector<Vertex> &dst_indices_v,
               thrust::device_vector<Weight> &weights_v);
} // namespace clustering
} // namespace ba
} // namespace sfm
