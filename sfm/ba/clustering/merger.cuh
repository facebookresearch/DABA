// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/ba/dataset.h>
#include <sfm/graph/graph.cuh>
#include <sfm/graph/types.hpp>
#include <sfm/utils/utils.cuh>

namespace sfm {
namespace ba {
namespace clustering {
template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
class Merger {
public:
  using Graph = sfm::graph::GraphCSRView<Vertex, Vertex, Weight>;

  Merger(sfm::graph::Handle const &handle)
      : m_handle(handle),
        m_dendrogram(std::make_unique<sfm::graph::Dendrogram<Vertex>>()),
        m_src_offsets_v(1), m_src_indices_v(0), m_dst_indices_v(0),
        m_weights_v(0), m_vertex_weights_v(0), m_cluster_weights_v(0),
        m_tmp_arr_v(0), m_cluster_inverse_v(0), m_number_of_vertices(0),
        m_number_of_edges(0), m_number_of_clusters(0),
        m_number_of_cluster_edges(0), m_vertex_scores_v(0),
        m_cluster_scores_v(0), m_max_cluster_socre(std::nullopt) {}

  Merger(Merger<Float, Size, Vertex, Weight, Score> const &) = delete;
  Merger<Float, Size, Vertex, Weight, Score> &
  operator=(Merger<Float, Size, Vertex, Weight, Score> const &) = delete;

  virtual ~Merger() {}

  sfm::graph::Dendrogram<Vertex> const &GetDendrogram() const {
    return *m_dendrogram;
  }

  Size GetNumberOfVertices() const { return m_number_of_vertices; }

  Size GetNumberOfEdges() const { return m_number_of_edges; }

  Size GetNumberOfClusters() const { return m_number_of_clusters; }

  Size GetNumberOfClusterEdges() const { return m_number_of_cluster_edges; }

  std::optional<Score> GetMaxClusterSocre() const {
    return m_max_cluster_socre;
  }

  thrust::device_vector<Score> const &GetClusterScores() const {
    return m_cluster_scores_v;
  }

  Float Merge(const Vertex *src_offsets, const Vertex *dst_indices,
              const Weight *weights, Size number_of_vertices,
              Size number_of_edges, Size max_level, Float resolution,
              Vertex *clustering, Size targeted_number_of_clusters,
              bool initialized) {
    GRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a "
                  "device pointer to "
                  "memory for storing the result");

    Setup(src_offsets, dst_indices, weights, number_of_vertices,
          number_of_edges, targeted_number_of_clusters, std::nullopt);

    thrust::fill(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), 0);

    thrust::copy(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), m_cluster_scores_v.begin());

    Float best_modularity =
        Solve(max_level, resolution, initialized ? clustering : nullptr);
    GetClustering(clustering);

    return best_modularity;
  }

  Float Merge(const Vertex *src_offsets, const Vertex *dst_indices,
              const Weight *weights, Size number_of_vertices,
              Size number_of_edges, Size max_level, Float resolution,
              Vertex *clustering, const Score *vertex_scores,
              Size targeted_number_of_clusters,
              std::optional<Score> max_cluster_socre = std::nullopt,
              bool initialized = false) {
    GRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a "
                  "device pointer to "
                  "memory for storing the result");

    Setup(src_offsets, dst_indices, weights, number_of_vertices,
          number_of_edges, targeted_number_of_clusters, max_cluster_socre);

    thrust::copy(m_handle.GetThrustPolicy(), vertex_scores,
                 vertex_scores + number_of_vertices, m_vertex_scores_v.begin());

    thrust::copy(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), m_cluster_scores_v.begin());

    Float best_modularity =
        Solve(max_level, resolution, initialized ? clustering : nullptr);
    GetClustering(clustering);

    return best_modularity;
  }

protected:
  virtual void Setup(const Vertex *src_offsets, const Vertex *dst_indices,
                     const Weight *weights, Size number_of_vertices,
                     Size number_of_edges, Size targeted_number_of_clusters,
                     std::optional<Score> max_cluster_socre);

  virtual Float Solve(size_t max_level, Float resolution,
                      Float total_edge_weight, Vertex *initial_clustering);

  virtual Float Solve(size_t max_level, Float resolution,
                      Vertex *initial_clustering);

  virtual void InitializeDendrogramLevel(Vertex num_vertices);

  virtual Float UpdateClustering(Float total_edge_weight, Float resolution,
                                 Graph const &graph);

  virtual void ShrinkGraph(Graph &graph);

  virtual void GetClustering(Vertex *clustering);

protected:
  sfm::graph::Handle const &m_handle;
  Size m_number_of_vertices;
  Size m_number_of_edges;
  Size m_number_of_clusters;
  Size m_number_of_cluster_edges;
  Size m_targeted_number_of_clusters;

  std::unique_ptr<sfm::graph::Dendrogram<Vertex>> m_dendrogram;

  //
  //  Copy of graph
  //
  thrust::device_vector<Vertex> m_src_offsets_v;
  thrust::device_vector<Vertex> m_src_indices_v;
  thrust::device_vector<Vertex> m_dst_indices_v;
  thrust::device_vector<Weight> m_weights_v;

  //
  //  Weights and clustering across iterations of algorithm
  //
  thrust::device_vector<Weight> m_vertex_weights_v;
  thrust::device_vector<Weight> m_cluster_weights_v;

  //
  //  Temporaries used within kernels.  Each iteration uses less
  //  of this memory
  //
  thrust::device_vector<Vertex> m_tmp_arr_v;
  thrust::device_vector<Vertex> m_cluster_inverse_v;

  //
  // Community scores
  //
  std::optional<Score> m_max_cluster_socre;
  thrust::device_vector<Score> m_vertex_scores_v;
  thrust::device_vector<Score> m_cluster_scores_v;
};
} // namespace clustering
} // namespace ba
} // namespace sfm