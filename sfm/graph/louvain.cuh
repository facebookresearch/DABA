// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <limits>
#include <optional>

#include <sfm/graph/converters/COOtoCSR.cuh>
#include <sfm/graph/types.hpp>
#include <sfm/graph/utilities/graph_utils.cuh>

#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace sfm {
namespace graph {
template <typename GraphType, typename Float = float, typename Score = float>
class Louvain {
public:
  using Graph = GraphType;
  using Vertex = typename GraphType::VertexType;
  using Edge = typename GraphType::EdgeType;
  using Weight = typename GraphType::WeightType;

  Louvain(Handle const &handle)
      : m_handle(handle), m_dendrogram(std::make_unique<Dendrogram<Vertex>>()),
        m_src_offsets_v(1), m_src_indices_v(0), m_dst_indices_v(0),
        m_weights_v(0), m_vertex_weights_v(0), m_cluster_weights_v(0),
        m_tmp_arr_v(0), m_cluster_inverse_v(0), m_number_of_vertices(0),
        m_number_of_edges(0), m_number_of_clusters(0),
        m_number_of_cluster_edges(0), m_vertex_scores_v(0),
        m_cluster_scores_v(0), m_max_cluster_socre(std::nullopt) {}

  Louvain(Louvain<Graph, Float, Score> const &) = delete;
  Louvain<Graph, Float, Score> &
  operator=(Louvain<Graph, Float, Score> const &) = delete;

  virtual ~Louvain() {}

  Dendrogram<Vertex> const &GetDendrogram() const { return *m_dendrogram; }

  Vertex GetNumberOfVertices() const { return m_number_of_vertices; }

  Edge GetNumberOfEdges() const { return m_number_of_edges; }

  Vertex GetNumberOfClusters() const { return m_number_of_clusters; }

  Edge GetNumberOfClusterEdges() const { return m_number_of_cluster_edges; }

  void GetClusterEdges(thrust::device_vector<Vertex> &cluster_src_indices_v,
                       thrust::device_vector<Vertex> &cluster_dst_indices_v,
                       thrust::device_vector<Weight> &cluster_weights_v) {
    cluster_src_indices_v.resize(m_number_of_cluster_edges);
    cluster_dst_indices_v.resize(m_number_of_cluster_edges);
    cluster_weights_v.resize(m_number_of_cluster_edges);
    GetClusterEdges(cluster_src_indices_v.data().get(),
                    cluster_dst_indices_v.data().get(),
                    cluster_weights_v.data().get());
  }

  std::optional<Score> GetMaxClusterSocre() const {
    return m_max_cluster_socre;
  }

  thrust::device_vector<Score> const &GetClusterScores() const {
    return m_cluster_scores_v;
  }

  Float Cluster(Graph const &graph, size_t max_level, Float resolution,
                Vertex *clustering, bool initialized) {
    GRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a "
                  "device pointer to "
                  "memory for storing the result");

    Setup(graph, std::nullopt);

    thrust::fill(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), 0);

    thrust::copy(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), m_cluster_scores_v.begin());

    Float best_modularity =
        Solve(max_level, resolution, initialized ? clustering : nullptr);
    GetClustering(clustering);

    return best_modularity;
  }

  template <typename score_function = ZeroScoreFunction<Vertex, Score>>
  Float Cluster(Graph const &graph, size_t max_level, Float resolution,
                Vertex *clustering, score_function vertex_score_func,
                std::optional<Score> max_cluster_socre = std::nullopt,
                bool initialized = false) {
    GRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a "
                  "device pointer to "
                  "memory for storing the result");

    Setup(graph, max_cluster_socre);

    thrust::transform(
        m_handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
        thrust::make_counting_iterator<Vertex>(graph.number_of_vertices),
        m_vertex_scores_v.begin(), vertex_score_func);

    thrust::copy(m_handle.GetThrustPolicy(), m_vertex_scores_v.begin(),
                 m_vertex_scores_v.end(), m_cluster_scores_v.begin());

    Float best_modularity =
        Solve(max_level, resolution, initialized ? clustering : nullptr);
    GetClustering(clustering);

    return best_modularity;
  }

protected:
  virtual void Setup(Graph const &graph,
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

  virtual void GetClusterEdges(Vertex *cluster_src_indices,
                               Vertex *cluster_dst_indices,
                               Weight *cluster_weights);

protected:
  Handle const &m_handle;
  Vertex m_number_of_vertices;
  Edge m_number_of_edges;
  Vertex m_number_of_clusters;
  Edge m_number_of_cluster_edges;

  std::unique_ptr<Dendrogram<Vertex>> m_dendrogram;

  //
  //  Copy of graph
  //
  thrust::device_vector<Edge> m_src_offsets_v;
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

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

template <typename Vertex, typename Edge, typename Weight, typename Float>
std::pair<size_t, Float> LouvainClustering(
    Handle const &handle, GraphCSRView<Vertex, Edge, Weight> const &graph_view,
    Vertex *clustering, size_t &cluster_size, size_t max_level = 100,
    Float resolution = 1, bool initialized = false);
} // namespace graph
} // namespace sfm
