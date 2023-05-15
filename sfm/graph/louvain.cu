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

#include <cstdint>

#include <sfm/graph/detail/common_methods.cuh>
#include <sfm/graph/louvain.cuh>
#include <sfm/graph/utilities/error.hpp>

#include <optional>
#include <sys/types.h>

namespace sfm {
namespace graph {
template <typename GraphType, typename Float, typename Score>
void Louvain<GraphType, Float, Score>::Setup(
    GraphType const &graph, std::optional<Score> max_cluster_socre) {
  m_dendrogram = std::make_unique<Dendrogram<Vertex>>();
  m_src_offsets_v.resize(graph.number_of_vertices + 1);
  m_dst_indices_v.resize(graph.number_of_edges);
  m_weights_v.resize(graph.number_of_edges);
  m_src_indices_v.resize(graph.number_of_edges);
  m_vertex_weights_v.resize(graph.number_of_vertices);
  m_cluster_weights_v.resize(graph.number_of_vertices);
  m_tmp_arr_v.resize(graph.number_of_vertices);
  m_cluster_inverse_v.resize(graph.number_of_vertices);
  m_number_of_vertices = graph.number_of_vertices;
  m_number_of_edges = graph.number_of_edges;
  m_number_of_clusters = graph.number_of_vertices;
  m_number_of_cluster_edges = graph.number_of_edges;
  m_vertex_scores_v.resize(graph.number_of_vertices);
  m_cluster_scores_v.resize(graph.number_of_vertices);
  m_max_cluster_socre = max_cluster_socre;

  thrust::copy(m_handle.GetThrustPolicy(), graph.offsets,
               graph.offsets + graph.number_of_vertices + 1,
               m_src_offsets_v.begin());

  thrust::copy(m_handle.GetThrustPolicy(), graph.indices,
               graph.indices + graph.number_of_edges, m_dst_indices_v.begin());

  thrust::copy(m_handle.GetThrustPolicy(), graph.edge_data,
               graph.edge_data + graph.number_of_edges, m_weights_v.begin());
}

template <typename GraphType, typename Float, typename Score>
Float Louvain<GraphType, Float, Score>::Solve(size_t max_level,
                                              Float resolution,
                                              Float total_edge_weight,
                                              Vertex *initial_clustering) {
  Float best_modularity = Float{-1};

  //
  //  Our copy of the graph.  Each iteration of the outer loop will
  //  shrink this copy of the graph.
  //
  GraphCSRView<Vertex, Edge, Weight> current_graph(
      m_src_offsets_v.data().get(), m_dst_indices_v.data().get(),
      m_weights_v.data().get(), m_number_of_vertices, m_number_of_edges);

  current_graph.GetSourceIndices(m_src_indices_v.data().get());

  if (initial_clustering != nullptr) {
    InitializeDendrogramLevel(current_graph.number_of_vertices);
    thrust::copy(m_handle.GetThrustPolicy(), initial_clustering,
                 initial_clustering + current_graph.number_of_vertices,
                 m_dendrogram->CurrentLevelBegin());
    ShrinkGraph(current_graph);
  }

  while (m_dendrogram->NumLevels() < max_level) {
    //
    //  Initialize every cluster to reference each vertex to itself
    //
    InitializeDendrogramLevel(current_graph.number_of_vertices);

    detail::ComputeVertexAndClusterWeights(m_handle, current_graph,
                                           m_vertex_weights_v.data().get(),
                                           m_cluster_weights_v.data().get());

    Float new_Q =
        UpdateClustering(total_edge_weight, resolution, current_graph);

    ShrinkGraph(current_graph);

    if (new_Q <= best_modularity) {
      break;
    }

    best_modularity = new_Q;
  }

  return best_modularity;
}

template <typename GraphType, typename Float, typename Score>
Float Louvain<GraphType, Float, Score>::Solve(size_t max_level,
                                              Float resolution,
                                              Vertex *initial_clustering) {
  Float total_edge_weight = thrust::reduce(
      m_handle.GetThrustPolicy(), m_weights_v.begin(), m_weights_v.end());

  return Solve(max_level, resolution, total_edge_weight, initial_clustering);
}

template <typename GraphType, typename Float, typename Score>
void Louvain<GraphType, Float, Score>::InitializeDendrogramLevel(
    Vertex num_vertices) {
  m_dendrogram->AddLevel(0, num_vertices);

  thrust::sequence(m_handle.GetThrustPolicy(),
                   m_dendrogram->CurrentLevelBegin(),
                   m_dendrogram->CurrentLevelEnd());
}

template <typename GraphType, typename Float, typename Score>
Float Louvain<GraphType, Float, Score>::UpdateClustering(
    Float total_edge_weight, Float resolution, GraphType const &graph) {
  thrust::device_vector<Vertex> next_cluster_v(
      m_dendrogram->CurrentLevelSize());
  thrust::device_vector<Float> delta_Q_v(graph.number_of_edges);
  thrust::device_vector<Vertex> cluster_hash_v(graph.number_of_edges);
  thrust::device_vector<Weight> old_cluster_sum_v(graph.number_of_vertices);
  thrust::device_vector<Weight> new_cluster_sum_v(graph.number_of_edges);
  thrust::device_vector<Weight> next_cluster_weights_v(
      graph.number_of_vertices);
  thrust::device_vector<Score> next_cluster_scores_v(graph.number_of_vertices);

  const Score max_cluster_socre = m_max_cluster_socre == std::nullopt
                                      ? std::numeric_limits<Score>::max()
                                      : m_max_cluster_socre.value();

  bool up_down = true;
  Float cur_Q = -1;

  for (int_t n = 0; n < 2; n++) {
    Vertex *d_cluster = m_dendrogram->CurrentLevelBegin();
    Float *d_delta_Q = delta_Q_v.data().get();

    thrust::copy(m_handle.GetThrustPolicy(), m_dendrogram->CurrentLevelBegin(),
                 m_dendrogram->CurrentLevelEnd(), next_cluster_v.data());
    thrust::copy(m_handle.GetThrustPolicy(), m_cluster_weights_v.begin(),
                 m_cluster_weights_v.end(), next_cluster_weights_v.data());
    thrust::copy(m_handle.GetThrustPolicy(), m_cluster_scores_v.begin(),
                 m_cluster_scores_v.end(), next_cluster_scores_v.data());

    Float new_Q = detail::Modularity(m_handle, total_edge_weight, resolution,
                                     graph, m_dendrogram->CurrentLevelBegin());

    cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop

    while (new_Q > cur_Q) {
      if (new_Q <= (cur_Q + 1e-6)) {
        cur_Q = new_Q;
        break;
      }

      cur_Q = new_Q;

      detail::ComputeDeltaModularity(
          m_handle, total_edge_weight, resolution, graph.number_of_vertices,
          graph.number_of_edges, m_src_offsets_v, m_src_indices_v,
          m_dst_indices_v, m_weights_v, m_vertex_weights_v,
          next_cluster_weights_v, m_vertex_scores_v, next_cluster_scores_v,
          *m_dendrogram, cluster_hash_v, old_cluster_sum_v, new_cluster_sum_v,
          delta_Q_v, max_cluster_socre);

      detail::AssignNodes(
          m_handle, graph.number_of_vertices, graph.number_of_edges,
          m_src_indices_v, m_vertex_weights_v, next_cluster_weights_v,
          m_vertex_scores_v, next_cluster_scores_v, cluster_hash_v,
          next_cluster_v, delta_Q_v, up_down, max_cluster_socre);

      up_down = !up_down;

      new_Q = detail::Modularity(m_handle, total_edge_weight, resolution, graph,
                                 next_cluster_v.data().get());

      if (new_Q > cur_Q) {
        thrust::copy(m_handle.GetThrustPolicy(), next_cluster_v.begin(),
                     next_cluster_v.end(), m_dendrogram->CurrentLevelBegin());
        thrust::copy(m_handle.GetThrustPolicy(), next_cluster_weights_v.begin(),
                     next_cluster_weights_v.end(), m_cluster_weights_v.data());
        thrust::copy(m_handle.GetThrustPolicy(), next_cluster_scores_v.begin(),
                     next_cluster_scores_v.end(), m_cluster_scores_v.data());
      }
    }
  }

  return cur_Q;
}

template <typename GraphType, typename Float, typename Score>
void Louvain<GraphType, Float, Score>::ShrinkGraph(Graph &graph) {
  // renumber the clusters to the range 0..(num_clusters-1)
  m_number_of_clusters = detail::RenumberClusters(
      m_handle, m_dendrogram->CurrentLevelBegin(),
      m_dendrogram->CurrentLevelSize(), m_cluster_inverse_v, m_tmp_arr_v);
  m_cluster_weights_v.resize(m_number_of_clusters);

  // shrink our graph to represent the graph of supervertices
  detail::GenerateSuperverticesGraph(
      m_handle, m_dendrogram->CurrentLevelBegin(), graph.number_of_vertices,
      graph.number_of_edges, m_src_offsets_v, m_src_indices_v, m_dst_indices_v,
      m_weights_v, m_vertex_scores_v, m_cluster_scores_v, m_tmp_arr_v);

  assert(m_number_of_clusters == graph.number_of_vertices);

  m_number_of_cluster_edges = graph.number_of_edges;
}

template <typename GraphType, typename Float, typename Score>
void Louvain<GraphType, Float, Score>::GetClustering(Vertex *clustering) {
  // flatten_dendrogram(m_handle, m_number_of_vertices, *dendrogram_,
  // clustering);
  detail::FlattenDendrogram(m_handle, *m_dendrogram);
  thrust::copy(m_handle.GetThrustPolicy(), m_dendrogram->CurrentLevelBegin(),
               m_dendrogram->CurrentLevelEnd(), clustering);
}

template <typename GraphType, typename Float, typename Score>
void Louvain<GraphType, Float, Score>::GetClusterEdges(
    Vertex *cluster_src_indices, Vertex *cluster_dst_indices,
    Weight *cluster_weights) {
  thrust::copy(m_handle.GetThrustPolicy(), m_src_indices_v.begin(),
               m_src_indices_v.begin() + m_number_of_cluster_edges,
               cluster_src_indices);
  thrust::copy(m_handle.GetThrustPolicy(), m_dst_indices_v.begin(),
               m_dst_indices_v.begin() + m_number_of_cluster_edges,
               cluster_dst_indices);
  thrust::copy(m_handle.GetThrustPolicy(), m_weights_v.begin(),
               m_weights_v.begin() + m_number_of_cluster_edges,
               cluster_weights);
}

template <typename Vertex, typename Edge, typename Weight, typename Float>
std::pair<size_t, Float>
LouvainClustering(Handle const &handle,
                  GraphCSRView<Vertex, Edge, Weight> const &graph_view,
                  Vertex *clustering, size_t &cluster_size, size_t max_level,
                  Float resolution, bool initialized) {
  GRAPH_EXPECTS(graph_view.HasData(), "Graph must be weighted");
  GRAPH_EXPECTS(clustering != nullptr,
                "Invalid input argument: clustering is null, should be a "
                "device pointer to "
                "memory for storing the result");

  Louvain<GraphCSRView<Vertex, Edge, Weight>, Float, unsigned long long> runner(
      handle);
  Float modularity = runner.Cluster(graph_view, max_level, resolution,
                                    clustering, initialized);

  cluster_size = runner.GetNumberOfClusters();

  return std::make_pair(runner.GetDendrogram().NumLevels(), modularity);
}

// Explicit template instantations
template class Louvain<GraphCSRView<int, int, float>, float,
                       unsigned long long>;
template class Louvain<GraphCSRView<int, int, double>, double,
                       unsigned long long>;
template class Louvain<GraphCSRView<int, int, unsigned long long>, float,
                       unsigned long long>;
template class Louvain<GraphCSRView<int, int, unsigned long long>, double,
                       unsigned long long>;
template class Louvain<GraphCSRView<int, int, float>, float, float>;
template class Louvain<GraphCSRView<int, int, double>, double, float>;
template class Louvain<GraphCSRView<int, int, unsigned long long>, float,
                       float>;
template class Louvain<GraphCSRView<int, int, unsigned long long>, double,
                       float>;

template std::pair<size_t, float>
LouvainClustering(Handle const &, GraphCSRView<int, int, float> const &, int *,
                  size_t &, size_t, float, bool);

template std::pair<size_t, double>
LouvainClustering(Handle const &, GraphCSRView<int, int, double> const &, int *,
                  size_t &, size_t, double, bool);

template std::pair<size_t, float>
LouvainClustering(Handle const &,
                  GraphCSRView<int, int, unsigned long long> const &, int *,
                  size_t &, size_t, float, bool);

template std::pair<size_t, double>
LouvainClustering(Handle const &,
                  GraphCSRView<int, int, unsigned long long> const &, int *,
                  size_t &, size_t, double, bool);
} // namespace graph
} // namespace sfm
