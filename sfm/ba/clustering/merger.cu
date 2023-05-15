// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <sfm/ba/clustering/common_methods.cuh>
#include <sfm/ba/clustering/merger.cuh>
#include <sfm/graph/detail/common_methods.cuh>
#include <sfm/graph/utilities/error.hpp>
#include <sfm/types.h>

#include <optional>

namespace sfm {
namespace ba {
namespace clustering {
template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Merger<Float, Size, Vertex, Weight, Score>::Setup(
    const Vertex *src_offsets, const Vertex *dst_indices, const Weight *weights,
    Size number_of_vertices, Size number_of_edges,
    Size targeted_number_of_clusters, std::optional<Score> max_cluster_socre) {
  m_dendrogram = std::make_unique<sfm::graph::Dendrogram<Vertex>>();
  m_src_offsets_v.resize(number_of_vertices + 1);
  m_dst_indices_v.resize(number_of_edges);
  m_weights_v.resize(number_of_edges);
  m_src_indices_v.resize(number_of_edges);
  m_vertex_weights_v.resize(number_of_vertices);
  m_cluster_weights_v.resize(number_of_vertices);
  m_tmp_arr_v.resize(number_of_vertices);
  m_cluster_inverse_v.resize(number_of_vertices);
  m_number_of_vertices = number_of_vertices;
  m_number_of_edges = number_of_edges;
  m_number_of_clusters = number_of_vertices;
  m_number_of_cluster_edges = number_of_edges;
  m_vertex_scores_v.resize(number_of_vertices);
  m_cluster_scores_v.resize(number_of_vertices);
  m_targeted_number_of_clusters = targeted_number_of_clusters;
  m_max_cluster_socre = max_cluster_socre;

  thrust::copy(m_handle.GetThrustPolicy(), src_offsets,
               src_offsets + number_of_vertices + 1, m_src_offsets_v.begin());

  thrust::copy(m_handle.GetThrustPolicy(), dst_indices,
               dst_indices + number_of_edges, m_dst_indices_v.begin());

  thrust::copy(m_handle.GetThrustPolicy(), weights, weights + number_of_edges,
               m_weights_v.begin());
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
Float Merger<Float, Size, Vertex, Weight, Score>::Solve(
    size_t max_level, Float resolution, Float total_edge_weight,
    Vertex *initial_clustering) {
  Float best_modularity = Float{-1};

  //
  //  Our copy of the graph.  Each iteration of the outer loop will
  //  shrink this copy of the graph.
  //
  sfm::graph::GraphCSRView<Vertex, Vertex, Weight> current_graph(
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

    sfm::graph::detail::ComputeVertexAndClusterWeights(
        m_handle, current_graph, m_vertex_weights_v.data().get(),
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

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
Float Merger<Float, Size, Vertex, Weight, Score>::Solve(
    size_t max_level, Float resolution, Vertex *initial_clustering) {
  Float total_edge_weight = thrust::reduce(
      m_handle.GetThrustPolicy(), m_weights_v.begin(), m_weights_v.end());

  return Solve(max_level, resolution, total_edge_weight, initial_clustering);
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Merger<Float, Size, Vertex, Weight, Score>::InitializeDendrogramLevel(
    Vertex num_vertices) {
  m_dendrogram->AddLevel(0, num_vertices);

  thrust::sequence(m_handle.GetThrustPolicy(),
                   m_dendrogram->CurrentLevelBegin(),
                   m_dendrogram->CurrentLevelEnd());
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
Float Merger<Float, Size, Vertex, Weight, Score>::UpdateClustering(
    Float total_edge_weight, Float resolution, Graph const &graph) {
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

  // To avoid the potential of having two vertices swap clusters
  // we will only allow vertices to move up (true) or down (false)
  // during each iteration of the loop
  bool up_down = false;
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

    Float new_Q = sfm::graph::detail::Modularity(
        m_handle, total_edge_weight, resolution, graph,
        m_dendrogram->CurrentLevelBegin());

    cur_Q = new_Q - 1;

    while (new_Q > cur_Q) {
      if (new_Q <= (cur_Q + 1e-6)) {
        cur_Q = new_Q;
        break;
      }

      cur_Q = new_Q;

      sfm::ba::clustering::ComputeDeltaModularity(
          m_handle, total_edge_weight, resolution, graph.number_of_vertices,
          graph.number_of_edges, m_src_offsets_v, m_src_indices_v,
          m_dst_indices_v, m_weights_v, m_vertex_weights_v,
          next_cluster_weights_v, m_vertex_scores_v, next_cluster_scores_v,
          *m_dendrogram, cluster_hash_v, old_cluster_sum_v, new_cluster_sum_v,
          delta_Q_v, m_targeted_number_of_clusters, max_cluster_socre);

      sfm::graph::detail::AssignNodes(
          m_handle, graph.number_of_vertices, graph.number_of_edges,
          m_src_indices_v, m_vertex_weights_v, next_cluster_weights_v,
          m_vertex_scores_v, next_cluster_scores_v, cluster_hash_v,
          next_cluster_v, delta_Q_v, up_down, max_cluster_socre);

      up_down = !up_down;

      new_Q = sfm::graph::detail::Modularity(m_handle, total_edge_weight,
                                             resolution, graph,
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

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Merger<Float, Size, Vertex, Weight, Score>::ShrinkGraph(Graph &graph) {
  // renumber the clusters to the range 0..(num_clusters-1)
  m_number_of_clusters = sfm::graph::detail::RenumberClusters(
      m_handle, m_dendrogram->CurrentLevelBegin(),
      m_dendrogram->CurrentLevelSize(), m_cluster_inverse_v, m_tmp_arr_v);
  m_cluster_weights_v.resize(m_number_of_clusters);

  // shrink our graph to represent the graph of supervertices
  sfm::graph::detail::GenerateSuperverticesGraph(
      m_handle, m_dendrogram->CurrentLevelBegin(), graph.number_of_vertices,
      graph.number_of_edges, m_src_offsets_v, m_src_indices_v, m_dst_indices_v,
      m_weights_v, m_vertex_scores_v, m_cluster_scores_v, m_tmp_arr_v);

  assert(m_number_of_clusters == graph.number_of_vertices);

  m_number_of_cluster_edges = graph.number_of_edges;
}

template <typename Float, typename Size, typename Vertex, typename Weight,
          typename Score>
void Merger<Float, Size, Vertex, Weight, Score>::GetClustering(
    Vertex *clustering) {
  // flatten_dendrogram(m_handle, m_number_of_vertices, *dendrogram_,
  // clustering);
  sfm::graph::detail::FlattenDendrogram(m_handle, *m_dendrogram);
  thrust::copy(m_handle.GetThrustPolicy(), m_dendrogram->CurrentLevelBegin(),
               m_dendrogram->CurrentLevelEnd(), clustering);
}

// Explicit template instantations
template class Merger<float, int, int, unsigned long long, unsigned long long>;
template class Merger<double, int, int, unsigned long long, unsigned long long>;
} // namespace clustering
} // namespace ba
} // namespace sfm
