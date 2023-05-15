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

#include <cassert>
#include <memory>
#include <sys/types.h>

#include <sfm/graph/converters/COOtoCSR.cuh>
#include <sfm/graph/types.hpp>
#include <sfm/graph/utilities/error.hpp>
#include <sfm/utils/utils.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
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
namespace detail {
template <typename Size, typename Vertex, typename Weight, typename Offset>
void COO2CSR(Handle const &handle, Size number_of_vertices, Vertex *src_indices,
             Vertex *dst_indices, Weight *weights, Size number_of_edges,
             Offset *src_offsets) {
  if (weights != nullptr) {
    thrust::stable_sort_by_key(
        handle.GetThrustPolicy(), dst_indices, dst_indices + number_of_edges,
        sfm::utils::MakeZipIterator(src_indices, weights));

    thrust::stable_sort_by_key(
        handle.GetThrustPolicy(), src_indices, src_indices + number_of_edges,
        sfm::utils::MakeZipIterator(dst_indices, weights));
  } else {
    thrust::stable_sort_by_key(handle.GetThrustPolicy(), dst_indices,
                               dst_indices + number_of_edges, src_indices);
    thrust::stable_sort_by_key(handle.GetThrustPolicy(), src_indices,
                               src_indices + number_of_edges, dst_indices);
  }
  sfm::graph::detail::FillOffset(src_indices, src_offsets, number_of_vertices,
                                 number_of_edges, handle.GetStream());
}

template <typename Vertex, typename Edge, typename Weight, typename Float,
          typename Score>
void ComputeDeltaModularity(
    Handle const &handle, Float total_edge_weight, Float resolution,
    Vertex number_of_vertices, Edge number_of_edges,
    thrust::device_vector<Vertex> const &offsets_v,
    thrust::device_vector<Vertex> const &src_indices_v,
    thrust::device_vector<Vertex> const &dst_indices_v,
    thrust::device_vector<Weight> const &weights_v,
    thrust::device_vector<Weight> const &vertex_weights_v,
    thrust::device_vector<Weight> const &cluster_weights_v,
    thrust::device_vector<Score> const &vertex_scores_v,
    thrust::device_vector<Score> const &cluster_scores_v,
    Dendrogram<Vertex> const &dendrogram,
    thrust::device_vector<Vertex> &cluster_hash_v,
    thrust::device_vector<Weight> &old_cluster_sum_v,
    thrust::device_vector<Weight> &new_cluster_sum_v,
    thrust::device_vector<Float> &delta_Q_v, Score max_cluster_socre) {
  Edge const *d_offsets = offsets_v.data().get();
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
                   thrust::make_counting_iterator<Edge>(0),
                   thrust::make_counting_iterator<Edge>(number_of_edges),
                   [d_src_indices = src_indices_v.data().get(),
                    d_dst_indices = dst_indices_v.begin(), d_cluster, d_offsets,
                    d_cluster_hash, d_new_cluster_sum, d_weights,
                    d_old_cluster_sum] __device__(Edge loc) {
                     Vertex src = d_src_indices[loc];
                     Vertex dst = d_dst_indices[loc];

                     if (src != dst) {
                       Vertex old_cluster = d_cluster[src];
                       Vertex new_cluster = d_cluster[dst];
                       Edge hash_base = d_offsets[src];
                       Edge n_edges = d_offsets[src + 1] - hash_base;

                       int h = (new_cluster % n_edges);
                       Edge offset = hash_base + h;
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
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Edge>(0),
      thrust::make_counting_iterator<Edge>(number_of_edges),
      [total_edge_weight, resolution, d_cluster_hash,
       d_src_indices = src_indices_v.data().get(), d_cluster, d_vertex_weights,
       d_delta_Q, d_new_cluster_sum, d_old_cluster_sum, d_cluster_weights,
       max_cluster_socre, d_vertex_scores,
       d_cluster_scores] __device__(Edge loc) {
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

          d_delta_Q[loc] =
              ((vertex_score + cluster_score) <= max_cluster_socre ||
               vertex_score == 0 || cluster_score == 0)
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

template <typename graph_t, typename Float>
Float Modularity(Handle const &handle, Float total_edge_weight,
                 Float resolution, graph_t const &graph,
                 typename graph_t::VertexType const *d_cluster) {
  using Vertex = typename graph_t::VertexType;
  using Edge = typename graph_t::EdgeType;
  using Weight = typename graph_t::WeightType;

  Vertex n_verts = graph.number_of_vertices;

  thrust::device_vector<Weight> inc(n_verts);
  thrust::device_vector<Weight> deg(n_verts);

  thrust::fill(handle.GetThrustPolicy(), inc.begin(), inc.end(), Weight{0});
  thrust::fill(handle.GetThrustPolicy(), deg.begin(), deg.end(), Weight{0});

  // FIXME:  Already have weighted degree computed in main loop,
  //         could pass that in rather than computing d_deg... which
  //         would save an atomicAdd (synchronization)
  //
  thrust::for_each(
      handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_vertices),
      [d_inc = inc.data().get(), d_deg = deg.data().get(),
       d_offsets = graph.offsets, d_indices = graph.indices,
       d_weights = graph.edge_data, d_cluster] __device__(Vertex v) {
        Vertex community = d_cluster[v];
        Weight increase{0};
        Weight degree{0};

        for (Edge loc = d_offsets[v]; loc < d_offsets[v + 1]; ++loc) {
          Vertex neighbor = d_indices[loc];
          degree += d_weights[loc];
          if (d_cluster[neighbor] == community) {
            increase += d_weights[loc];
          }
        }

        if (degree > Weight{0})
          atomicAdd(d_deg + community, degree);
        if (increase > Weight{0})
          atomicAdd(d_inc + community, increase);
      });

  Float Q = thrust::transform_reduce(
      handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_vertices),
      [d_deg = deg.data().get(), d_inc = inc.data().get(), total_edge_weight,
       resolution] __device__(Vertex community) -> Float {
        return ((d_inc[community] / total_edge_weight) -
                resolution * (d_deg[community] / total_edge_weight) *
                    (d_deg[community] / total_edge_weight));
      },
      Float{0}, thrust::plus<Float>());

  return Q;
}

template <typename graph_t>
void ComputeVertexAndClusterWeights(
    Handle const &handle, graph_t const &graph,
    typename graph_t::WeightType *vertex_weights,
    typename graph_t::WeightType *cluster_weights) {
  using Vertex = typename graph_t::VertexType;
  using Edge = typename graph_t::EdgeType;
  using Weight = typename graph_t::WeightType;

  Edge const *d_offsets = graph.offsets;
  Vertex const *d_indices = graph.indices;
  Weight const *d_weights = graph.edge_data;
  Weight *d_vertex_weights = vertex_weights;
  Weight *d_cluster_weights = cluster_weights;

  //
  // MNMG:  per_v_transform_reduce_outgoing_e, then copy
  //
  thrust::for_each(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Edge>(0),
      thrust::make_counting_iterator<Edge>(graph.number_of_vertices),
      [d_offsets, d_indices, d_weights, d_vertex_weights,
       d_cluster_weights] __device__(Vertex src) {
        Weight sum = thrust::reduce(thrust::seq, d_weights + d_offsets[src],
                                    d_weights + d_offsets[src + 1]);

        d_vertex_weights[src] = sum;
        d_cluster_weights[src] = sum;
      });
}

template <typename Vertex, typename count_t>
count_t RenumberClusters(Handle const &handle, Vertex *d_cluster,
                         count_t old_num_clusters,
                         thrust::device_vector<Vertex> &cluster_inverse_v,
                         thrust::device_vector<Vertex> &tmp_arr_v) {
  Vertex *d_tmp_array = tmp_arr_v.data().get();
  Vertex *d_cluster_inverse = cluster_inverse_v.data().get();

  //
  //  New technique.  Initialize cluster_inverse_v_ to 0
  //
  thrust::fill(handle.GetThrustPolicy(), cluster_inverse_v.begin(),
               cluster_inverse_v.end(), Vertex{0});

  //
  // Iterate over every element c in the current clustering and set
  // cluster_inverse_v to 1
  //
  auto first_1 = thrust::make_constant_iterator<Vertex>(1);
  auto last_1 = first_1 + old_num_clusters;

  thrust::scatter(handle.GetThrustPolicy(), first_1, last_1, d_cluster,
                  cluster_inverse_v.begin());

  //
  // Now we'll copy all of the clusters that have a value of 1 into a
  // temporary array
  //
  auto copy_end = thrust::copy_if(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
      thrust::make_counting_iterator<Vertex>(old_num_clusters),
      tmp_arr_v.begin(), [d_cluster_inverse] __device__(const Vertex idx) {
        return d_cluster_inverse[idx] == 1;
      });

  count_t new_num_clusters = thrust::distance(tmp_arr_v.begin(), copy_end);
  tmp_arr_v.resize(new_num_clusters);

  //
  // Now we can set each value in cluster_inverse of a cluster to its index
  //
  thrust::for_each(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
      thrust::make_counting_iterator<Vertex>(new_num_clusters),
      [d_cluster_inverse, d_tmp_array] __device__(const Vertex idx) {
        d_cluster_inverse[d_tmp_array[idx]] = idx;
      });

  thrust::for_each(handle.GetThrustPolicy(),
                   thrust::make_counting_iterator<Vertex>(0),
                   thrust::make_counting_iterator<Vertex>(old_num_clusters),
                   [d_cluster, d_cluster_inverse] __device__(Vertex i) {
                     d_cluster[i] = d_cluster_inverse[d_cluster[i]];
                   });

  cluster_inverse_v.resize(new_num_clusters);

  return new_num_clusters;
}

template <typename Vertex, typename Weight, typename Score, typename Size>
void GenerateSuperverticesGraph(Handle const &handle, const Vertex *d_clusters,
                                Size &number_of_vertices, Size &number_of_edges,
                                thrust::device_vector<Vertex> &src_offsets_v,
                                thrust::device_vector<Vertex> &src_indices_v,
                                thrust::device_vector<Vertex> &dst_indices_v,
                                thrust::device_vector<Weight> &weights_v,
                                thrust::device_vector<Score> &vertex_scores_v,
                                thrust::device_vector<Score> &cluster_scores_v,
                                thrust::device_vector<Vertex> &tmp_arr_v) {
  // Update cluster scores
  //
  // Reuse graph.offsets to save memory
  src_offsets_v.resize(number_of_vertices);
  cluster_scores_v.resize(number_of_vertices);
  Vertex *d_tmp_clusters = src_offsets_v.data().get();
  thrust::copy(handle.GetThrustPolicy(), d_clusters,
               d_clusters + number_of_vertices, d_tmp_clusters);
  thrust::sort_by_key(handle.GetThrustPolicy(), d_tmp_clusters,
                      d_tmp_clusters + number_of_vertices,
                      vertex_scores_v.begin());
  auto reduce_end = thrust::reduce_by_key(
      handle.GetThrustPolicy(), d_tmp_clusters,
      d_tmp_clusters + number_of_vertices, vertex_scores_v.begin(),
      tmp_arr_v.begin(), cluster_scores_v.begin());
  Size number_of_clusters =

      thrust::distance(tmp_arr_v.begin(), reduce_end.first);
  cluster_scores_v.resize(number_of_clusters);
  vertex_scores_v.resize(number_of_clusters);
  thrust::copy(handle.GetThrustPolicy(), cluster_scores_v.begin(),
               cluster_scores_v.end(), vertex_scores_v.begin());

  //
  // Update vertices and edges
  //
  thrust::device_vector<Vertex> new_src_v(number_of_edges);
  thrust::device_vector<Vertex> new_dst_v(number_of_edges);
  thrust::device_vector<Weight> new_weight_v(number_of_edges);

  //
  //  Renumber the COO
  //
  thrust::for_each(
      handle.GetThrustPolicy(),
      thrust::make_counting_iterator<long long int>(0),
      thrust::make_counting_iterator<long long int>(number_of_edges),
      [d_old_src = src_indices_v.data().get(),
       d_old_dst = dst_indices_v.data().get(),
       d_old_weight = weights_v.data().get(),
       d_new_src = new_src_v.data().get(), d_new_dst = new_dst_v.data().get(),
       d_new_weight = new_weight_v.data().get(),
       d_clusters] __device__(auto e) {
        d_new_src[e] = d_clusters[d_old_src[e]];
        d_new_dst[e] = d_clusters[d_old_dst[e]];
        d_new_weight[e] = d_old_weight[e];
      });

  thrust::stable_sort_by_key(handle.GetThrustPolicy(), new_dst_v.begin(),
                             new_dst_v.end(),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 new_src_v.begin(), new_weight_v.begin())));
  thrust::stable_sort_by_key(handle.GetThrustPolicy(), new_src_v.begin(),
                             new_src_v.end(),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 new_dst_v.begin(), new_weight_v.begin())));

  //
  //  Now we reduce by key to combine the weights of duplicate
  //  edges.
  //
  auto start = thrust::make_zip_iterator(
      thrust::make_tuple(new_src_v.begin(), new_dst_v.begin()));
  auto new_start = thrust::make_zip_iterator(thrust::make_tuple(
      src_indices_v.data().get(), dst_indices_v.data().get()));
  auto new_end = thrust::reduce_by_key(
      handle.GetThrustPolicy(), start, start + number_of_edges,
      new_weight_v.begin(), new_start, weights_v.data().get(),
      thrust::equal_to<thrust::tuple<Vertex, Vertex>>(),
      thrust::plus<Weight>());

  number_of_edges = thrust::distance(new_start, new_end.first);
  number_of_vertices = number_of_clusters;

  detail::FillOffset(src_indices_v.data().get(), src_offsets_v.data().get(),
                     number_of_clusters, number_of_edges, handle.GetStream());

  src_indices_v.resize(number_of_edges);
  dst_indices_v.resize(number_of_edges);
  weights_v.resize(number_of_edges);
}

template <typename Vertex, typename Edge, typename Weight, typename Float,
          typename Score>
void AssignNodes(Handle const &handle, Vertex number_of_vertices,
                 Edge number_of_edges,
                 thrust::device_vector<Vertex> const &src_indices_v,
                 thrust::device_vector<Weight> const &vertex_weights_v,
                 thrust::device_vector<Weight> &cluster_weights_v,
                 thrust::device_vector<Score> const &vertex_scores_v,
                 thrust::device_vector<Score> &cluster_scores,
                 thrust::device_vector<Vertex> const &cluster_hash_v,
                 thrust::device_vector<Vertex> &next_cluster_v,
                 thrust::device_vector<Float> const &delta_Q_v, bool up_down) {
  thrust::device_vector<Vertex> temp_vertices_v(number_of_vertices);
  thrust::device_vector<Vertex> temp_cluster_v(number_of_vertices);
  thrust::device_vector<Float> temp_delta_Q_v(number_of_vertices);

  thrust::fill(handle.GetThrustPolicy(), temp_cluster_v.begin(),
               temp_cluster_v.end(), Vertex{-1});

  thrust::fill(handle.GetThrustPolicy(), temp_delta_Q_v.begin(),
               temp_delta_Q_v.end(), Float{0});

  auto cluster_reduce_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(cluster_hash_v.begin(), delta_Q_v.begin()));

  auto output_edge_iterator2 = thrust::make_zip_iterator(
      thrust::make_tuple(temp_cluster_v.begin(), temp_delta_Q_v.begin()));

  auto cluster_reduce_end = thrust::reduce_by_key(
      handle.GetThrustPolicy(), src_indices_v.begin(), src_indices_v.end(),
      cluster_reduce_iterator, temp_vertices_v.data().get(),
      output_edge_iterator2, thrust::equal_to<Vertex>(),
      [] __device__(auto pair1, auto pair2) {
        if (thrust::get<1>(pair1) > thrust::get<1>(pair2))
          return pair1;
        else if ((thrust::get<1>(pair1) == thrust::get<1>(pair2)) &&
                 (thrust::get<0>(pair1) < thrust::get<0>(pair2)))
          return pair1;
        else
          return pair2;
      });

  Vertex final_size =
      thrust::distance(temp_vertices_v.data().get(), cluster_reduce_end.first);

  thrust::for_each(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
      thrust::make_counting_iterator<Vertex>(final_size),
      [up_down, d_temp_delta_Q = temp_delta_Q_v.data().get(),
       d_next_cluster = next_cluster_v.data().get(),
       d_temp_vertices = temp_vertices_v.data().get(),
       d_vertex_weights = vertex_weights_v.data().get(),
       d_vertex_scores = vertex_scores_v.data().get(),
       d_temp_clusters = temp_cluster_v.data().get(),
       d_cluster_weights = cluster_weights_v.data().get(),
       d_cluster_scores = cluster_scores.data().get()] __device__(Vertex id) {
        if ((d_temp_clusters[id] >= 0) && (d_temp_delta_Q[id] > Float{0})) {
          Vertex new_cluster = d_temp_clusters[id];
          Vertex old_cluster = d_next_cluster[d_temp_vertices[id]];

          if ((new_cluster > old_cluster) == up_down) {
            Weight src_weight = d_vertex_weights[d_temp_vertices[id]];
            Score src_score = d_vertex_scores[d_temp_vertices[id]];
            d_next_cluster[d_temp_vertices[id]] = d_temp_clusters[id];

            atomicAdd(d_cluster_weights + new_cluster, src_weight);
            atomicAdd(d_cluster_weights + old_cluster, -src_weight);
            atomicAdd(d_cluster_scores + new_cluster, src_score);
            atomicAdd(d_cluster_scores + old_cluster, -src_score);
          }
        }
      });
}

template <typename Vertex, typename Edge, typename Weight, typename Float,
          typename Score>
void AssignNodes(Handle const &handle, Vertex number_of_vertices,
                 Edge number_of_edges,
                 thrust::device_vector<Vertex> const &src_indices_v,
                 thrust::device_vector<Weight> const &vertex_weights_v,
                 thrust::device_vector<Weight> &cluster_weights_v,
                 thrust::device_vector<Score> const &vertex_scores_v,
                 thrust::device_vector<Score> &cluster_scores_v,
                 thrust::device_vector<Vertex> const &cluster_hash_v,
                 thrust::device_vector<Vertex> &next_cluster_v,
                 thrust::device_vector<Float> const &delta_Q_v, bool up_down,
                 Score max_cluster_score) {
  thrust::device_vector<Vertex> temp_vertices_v(number_of_vertices);
  thrust::device_vector<Vertex> temp_cluster_v(number_of_vertices);
  thrust::device_vector<Float> temp_delta_Q_v(number_of_vertices);

  thrust::fill(handle.GetThrustPolicy(), temp_cluster_v.begin(),
               temp_cluster_v.end(), Vertex{-1});

  thrust::fill(handle.GetThrustPolicy(), temp_delta_Q_v.begin(),
               temp_delta_Q_v.end(), Float{0});

  auto cluster_reduce_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(cluster_hash_v.begin(), delta_Q_v.begin()));

  auto output_edge_iterator2 = thrust::make_zip_iterator(
      thrust::make_tuple(temp_cluster_v.begin(), temp_delta_Q_v.begin()));

  auto cluster_reduce_end = thrust::reduce_by_key(
      handle.GetThrustPolicy(), src_indices_v.begin(), src_indices_v.end(),
      cluster_reduce_iterator, temp_vertices_v.data().get(),
      output_edge_iterator2, thrust::equal_to<Vertex>(),
      [] __device__(auto pair1, auto pair2) {
        if (thrust::get<1>(pair1) > thrust::get<1>(pair2))
          return pair1;
        else if ((thrust::get<1>(pair1) == thrust::get<1>(pair2)) &&
                 (thrust::get<0>(pair1) < thrust::get<0>(pair2)))
          return pair1;
        else
          return pair2;
      });

  Vertex final_size =
      thrust::distance(temp_vertices_v.data().get(), cluster_reduce_end.first);

  thrust::device_vector<Score> temp_cluster_scores_v(cluster_scores_v.begin(),
                                                     cluster_scores_v.end());
  thrust::device_vector<Score> cached_cluster_scores_v(cluster_scores_v.begin(),
                                                       cluster_scores_v.end());
  thrust::device_vector<Vertex> temp_selected_vertices_v(final_size);
  auto temp_selected_vertices_begin = temp_selected_vertices_v.begin();
  auto temp_selected_vertices_end = thrust::copy_if(
      handle.GetThrustPolicy(), thrust::make_counting_iterator<Vertex>(0),
      thrust::make_counting_iterator<Vertex>(final_size),
      temp_selected_vertices_begin,
      [up_down, d_temp_delta_Q = temp_delta_Q_v.data().get(),
       d_next_cluster = next_cluster_v.data().get(),
       d_temp_vertices = temp_vertices_v.data().get(),
       d_vertex_scores = vertex_scores_v.data().get(),
       d_temp_clusters = temp_cluster_v.data().get(),
       d_temp_cluster_scores =
           temp_cluster_scores_v.data().get()] __device__(Vertex id) {
        if ((d_temp_clusters[id] >= 0) && (d_temp_delta_Q[id] > Float{0})) {
          Vertex new_cluster = d_temp_clusters[id];
          Vertex old_cluster = d_next_cluster[d_temp_vertices[id]];
          if ((new_cluster > old_cluster) == up_down) {
            Score src_score = d_vertex_scores[d_temp_vertices[id]];
            atomicAdd(d_temp_cluster_scores + new_cluster, src_score);
            return true;
          }
        }

        return false;
      });
  Vertex temp_selected_vertices_size =
      temp_selected_vertices_end - temp_selected_vertices_begin;

  thrust::for_each(handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(number_of_vertices),
                   [d_cluster_scores = cluster_scores_v.data().get(),
                    d_temp_cluster_scores = temp_cluster_scores_v.data().get(),
                    max_cluster_score] __device__(Vertex n) {
                     Score temp_cluster_score = d_temp_cluster_scores[n];
                     Score cluster_score = d_cluster_scores[n];
                     if (temp_cluster_score <= max_cluster_score ||
                         temp_cluster_score == cluster_score) {
                       d_cluster_scores[n] = temp_cluster_score;
                     }
                   });

  thrust::device_vector<Vertex> selected_vertices_v(
      temp_selected_vertices_size);
  auto selected_vertices_begin = selected_vertices_v.begin();
  auto selected_vertices_end = thrust::copy_if(
      handle.GetThrustPolicy(), temp_selected_vertices_begin,
      temp_selected_vertices_end, selected_vertices_begin,
      [up_down, max_cluster_score, d_temp_delta_Q = temp_delta_Q_v.data().get(),
       d_next_cluster = next_cluster_v.data().get(),
       d_temp_vertices = temp_vertices_v.data().get(),
       d_vertex_weights = vertex_weights_v.data().get(),
       d_vertex_scores = vertex_scores_v.data().get(),
       d_temp_clusters = temp_cluster_v.data().get(),
       d_cluster_weights = cluster_weights_v.data().get(),
       d_cluster_scores = cluster_scores_v.data().get(),
       d_temp_cluster_scores =
           temp_cluster_scores_v.data().get()] __device__(Vertex id) {
        Vertex new_cluster = d_temp_clusters[id];
        Vertex old_cluster = d_next_cluster[d_temp_vertices[id]];
        Score temp_cluster_score = d_temp_cluster_scores[new_cluster];
        Score cluster_score = d_cluster_scores[new_cluster];

        if (temp_cluster_score <= max_cluster_score ||
            cluster_score == temp_cluster_score) {
          Weight src_weight = d_vertex_weights[d_temp_vertices[id]];
          Score src_score = d_vertex_scores[d_temp_vertices[id]];
          d_next_cluster[d_temp_vertices[id]] = d_temp_clusters[id];

          atomicAdd(d_cluster_weights + new_cluster, src_weight);
          atomicAdd(d_cluster_weights + old_cluster, -src_weight);
          atomicAdd(d_cluster_scores + old_cluster, -src_score);
          return false;
        }

        return true;
      });
  Vertex selected_vertices_size =
      selected_vertices_end - selected_vertices_begin;

  if (selected_vertices_size != 0) {
    thrust::sort(
        handle.GetThrustPolicy(), selected_vertices_begin,
        selected_vertices_end,
        [d_temp_clusters = temp_cluster_v.data().get(),
         d_temp_delta_Q = temp_delta_Q_v.data().get()] __device__(Vertex id1,
                                                                  Vertex id2) {
          Vertex new_cluster1 = d_temp_clusters[id1];
          Vertex new_cluster2 = d_temp_clusters[id2];
          Score delta_Q1 = d_temp_delta_Q[id1];
          Score delta_Q2 = d_temp_delta_Q[id2];

          if (new_cluster1 == new_cluster2) {
            return delta_Q1 == delta_Q2 ? id1 < id2 : delta_Q1 > delta_Q2;
          } else {
            return new_cluster1 < new_cluster2;
          }
        });

    auto inclusive_scan_key_begin = sfm::utils::MakeTransformIterator<Vertex>(
        selected_vertices_begin,
        [d_temp_clusters = temp_cluster_v.data().get()] __device__(Vertex id) {
          return d_temp_clusters[id];
        });
    auto inclusive_scan_key_end =
        inclusive_scan_key_begin + selected_vertices_size;

    auto inclusive_scan_val_begin = sfm::utils::MakeTransformIterator<Score>(
        selected_vertices_begin,
        [d_vertex_scores = vertex_scores_v.data().get(),
         d_temp_vertices = temp_vertices_v.data().get()] __device__(Vertex id)
            -> Score { return d_vertex_scores[d_temp_vertices[id]]; });

    thrust::device_vector<Score> inclusive_scan_results_v(
        selected_vertices_size);
    thrust::inclusive_scan_by_key(
        handle.GetThrustPolicy(), inclusive_scan_key_begin,
        inclusive_scan_key_end, inclusive_scan_val_begin,
        inclusive_scan_results_v.begin());

    thrust::device_vector<Vertex> final_vertex_indices_v(
        selected_vertices_size);
    auto final_vertex_indices_begin = final_vertex_indices_v.begin();
    auto final_vertex_indices_end = thrust::copy_if(
        handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(selected_vertices_size),
        final_vertex_indices_begin,
        [d_selected_vertices = selected_vertices_v.data().get(),
         d_inclusive_scans = inclusive_scan_results_v.data().get(),
         d_temp_clusters = temp_cluster_v.data().get(),
         d_cached_cluster_scores = cached_cluster_scores_v.data().get(),
         max_cluster_score] __device__(Vertex n) {
          Vertex id = d_selected_vertices[n];
          Vertex new_cluster = d_temp_clusters[id];
          Score new_cached_cluster_score = d_cached_cluster_scores[new_cluster];

          return (new_cached_cluster_score + d_inclusive_scans[n]) <=
                 max_cluster_score;
        });

    thrust::for_each(
        handle.GetThrustPolicy(), final_vertex_indices_begin,
        final_vertex_indices_end,
        [up_down, d_selected_vertices = selected_vertices_v.data().get(),
         d_temp_delta_Q = temp_delta_Q_v.data().get(),
         d_next_cluster = next_cluster_v.data().get(),
         d_temp_vertices = temp_vertices_v.data().get(),
         d_vertex_weights = vertex_weights_v.data().get(),
         d_vertex_scores = vertex_scores_v.data().get(),
         d_temp_clusters = temp_cluster_v.data().get(),
         d_cluster_weights = cluster_weights_v.data().get(),
         d_cluster_scores =
             cluster_scores_v.data().get()] __device__(Vertex n) {
          Vertex id = d_selected_vertices[n];
          Vertex new_cluster = d_temp_clusters[id];
          Vertex old_cluster = d_next_cluster[d_temp_vertices[id]];

          Weight src_weight = d_vertex_weights[d_temp_vertices[id]];
          Score src_score = d_vertex_scores[d_temp_vertices[id]];
          d_next_cluster[d_temp_vertices[id]] = d_temp_clusters[id];

          atomicAdd(d_cluster_weights + new_cluster, src_weight);
          atomicAdd(d_cluster_weights + old_cluster, -src_weight);
          atomicAdd(d_cluster_scores + new_cluster, src_score);
          atomicAdd(d_cluster_scores + old_cluster, -src_score);
        });
  }
}

template <typename Vertex>
void FlattenDendrogram(Handle const &handle,
                       Dendrogram<Vertex> const &dendrogram,
                       thrust::device_vector<Vertex> &tmp_arr_v,
                       Vertex *clustering) {
  Vertex num_verts = dendrogram.get_level_size_nocheck(0);
  Vertex num_levels = dendrogram.num_levels();

  thrust::copy(handle.GetThrustPolicy(), dendrogram.CurrentLevelBegin(),
               dendrogram.CurrentLevelEnd(), clustering);

  if (num_levels <= 1) {
    return;
  }

  tmp_arr_v.resize(dendrogram.get_level_size_nocheck(1));

  for (Vertex curr_level = num_levels - 1; curr_level > 0; curr_level--) {
    Vertex curr_num_verts = dendrogram.get_level_size_nocheck(curr_level);
    Vertex prev_num_verts = dendrogram.get_level_size_nocheck(curr_level - 1);
    thrust::copy(handle.GetThrustPolicy(), clustering,
                 clustering + curr_num_verts, tmp_arr_v.begin());

    thrust::for_each(
        handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(prev_num_verts),
        [d_local_clustering = dendrogram.get_level_ptr_nocheck(curr_level - 1),
         d_curr_clustering = tmp_arr_v.data().get(),
         d_prev_clustering = clustering] __device__(Vertex idx) {
          d_prev_clustering[idx] = d_curr_clustering[d_local_clustering[idx]];
        });
  }
}

template <typename Vertex>
void FlattenDendrogram(Handle const &handle, Dendrogram<Vertex> &dendrogram) {
  Vertex num_verts = dendrogram.GetLevelSizeNoCheck(0);
  Vertex num_levels = dendrogram.NumLevels();

  if (num_levels <= 1) {
    return;
  }

  for (Vertex curr_level = num_levels - 1; curr_level > 0; curr_level--) {
    Vertex curr_num_verts = dendrogram.GetLevelSizeNoCheck(curr_level);
    Vertex prev_num_verts = dendrogram.GetLevelSizeNoCheck(curr_level - 1);

    thrust::for_each(
        handle.GetThrustPolicy(), thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(prev_num_verts),
        [d_prev_clustering = dendrogram.GetLevelPtrNoCheck(curr_level - 1),
         d_curr_clustering =
             dendrogram.GetLevelPtrNoCheck(curr_level)] __device__(Vertex idx) {
          d_prev_clustering[idx] = d_curr_clustering[d_prev_clustering[idx]];
        });

    dendrogram.PopBack();
  }
}
} // namespace detail
} // namespace graph
} // namespace sfm
