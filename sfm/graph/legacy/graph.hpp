// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unistd.h>

namespace sfm {
namespace graph {
namespace legacy {
enum class PropType { PROP_UNDEF, PROP_FALSE, PROP_TRUE };

struct GraphProperties {
  bool directed{false};
  bool weighted{false};
  bool multigraph{false};
  bool bipartite{false};
  bool tree{false};
  PropType has_negative_edges{PropType::PROP_UNDEF};
  GraphProperties() = default;
};

/**
 * @brief       Base class graphs, all but vertices and edges
 *
 * @tparam Vertex   Type of vertex id
 * @tparam Edge     Type of edge id
 * @tparam Weight   Type of weight
 */
template <typename Vertex, typename Edge, typename Weight> class GraphViewBase {
public:
  using VertexType = Vertex;
  using EdgeType = Edge;
  using WeightType = Weight;

  Weight *edge_data; ///< edge weight

  GraphProperties prop;

  Vertex number_of_vertices;
  Edge number_of_edges;

  Vertex *local_vertices;
  Edge *local_edges;
  Vertex *local_offsets;

  Vertex GetNumberOfVertices() const { return number_of_vertices; }

  Vertex GetLocalVertexFirst() const { return Vertex{0}; }

  /**
   * @brief      Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifiers      Pointer to device memory to store the
   * vertex identifiers
   */
  void GetVertexIdentifiers(Vertex *identifiers) const;

  void SetLocalData(Vertex *vertices, Edge *edges, Vertex *offsets) {
    local_vertices = vertices;
    local_edges = edges;
    local_offsets = offsets;
  }

  GraphViewBase(Weight *edge_data, Vertex number_of_vertices,
                Edge number_of_edges)
      : edge_data(edge_data), prop(), number_of_vertices(number_of_vertices),
        number_of_edges(number_of_edges), local_vertices(nullptr),
        local_edges(nullptr), local_offsets(nullptr) {}

  bool HasData(void) const { return edge_data != nullptr; }
};

/**
 * @brief       A graph stored in COO (COOrdinate) format.
 *
 * @tparam Vertex   Type of vertex id
 * @tparam Edge     Type of edge id
 * @tparam Weight   Type of weight
 */
template <typename Vertex, typename Edge, typename Weight>
class GraphCOOView : public GraphViewBase<Vertex, Edge, Weight> {
public:
  Vertex *src_indices{nullptr}; ///< rowInd
  Vertex *dst_indices{nullptr}; ///< colInd

  /**
   * @brief      Default constructor
   */
  GraphCOOView() : GraphViewBase<Vertex, Edge, Weight>(nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing an edge list in a Graph.
   *
   *             GraphCOOView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  source_indices        This array of size E (number of edges)
   * contains the index of the
   * source for each edge. Indices must be in the range [0, V-1].
   * @param  destination_indices   This array of size E (number of edges)
   * contains the index of the
   * destination for each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array size E (number of edges) contains
   * the weight for each
   * edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCOOView(Vertex *src_indices, Vertex *dst_indices, Weight *edge_data,
               Vertex number_of_vertices, Edge number_of_edges)
      : GraphViewBase<Vertex, Edge, Weight>(edge_data, number_of_vertices,
                                            number_of_edges),
        src_indices(src_indices), dst_indices(dst_indices) {}
};

/**
 * @brief       Base class for graph stored in CSR (Compressed Sparse Row)
 * format or CSC (Compressed
 * Sparse Column) format
 *
 * @tparam Vertex   Type of vertex id
 * @tparam Edge     Type of edge id
 * @tparam Weight   Type of weight
 */
template <typename Vertex, typename Edge, typename Weight>
class GraphCompressedSparseBaseView
    : public GraphViewBase<Vertex, Edge, Weight> {
public:
  Edge *offsets{nullptr};   ///< CSR offsets
  Vertex *indices{nullptr}; ///< CSR indices

  /**
   * @brief      Fill the identifiers in the array with the source vertex
   * identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the
   * source vertex identifiers
   */
  void GetSourceIndices(Vertex *src_indices) const;

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCompressedSparseBaseView(Edge *offsets, Vertex *indices,
                                Weight *edge_data, Vertex number_of_vertices,
                                Edge number_of_edges)
      : GraphViewBase<Vertex, Edge, Weight>(edge_data, number_of_vertices,
                                            number_of_edges),
        offsets{offsets}, indices{indices} {}
};

/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam Vertex   Type of vertex id
 * @tparam Edge   Type of edge id
 * @tparam Weight   Type of weight
 */
template <typename Vertex, typename Edge, typename Weight>
class GraphCSRView
    : public GraphCompressedSparseBaseView<Vertex, Edge, Weight> {
public:
  /**
   * @brief      Default constructor
   */
  GraphCSRView()
      : GraphCompressedSparseBaseView<Vertex, Edge, Weight>(nullptr, nullptr,
                                                            nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSRView(Edge *offsets, Vertex *indices, Weight *edge_data,
               Vertex number_of_vertices, Edge number_of_edges)
      : GraphCompressedSparseBaseView<Vertex, Edge, Weight>(
            offsets, indices, edge_data, number_of_vertices, number_of_edges) {}
};
} // namespace legacy
} // namespace graph
} // namespace sfm
