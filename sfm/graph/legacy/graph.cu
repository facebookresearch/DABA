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

#include <sfm/graph/legacy/graph.hpp>
#include <sfm/graph/utilities/error.hpp>
#include <sfm/graph/utilities/graph_utils.cuh>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace sfm {
namespace graph {
namespace legacy {
template <typename VT, typename ET, typename WT>
void GraphViewBase<VT, ET, WT>::GetVertexIdentifiers(VT *identifiers) const {
  graph::detail::sequence<VT>(number_of_vertices, identifiers);
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBaseView<VT, ET, WT>::GetSourceIndices(
    VT *src_indices) const {
  GRAPH_EXPECTS(offsets != nullptr, "No graph specified");
  graph::detail::offsets_to_indices<VT>(
      offsets, GraphViewBase<VT, ET, WT>::number_of_vertices, src_indices);
}

// explicit instantiation
template class GraphViewBase<int32_t, int32_t, float>;
template class GraphViewBase<int32_t, int32_t, double>;
template class GraphViewBase<int32_t, int32_t, unsigned long long>;
template class GraphCOOView<int32_t, int32_t, float>;
template class GraphCOOView<int32_t, int32_t, double>;
template class GraphCOOView<int32_t, int32_t, unsigned long long>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, float>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, double>;
template class GraphCompressedSparseBaseView<int32_t, int32_t,
                                             unsigned long long>;
template class GraphCSRView<int32_t, int32_t, float>;
template class GraphCSRView<int32_t, int32_t, double>;
template class GraphCSRView<int32_t, int32_t, unsigned long long>;
} // namespace legacy
} // namespace graph
} // namespace sfm
