// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
/*
 * COOtoCSR_kernels.cuh
 *
 *  Created on: Mar 8, 2018
 *      Author: jwyles
 */

#pragma once

#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <sfm/graph/legacy/graph.hpp>
#include <sfm/graph/utilities/error.hpp>
#include <sfm/graph/utilities/graph_utils.cuh>

namespace sfm {
namespace graph {
namespace detail {
/**
 * @brief     Sort input graph and find the total number of vertices
 *
 * Lexicographically sort a COO view and find the total number of vertices
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int
 * (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int
 * (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or
 * double.
 *
 * @param[in] graph        The input graph object
 * @param[in] stream_view  The cuda stream for kernel calls
 *
 * @param[out] result      Total number of vertices
 */
template <typename VT, typename ET, typename WT>
VT Sort(legacy::GraphCOOView<VT, ET, WT> &graph, cudaStream_t stream) {
  VT max_src_id;
  VT max_dst_id;
  if (graph.HasData()) {
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), graph.dst_indices,
                               graph.dst_indices + graph.number_of_edges,
                               thrust::make_zip_iterator(thrust::make_tuple(
                                   graph.src_indices, graph.edge_data)));
    CHECK_CUDA(cudaMemcpy(&max_dst_id,
                          &(graph.dst_indices[graph.number_of_edges - 1]),
                          sizeof(VT), cudaMemcpyDefault));
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), graph.src_indices,
                               graph.src_indices + graph.number_of_edges,
                               thrust::make_zip_iterator(thrust::make_tuple(
                                   graph.dst_indices, graph.edge_data)));
    CHECK_CUDA(cudaMemcpy(&max_src_id,
                          &(graph.src_indices[graph.number_of_edges - 1]),
                          sizeof(VT), cudaMemcpyDefault));
  } else {
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), graph.dst_indices,
                               graph.dst_indices + graph.number_of_edges,
                               graph.src_indices);
    CHECK_CUDA(cudaMemcpy(&max_dst_id,
                          &(graph.dst_indices[graph.number_of_edges - 1]),
                          sizeof(VT), cudaMemcpyDefault));
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), graph.src_indices,
                               graph.src_indices + graph.number_of_edges,
                               graph.dst_indices);
    CHECK_CUDA(cudaMemcpy(&max_src_id,
                          &(graph.src_indices[graph.number_of_edges - 1]),
                          sizeof(VT), cudaMemcpyDefault));
  }
  return std::max(max_src_id, max_dst_id) + 1;
}

template <typename VT, typename ET>
void FillOffset(VT *source, ET *offsets, VT number_of_vertices,
                ET number_of_edges, cudaStream_t stream) {
  thrust::fill(thrust::cuda::par.on(stream), offsets,
               offsets + number_of_vertices + 1, number_of_edges);
  if (number_of_edges > 0) {
    thrust::for_each(thrust::cuda::par.on(stream),
                     thrust::make_counting_iterator<ET>(1),
                     thrust::make_counting_iterator<ET>(number_of_edges),
                     [source, offsets] __device__(ET index) {
                       VT id = source[index];
                       if (id != source[index - 1]) {
                         offsets[id] = index;
                       }
                     });
    thrust::device_ptr<VT> src = thrust::device_pointer_cast(source);
    thrust::device_ptr<ET> off = thrust::device_pointer_cast(offsets);
    off[src[0]] = ET{0};
  }

  auto iter = thrust::make_reverse_iterator(offsets + number_of_vertices + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), iter,
                         iter + number_of_vertices + 1, iter,
                         thrust::minimum<ET>());
}

} // namespace detail

template <typename VT, typename ET, typename WT>
void CooToCsrInplace(legacy::GraphCOOView<VT, ET, WT> &graph,
                     legacy::GraphCSRView<VT, ET, WT> &result) {
  cudaStream_t stream{nullptr};

  detail::Sort(graph, stream);
  detail::FillOffset(graph.src_indices, result.offsets,
                     graph.number_of_vertices, graph.number_of_edges, stream);

  CHECK_CUDA(cudaMemcpy(result.indices, graph.dst_indices,
                        sizeof(VT) * graph.number_of_edges, cudaMemcpyDefault));
  if (graph.HasData())
    CHECK_CUDA(cudaMemcpy(result.edge_data, graph.edge_data,
                          sizeof(WT) * graph.number_of_edges,
                          cudaMemcpyDefault));
}
} // namespace graph
} // namespace sfm