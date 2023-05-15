// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdlib>
#include <limits>
#include <sfm/graph/graph.cuh>
#include <sfm/utils/utils.cuh>

namespace sfm {
namespace ba {
namespace clustering {
template <typename Float, typename Size, typename Vertex,
          typename Weight = uint64_t, typename Score = uint64_t>
void Cluster(const sfm::graph::Handle &handle, Size number_of_cameras,
             Size number_of_points, const Vertex *src_indices,
             const Vertex *dst_indices, const Weight *weights,
             Size number_of_edges, Size targeted_number_of_clusters,
             Vertex *camera_clustering, Vertex *point_clustering,
             Float &modularity, Size &number_of_clusters,
             Float initial_resolution = 1.0, Float refined_resolution = 2.0,
             bool memory_efficient = false, Size max_iters = 1000);
} // namespace clustering
} // namespace ba
} // namespace sfm
