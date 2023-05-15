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

#include "COOtoCSR.cuh"

namespace sfm {
namespace graph {
// in-place versions:
//
// Explicit instantiation for int + float
template void
CooToCsrInplace<int, int, float>(legacy::GraphCOOView<int, int, float> &graph,
                                 legacy::GraphCSRView<int, int, float> &result);

// Explicit instantiation for int + double
template void CooToCsrInplace<int, int, double>(
    legacy::GraphCOOView<int, int, double> &graph,
    legacy::GraphCSRView<int, int, double> &result);

// Explicit instantiation for int + unsigned long long
template void CooToCsrInplace<int, int, unsigned long long>(
    legacy::GraphCOOView<int, int, unsigned long long> &graph,
    legacy::GraphCSRView<int, int, unsigned long long> &result);
} // namespace graph
} // namespace sfm