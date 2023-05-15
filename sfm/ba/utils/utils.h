// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/types.h>
#include <vector>

namespace sfm {
namespace ba {
int BundlerDatasetToBALDataset(std::string bundler_file, std::string bal_file,
                               int_t &num_cameras, int_t &num_points,
                               int_t &num_measurements);
} // namespace ba
} // namespace sfm