// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <glog/logging.h>

namespace sfm {
namespace graph {
/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a
 * condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected
 * to be true with optinal format tagas
 */
#define GRAPH_EXPECTS(cond, fmt, ...)                                          \
  if (!(cond)) {                                                               \
    LOG(ERROR) << fmt << std::endl;                                           \
    exit(-1);                                                                  \
  }

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path
 * is erroneous with optinal format tagas
 * @throw always throws cugraph::logic_error
 */
#define GRAPH_FAIL(fmt, ...)                                                   \
  if (!(cond)) {                                                               \
    LOG(ERROR) << fmt << std::endl;                                           \
    exit(-1);                                                                  \
  }
} // namespace graph
} // namespace sfm