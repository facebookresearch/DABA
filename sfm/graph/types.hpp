// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/graph/legacy/graph.hpp>
#include <thrust/device_vector.h>

namespace sfm {
namespace graph {
template <typename Vertex, typename Score> struct ScoreFunction {
  virtual __host__ __device__ Score operator()(Vertex) const = 0;
};

class Handle {
public:
  Handle() {
    cudaGetDevice(&m_device);
    cudaStreamCreate(&m_stream);
  }

  cudaStream_t const &GetStream() const { return m_stream; }

  int GetDevice() const {
    thrust::cuda::par.on(m_stream);
    return m_device;
  }

  thrust::cuda_cub::execute_on_stream GetThrustPolicy() const {
    return thrust::cuda::par.on(m_stream);
  }

private:
  int m_device;
  cudaStream_t m_stream;
};

template <typename Vertex, typename Score>
struct ConstantScoreFunction : public ScoreFunction<Vertex, Score> {
  ConstantScoreFunction(Score val) : m_val(val) {}

  virtual __host__ __device__ Score operator()(Vertex) const { return m_val; }

private:
  Score m_val;
};

template <typename Vertex, typename Score>
struct ZeroScoreFunction : public ConstantScoreFunction<Vertex, Score> {
  ZeroScoreFunction() : ConstantScoreFunction<Vertex, Score>(0) {}
};

template <typename Vertex, typename Score>
struct OneScoreFunction : public ConstantScoreFunction<Vertex, Score> {
  OneScoreFunction() : ConstantScoreFunction<Vertex, Score>(1) {}
};

template <typename Vertex> class Dendrogram {
public:
  void AddLevel(Vertex first_index, Vertex num_verts) {
    m_level_ptr.push_back(
        std::make_unique<thrust::device_vector<Vertex>>(num_verts));
    m_level_first_index.push_back(first_index);
  }

  void PopBack() {
    m_level_ptr.pop_back();
    m_level_first_index.pop_back();
  }

  size_t CurrentLevel() const { return m_level_ptr.size() - 1; }

  size_t NumLevels() const { return m_level_ptr.size(); }

  Vertex const *GetLevelPtrNoCheck(size_t level) const {
    return m_level_ptr[level]->data().get();
  }

  Vertex *GetLevelPtrNoCheck(size_t level) {
    return m_level_ptr[level]->data().get();
  }

  size_t GetLevelSizeNoCheck(size_t level) const {
    return m_level_ptr[level]->size();
  }

  Vertex GetLevelFirstIndexNoCheck(size_t level) const {
    return m_level_first_index[level];
  }

  Vertex const *CurrentLevelBegin() const {
    return GetLevelPtrNoCheck(CurrentLevel());
  }

  Vertex const *CurrentLevelEnd() const {
    return CurrentLevelBegin() + CurrentLevelSize();
  }

  Vertex *CurrentLevelBegin() { return GetLevelPtrNoCheck(CurrentLevel()); }

  Vertex *CurrentLevelEnd() { return CurrentLevelBegin() + CurrentLevelSize(); }

  size_t CurrentLevelSize() const {
    return GetLevelSizeNoCheck(CurrentLevel());
  }

  Vertex CurrentLevelFirstIndex() const {
    return GetLevelFirstIndexNoCheck(CurrentLevel());
  }

private:
  std::vector<Vertex> m_level_first_index;
  std::vector<std::unique_ptr<thrust::device_vector<Vertex>>> m_level_ptr;
};

template <typename Vertex, typename Edge, typename Weight>
using GraphCSRView = typename legacy::GraphCSRView<Vertex, Edge, Weight>;

template <typename Vertex, typename Edge, typename Weight>
using GraphCOOView = typename legacy::GraphCOOView<Vertex, Edge, Weight>;
} // namespace graph
} // namespace sfm