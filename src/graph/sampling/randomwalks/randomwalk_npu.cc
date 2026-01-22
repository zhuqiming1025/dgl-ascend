/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/sampling/randomwalks/randomwalk_npu.cc
 * @brief DGL sampler - NPU implementation of random walk (placeholder)
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dmlc/logging.h>

#include <utility>
#include <vector>

#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "RandomWalk on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/sampling/randomwalks/randomwalk_npu.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SelectPinSageNeighbors on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/sampling/randomwalks/randomwalk_npu.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template std::pair<IdArray, IdArray> RandomWalk<kDGLAscend, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template std::pair<IdArray, IdArray> RandomWalk<kDGLAscend, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors<kDGLAscend, int32_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);
template std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors<kDGLAscend, int64_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);

}  // namespace impl

}  // namespace sampling

}  // namespace dgl
