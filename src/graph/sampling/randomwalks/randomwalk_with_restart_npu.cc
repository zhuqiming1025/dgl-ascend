/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/sampling/randomwalks/randomwalk_with_restart_npu.cc
 * @brief DGL sampler - NPU implementation of metapath-based random walk with restart (placeholder)
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
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "RandomWalkWithRestart on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/sampling/randomwalks/randomwalk_with_restart_npu.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "RandomWalkWithStepwiseRestart on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/sampling/randomwalks/randomwalk_with_restart_npu.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template std::pair<IdArray, IdArray> RandomWalkWithRestart<kDGLAscend, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);
template std::pair<IdArray, IdArray> RandomWalkWithRestart<kDGLAscend, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);
template std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDGLAscend, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);
template std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDGLAscend, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);

}  // namespace impl

}  // namespace sampling

}  // namespace dgl
