/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/transform/npu/compact.cc
 * @brief Compact graphs NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include "../../transform/compact.h"
#include <dgl/base_heterograph.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <utility>
#include <vector>

namespace dgl {
namespace transform {

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLAscend, int32_t>(
    const std::vector<HeteroGraphPtr>& graphs,
    const std::vector<IdArray>& always_preserve) {
  LOG(FATAL) << "CompactGraphs on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/compact.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLAscend, int64_t>(
    const std::vector<HeteroGraphPtr>& graphs,
    const std::vector<IdArray>& always_preserve) {
  LOG(FATAL) << "CompactGraphs on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/compact.cc using Ascend kernels.";
  __builtin_unreachable();
}

}  // namespace transform
}  // namespace dgl
