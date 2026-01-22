/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/transform/npu/to_block.cc
 * @brief ToBlock NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include "../../transform/to_block.h"
#include <dgl/base_heterograph.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLAscend, int32_t>(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes) {
  LOG(FATAL) << "ToBlock on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/to_block.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLAscend, int64_t>(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes) {
  LOG(FATAL) << "ToBlock on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/to_block.cc using Ascend kernels.";
  __builtin_unreachable();
}

}  // namespace transform
}  // namespace dgl
