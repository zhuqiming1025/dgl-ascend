/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/disjoint_union.cc
 * @brief Disjoint union NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>


namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
std::tuple<IdArray, IdArray, IdArray> _ComputePrefixSums(
    const std::vector<COOMatrix>& coos) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "_ComputePrefixSums on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/disjoint_union.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "DisjointUnionCoo on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/disjoint_union.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template std::tuple<IdArray, IdArray, IdArray> _ComputePrefixSums<kDGLAscend, int32_t>(
    const std::vector<COOMatrix>&);
template std::tuple<IdArray, IdArray, IdArray> _ComputePrefixSums<kDGLAscend, int64_t>(
    const std::vector<COOMatrix>&);
template COOMatrix DisjointUnionCoo<kDGLAscend, int32_t>(const std::vector<COOMatrix>&);
template COOMatrix DisjointUnionCoo<kDGLAscend, int64_t>(const std::vector<COOMatrix>&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
