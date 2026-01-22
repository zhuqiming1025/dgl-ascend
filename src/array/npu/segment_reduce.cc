/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/segment_reduce.cc
 * @brief Segment reduce NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dmlc/logging.h>

#include <string>
#include <vector>

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
void SegmentReduce(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SegmentReduce on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/segment_reduce.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "ScatterAdd on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/segment_reduce.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "UpdateGradMinMax_hetero on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/segment_reduce.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "BackwardSegmentCmp on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/segment_reduce.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template void SegmentReduce<kDGLAscend, int32_t, float>(
    const std::string&, NDArray, NDArray, NDArray, NDArray);
template void SegmentReduce<kDGLAscend, int64_t, float>(
    const std::string&, NDArray, NDArray, NDArray, NDArray);
template void SegmentReduce<kDGLAscend, int32_t, double>(
    const std::string&, NDArray, NDArray, NDArray, NDArray);
template void SegmentReduce<kDGLAscend, int64_t, double>(
    const std::string&, NDArray, NDArray, NDArray, NDArray);

template void ScatterAdd<kDGLAscend, int32_t, float>(NDArray, NDArray, NDArray);
template void ScatterAdd<kDGLAscend, int64_t, float>(NDArray, NDArray, NDArray);
template void ScatterAdd<kDGLAscend, int32_t, double>(NDArray, NDArray, NDArray);
template void ScatterAdd<kDGLAscend, int64_t, double>(NDArray, NDArray, NDArray);

template void UpdateGradMinMax_hetero<kDGLAscend, int32_t, float>(
    const HeteroGraphPtr&, const std::string&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*);
template void UpdateGradMinMax_hetero<kDGLAscend, int64_t, float>(
    const HeteroGraphPtr&, const std::string&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*);
template void UpdateGradMinMax_hetero<kDGLAscend, int32_t, double>(
    const HeteroGraphPtr&, const std::string&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*);
template void UpdateGradMinMax_hetero<kDGLAscend, int64_t, double>(
    const HeteroGraphPtr&, const std::string&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*);

template void BackwardSegmentCmp<kDGLAscend, int32_t, float>(NDArray, NDArray, NDArray);
template void BackwardSegmentCmp<kDGLAscend, int64_t, float>(NDArray, NDArray, NDArray);
template void BackwardSegmentCmp<kDGLAscend, int32_t, double>(NDArray, NDArray, NDArray);
template void BackwardSegmentCmp<kDGLAscend, int64_t, double>(NDArray, NDArray, NDArray);

}  // namespace aten
}  // namespace dgl
