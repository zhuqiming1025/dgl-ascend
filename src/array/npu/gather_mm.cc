/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/gather_mm.cc
 * @brief GatherMM and SegmentMM NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
void GatherMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GatherMM on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/gather_mm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GatherMMScatter on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/gather_mm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SegmentMM on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/gather_mm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SegmentMMBackwardB on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/gather_mm.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template void GatherMM<kDGLAscend, int32_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray);
template void GatherMM<kDGLAscend, int64_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray);
template void GatherMM<kDGLAscend, int32_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray);
template void GatherMM<kDGLAscend, int64_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray);

template void GatherMMScatter<kDGLAscend, int32_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray, const NDArray);
template void GatherMMScatter<kDGLAscend, int64_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray, const NDArray);
template void GatherMMScatter<kDGLAscend, int32_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray, const NDArray);
template void GatherMMScatter<kDGLAscend, int64_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, const NDArray, const NDArray);

template void SegmentMM<kDGLAscend, int32_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, bool, bool);
template void SegmentMM<kDGLAscend, int64_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray, bool, bool);
template void SegmentMM<kDGLAscend, int32_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, bool, bool);
template void SegmentMM<kDGLAscend, int64_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray, bool, bool);

template void SegmentMMBackwardB<kDGLAscend, int32_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray);
template void SegmentMMBackwardB<kDGLAscend, int64_t, float>(
    const NDArray, const NDArray, NDArray, const NDArray);
template void SegmentMMBackwardB<kDGLAscend, int32_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray);
template void SegmentMMBackwardB<kDGLAscend, int64_t, double>(
    const NDArray, const NDArray, NDArray, const NDArray);

}  // namespace aten
}  // namespace dgl
