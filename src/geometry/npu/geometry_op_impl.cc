/**
 *  Copyright (c) 2020 by Contributors
 * @file geometry/npu/geometry_op_impl.cc
 * @brief Geometry operations NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>
#include "../../geometry/geometry_op.h"

namespace dgl {
using runtime::NDArray;
using aten::CSRMatrix;
namespace geometry {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
void NeighborMatching(const CSRMatrix& csr, NDArray result) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "NeighborMatching on Ascend NPU is not implemented yet. "
             << "Please implement src/geometry/npu/geometry_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for NeighborMatching - use NDArray to match linker expectations
template void NeighborMatching<kDGLAscend, int32_t>(const CSRMatrix&, NDArray);
template void NeighborMatching<kDGLAscend, int64_t>(const CSRMatrix&, NDArray);

template <DGLDeviceType XPU, typename DType, typename IdType>
void FarthestPointSampler(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    NDArray start_idx, NDArray result) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "FarthestPointSampler on Ascend NPU is not implemented yet. "
             << "Please implement src/geometry/npu/geometry_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for FarthestPointSampler - use NDArray to match linker expectations
template void FarthestPointSampler<kDGLAscend, double, int32_t>(
    NDArray, int64_t, int64_t, NDArray, NDArray, NDArray);
template void FarthestPointSampler<kDGLAscend, double, int64_t>(
    NDArray, int64_t, int64_t, NDArray, NDArray, NDArray);
template void FarthestPointSampler<kDGLAscend, float, int32_t>(
    NDArray, int64_t, int64_t, NDArray, NDArray, NDArray);
template void FarthestPointSampler<kDGLAscend, float, int64_t>(
    NDArray, int64_t, int64_t, NDArray, NDArray, NDArray);

template <DGLDeviceType XPU, typename DType, typename IdType>
void WeightedNeighborMatching(
    const CSRMatrix& csr, NDArray weight, NDArray result) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "WeightedNeighborMatching on Ascend NPU is not implemented yet. "
             << "Please implement src/geometry/npu/geometry_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for WeightedNeighborMatching - use NDArray to match linker expectations
template void WeightedNeighborMatching<kDGLAscend, double, int32_t>(
    const CSRMatrix&, NDArray, NDArray);
template void WeightedNeighborMatching<kDGLAscend, double, int64_t>(
    const CSRMatrix&, NDArray, NDArray);
template void WeightedNeighborMatching<kDGLAscend, float, int32_t>(
    const CSRMatrix&, NDArray, NDArray);
template void WeightedNeighborMatching<kDGLAscend, float, int64_t>(
    const CSRMatrix&, NDArray, NDArray);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
