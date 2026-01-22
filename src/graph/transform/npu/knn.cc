/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/transform/npu/knn.cc
 * @brief KNN and NNDescent NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <string>

namespace dgl {
using runtime::NDArray;
namespace transform {

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void KNN(
    const NDArray& data_points, const NDArray& query_points,
    const NDArray& data_offsets, const NDArray& query_offsets, int k,
    NDArray result, const std::string& algorithm) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "KNN on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/knn.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for KNN
template void KNN<kDGLAscend, float, int32_t>(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&, int,
    NDArray, const std::string&);
template void KNN<kDGLAscend, float, int64_t>(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&, int,
    NDArray, const std::string&);
template void KNN<kDGLAscend, double, int32_t>(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&, int,
    NDArray, const std::string&);
template void KNN<kDGLAscend, double, int64_t>(
    const NDArray&, const NDArray&, const NDArray&, const NDArray&, int,
    NDArray, const std::string&);

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void NNDescent(
    const NDArray& points, const NDArray& offsets, NDArray result,
    int k, int num_iters, int num_candidates, double delta) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "NNDescent on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/transform/npu/knn.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for NNDescent
template void NNDescent<kDGLAscend, float, int32_t>(
    const NDArray&, const NDArray&, NDArray, int, int, int, double);
template void NNDescent<kDGLAscend, float, int64_t>(
    const NDArray&, const NDArray&, NDArray, int, int, int, double);
template void NNDescent<kDGLAscend, double, int32_t>(
    const NDArray&, const NDArray&, NDArray, int, int, int, double);
template void NNDescent<kDGLAscend, double, int64_t>(
    const NDArray&, const NDArray&, NDArray, int, int, int, double);

}  // namespace transform
}  // namespace dgl
