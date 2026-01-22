/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/rowwise_sampling.cc
 * @brief rowwise sampling

 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>


namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename... Args>
auto DoubleSlice(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "DoubleSlice on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingNumPicksFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingNumPicksFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingPickFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingPickFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingRangePickFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingRangePickFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingUniformNumPicksFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingUniformNumPicksFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingUniformPickFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingUniformPickFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingUniformRangePickFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingUniformRangePickFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingBiasedNumPicksFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingBiasedNumPicksFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto GetSamplingBiasedPickFn(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetSamplingBiasedPickFn on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdxType, typename DType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWiseSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRRowWiseSampling
template COOMatrix CSRRowWiseSampling<kDGLAscend, int32_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int64_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int32_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int64_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int32_t, int8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int64_t, int8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int32_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int64_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);

template <DGLDeviceType XPU, typename... Args>
auto CSRRowWiseSamplingFused(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWiseSamplingFused on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto CSRRowWisePerEtypeSampling(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWisePerEtypeSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWiseSamplingUniform on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRRowWiseSamplingUniform
template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLAscend, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

template <DGLDeviceType XPU, typename... Args>
auto CSRRowWiseSamplingUniformFused(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWiseSamplingUniformFused on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto CSRRowWisePerEtypeSamplingUniform(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWisePerEtypeSamplingUniform on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto CSRRowWiseSamplingBiased(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRRowWiseSamplingBiased on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COORowWiseSampling(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COORowWiseSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COORowWisePerEtypeSampling(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COORowWisePerEtypeSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COORowWiseSamplingUniform(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COORowWiseSamplingUniform on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COORowWisePerEtypeSamplingUniform(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COORowWisePerEtypeSamplingUniform on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/rowwise_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl
