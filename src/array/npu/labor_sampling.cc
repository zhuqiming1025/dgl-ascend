/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/labor_sampling.cc
 * @brief Labor sampling NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>


namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRLaborSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/labor_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOLaborSampling on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/labor_sampling.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLAscend, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLAscend, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLAscend, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLAscend, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);

template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLAscend, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLAscend, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLAscend, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLAscend, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
