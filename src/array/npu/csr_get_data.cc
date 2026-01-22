/**
 *  Copyright (c) 2021 by Contributors
 * @file array/npu/csr_get_data.cc
 * @brief Retrieve entries of a CSR matrix

 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <vector>

// Note: array_utils.h is not needed for placeholder implementation

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
void CollectDataFromSorted(
    const IdType* indices_data, const IdType* data, const IdType start,
    const IdType end, const IdType col, std::vector<IdType>* ret_vec) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CollectDataFromSorted on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_get_data.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids,
    NDArray weights, DType filler) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRGetData on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_get_data.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template NDArray CSRGetData<kDGLAscend, int32_t, float>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, float);
template NDArray CSRGetData<kDGLAscend, int64_t, float>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, float);
template NDArray CSRGetData<kDGLAscend, int32_t, double>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, double);
template NDArray CSRGetData<kDGLAscend, int64_t, double>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, double);
template NDArray CSRGetData<kDGLAscend, int32_t, int32_t>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, int32_t);
template NDArray CSRGetData<kDGLAscend, int64_t, int64_t>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, int64_t);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
