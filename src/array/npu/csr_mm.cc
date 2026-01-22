/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/csr_mm.cc
 * @brief CSR Matrix Multiplication NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A, NDArray A_weights, const CSRMatrix& B,
    NDArray B_weights) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRMM on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_mm.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common type pairs
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLAscend, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLAscend, int64_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLAscend, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLAscend, int64_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

}  // namespace aten
}  // namespace dgl
