/**
 *  Copyright (c) 2019 by Contributors
 * @file array/npu/csr_sum.cc
 * @brief CSR Summation

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

namespace {

template <typename IdType>
void CountNNZPerRow(
    const std::vector<const IdType*>& A_indptr,
    const std::vector<const IdType*>& A_indices, IdType* C_indptr_data,
    int64_t M) {
  LOG(FATAL) << "CountNNZPerRow on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_sum.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <typename IdType>
IdType ComputeIndptrInPlace(IdType* C_indptr_data, int64_t M) {
  LOG(FATAL) << "ComputeIndptrInPlace on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_sum.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <typename IdType, typename DType>
void ComputeIndicesAndData(
    const std::vector<const IdType*>& A_indptr,
    const std::vector<const IdType*>& A_indices,
    const std::vector<const IdType*>& A_eids,
    const std::vector<const DType*>& A_data, IdType* C_indptr_data,
    IdType* C_indices_data, DType* C_weights_data, int64_t M) {
  LOG(FATAL) << "ComputeIndicesAndData on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_sum.cc using Ascend kernels.";
  __builtin_unreachable();
}

}  // namespace

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A, const std::vector<NDArray>& A_weights) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRSum on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/csr_sum.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLAscend, int32_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLAscend, int64_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLAscend, int32_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLAscend, int64_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);

}  // namespace aten
}  // namespace dgl
