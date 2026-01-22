/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/spmat_op_impl_csr.cc
 * @brief CSR matrix operations NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <vector>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

///////////////////////////// CSRIsNonZero /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRIsNonZero (scalar) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType>
NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRIsNonZero (array) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRIsNonZero
template bool CSRIsNonZero<kDGLAscend, int32_t>(CSRMatrix, int64_t, int64_t);
template bool CSRIsNonZero<kDGLAscend, int64_t>(CSRMatrix, int64_t, int64_t);
template NDArray CSRIsNonZero<kDGLAscend, int32_t>(CSRMatrix, NDArray, NDArray);
template NDArray CSRIsNonZero<kDGLAscend, int64_t>(CSRMatrix, NDArray, NDArray);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRSliceRows on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRSliceRows (with NDArray) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, runtime::NDArray);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, runtime::NDArray);

///////////////////////////// CSRGetRowNNZ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRGetRowNNZ on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRGetRowNNZ
template NDArray CSRGetRowNNZ<kDGLAscend, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDGLAscend, int64_t>(CSRMatrix, NDArray);

///////////////////////////// CSRTranspose /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRTranspose on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRTranspose
template CSRMatrix CSRTranspose<kDGLAscend, int32_t>(CSRMatrix);
template CSRMatrix CSRTranspose<kDGLAscend, int64_t>(CSRMatrix);

///////////////////////////// CSRSliceMatrix /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, NDArray rows, NDArray cols) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRSliceMatrix on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRSliceMatrix
template CSRMatrix CSRSliceMatrix<kDGLAscend, int32_t>(CSRMatrix, NDArray, NDArray);
template CSRMatrix CSRSliceMatrix<kDGLAscend, int64_t>(CSRMatrix, NDArray, NDArray);

///////////////////////////// CSRToCOO /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRToCOO on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRToCOO
template COOMatrix CSRToCOO<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOO<kDGLAscend, int64_t>(CSRMatrix);

///////////////////////////// CSRToCOODataAsOrder /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRToCOODataAsOrder on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRToCOODataAsOrder
template COOMatrix CSRToCOODataAsOrder<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOODataAsOrder<kDGLAscend, int64_t>(CSRMatrix);

///////////////////////////// CSRGetDataAndIndices /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
std::vector<NDArray> CSRGetDataAndIndices(CSRMatrix csr, NDArray rows, NDArray cols) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRGetDataAndIndices on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_csr.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRGetDataAndIndices
template std::vector<NDArray> CSRGetDataAndIndices<kDGLAscend, int32_t>(CSRMatrix, NDArray, NDArray);
template std::vector<NDArray> CSRGetDataAndIndices<kDGLAscend, int64_t>(CSRMatrix, NDArray, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
