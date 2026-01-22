/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/sddmm.cc
 * @brief SDDMM NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include "../kernel_decl.h"

namespace dgl {
namespace aten {

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SDDMMCsr on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/sddmm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void SDDMMCsrHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& ufeat_eid,
    const std::vector<dgl_type_t>& out_eid) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SDDMMCsrHetero on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/sddmm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SDDMMCoo on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/sddmm.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <int XPU, typename IdType, typename DType>
void SDDMMCooHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& lhs_eid,
    const std::vector<dgl_type_t>& rhs_eid) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SDDMMCooHetero on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/sddmm.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common IdType/DType pairs
// SDDMMCsr
template void SDDMMCsr<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCsr<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCsr<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCsr<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, int, int);

// SDDMMCsrHetero
template void SDDMMCsrHetero<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCsrHetero<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCsrHetero<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCsrHetero<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);

// SDDMMCoo
template void SDDMMCoo<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCoo<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCoo<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, int, int);
template void SDDMMCoo<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, int, int);

// SDDMMCooHetero
template void SDDMMCooHetero<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCooHetero<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCooHetero<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SDDMMCooHetero<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);

}  // namespace aten
}  // namespace dgl
