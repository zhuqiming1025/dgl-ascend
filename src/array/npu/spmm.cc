#include <dgl/array.h>
#include <dmlc/logging.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include "../cpu/spmm.h"
#include "../cpu/spmm_binary_ops.h"
#include <algorithm>
#ifdef DGL_USE_ASCEND
#include <acl/acl_rt.h>
#endif

namespace dgl {
namespace aten {
using namespace cpu;
 
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  
  // For NPU, we use CPU implementation as a fallback by copying data to CPU,
  // processing there, and copying back. This ensures correctness.
  // TODO: Optimize with native NPU kernels in the future
  
  const DGLContext cpu_ctx = DGLContext{kDGLCPU, 0};
  
  // Copy inputs to CPU
  CSRMatrix csr_cpu = csr.CopyTo(cpu_ctx);
  NDArray ufeat_cpu = ufeat.CopyTo(cpu_ctx);
  NDArray efeat_cpu = IsNullArray(efeat) ? NullArray() : efeat.CopyTo(cpu_ctx);
  
  // Create output array on CPU (don't copy from NPU, create fresh and initialize to zero)
  const int64_t dim = bcast.out_len;
  NDArray out_cpu = NDArray::Empty({csr.num_rows, dim}, out->dtype, cpu_ctx);
  
  // Initialize output to zero on CPU (required for sum reduction)
  DType* out_cpu_ptr = out_cpu.Ptr<DType>();
  std::fill(out_cpu_ptr, out_cpu_ptr + csr.num_rows * dim, static_cast<DType>(0));
  
  // Copy out_aux to CPU if needed
  std::vector<NDArray> out_aux_cpu;
  if (!out_aux.empty()) {
    out_aux_cpu.reserve(out_aux.size());
    for (const auto& aux : out_aux) {
      out_aux_cpu.push_back(IsNullArray(aux) ? NullArray() : aux.CopyTo(cpu_ctx));
    }
  }
  
  // Call CPU implementation
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr_cpu, ufeat_cpu, efeat_cpu, out_cpu);
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_OP(op, Op, {
      DType* out_off = out_cpu.Ptr<DType>();
      if (reduce == "max") {
        std::fill(
            out_off, out_off + csr_cpu.num_rows * dim, cpu::op::Max<DType>::zero);
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
            bcast, csr_cpu, ufeat_cpu, efeat_cpu, out_cpu, out_aux_cpu[0], out_aux_cpu[1]);
      } else {
        std::fill(
            out_off, out_off + csr_cpu.num_rows * dim, cpu::op::Min<DType>::zero);
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
            bcast, csr_cpu, ufeat_cpu, efeat_cpu, out_cpu, out_aux_cpu[0], out_aux_cpu[1]);
      }
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
  
  // Copy results back to NPU
  // Use CopyFromTo for explicit copy, which handles device-to-device transfers correctly
  // Note: CopyDataFromTo in ascend_device_api.cc uses synchronous aclrtMemcpy for CPU->NPU,
  // so the copy is already complete when CopyFromTo returns. However, we still call StreamSync
  // to ensure any other pending operations on the stream are completed.
  DGLArray* from_array = const_cast<DGLArray*>(out_cpu.operator->());
  DGLArray* to_array = const_cast<DGLArray*>(out.operator->());
  NDArray::CopyFromTo(from_array, to_array);
  
  // Synchronize stream to ensure all operations are complete
  // This ensures PyTorch can see the updated data when it accesses the tensor
  runtime::DeviceAPI::Get(out->ctx)->StreamSync(out->ctx, nullptr);
  
  if (!out_aux.empty() && !out_aux_cpu.empty()) {
    for (size_t i = 0; i < out_aux.size(); ++i) {
      if (!IsNullArray(out_aux[i]) && !IsNullArray(out_aux_cpu[i])) {
        out_aux[i].CopyFrom(out_aux_cpu[i]);
      }
    }
  }
}
 
template <int XPU, typename IdType, typename DType>
void SpMMCsrHetero(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_eid,
    const std::vector<dgl_type_t>& out_eid) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SpMMCsrHetero (gspmm_hetero) on Ascend NPU is not implemented yet.";
}
 
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SpMMCoo (gspmm) on Ascend NPU is not implemented yet.";
}
 
// Force instantiation for common IdType/DType pairs so the linker finds symbols.
template void SpMMCsr<kDGLAscend, int32_t, float>(
    const std::string&, const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCsr<kDGLAscend, int64_t, float>(
    const std::string&, const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCsr<kDGLAscend, int32_t, double>(
    const std::string&, const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCsr<kDGLAscend, int64_t, double>(
    const std::string&, const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
 
template void SpMMCoo<kDGLAscend, int32_t, float>(
    const std::string&, const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCoo<kDGLAscend, int64_t, float>(
    const std::string&, const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCoo<kDGLAscend, int32_t, double>(
    const std::string&, const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);
template void SpMMCoo<kDGLAscend, int64_t, double>(
    const std::string&, const std::string&, const BcastOff&, const COOMatrix&,
    NDArray, NDArray, NDArray, std::vector<NDArray>);

template void SpMMCsrHetero<kDGLAscend, int32_t, float>(
    const std::string&, const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*,
    std::vector<std::vector<NDArray>>*,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SpMMCsrHetero<kDGLAscend, int64_t, float>(
    const std::string&, const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*,
    std::vector<std::vector<NDArray>>*,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SpMMCsrHetero<kDGLAscend, int32_t, double>(
    const std::string&, const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*,
    std::vector<std::vector<NDArray>>*,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
template void SpMMCsrHetero<kDGLAscend, int64_t, double>(
    const std::string&, const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&,
    const std::vector<NDArray>&, std::vector<NDArray>*,
    std::vector<std::vector<NDArray>>*,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&);
 
}  // namespace aten
}  // namespace dgl

