/**
 *  Copyright (c) 2019 by Contributors
 * @file array/npu/spmat_op_impl.cc
 * @brief NPU implementation of COO sparse matrix operators

 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

#include <cstddef>
// Note: array_utils.h is not needed for placeholder implementation

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename... Args>
auto COOIsNonZero(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOIsNonZero on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOHasDuplicate(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOHasDuplicate on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOGetRowNNZ(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOGetRowNNZ on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOGetRowDataAndIndices(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOGetRowDataAndIndices on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOGetData(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOGetData on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOGetDataAndIndices(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOGetDataAndIndices on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOTranspose(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOTranspose on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto SortedCOOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "SortedCOOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto UnSortedSparseCOOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "UnSortedSparseCOOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto UnSortedDenseCOOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "UnSortedDenseCOOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto UnSortedSmallCOOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "UnSortedSmallCOOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto WhichCOOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "WhichCOOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOToCSR(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOToCSR on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOSliceRows(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOSliceRows on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOSliceMatrix(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOSliceMatrix on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename... Args>
auto COOReorder(Args&&... args) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOReorder on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

///////////////////////////// COOToCSR /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  
  const int64_t N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const DGLContext& ctx = coo.row->ctx;
  
  // For NPU, we use CPU implementation as a fallback by copying data to CPU,
  // processing there, and copying back. This ensures correctness.
  // TODO: Optimize with native NPU kernels in the future
  const DGLContext cpu_ctx = DGLContext{kDGLCPU, 0};
  
  // Copy COO to CPU for processing
  COOMatrix coo_cpu = coo.CopyTo(cpu_ctx);
  
  // Process on CPU using the existing CPU implementation logic
  // Allocate output arrays on CPU first
  NDArray ret_indptr_cpu = NDArray::Empty({N + 1}, coo_cpu.row->dtype, cpu_ctx);
  NDArray ret_indices_cpu = NDArray::Empty({NNZ}, coo_cpu.row->dtype, cpu_ctx);
  NDArray ret_data_cpu;
  
  const bool has_data = COOHasData(coo_cpu);
  if (has_data) {
    ret_data_cpu = NDArray::Empty({NNZ}, coo_cpu.data->dtype, cpu_ctx);
  } else {
    ret_data_cpu = NDArray::Empty({NNZ}, coo_cpu.row->dtype, cpu_ctx);
  }
  
  // Get CPU data pointers
  IdType* indptr_data = static_cast<IdType*>(ret_indptr_cpu->data);
  IdType* indices_data = static_cast<IdType*>(ret_indices_cpu->data);
  IdType* data_ptr = static_cast<IdType*>(ret_data_cpu->data);
  
  const IdType* row_data = static_cast<IdType*>(coo_cpu.row->data);
  const IdType* col_data = static_cast<IdType*>(coo_cpu.col->data);
  const IdType* coo_data = has_data ? static_cast<IdType*>(coo_cpu.data->data) : nullptr;
  
  if (coo_cpu.row_sorted) {
    // Sorted COO: directly build indptr (similar to SortedCOOToCSR)
    // Initialize indptr
    std::fill(indptr_data, indptr_data + N + 1, static_cast<IdType>(0));
    
    if (NNZ > 0) {
      // Build indptr by detecting row changes
      int64_t row = 0;
      for (int64_t i = 0; i < NNZ; ++i) {
        while (row != row_data[i]) {
          ++row;
          indptr_data[row] = static_cast<IdType>(i);
        }
      }
      // Fill remaining rows
      while (row < N) {
        ++row;
        indptr_data[row] = static_cast<IdType>(NNZ);
      }
      
      // Copy indices and data (already in sorted order)
      for (int64_t i = 0; i < NNZ; ++i) {
        indices_data[i] = col_data[i];
        data_ptr[i] = coo_data ? coo_data[i] : static_cast<IdType>(i);
      }
    }
  } else {
    // Unsorted COO: use counting method (similar to UnSortedSmallCOOToCSR)
    // Initialize indptr to zero
    std::fill(indptr_data, indptr_data + N + 1, static_cast<IdType>(0));
    
    // Count elements in each row
    for (int64_t i = 0; i < NNZ; ++i) {
      indptr_data[row_data[i]]++;
    }
    
    // Convert counts to cumulative indices (prefix sum)
    IdType cumsum = 0;
    for (int64_t i = 0; i < N; ++i) {
      const IdType temp = indptr_data[i];
      indptr_data[i] = cumsum;
      cumsum += temp;
    }
    indptr_data[N] = cumsum;
    
    // Fill indices and data arrays
    for (int64_t i = 0; i < NNZ; ++i) {
      const IdType r = row_data[i];
      const IdType pos = indptr_data[r];
      indices_data[pos] = col_data[i];
      data_ptr[pos] = coo_data ? coo_data[i] : static_cast<IdType>(i);
      indptr_data[r]++;
    }
    
    // Restore indptr (shift back)
    for (int64_t i = N; i > 0; --i) {
      indptr_data[i] = indptr_data[i - 1];
    }
    indptr_data[0] = 0;
  }
  
  // Copy results back to NPU
  NDArray ret_indptr = ret_indptr_cpu.CopyTo(ctx);
  NDArray ret_indices = ret_indices_cpu.CopyTo(ctx);
  NDArray ret_data = ret_data_cpu.CopyTo(ctx);
  
  return CSRMatrix(
      coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data,
      coo.col_sorted);
}

// Force instantiation for COOToCSR
template CSRMatrix COOToCSR<kDGLAscend, int32_t>(COOMatrix);
template CSRMatrix COOToCSR<kDGLAscend, int64_t>(COOMatrix);

///////////////////////////// COOSort_ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "COOSort_ on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for COOSort_
template void COOSort_<kDGLAscend, int32_t>(COOMatrix*, bool);
template void COOSort_<kDGLAscend, int64_t>(COOMatrix*, bool);

///////////////////////////// CSRToCOO /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CSRToCOO on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/spmat_op_impl_coo.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for CSRToCOO
template COOMatrix CSRToCOO<kDGLAscend, int32_t>(CSRMatrix);
template COOMatrix CSRToCOO<kDGLAscend, int64_t>(CSRMatrix);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
