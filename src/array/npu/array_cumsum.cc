/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/array_cumsum.cc
 * @brief Array cumsum NPU implementation
  * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
IdArray CumSum(IdArray array, bool prepend_zero) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "CumSum on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_cumsum.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common type pairs
template IdArray CumSum<kDGLAscend, int32_t>(IdArray, bool);
template IdArray CumSum<kDGLAscend, int64_t>(IdArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
