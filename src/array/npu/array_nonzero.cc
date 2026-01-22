/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/array_nonzero.cc
 * @brief Array nonzero NPU implementation
  * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
IdArray NonZero(IdArray array) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "NonZero on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_nonzero.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common type pairs
template IdArray NonZero<kDGLAscend, int32_t>(IdArray);
template IdArray NonZero<kDGLAscend, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
