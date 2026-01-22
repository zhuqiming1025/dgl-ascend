/**
 *  Copyright (c) 2019 by Contributors
 * @file array/npu/array_index_select.cc
 * @brief Array index select NPU implementation
  * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "IndexSelect on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_index_select.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "IndexSelect (scalar) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_index_select.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common type pairs
template NDArray IndexSelect<kDGLAscend, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDGLAscend, double, int64_t>(NDArray, IdArray);

template int32_t IndexSelect<kDGLAscend, int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<kDGLAscend, int64_t>(NDArray array, int64_t index);
template float IndexSelect<kDGLAscend, float>(NDArray array, int64_t index);
template double IndexSelect<kDGLAscend, double>(NDArray array, int64_t index);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
