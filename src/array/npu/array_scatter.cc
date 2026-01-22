/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/array_scatter.cc
 * @brief Array scatter NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
void Scatter_(IdArray index, NDArray value, NDArray out) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "Scatter_ on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_scatter.cc using Ascend kernels.";
}

// Force instantiation for common type pairs
template void Scatter_<kDGLAscend, int32_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, int64_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, float, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, double, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, int32_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, int64_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, float, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLAscend, double, int64_t>(IdArray, NDArray, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
