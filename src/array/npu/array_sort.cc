/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/array_sort.cc
 * @brief Array sort NPU implementation (placeholder)
 * @note This file is a placeholder. Replace LOG(FATAL) with actual Ascend NPU kernel implementation.
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> Sort(IdArray array, int dim) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "Sort on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_sort.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for Sort
template std::pair<IdArray, IdArray> Sort<kDGLAscend, int32_t>(IdArray, int);
template std::pair<IdArray, IdArray> Sort<kDGLAscend, int64_t>(IdArray, int);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
