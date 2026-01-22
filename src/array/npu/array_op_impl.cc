/**
 *  Copyright (c) 2020 by Contributors
 * @file array/npu/array_op_impl.cc
 * @brief Array operator NPU implementation (placeholder)
 */
#include <dgl/array.h>
#include <dmlc/logging.h>
#include <cstddef>
#include <vector>

#include "../arith.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "BinaryElewise (array, array) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "BinaryElewise (array, scalar) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "BinaryElewise (scalar, array) on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for common operations and types
// BinaryElewise(IdArray, IdArray)
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LT>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LT>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::EQ>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::EQ>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GT>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GT>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::NE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::NE>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Add>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Add>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Sub>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Sub>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mul>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mul>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Div>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Div>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mod>(IdArray, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mod>(IdArray, IdArray);

// BinaryElewise(IdArray, IdType)
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LT>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LT>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::EQ>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::EQ>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GT>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GT>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GE>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GE>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LE>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LE>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::NE>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::NE>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Add>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Add>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Sub>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Sub>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mul>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mul>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Div>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Div>(IdArray, int64_t);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mod>(IdArray, int32_t);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mod>(IdArray, int64_t);

// BinaryElewise(IdType, IdArray)
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LT>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LT>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::EQ>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::EQ>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GT>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GT>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::GE>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::GE>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::LE>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::LE>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::NE>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::NE>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Add>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Add>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Sub>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Sub>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mul>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mul>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Div>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Div>(int64_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int32_t, arith::Mod>(int32_t, IdArray);
template IdArray BinaryElewise<kDGLAscend, int64_t, arith::Mod>(int64_t, IdArray);

///////////////////////////// UnaryElewise /////////////////////////////

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "UnaryElewise on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for UnaryElewise
template IdArray UnaryElewise<kDGLAscend, int32_t, arith::Neg>(IdArray);
template IdArray UnaryElewise<kDGLAscend, int64_t, arith::Neg>(IdArray);

///////////////////////////// Full /////////////////////////////

template <DGLDeviceType XPU, typename DType>
NDArray Full(DType fill_value, int64_t length, DGLContext ctx) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "Full on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for Full
template NDArray Full<kDGLAscend, int32_t>(int32_t, int64_t, DGLContext);
template NDArray Full<kDGLAscend, int64_t>(int64_t, int64_t, DGLContext);
template NDArray Full<kDGLAscend, float>(float, int64_t, DGLContext);
template NDArray Full<kDGLAscend, double>(double, int64_t, DGLContext);

///////////////////////////// Range /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DGLContext ctx) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "Range on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for Range
template IdArray Range<kDGLAscend, int32_t>(int32_t, int32_t, DGLContext);
template IdArray Range<kDGLAscend, int64_t>(int64_t, int64_t, DGLContext);

///////////////////////////// AsNumBits /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "AsNumBits on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for AsNumBits
template IdArray AsNumBits<kDGLAscend, int32_t>(IdArray, uint8_t);
template IdArray AsNumBits<kDGLAscend, int64_t>(IdArray, uint8_t);

///////////////////////////// Relabel_ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "Relabel_ on Ascend NPU is not implemented yet. "
             << "Please implement src/array/npu/array_op_impl.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation for Relabel_
template IdArray Relabel_<kDGLAscend, int32_t>(const std::vector<IdArray>&);
template IdArray Relabel_<kDGLAscend, int64_t>(const std::vector<IdArray>&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
