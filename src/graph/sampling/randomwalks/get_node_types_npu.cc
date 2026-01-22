/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/sampling/randomwalks/get_node_types_npu.cc
 * @brief DGL sampler - NPU implementation of GetNodeTypesFromMetapath (placeholder)
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dmlc/logging.h>

#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template <DGLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg, const TypeArray metapath) {
  static_assert(XPU == kDGLAscend, "This file should only be compiled for kDGLAscend.");
  LOG(FATAL) << "GetNodeTypesFromMetapath on Ascend NPU is not implemented yet. "
             << "Please implement src/graph/sampling/randomwalks/get_node_types_npu.cc using Ascend kernels.";
  __builtin_unreachable();
}

// Force instantiation
template TypeArray GetNodeTypesFromMetapath<kDGLAscend, int32_t>(
    const HeteroGraphPtr hg, const TypeArray metapath);
template TypeArray GetNodeTypesFromMetapath<kDGLAscend, int64_t>(
    const HeteroGraphPtr hg, const TypeArray metapath);

}  // namespace impl

}  // namespace sampling

}  // namespace dgl
