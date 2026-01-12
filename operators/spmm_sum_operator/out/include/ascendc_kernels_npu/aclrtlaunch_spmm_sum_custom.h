#ifndef HEADER_ACLRTLAUNCH_SPMM_SUM_CUSTOM_H
#define HEADER_ACLRTLAUNCH_SPMM_SUM_CUSTOM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_spmm_sum_custom(uint32_t blockDim, aclrtStream stream, void* row_ptr, void* col_ind, void* values, void* dense_matrix, void* output, SpmmSumCustomTilingData* tiling);
#endif
