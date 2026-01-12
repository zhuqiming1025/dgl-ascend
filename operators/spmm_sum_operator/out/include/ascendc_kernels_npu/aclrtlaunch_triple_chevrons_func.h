
#ifndef HEADER_ACLRTLAUNCH_SPMM_SUM_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SPMM_SUM_CUSTOM_HKERNEL_H_


struct SpmmSumCustomTilingData;


extern "C" uint32_t aclrtlaunch_spmm_sum_custom(uint32_t blockDim, void* stream, void* row_ptr, void* col_ind, void* values, void* dense_matrix, void* output, SpmmSumCustomTilingData* tiling);

inline uint32_t spmm_sum_custom(uint32_t blockDim, void* hold, void* stream, void* row_ptr, void* col_ind, void* values, void* dense_matrix, void* output, SpmmSumCustomTilingData* tiling)
{
    (void)hold;
    return aclrtlaunch_spmm_sum_custom(blockDim, stream, row_ptr, col_ind, values, dense_matrix, output, tiling);
}

#endif
