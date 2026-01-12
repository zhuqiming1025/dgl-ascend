/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "spmm_sum_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_spmm_sum.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void spmm_sum(GM_ADDR row_ptr, GM_ADDR col_ind, 
                                                       GM_ADDR values, GM_ADDR dense_matrix,
                                                       GM_ADDR output, SpmmSumTilingData tiling);
#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t tilingSize = sizeof(SpmmSumTilingData);
    
    // 首先读取tiling数据以获取矩阵维度
    SpmmSumTilingData tilingData;
    size_t fileSize = 0;
    ReadFile("./input/input_tiling.bin", fileSize, &tilingData, tilingSize);
    
    uint32_t numSparseRows = tilingData.numSparseRows;   // 稀疏矩阵行数
    uint32_t numSparseCols = tilingData.numSparseCols;   // 稀疏矩阵列数/密集矩阵行数
    uint32_t numDenseCols = tilingData.numDenseCols;     // 密集矩阵列数
    uint32_t nnz = tilingData.nnz;                       // 非零元素个数
    
    size_t rowPtrByteSize = (numSparseRows + 1) * sizeof(uint32_t);
    size_t colIndByteSize = nnz * sizeof(uint32_t);
    size_t valuesByteSize = nnz * sizeof(float);
    size_t denseByteSize = numSparseCols * numDenseCols * sizeof(float);
    size_t outputByteSize = numSparseRows * numDenseCols * sizeof(float);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    size_t tilingFileSize = 0;
    ReadFile("./input/input_tiling.bin", tilingFileSize, tiling, tilingSize);
    
    uint8_t *row_ptr = (uint8_t *)AscendC::GmAlloc(rowPtrByteSize);
    uint8_t *col_ind = (uint8_t *)AscendC::GmAlloc(colIndByteSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesByteSize);
    uint8_t *dense_matrix = (uint8_t *)AscendC::GmAlloc(denseByteSize);
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_row_ptr.bin", rowPtrByteSize, row_ptr, rowPtrByteSize);
    ReadFile("./input/input_col_ind.bin", colIndByteSize, col_ind, colIndByteSize);
    ReadFile("./input/input_values.bin", valuesByteSize, values, valuesByteSize);
    ReadFile("./input/input_dense.bin", denseByteSize, dense_matrix, denseByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(spmm_sum, blockDim, row_ptr, col_ind, values, dense_matrix, output,
                *reinterpret_cast<SpmmSumTilingData *>(tiling)); // use this macro for cpu debug

    WriteFile("./output/output_sum.bin", output, outputByteSize);

    AscendC::GmFree((void *)row_ptr);
    AscendC::GmFree((void *)col_ind);
    AscendC::GmFree((void *)values);
    AscendC::GmFree((void *)dense_matrix);
    AscendC::GmFree((void *)output);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    SpmmSumTilingData *tiling;
    uint8_t *rowPtrHost, *colIndHost, *valuesHost, *denseHost, *outputHost;
    uint8_t *rowPtrDevice, *colIndDevice, *valuesDevice, *denseDevice, *outputDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    size_t tilingFileSize = 0;
    ReadFile("./input/input_tiling.bin", tilingFileSize, tiling, tilingSize);

    CHECK_ACL(aclrtMallocHost((void **)(&rowPtrHost), rowPtrByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&colIndHost), colIndByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&valuesHost), valuesByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&denseHost), denseByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&outputHost), outputByteSize));
    
    CHECK_ACL(aclrtMalloc((void **)&rowPtrDevice, rowPtrByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&colIndDevice, colIndByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&valuesDevice, valuesByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&denseDevice, denseByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outputDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_row_ptr.bin", rowPtrByteSize, rowPtrHost, rowPtrByteSize);
    ReadFile("./input/input_col_ind.bin", colIndByteSize, colIndHost, colIndByteSize);
    ReadFile("./input/input_values.bin", valuesByteSize, valuesHost, valuesByteSize);
    ReadFile("./input/input_dense.bin", denseByteSize, denseHost, denseByteSize);

    CHECK_ACL(aclrtMemcpy(rowPtrDevice, rowPtrByteSize, rowPtrHost, rowPtrByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(colIndDevice, colIndByteSize, colIndHost, colIndByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(valuesDevice, valuesByteSize, valuesHost, valuesByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(denseDevice, denseByteSize, denseHost, denseByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(spmm_sum)(blockDim, stream, rowPtrDevice, colIndDevice, 
                                         valuesDevice, denseDevice, outputDevice, tiling);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(outputHost, outputByteSize, outputDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_sum.bin", outputHost, outputByteSize);

    CHECK_ACL(aclrtFree(rowPtrDevice));
    CHECK_ACL(aclrtFree(colIndDevice));
    CHECK_ACL(aclrtFree(valuesDevice));
    CHECK_ACL(aclrtFree(denseDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFreeHost(rowPtrHost));
    CHECK_ACL(aclrtFreeHost(colIndHost));
    CHECK_ACL(aclrtFreeHost(valuesHost));
    CHECK_ACL(aclrtFreeHost(denseHost));
    CHECK_ACL(aclrtFreeHost(outputHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
