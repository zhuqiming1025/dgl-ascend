/**
 * @file spmm_sum_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef SPMM_SUM_TILING_H
#define SPMM_SUM_TILING_H
#include <cstdint>

struct SpmmSumTilingData {
    uint32_t numSparseRows;  // 稀疏矩阵行数
    uint32_t numSparseCols;  // 稀疏矩阵列数（也是密集矩阵行数）
    uint32_t numDenseCols;   // 密集矩阵列数
    uint32_t nnz;            // 非零元素个数
    uint32_t maxNnzPerRow;   // 每行最大非零元素个数
};
#endif  // SPMM_SUM_TILING_H
