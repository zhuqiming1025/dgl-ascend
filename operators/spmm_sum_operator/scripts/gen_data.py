#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import scipy.sparse as sp


def gen_csr_matrix(M, K, density=0.1):
    """生成CSR格式的稀疏矩阵"""
    # 生成随机稀疏矩阵
    np.random.seed(42)
    sparse_matrix = sp.random(M, K, density=density, format='csr', dtype=np.float32)
    
    # 转换为CSR格式的数组
    row_ptr = sparse_matrix.indptr.astype(np.uint32)
    col_ind = sparse_matrix.indices.astype(np.uint32)
    # 将值保持为float32
    values = sparse_matrix.data.astype(np.float32)
    
    return row_ptr, col_ind, values, sparse_matrix.nnz


def gen_golden_data_simple():
    """生成测试数据和真值数据"""
    # 矩阵维度
    numSparseRows = 64   # 稀疏矩阵行数
    numSparseCols = 128  # 稀疏矩阵列数/密集矩阵行数
    numDenseCols = 32   # 密集矩阵列数
    density = 0.5  # 稀疏矩阵密度
    
    # 生成CSR格式的稀疏矩阵
    row_ptr, col_ind, values, nnz = gen_csr_matrix(numSparseRows, numSparseCols, density)
    
    # 设置所有权重为1
    values = np.ones(nnz, dtype=np.float32)
    
    # 生成密集矩阵，所有特征值设置为1
    dense_matrix = np.ones([numSparseCols, numDenseCols], dtype=np.float32)
    
    # 计算真值：稀疏矩阵 @ 密集矩阵，然后对每行求和
    # 将CSR格式转换为密集矩阵进行计算
    sparse_matrix_dense = np.zeros([numSparseRows, numSparseCols], dtype=np.float32)
    for i in range(numSparseRows):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        for j in range(start, end):
            col = col_ind[j]
            val = values[j]
            sparse_matrix_dense[i, col] = val
    
    # 执行矩阵乘法
    result = sparse_matrix_dense @ dense_matrix  # numSparseRows x numDenseCols
    
    # 对每行求和：直接使用结果，不需要除以非零元素个数
    golden = result.copy()
    
    # 计算每行最大非零元素个数
    maxNnzPerRow = 0
    for i in range(numSparseRows):
        rowNnz = row_ptr[i + 1] - row_ptr[i]
        if rowNnz > maxNnzPerRow:
            maxNnzPerRow = rowNnz
    
    # 生成tiling数据（按照C++结构体字段顺序：numSparseRows, numSparseCols, numDenseCols, nnz, maxNnzPerRow）
    # tileLength 在 kernel 中固定为 128 字节，不再作为 tiling 参数
    tiling = np.array([numSparseRows, numSparseCols, numDenseCols, nnz, maxNnzPerRow], dtype=np.uint32)
    
    # 保存数据到文件
    tiling.tofile("./input/input_tiling.bin")
    row_ptr.tofile("./input/input_row_ptr.bin")
    col_ind.tofile("./input/input_col_ind.bin")
    values.tofile("./input/input_values.bin")
    dense_matrix.tofile("./input/input_dense.bin")
    golden.tofile("./output/golden.bin")
    
    print(f"Generated test data:")
    print(f"  Sparse matrix: {numSparseRows} x {numSparseCols}, nnz={nnz}")
    print(f"  Dense matrix: {numSparseCols} x {numDenseCols}")
    print(f"  Output: {numSparseRows} x {numDenseCols} matrix")
    print(f"  Tiling: numSparseRows={numSparseRows}, numSparseCols={numSparseCols}, numDenseCols={numDenseCols}, nnz={nnz}, maxNnzPerRow={maxNnzPerRow} (tileLength fixed to 128B in kernel)")


if __name__ == "__main__":
    gen_golden_data_simple()
