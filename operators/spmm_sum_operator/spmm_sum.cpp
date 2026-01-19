/**
 * @file spmm_sum.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "spmm_sum_tiling.h"
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t TILE_LENGTH = 512;

class KernelSpmmSum {
public:
    __aicore__ inline KernelSpmmSum() {}
    __aicore__ inline void Init(GM_ADDR row_ptr, GM_ADDR col_ind, GM_ADDR values, 
                                GM_ADDR dense_matrix, GM_ADDR output,
                                uint32_t numSparseRows, uint32_t numSparseCols, uint32_t numDenseCols, 
                                uint32_t nnz, uint32_t maxNnzPerRow)
    {
        this->numSparseRows = numSparseRows;
        this->numSparseCols = numSparseCols;
        this->numDenseCols = numDenseCols;
        this->nnz = nnz;
        this->maxNnzPerRow = maxNnzPerRow;
        
        
        // 计算每个tile的特征维度数量
        this->tileCols = TILE_LENGTH / sizeof(half);

        // 计算每个block需要完成稀疏矩阵的行数
        uint32_t totalBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerBlock = (numSparseRows + totalBlocks - 1) / totalBlocks;
        this->startRow = blockIdx * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock > numSparseRows ? numSparseRows : this->startRow + rowsPerBlock;

        // 设置全局内存缓冲区
        rowPtrGm.SetGlobalBuffer((__gm__ uint32_t *)row_ptr, numSparseRows + 1);
        colIndGm.SetGlobalBuffer((__gm__ uint32_t *)col_ind, nnz);
        valuesGm.SetGlobalBuffer((__gm__ half *)values, nnz);
        denseGm.SetGlobalBuffer((__gm__ half *)dense_matrix, numSparseCols * numDenseCols);
        outputGm.SetGlobalBuffer((__gm__ half *)output, numSparseRows * numDenseCols);

        // 初始化队列缓冲区
        // 每个tile处理 128 字节的特征，即 tileCols 个特征维度
        pipe.InitBuffer(inQueueDense, BUFFER_NUM, maxNnzPerRow * TILE_LENGTH);
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, maxNnzPerRow * sizeof(half));
        pipe.InitBuffer(accumQueue, BUFFER_NUM, TILE_LENGTH);
    }
    
    __aicore__ inline void Process()
    {
        for(uint32_t rowIdx = this->startRow; rowIdx < this->endRow; rowIdx++)
        {
            uint32_t rowStart = rowPtrGm.GetValue(rowIdx);
            uint32_t rowEnd = rowPtrGm.GetValue(rowIdx + 1);
            uint32_t rowNnz = rowEnd - rowStart;
            for (uint32_t colStart = 0; colStart < numDenseCols; colStart += tileCols) {
                uint32_t colLen = colStart + tileCols > numDenseCols ? numDenseCols - colStart : tileCols;
                CopyIn(rowIdx, rowStart, rowEnd, colStart, colLen);
                Compute(rowIdx, rowStart, rowEnd, colStart, colLen);
                CopyOut(rowIdx, rowStart, rowEnd, colStart, colLen);
            }
        }
    }

private:

    __aicore__ inline void CopyIn(int32_t rowIdx, uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colLen){
        uint32_t rowNnz = rowEnd - rowStart;
        AscendC::LocalTensor<half> denseRows = inQueueDense.AllocTensor<half>();
        AscendC::LocalTensor<half> weights = inQueueWeight.AllocTensor<half>();
        // 只复制当前tile的特征维度 [colStart, colStart + colLen)
        for (uint32_t i = 0; i < rowNnz; ++i) {
            uint32_t idx = rowStart + i;
            uint32_t col = colIndGm.GetValue(idx);
        AscendC::DataCopy(denseRows[i * tileCols], denseGm[col * numDenseCols + colStart], colLen);
        if (rowIdx == 1358 && col == 1741 && colStart == 0){
            DumpTensor(denseRows[i*tileCols], 0, colLen);
        }
        weights.SetValue(i, valuesGm.GetValue(idx));
        }

        
    inQueueDense.EnQue<half>(denseRows);
    inQueueWeight.EnQue<half>(weights);
    }

    __aicore__ inline void Compute(int32_t rowIdx, uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colLen){
        uint32_t rowNnz = rowEnd - rowStart;
        // 初始化累加器（只处理当前tile的特征维度，大小为colLen）
        AscendC::LocalTensor<half> accum = accumQueue.AllocTensor<half>();
        AscendC::Duplicate<half>(accum, half(0.0f), colLen);
        
        AscendC::LocalTensor<half> denseRows = inQueueDense.DeQue<half>();
        AscendC::LocalTensor<half> weights = inQueueWeight.DeQue<half>();

        // 对当前tile的特征维度进行计算
        for (uint32_t i = 0; i < rowNnz; ++i) {
            half weight = weights.GetValue(i);
            // 使用Axpy：accum = weight * denseRows[i] + accum
            AscendC::Axpy(accum, denseRows[i * tileCols], weight, colLen);
        }
        if (rowIdx == 1358 && colStart == 0){
            DumpTensor(accum, 1, colLen);
        }
        accumQueue.EnQue<half>(accum);
        inQueueDense.FreeTensor(denseRows);
        inQueueWeight.FreeTensor(weights);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colLen)
    {
        AscendC::LocalTensor<half> accum = accumQueue.DeQue<half>();
        AscendC::DataCopy(outputGm[rowIdx * numDenseCols + colStart], accum, colLen);
        if (rowIdx == 1358 && colStart == 0){
            DumpTensor(outputGm[rowIdx * numDenseCols + colStart], 2, colLen);
        }
        accumQueue.FreeTensor(accum);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueDense;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;  // 从 Compute 传递到 CopyOut，使用 VECOUT 类型
    
    // 全局内存缓冲区
    AscendC::GlobalTensor<uint32_t> rowPtrGm;
    AscendC::GlobalTensor<uint32_t> colIndGm;
    AscendC::GlobalTensor<half> valuesGm;
    AscendC::GlobalTensor<half> denseGm;  // 密集矩阵，大小为numSparseCols*numDenseCols
    AscendC::GlobalTensor<half> outputGm;
    uint32_t numSparseRows;  // 稀疏矩阵行数
    uint32_t numSparseCols;  // 稀疏矩阵列数（也是密集矩阵行数）
    uint32_t numDenseCols;   // 密集矩阵列数
    uint32_t nnz;
    uint32_t maxNnzPerRow;   // 每行最大非零元素个数
    uint32_t tileCols;       // 每个tile的特征维度数量（固定为 512/sizeof(half) = 256）
    uint32_t startRow, endRow;
};

extern "C" __global__ __aicore__ void spmm_sum(GM_ADDR row_ptr, GM_ADDR col_ind, 
                                                    GM_ADDR values, GM_ADDR dense_matrix,
                                                    GM_ADDR output, SpmmSumTilingData tiling)
{
    KernelSpmmSum op;
    op.Init(row_ptr, col_ind, values, dense_matrix, output,
            tiling.numSparseRows, tiling.numSparseCols, tiling.numDenseCols, 
            tiling.nnz, tiling.maxNnzPerRow);
    op.Process();
}
 