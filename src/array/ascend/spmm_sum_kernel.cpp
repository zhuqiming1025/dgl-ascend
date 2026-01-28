#ifndef SPMM_SUM_TILING_H
#define SPMM_SUM_TILING_H
#include <cstdint>

struct SpmmSumTilingData {
    uint32_t numSparseRows;
    uint32_t numSparseCols;
    uint32_t numDenseCols;
    uint32_t nnz;
    uint32_t maxNnzPerRow;
};
#endif

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 512;
class KernelSpmmSum {
public:
    __aicore__ inline KernelSpmmSum() {}
    __aicore__ inline void Init(GM_ADDR row_ptr, GM_ADDR col_ind, 
                                GM_ADDR dense_matrix, GM_ADDR output,
                                uint32_t numSparseRows, uint32_t numSparseCols, uint32_t numDenseCols, 
                                uint32_t nnz, uint32_t maxNnzPerRow)
    {
        this->numSparseRows = numSparseRows;
        this->numSparseCols = numSparseCols;
        this->numDenseCols = numDenseCols;
        this->nnz = nnz;
        this->maxNnzPerRow = maxNnzPerRow;
        
        
        this->tileCols = TILE_LENGTH / sizeof(float);
        uint32_t totalBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerBlock = (numSparseRows + totalBlocks - 1) / totalBlocks;
        this->startRow = blockIdx * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock > numSparseRows ? numSparseRows : this->startRow + rowsPerBlock;
        
        rowPtrGm.SetGlobalBuffer((__gm__ uint32_t *)row_ptr, numSparseRows + 1);
        colIndGm.SetGlobalBuffer((__gm__ uint32_t *)col_ind, nnz);
        denseGm.SetGlobalBuffer((__gm__ float *)dense_matrix, numSparseCols * numDenseCols);
        outputGm.SetGlobalBuffer((__gm__ float *)output, numSparseRows * numDenseCols);
        pipe.InitBuffer(inQueueDense, BUFFER_NUM, maxNnzPerRow * TILE_LENGTH);
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
        AscendC::LocalTensor<float> denseRows = inQueueDense.AllocTensor<float>();
        for (uint32_t i = 0; i < rowNnz; ++i) {
            uint32_t idx = rowStart + i;
            uint32_t col = colIndGm.GetValue(idx);
            AscendC::DataCopy(denseRows[i * tileCols], denseGm[col * numDenseCols + colStart], colLen);
        }
        
        inQueueDense.EnQue<float>(denseRows);
    }

    __aicore__ inline void Compute(int32_t rowIdx, uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colLen){
        uint32_t rowNnz = rowEnd - rowStart;
        AscendC::LocalTensor<float> accum = accumQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> denseRows = inQueueDense.DeQue<float>();
        AscendC::Duplicate<float>(accum, float(0.0f), colLen);

        for (uint32_t i = 0; i < rowNnz; ++i) {
            AscendC::Add(accum, accum, denseRows[i * tileCols], colLen);
        }
        accumQueue.EnQue<float>(accum);
        inQueueDense.FreeTensor(denseRows);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colLen)
    {
        AscendC::LocalTensor<float> accum = accumQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[rowIdx * numDenseCols + colStart], accum, colLen);
        accumQueue.FreeTensor(accum);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueDense;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue; 
    
    AscendC::GlobalTensor<uint32_t> rowPtrGm;
    AscendC::GlobalTensor<uint32_t> colIndGm;
    AscendC::GlobalTensor<float> denseGm; 
    AscendC::GlobalTensor<float> outputGm;
    uint32_t numSparseRows;
    uint32_t numSparseCols; 
    uint32_t numDenseCols;
    uint32_t nnz;
    uint32_t maxNnzPerRow;
    uint32_t tileCols;
    uint32_t startRow, endRow;
};

extern "C" __global__ __aicore__ void spmm_sum(GM_ADDR row_ptr, GM_ADDR col_ind, 
                                                    GM_ADDR dense_matrix,
                                                    GM_ADDR output, GM_ADDR tiling_ptr)
{

    AscendC::GlobalTensor<uint32_t> tilingGm;
    tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
    
    uint32_t numSparseRows = tilingGm.GetValue(0);
    uint32_t numSparseCols = tilingGm.GetValue(1);
    uint32_t numDenseCols = tilingGm.GetValue(2);
    uint32_t nnz = tilingGm.GetValue(3);
    uint32_t maxNnzPerRow = tilingGm.GetValue(4);
    KernelSpmmSum op;
    op.Init(row_ptr, col_ind, dense_matrix, output,
            numSparseRows, numSparseCols, numDenseCols,
            nnz, maxNnzPerRow);
    op.Process();
}
