/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
 #include <pybind11/pybind11.h>
 #include <torch/extension.h>
 
 #include "spmm_sum_tiling.h"
 #include "aclrtlaunch_spmm_sum.h"
 #include "torch_npu/csrc/core/npu/NPUStream.h"
 #include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
 
 namespace my_spmm_sum {
 at::Tensor run_spmm_sum(const at::Tensor &row_ptr, const at::Tensor &col_ind,
                                const at::Tensor &values, const at::Tensor &dense_matrix,
                                uint32_t numSparseRows, uint32_t numSparseCols, uint32_t numDenseCols,
                                uint32_t maxNnzPerRow)
 {
     auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
     
     // 将输入从 float 转换为 half (float16)
     auto values_half = values.to(at::kHalf);
     auto dense_matrix_half = dense_matrix.to(at::kHalf);
     
     // 计算输出张量大小，使用 half 类型
     uint32_t nnz = values.size(0);
     auto output_half = at::empty({numSparseRows, numDenseCols}, dense_matrix_half.options());
     
     
     SpmmSumTilingData tiling;
     tiling.numSparseRows = numSparseRows;
     tiling.numSparseCols = numSparseCols;
     tiling.numDenseCols = numDenseCols;
     tiling.nnz = nnz;
     tiling.maxNnzPerRow = maxNnzPerRow;
     
     uint32_t blockDim = 32;
     
     ACLRT_LAUNCH_KERNEL(spmm_sum)
     (blockDim, acl_stream,
      const_cast<void *>(row_ptr.data_ptr()),
      const_cast<void *>(col_ind.data_ptr()),
      const_cast<void *>(values_half.data_ptr()),
      const_cast<void *>(dense_matrix_half.data_ptr()),
      const_cast<void *>(output_half.data_ptr()),
      &tiling);
     
     // 将结果从 half 转换为 float
     auto output = output_half.to(at::kFloat);
     
     return output;
 }
 } // namespace my_spmm_sum

// namespace my_spmm_sum {
//     at::Tensor run_spmm_sum(const at::Tensor &row_ptr, const at::Tensor &col_ind,
//                             const at::Tensor &values, const at::Tensor &dense_matrix,
//                             uint32_t numSparseRows, uint32_t numSparseCols, uint32_t numDenseCols,
//                             uint32_t maxNnzPerRow)
//     {
//         // 1. 获取 Stream
//         auto stream_wrapper = c10_npu::getCurrentNPUStream();
//         auto acl_stream = stream_wrapper.stream(false);
    
//         // 2. 计算对齐
//         int64_t original_dense_cols = dense_matrix.size(1); 
//         int64_t aligned_dense_cols = ((original_dense_cols + 15) / 16) * 16;
//         int64_t padding_size = aligned_dense_cols - original_dense_cols;
    
//         // 3. 准备输入：使用 at::pad 替代手动的 zeros+copy
//         //    at::pad 会自动处理内存申请、拷贝和 0 填充，且保证内存连续
//         at::Tensor dense_matrix_aligned;
//         if (padding_size > 0) {
//             // pad 参数格式: (pad_left, pad_right, pad_top, pad_bottom, ...)
//             // 这里只在最后一维的右边填充 padding_size
//             dense_matrix_aligned = at::pad(dense_matrix, {0, padding_size}, "constant", 0);
//         } else {
//             dense_matrix_aligned = dense_matrix;
//         }
    
//         // 确保类型转换 (Float -> Half) 和内存连续
//         // [关键]: 加上 .contiguous() 防止因为 pad 产生的 stride 问题
//         at::Tensor dense_matrix_half = dense_matrix_aligned.to(at::kHalf).contiguous();
//         at::Tensor values_half = values.to(at::kHalf).contiguous();
        
//         // 准备输出
//         at::Tensor output_half = at::empty({numSparseRows, aligned_dense_cols}, dense_matrix_half.options());
    
//         // 4. Tiling 设置
//         SpmmSumTilingData tiling;
//         tiling.numSparseRows = numSparseRows;
//         tiling.numSparseCols = numSparseCols;
//         tiling.numDenseCols = aligned_dense_cols; 
//         tiling.nnz = values.size(0);
//         tiling.maxNnzPerRow = maxNnzPerRow;
    
//         // 你确认 blockDim 没问题，那就保持原样 (但建议至少设为 32 或 64)
//         uint32_t blockDim = 64; 
    
//         // 5. 启动 Kernel
//         ACLRT_LAUNCH_KERNEL(spmm_sum)
//         (blockDim, acl_stream,
//          const_cast<void *>(row_ptr.data_ptr()),
//          const_cast<void *>(col_ind.data_ptr()),
//          const_cast<void *>(values_half.data_ptr()),
//          const_cast<void *>(dense_matrix_half.data_ptr()),
//          const_cast<void *>(output_half.data_ptr()),
//          &tiling);
    
//         // 6. 延长生命周期 (必须做！)
//         // 即使用了 at::pad，这些中间变量依然是局部变量，必须 recordStream
//         c10_npu::NPUCachingAllocator::recordStream(dense_matrix_half.storage().data_ptr(), stream_wrapper);
//         c10_npu::NPUCachingAllocator::recordStream(values_half.storage().data_ptr(), stream_wrapper);
//         c10_npu::NPUCachingAllocator::recordStream(output_half.storage().data_ptr(), stream_wrapper);
    
//         // 7. 输出切片还原
//         at::Tensor output = output_half.to(at::kFloat);
//         if (padding_size > 0) {
//             output = output.slice(1, 0, original_dense_cols);
//         }
    
//         return output;
//     }
//     }
 
 PYBIND11_MODULE(spmm_sum, m)
 {
     m.doc() = "spmm_sum pybind11 interfaces"; // optional module docstring
     m.def("run_spmm_sum", &my_spmm_sum::run_spmm_sum,
           "Run SPMM sum  kernel",
           pybind11::arg("row_ptr"),
           pybind11::arg("col_ind"),
           pybind11::arg("values"),
           pybind11::arg("dense_matrix"),
           pybind11::arg("numSparseRows"),
           pybind11::arg("numSparseCols"),
           pybind11::arg("numDenseCols"),
           pybind11::arg("maxNnzPerRow"));
 }
 