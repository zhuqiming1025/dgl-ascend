## 目录结构介绍
```
├── spmm_sum_operator
│   ├── cmake                   // 编译工程文件
│   ├── spmm_sum_benchmark
│   │   ├── compare_cpu_npu.py         // 算子精度计算
│   │   └── dgl_benchmark_npu.py    // 算子计算时间统计
│   ├── spmm_sum_custom_tiling.h     // 算子tiling实现
│   ├── spmm_sum_custom.cpp          // 算子kernel实现
│   ├── CMakeLists.txt               // 编译工程文件
│   └── test.sh                       // 编译以及运行compare_cpu_npu.py和dgl_benchmark_npu.py的脚本
```
  - 样例执行

    ```bash
    bash test.sh 
    ```