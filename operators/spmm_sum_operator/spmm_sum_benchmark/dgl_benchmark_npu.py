"""
在 NPU 上运行 spmm_sum 操作的基准测试。
使用 Cora 和 PubMed 数据集，运行 100 次测试并输出平均值和方差。
"""

import time
import torch
import torch_npu
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.data import CoraGraphDataset, PubmedGraphDataset
import pandas as pd
import sys
import os

# 添加项目根目录和 build 目录到路径，以便导入 spmm_sum 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_root, "build")
build_lib_dir = os.path.join(build_dir, "lib")
sys.path.insert(0, project_root)
sys.path.insert(0, build_dir)

# 设置库搜索路径，以便找到 libascendc_kernels_npu.so
# 必须在导入 spmm_sum 之前设置
if os.path.exists(build_lib_dir):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if build_lib_dir not in current_ld_path:
        if current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{build_lib_dir}:{current_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = build_lib_dir

try:
    # 使用 ctypes 预先加载依赖库，这样 spmm_sum 导入时就能找到它
    import ctypes
    if sys.platform == "linux":
        lib_path = os.path.join(build_lib_dir, "libascendc_kernels_npu.so")
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
            except (OSError, Exception):
                # 如果预加载失败，继续尝试正常导入（可能库已经在路径中）
                pass
    import spmm_sum
except ImportError as e:
    print(f"错误：无法导入 spmm_sum 模块: {e}")
    print("请确保已编译 pybind11 扩展：")
    print("  cd build")
    print("  cmake .. -DBUILD_PYBIND11=ON -DRUN_MODE=npu")
    print("  make -j")
    print(f"\n请确保 {build_lib_dir} 目录中有 libascendc_kernels_npu.so 文件")
    print("如果库文件存在但仍然报错，请尝试在运行前设置环境变量：")
    print(f"  export LD_LIBRARY_PATH={build_lib_dir}:$LD_LIBRARY_PATH")
    sys.exit(1)


def benchmark_spmm_single_run(graph, features, row_ptr, col_ind, values, 
                                      max_nnz_per_row):
    """单次运行 spmm_sum 操作的性能测试"""
    # 获取图的基本信息
    num_nodes = graph.num_nodes()
    feat_dim = features.shape[1]  # 使用对齐后的特征维度
    
    # 将数据移到 NPU
    row_ptr_npu = row_ptr.npu()
    col_ind_npu = col_ind.npu()
    values_npu = values.npu()
    features_npu = features.npu()
    
    # 同步
    torch.npu.synchronize()
    
    # 执行计算（tileLength 在 kernel 中固定为 128 字节）
    start = time.time()
    output = spmm_sum.run_spmm_sum(
        row_ptr_npu, col_ind_npu, values_npu, features_npu,
        num_nodes, num_nodes, feat_dim,
        max_nnz_per_row
    )
    torch.npu.synchronize()
    end = time.time()
    
    return (end - start) * 1000  # 转换为毫秒


def benchmark_with_stats(graph, features, row_ptr, col_ind, values, 
                         max_nnz_per_row, num_iterations=100):
    """运行多次测试并计算平均值和方差"""
    # 预热
    for _ in range(5):
        _ = benchmark_spmm_single_run(graph, features, row_ptr, col_ind, values,
                                              max_nnz_per_row)
    torch.npu.synchronize()
    
    # 存储所有运行的结果
    times = []
    
    # 运行多次测试
    for i in range(num_iterations):
        elapsed = benchmark_spmm_single_run(graph, features, row_ptr, col_ind, values,
                                                    max_nnz_per_row)
        times.append(elapsed)
    
    # 计算统计信息
    stats = {
        'mean': np.mean(times),
        'std': np.std(times),
        'variance': np.var(times)
    }
    
    return stats


def run_benchmark_for_dataset(dataset_name, dataset_loader, max_nnz_per_row=32, 
                               num_iterations=100):
    """为指定数据集运行基准测试"""
    print(f"\n正在加载{dataset_name}数据集...")
    dataset = dataset_loader()
    graph = dataset[0]
    
    print(f"图信息:")
    print(f"  节点数: {graph.num_nodes()}")
    print(f"  边数: {graph.num_edges()}")
    original_feat_dim = graph.ndata['feat'].shape[1]
    print(f"  原始特征维度: {original_feat_dim}")

    features = graph.ndata['feat']
        
    # 计算对齐后的维度（向上取整到 32 的倍数，对应 128 字节对齐）
    aligned_feat_dim = ((original_feat_dim + 15) // 16) * 16
    if aligned_feat_dim > original_feat_dim:
        padding_size = aligned_feat_dim - original_feat_dim
        features_aligned = torch.nn.functional.pad(features, (0, padding_size), mode='constant', value=0.0)
        print(f"  对齐后特征维度: {aligned_feat_dim} (填充了 {padding_size} 维)")
    else:
        features_aligned = features
        print(f"  特征维度已对齐: {aligned_feat_dim}")
    

    # 将图转换为 CSR 格式
    # 从图的边直接构建 scipy CSR 矩阵
    # 注意：DGL 的 copy_u_sum 操作对应稀疏矩阵乘法
    # 其中稀疏矩阵的行是目标节点（dst），列是源节点（src）
    src, dst = graph.edges()
    num_nodes = graph.num_nodes()
    # 创建全1的边权重
    edge_weights = np.ones(graph.num_edges(), dtype=np.float32)
    # 构建 CSR 矩阵（行是 dst，列是 src）
    adj_scipy = sp.csr_matrix((edge_weights, (dst.numpy(), src.numpy())), 
                               shape=(num_nodes, num_nodes))
    
    # 获取 CSR 格式的数据并转换为 torch tensor
    row_ptr = torch.from_numpy(adj_scipy.indptr).int()  # indptr
    col_ind = torch.from_numpy(adj_scipy.indices).int()  # indices
    values = torch.from_numpy(adj_scipy.data).float()  # data
    
    
    # 计算每行最大非零元素个数
    max_nnz = 0
    for i in range(graph.num_nodes()):
        row_start = row_ptr[i].item()
        row_end = row_ptr[i + 1].item()
        max_nnz = max(max_nnz, row_end - row_start)
    max_nnz_per_row = max(max_nnz_per_row, max_nnz)
    print(f"  每行最大非零元素个数: {max_nnz}")
    
    print(f"\n在NPU上运行{num_iterations}次测试...")
    stats = benchmark_with_stats(graph, features_aligned, row_ptr, col_ind, values,
                                 max_nnz_per_row, num_iterations)
    
    return stats, dataset_name


def main():
    print("=" * 80)
    print("spmm_sum 操作性能测试（NPU）")
    print("数据集: Cora, PubMed")
    print("测试次数: 100 次")
    print("=" * 80)

    # 测试参数
    num_iterations = 100
    datasets = [
        # ("Cora", CoraGraphDataset),
        ("PubMed", PubmedGraphDataset),
    ]

    all_results = []
    torch_npu.npu.set_device(2)
    for dataset_name, dataset_loader in datasets:
        print("\n" + "=" * 80)
        print(f"{dataset_name} - NPU 测试")
        print("=" * 80)

        stats, _ = run_benchmark_for_dataset(
            dataset_name,
            dataset_loader,
            max_nnz_per_row=32,
            num_iterations=num_iterations,
        )

        all_results.append(
            {
                "数据集": dataset_name,
                "操作": "spmm_sum",
                "设备": "NPU",
                "平均耗时(ms)": stats["mean"],
                "标准差(ms)": stats["std"],
                "方差(ms²)": stats["variance"],
            }
        )

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("NPU 性能测试结果（100 次测试的平均值、标准差和方差）")
    print("=" * 80)
    print(df.to_string(index=False))

    # 确保结果目录存在
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "dgl_benchmark_npu_results.csv")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nNPU 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
