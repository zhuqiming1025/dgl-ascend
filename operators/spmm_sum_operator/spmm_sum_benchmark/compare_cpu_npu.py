"""
比较 CPU 和 NPU 版本的 spmm_sum 操作结果，检查是否存在误差。
使用相同的输入数据，分别运行 CPU（DGL）和 NPU（自定义kernel）版本，然后比较输出。
"""

import torch
import torch_npu
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.data import CoraGraphDataset, PubmedGraphDataset
import sys
import os
import gc

# 添加项目根目录和 build 目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_root, "build")
build_lib_dir = os.path.join(build_dir, "lib")
sys.path.insert(0, project_root)
sys.path.insert(0, build_dir)

# 设置库搜索路径
if os.path.exists(build_lib_dir):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if build_lib_dir not in current_ld_path:
        if current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{build_lib_dir}:{current_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = build_lib_dir

try:
    import ctypes
    if sys.platform == "linux":
        lib_path = os.path.join(build_lib_dir, "libascendc_kernels_npu.so")
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
            except (OSError, Exception):
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


def compute_cpu_result(graph, features):
    """使用 DGL 的 copy_u_sum 计算 CPU 结果"""
    graph_cpu = graph.to('cpu')
    features_cpu = features.to('cpu')
    result_cpu = dgl.ops.copy_u_sum(graph_cpu, features_cpu)
    return result_cpu


def compute_npu_result(graph, features, row_ptr, col_ind, values, max_nnz_per_row):
    """使用自定义 kernel 计算 NPU 结果"""
    num_nodes = graph.num_nodes()
    feat_dim = features.shape[1]  # 使用对齐后的特征维度
    
    # 将数据移到 NPU
    row_ptr_npu = row_ptr.npu()
    col_ind_npu = col_ind.npu()
    values_npu = values.npu()
    features_npu = features.npu()
    
    # 同步
    torch.npu.synchronize()
    
    # 执行计算
    result_npu = spmm_sum.run_spmm_sum(
        row_ptr_npu, col_ind_npu, values_npu, features_npu,
        num_nodes, num_nodes, feat_dim,
        max_nnz_per_row
    )
    
    torch.npu.synchronize()
    return result_npu.cpu()  # 转回 CPU 以便比较


def compare_results(cpu_result, npu_result, dataset_name, original_feat_dim, aligned_feat_dim):
    """比较 CPU 和 NPU 的结果"""
    print(f"\n{'='*80}")
    print(f"结果比较: {dataset_name}")
    print(f"{'='*80}")
    
    # 检查形状
    print(f"CPU 结果形状: {cpu_result.shape}")
    print(f"NPU 结果形状: {npu_result.shape}")
    
    # 如果特征维度被对齐了，需要截取到原始维度
    if aligned_feat_dim > original_feat_dim:
        print(f"注意: NPU 结果的特征维度已对齐到 {aligned_feat_dim}，需要截取到原始维度 {original_feat_dim}")
        npu_result = npu_result[:, :original_feat_dim]
        print(f"截取后 NPU 结果形状: {npu_result.shape}")
    
    # 确保形状一致
    if cpu_result.shape != npu_result.shape:
        print(f"错误: 形状不匹配！")
        print(f"  CPU: {cpu_result.shape}")
        print(f"  NPU: {npu_result.shape}")
        return False
    
    # 转换为 numpy 进行比较
    cpu_np = cpu_result.numpy().astype(np.float32)
    npu_np = npu_result.numpy().astype(np.float32)
    
    # 计算误差
    abs_diff = np.abs(cpu_np - npu_np)
    rel_diff = np.abs(cpu_np - npu_np) / (np.abs(cpu_np) + 1e-8)  # 避免除零
    
    # 统计信息
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # 计算匹配的元素数量
    # 使用相对误差阈值 1e-4 和绝对误差阈值 1e-5
    rtol = 1e-4
    atol = 1e-5
    matches = np.isclose(cpu_np, npu_np, rtol=rtol, atol=atol)
    match_ratio = np.sum(matches) / matches.size
    
    print(f"\n误差统计:")
    print(f"  最大绝对误差: {max_abs_diff:.6e}")
    print(f"  平均绝对误差: {mean_abs_diff:.6e}")
    print(f"  最大相对误差: {max_rel_diff:.6e}")
    print(f"  平均相对误差: {mean_rel_diff:.6e}")
    print(f"\n匹配统计 (rtol={rtol}, atol={atol}):")
    print(f"  匹配元素数: {np.sum(matches)} / {matches.size}")
    print(f"  匹配比例: {match_ratio*100:.2f}%")
    
    # 找出误差最大的位置
    max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(f"\n最大误差位置: 行={max_diff_idx[0]}, 列={max_diff_idx[1]}")
    print(f"  CPU 值: {cpu_np[max_diff_idx]:.6f}")
    print(f"  NPU 值: {npu_np[max_diff_idx]:.6f}")
    print(f"  绝对误差: {abs_diff[max_diff_idx]:.6e}")
    print(f"  相对误差: {rel_diff[max_diff_idx]:.6e}")
    
    # 判断是否通过
    if match_ratio >= 0.99:  # 99% 以上匹配认为通过
        print(f"\n✓ 结果匹配良好 (匹配率 {match_ratio*100:.2f}%)")
        return True
    else:
        print(f"\n✗ 结果存在较大误差 (匹配率 {match_ratio*100:.2f}%)")
        return False


def run_comparison(dataset_name, dataset_loader):
    """运行单个数据集的比较"""
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = dataset_loader()
    graph = dataset[0]
    
    print(f"图信息:")
    print(f"  节点数: {graph.num_nodes()}")
    print(f"  边数: {graph.num_edges()}")
    original_feat_dim = graph.ndata['feat'].shape[1]
    print(f"  原始特征维度: {original_feat_dim}")
    
    # 准备特征（NPU 版本需要对齐）
    features = graph.ndata['feat']
    
    aligned_feat_dim = ((original_feat_dim + 15) // 16) * 16
    if aligned_feat_dim > original_feat_dim:
        padding_size = aligned_feat_dim - original_feat_dim
        features_aligned = torch.nn.functional.pad(features, (0, padding_size), mode='constant', value=0.0)
        print(f"  对齐后特征维度: {aligned_feat_dim} (填充了 {padding_size} 维)")
    else:
        features_aligned = features
        print(f"  特征维度已对齐: {aligned_feat_dim}")
    
    # 将图转换为 CSR 格式（用于 NPU 版本）
    src, dst = graph.edges()
    num_nodes = graph.num_nodes()
    edge_weights = np.ones(graph.num_edges(), dtype=np.float32)
    adj_scipy = sp.csr_matrix((edge_weights, (dst.numpy(), src.numpy())), 
                               shape=(num_nodes, num_nodes))
    
    row_ptr = torch.from_numpy(adj_scipy.indptr).int()
    col_ind = torch.from_numpy(adj_scipy.indices).int()
    values = torch.from_numpy(adj_scipy.data).float()
    
    # 计算每行最大非零元素个数
    max_nnz = 0
    for i in range(graph.num_nodes()):
        row_start = row_ptr[i].item()
        row_end = row_ptr[i + 1].item()
        max_nnz = max(max_nnz, row_end - row_start)
    max_nnz_per_row = max(32, max_nnz)
    print(f"  每行最大非零元素个数: {max_nnz}")
    
    # 计算 CPU 结果
    print("\n计算 CPU 结果 (DGL copy_u_sum)...")
    cpu_result = compute_cpu_result(graph, features)
    print(f"CPU 计算完成")
    
    # 计算 NPU 结果
    print("\n计算 NPU 结果 (spmm_sum)...")
    npu_result = compute_npu_result(graph, features_aligned, row_ptr, col_ind, values, max_nnz_per_row)
    print(f"NPU 计算完成")
    
    # 比较结果
    success = compare_results(cpu_result, npu_result, dataset_name, original_feat_dim, aligned_feat_dim)


    return success


def main():
    print("=" * 80)
    print("CPU vs NPU 结果比较")
    print("=" * 80)
    print("比较 DGL 的 copy_u_sum (CPU) 和 spmm_sum (NPU) 的结果")
    print("=" * 80)
    
    datasets = [
        
        ("PubMed", PubmedGraphDataset),
        # ("Cora", CoraGraphDataset),
    ]
    
    all_success = True
    for dataset_name, dataset_loader in datasets:
        success = run_comparison(dataset_name, dataset_loader)
        all_success = all_success and success
    
    print(f"\n{'='*80}")
    if all_success:
        print("✓ 所有数据集的结果匹配良好")
    else:
        print("✗ 部分数据集的结果存在误差")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
