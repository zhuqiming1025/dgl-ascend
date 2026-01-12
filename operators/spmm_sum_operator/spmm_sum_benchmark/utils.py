import time
import torch
import dgl
import numpy as np
from dgl.data import CoraGraphDataset, PubmedGraphDataset
import pandas as pd

def benchmark_spmm_ops_single_run(graph, features, device='cpu'):
    """单次运行spmm_sum操作的性能测试"""
    results = {}
    
    # 将数据移到指定设备
    features = features.to(device)
    graph = graph.to(device)
    
    # 同步（如果是GPU）
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测试spmm_sum (copy_u_sum)
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    result = dgl.ops.copy_u_sum(graph, features)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    results['spmm_sum'] = (end - start) * 1000  # 转换为毫秒
    
    return results


def benchmark_with_stats(graph, features, device='cpu', num_iterations=10):
    """运行多次测试并计算平均值和方差"""
    # 预热 spmm_sum
    features_warmup = features.to(device)
    graph_warmup = graph.to(device)
    
    for _ in range(5):
        # 预热 spmm_sum（使用 copy_u_sum）
        _ = dgl.ops.copy_u_sum(graph_warmup, features_warmup)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 存储所有运行的结果
    spmm_results = {'spmm_sum': []}
    
    # 运行多次测试
    for i in range(num_iterations):
        spmm_res = benchmark_spmm_ops_single_run(graph, features, device)
        for key in spmm_results.keys():
            spmm_results[key].append(spmm_res[key])
    
    # 计算统计信息
    stats = {}
    for op_name, times in spmm_results.items():
        stats[op_name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'variance': np.var(times)
        }
    
    return stats


def run_benchmark_for_dataset(dataset_name, dataset_loader, device='cpu', num_iterations=10):
    """为指定数据集运行基准测试"""
    print(f"\n正在加载{dataset_name}数据集...")
    dataset = dataset_loader()
    graph = dataset[0]
    
    print(f"图信息:")
    print(f"  节点数: {graph.num_nodes()}")
    print(f"  边数: {graph.num_edges()}")
    print(f"  特征维度: {graph.ndata['feat'].shape[1]}")
    
    # 准备特征
    features = graph.ndata['feat']
    
    print(f"\n在{device.upper()}上运行{num_iterations}次测试...")
    stats = benchmark_with_stats(graph, features, device=device, num_iterations=num_iterations)
    
    return stats, dataset_name
