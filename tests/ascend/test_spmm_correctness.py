"""
使用 Cora 数据集测试 SpMM 操作在 CPU 和 NPU 上的精度
比较两者的输出结果，检查是否有误差
"""
import os
os.environ['LD_LIBRARY_PATH'] = '/home/zqm1/dgl-ascend/build:' + os.environ.get('LD_LIBRARY_PATH', '')
import torch
import dgl
import numpy as np
import time

# Try to import torch_npu
try:
    import torch_npu
    # Check if NPU is available
    NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
except (ImportError, AttributeError):
    NPU_AVAILABLE = False

def test_spmm_cpu(g, features, device='cpu', verbose=True):
    """在 CPU 上运行 SpMM"""
    if verbose:
        print(f"\n=== 在 {device.upper()} 上运行 SpMM ===")
    
    # 确保数据在 CPU 上
    g_cpu = g.to('cpu')
    features_cpu = features.to('cpu')
    
    # 创建输出张量
    out_cpu = torch.zeros(g.num_nodes(), features.shape[1], dtype=features.dtype)
    
    # 使用 DGL 的 SpMM 操作
    # copy_u + sum 操作
    with g_cpu.local_scope():
        g_cpu.ndata['h'] = features_cpu
        g_cpu.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
        out_cpu = g_cpu.ndata['h']
    
    return out_cpu

def test_spmm_npu(g, features, device='npu:0', verbose=True):
    """在 NPU 上运行 SpMM"""
    if verbose:
        print(f"\n=== 在 {device.upper()} 上运行 SpMM ===")
    
    if not NPU_AVAILABLE:
        if verbose:
            print(f"错误: NPU 不可用")
        return None
    
    # 确保数据在 NPU 上
    g_npu = g.to(device)
    features_npu = features.to(device)
    
    # 同步确保数据已传输
    torch.npu.synchronize()
    
    # 使用 DGL 的 SpMM 操作
    # copy_u + sum 操作
    with g_npu.local_scope():
        g_npu.ndata['h'] = features_npu
        g_npu.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
        out_npu = g_npu.ndata['h']
    
    # 同步确保计算完成
    torch.npu.synchronize()
    
    return out_npu

def compare_results(cpu_result, npu_result, rtol=1e-4, atol=1e-5):
    """比较 CPU 和 NPU 的结果"""
    print("\n=== 结果比较 ===")
    
    if npu_result is None:
        print("NPU 结果不可用，跳过比较")
        return False
    
    # 将 NPU 结果移到 CPU 进行比较
    npu_result_cpu = npu_result.cpu()
    
    # 检查形状
    if cpu_result.shape != npu_result_cpu.shape:
        print(f"错误: 形状不匹配!")
        print(f"  CPU 形状: {cpu_result.shape}")
        print(f"  NPU 形状: {npu_result_cpu.shape}")
        return False
    
    print(f"输出形状: {cpu_result.shape}")
    
    # 计算差异
    diff = torch.abs(cpu_result - npu_result_cpu)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    # 计算相对误差
    cpu_abs = torch.abs(cpu_result)
    relative_diff = diff / (cpu_abs + 1e-8)  # 避免除零
    max_rel_diff = torch.max(relative_diff).item()
    mean_rel_diff = torch.mean(relative_diff).item()
    
    print(f"最大绝对误差: {max_diff:.6e}")
    print(f"平均绝对误差: {mean_diff:.6e}")
    print(f"最大相对误差: {max_rel_diff:.6e}")
    print(f"平均相对误差: {mean_rel_diff:.6e}")
    
    # 使用 torch.allclose 检查是否接近
    is_close = torch.allclose(cpu_result, npu_result_cpu, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✓ 结果匹配! (rtol={rtol}, atol={atol})")
    else:
        print(f"✗ 结果不匹配! (rtol={rtol}, atol={atol})")
        
        # 找出差异最大的位置
        max_diff_idx = torch.argmax(diff)
        max_diff_idx_flat = max_diff_idx.item()
        row = max_diff_idx_flat // cpu_result.shape[1]
        col = max_diff_idx_flat % cpu_result.shape[1]
        
        print(f"\n最大差异位置: [{row}, {col}]")
        print(f"  CPU 值: {cpu_result[row, col].item():.6f}")
        print(f"  NPU 值: {npu_result_cpu[row, col].item():.6f}")
        print(f"  差异: {diff[row, col].item():.6e}")
    
    return is_close

def test_spmm_with_cora(rtol=1e-4, atol=1e-5, verbose=True):
    """使用 Cora 数据集进行精度测试"""
    print("=" * 60)
    print("使用 Cora 数据集测试 SpMM 精度")
    print("=" * 60)
    
    os.environ['DGLBACKEND'] = 'pytorch'
    
    try:
        from dgl.data import CoraGraphDataset
        from dgl import AddSelfLoop
        
        # 加载 Cora 数据集
        print("\n加载 Cora 数据集...")
        transform = AddSelfLoop()
        data = CoraGraphDataset(transform=transform)
        g = data[0]
        
        features = g.ndata['feat']
        print(f"\n图信息:")
        print(f"  节点数: {g.num_nodes()}")
        print(f"  边数: {g.num_edges()}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  特征数据类型: {features.dtype}")
        
        # 在 CPU 上预先准备图格式
        print("\n在 CPU 上准备图格式...")
        g = g.int()
        g = g.formats(['csc', 'csr'])
        g.create_formats_()
        print(f"  图 ID 类型: {g.idtype}")
        
        # 检查 CSR 格式
        if hasattr(g, 'adj_sparse'):
            csr = g.adj_sparse('csr')
            print(f"  CSR 格式: indptr shape={csr[0].shape}, indices shape={csr[1].shape}")
            if len(csr) > 2 and csr[2] is not None:
                print(f"  CSR data shape={csr[2].shape}")
        
        # 在 CPU 上运行
        print("\n在 CPU 上运行 SpMM...")
        cpu_start = time.time()
        cpu_result = test_spmm_cpu(g, features, verbose=verbose)
        cpu_time = time.time() - cpu_start
        print(f"  CPU 执行时间: {cpu_time*1000:.4f} 毫秒")
        
        # 在 NPU 上运行
        if NPU_AVAILABLE:
            print("\n在 NPU 上运行 SpMM...")
            npu_start = time.time()
            npu_result = test_spmm_npu(g, features, verbose=verbose)
            torch.npu.synchronize()
            npu_time = time.time() - npu_start
            print(f"  NPU 执行时间: {npu_time*1000:.4f} 毫秒")
            
            if npu_time > 0:
                speedup = cpu_time / npu_time
                print(f"  加速比: {speedup:.2f}x")
            
            # 详细的精度比较
            print("\n" + "=" * 60)
            print("精度分析")
            print("=" * 60)
            passed = compare_results(cpu_result, npu_result, rtol=rtol, atol=atol)
            
            # 额外的统计信息
            if verbose:
                npu_result_cpu = npu_result.cpu()
                diff = torch.abs(cpu_result - npu_result_cpu)
                
                # 计算不同误差阈值下的匹配率
                thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
                print("\n不同误差阈值下的匹配率:")
                for threshold in thresholds:
                    match_rate = (diff < threshold).float().mean().item() * 100
                    print(f"  < {threshold:.0e}: {match_rate:.2f}%")
                
                # 输出一些样本值用于验证
                print("\n样本值对比 (前5个节点, 前5个特征维度):")
                print(f"{'节点':<6} {'维度':<6} {'CPU值':<15} {'NPU值':<15} {'差异':<15}")
                print("-" * 60)
                for i in range(min(5, cpu_result.shape[0])):
                    for j in range(min(5, cpu_result.shape[1])):
                        cpu_val = cpu_result[i, j].item()
                        npu_val = npu_result_cpu[i, j].item()
                        diff_val = diff[i, j].item()
                        print(f"{i:<6} {j:<6} {cpu_val:<15.6f} {npu_val:<15.6f} {diff_val:<15.6e}")
            
            return passed
        else:
            print("\n跳过 NPU 测试 (NPU 不可用)")
            return True
            
    except Exception as e:
        print(f"\n错误: 无法加载 Cora 数据集: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set LD_LIBRARY_PATH to include build directory for AscendC kernel library
    test_spmm_with_cora(verbose=True)
