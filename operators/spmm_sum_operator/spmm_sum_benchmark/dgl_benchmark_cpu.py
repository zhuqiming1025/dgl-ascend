"""
仅在 CPU 上运行 DGL spmm_sum 操作的基准测试。
使用 Cora 和 PubMed 数据集，运行 100 次测试并输出平均值和方差。
"""

import time
import torch
import dgl
import numpy as np
import os
from dgl.data import CoraGraphDataset, PubmedGraphDataset
import pandas as pd
from utils import run_benchmark_for_dataset


def main():
    print("=" * 80)
    print("DGL SPMM Sum 操作性能测试（CPU）")
    print("数据集: Cora, PubMed")
    print("测试次数: 100 次")
    print("=" * 80)

    # 测试参数
    num_iterations = 100
    datasets = [
        ("Cora", CoraGraphDataset),
        # ("PubMed", PubmedGraphDataset),
    ]

    all_results = []

    for dataset_name, dataset_loader in datasets:
        print("\n" + "=" * 80)
        print(f"{dataset_name} - CPU 测试")
        print("=" * 80)

        stats, _ = run_benchmark_for_dataset(
            dataset_name,
            dataset_loader,
            device="cpu",
            num_iterations=num_iterations,
        )

        for op_name, stat in stats.items():
            all_results.append(
                {
                    "数据集": dataset_name,
                    "操作": op_name,
                    "设备": "CPU",
                    "平均耗时(ms)": stat["mean"],
                    "标准差(ms)": stat["std"],
                    "方差(ms²)": stat["variance"],
                }
            )

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("CPU 性能测试结果（100 次测试的平均值、标准差和方差）")
    print("=" * 80)
    print(df.to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "dgl_benchmark_cpu_results.csv")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nCPU 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
