import os
import dgl
import torch
import numpy as np
from tqdm import tqdm

# ========================================
# 配置：你的 .bin 文件所在目录
# ========================================
bin_dir = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology"  # 修改成你的目录

# 收集所有 shortest_distance 统计值
all_min = []
all_max = []
all_mean = []

# 找到所有 bin 文件
bin_files = [f for f in os.listdir(bin_dir) if f.endswith(".bin")]

print(">>> 扫描目录：", bin_dir)
print(f">>> 共发现 {len(bin_files)} 个 .bin 文件\n")

# tqdm 进度条
for fname in tqdm(bin_files, desc="Processing BIN files"):
    file_path = os.path.join(bin_dir, fname)

    try:
        graphs, geo = dgl.data.utils.load_graphs(file_path)
    except Exception as e:
        print(f"\n[ERROR] 无法读取 {fname}: {e}")
        continue

    if "shortest_distance_matrix" not in geo:
        print(f"\n[WARNING] {fname} 中不含 shortest_distance_matrix")
        continue

    shortest_dist = geo["shortest_distance_matrix"].float()

    # 忽略自环距离 0
    mask = shortest_dist > 0
    valid_values = shortest_dist[mask]

    if valid_values.numel() == 0:
        print(f"\n[WARNING] {fname} 没有有效 shortest distance 值")
        continue

    # 单图统计
    min_d = valid_values.min().item()
    max_d = valid_values.max().item()
    mean_d = valid_values.mean().item()

    # 保存全局统计
    all_min.append(min_d)
    all_max.append(max_d)
    all_mean.append(mean_d)

# ========================================
# 输出全局统计结果
# ========================================
print("\n================ 最终统计结果 ================")
if all_min:
    print(f">>> 所有文件 shortest_distance 最小值中的最小： {np.min(all_min):.4f}")
    print(f">>> 所有文件 shortest_distance 最大值中的最大： {np.max(all_max):.4f}")
    print(f">>> 所有文件 shortest_distance 平均值（文件平均）： {np.mean(all_mean):.4f}")
else:
    print("未读取到任何 shortest_distance_matrix 数据！")
