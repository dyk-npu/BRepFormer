import os
import dgl
import torch
import json
from tqdm import tqdm

# 文件夹路径
bin_dir = "../../../CADDataSet/CBF_20000_best/bin"
labels_dir = "../../../CADDataSet/CBF_20000_best/label"

# 获取所有的文件名
file_names = os.listdir(bin_dir)

for file_name in tqdm(file_names, desc="Processing WorkData"):
    if not file_name.endswith('.bin'):
        continue

    # 对应的标签文件
    label_file_name = file_name.replace('.bin', '.json')

    bin_path = os.path.join(bin_dir, file_name)
    label_path = os.path.join(labels_dir, label_file_name)

    # --- 标签文件不存在就跳过 ---
    if not os.path.exists(label_path):
        print(f"[Skip] Label file not found: {label_path}")
        continue

    try:
        # --- 读取 JSON ---
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            labels_dict = json.load(f)

        # --- 同时读取 graph + geo（必须！）---
        graphs, geo = dgl.load_graphs(bin_path)
        g = graphs[0]

        num_nodes = g.num_nodes()

        # --- 提取标签 ---
        values = [labels_dict[str(i)] for i in range(num_nodes)]

        if len(values) != num_nodes:
            print(f"[Error] Label count mismatch in {file_name}")
            continue

        # --- 转为 tensor ---
        node_labels = torch.tensor(values)

        # --- 填入标签 ---
        g.ndata['l'] = node_labels

        # --- 保存 graph + geo（关键，防止丢失字典特征）---
        dgl.save_graphs(bin_path, [g], geo)

    except Exception as e:
        print(f"[Error processing {file_name}] {e}")
        continue

print("All files processed successfully.")
