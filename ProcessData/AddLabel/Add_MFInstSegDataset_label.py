import os
import dgl
import torch
import json
from tqdm import tqdm

# 文件夹路径
bin_dir = '../../BRepFormer/dataset/MFInstSeg/bin/'
labels_dir = '../../BRepFormer/dataset/MFInstSeg/label/'

file_names = os.listdir(bin_dir)

for file_name in tqdm(file_names, desc="Processing MFInstSeg"):
    if not file_name.endswith('.bin'):
        continue

    label_file_name = file_name.replace('.bin', '.json')
    bin_path = os.path.join(bin_dir, file_name)
    label_path = os.path.join(labels_dir, label_file_name)

    # --- 标签文件不存在则跳过 ---
    if not os.path.exists(label_path):
        print(f"[Skip] Label file not found: {label_path}")
        continue

    try:
        # --- 加载 JSON ---
        with open(label_path, 'r') as f:
            labels = json.load(f)

        # MFInstSeg 的 JSON 结构是： [ [id, {"seg": {...}}] ... ]
        seg_dict = labels[0][1]['seg']
        seg_values = list(seg_dict.values())

        # --- 加载 graph + geo ---
        graphs, geo = dgl.load_graphs(bin_path)
        g = graphs[0]

        num_nodes = g.num_nodes()

        # 节点数校验
        if len(seg_values) != num_nodes:
            print(f"[Error] Node count mismatch in {file_name}: labels={len(seg_values)}, nodes={num_nodes}")
            continue

        # 转为 tensor
        node_labels = torch.tensor(seg_values)

        # 添加到图里
        g.ndata['l'] = node_labels

        # 删除 f 特征（你的代码提到过）
        if 'f' in g.ndata:
            del g.ndata['f']

        # --- 保存 graph + geo（关键，不丢失任何附加特征！）---
        dgl.save_graphs(bin_path, [g], geo)

    except Exception as e:
        print(f"[Error processing {file_name}] {e}")
        continue

print("All files processed.")
