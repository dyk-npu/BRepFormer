import os
import dgl
import torch
import json
from tqdm import tqdm

# --- 配置路径 ---
bin_dir = '/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFTRCAD/bin'
labels_dir = '/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFTRCAD/labels'

output_bin_dir = '/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/MFTRCAD/bin_with_labels'
os.makedirs(output_bin_dir, exist_ok=True)
# --- 结束配置 ---


file_names = os.listdir(bin_dir)
processed_count = 0
skipped_count = 0

for file_name in tqdm(file_names, desc="Processing AddMFTRCAD"):
    if not file_name.endswith('.bin'):
        continue

    label_file_name = file_name.replace('.bin', '.json')

    bin_path = os.path.join(bin_dir, file_name)
    label_path = os.path.join(labels_dir, label_file_name)
    output_bin_path = os.path.join(output_bin_dir, file_name)

    if not os.path.exists(label_path):
        skipped_count += 1
        continue

    try:
        # 读取标签
        with open(label_path, 'r') as f:
            labels = json.load(f)
        label = list(labels['cls'].values())
        node_labels = torch.tensor(label)

        # 关键：同时读取 graph + geo
        graphs, geo = dgl.load_graphs(bin_path)
        g = graphs[0]

        if len(node_labels) != g.number_of_nodes():
            skipped_count += 1
            continue

        # 添加标签
        g.ndata['l'] = node_labels

        # 关键：保存 graph + geo（否则附加字典会丢失）
        dgl.save_graphs(output_bin_path, [g], geo)

        processed_count += 1

    except Exception as e:
        skipped_count += 1
        continue

print("\n" + "="*30)
print("      处理完成")
print("="*30)
print(f"成功处理并保存的文件数: {processed_count}")
print(f"因缺少标签或错误而跳过的文件数: {skipped_count}")
print(f"处理好的文件已保存至: {output_bin_dir}")
