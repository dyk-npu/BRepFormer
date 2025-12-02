import os
import dgl
import torch
import json
from tqdm import tqdm

# --- 配置路径 ---
# 修改为你的分类数据集路径
bin_dir = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology" 
output_bin_dir = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology" 

os.makedirs(output_bin_dir, exist_ok=True)
# --- 结束配置 ---

def get_label_from_filename(filename):
    """
    解析文件名获取标签索引 (0-51)
    规则: 
    a-z (lower) -> 0-25
    a-z (upper) -> 26-51
    文件名示例: a_ABeeZee_lower.step -> .bin
    """
    try:
        # 去除扩展名
        name_part = os.path.splitext(filename)[0] 
        parts = name_part.split('_')
        
        # 获取首字母 (即字符本身)
        char = parts[0].lower() 
        # 获取大小写标识 (通常在最后)
        case_type = parts[-1] # 'lower' or 'upper'
        
        if len(char) != 1:
            return None
            
        # 计算基础索引 (a=0, b=1...)
        base_idx = ord(char) - ord('a')
        
        if base_idx < 0 or base_idx > 25:
            return None
            
        # 如果是 upper，偏移 26
        final_idx = base_idx + 26 if 'upper' in case_type else base_idx
        
        return final_idx
    except:
        return None

file_names = os.listdir(bin_dir)
processed_count = 0
skipped_count = 0

# 类别映射表，用于打印确认
class_map = {} 

for file_name in tqdm(file_names, desc="Generating Classification Labels"):
    if not file_name.endswith('.bin'):
        continue

    bin_path = os.path.join(bin_dir, file_name)
    output_bin_path = os.path.join(output_bin_dir, file_name)

    # 1. 解析标签
    label_idx = get_label_from_filename(file_name)
    
    if label_idx is None:
        print(f"Warning: Could not parse label for {file_name}")
        skipped_count += 1
        continue

    class_map[label_idx] = class_map.get(label_idx, 0) + 1

    try:
        # 2. 加载图数据
        graphs, geo = dgl.load_graphs(bin_path)
        g = graphs[0]
        num_nodes = g.number_of_nodes()

        # 3. 构造标签 Tensor
        # 分类任务中，虽然是图分类，但为了兼容 collator，我们给每个节点都打上相同的图标签
        node_labels = torch.full((num_nodes,), label_idx, dtype=torch.long)

        # 4. 赋值给 ndata['l']
        g.ndata['l'] = node_labels

        # 5. 保存
        dgl.save_graphs(output_bin_path, [g], geo)

        processed_count += 1

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        skipped_count += 1
        continue

print("\n" + "="*30)
print("      处理完成")
print("="*30)
print(f"成功处理: {processed_count}")
print(f"跳过: {skipped_count}")
print(f"输出目录: {output_bin_dir}")
print(f"检测到的类别数量: {len(class_map)} / 52")


