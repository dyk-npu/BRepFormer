import os
import argparse
import torch
import dgl
from tqdm import tqdm

def check_tensor_validity(tensor, name, file_name):
    """
    检查张量是否包含 NaN 或 Inf
    """
    if torch.isnan(tensor).any():
        return False, f"{name} contains NaN"
    if torch.isinf(tensor).any():
        return False, f"{name} contains Inf"
    return True, ""

def clean_dataset(data_folder, delete_files=False):
    print(f"正在扫描文件夹: {data_folder}")
    if delete_files:
        print("!!! 警告: 检测到 --delete 参数，有问题的文件将被永久删除 !!!")
    else:
        print("--- 预演模式 (Dry Run): 不会删除文件，仅列出问题文件 ---")

    bin_files = [f for f in os.listdir(data_folder) if f.endswith('.bin')]
    
    total_files = len(bin_files)
    corrupt_count = 0
    valid_count = 0
    error_details = []

    # 需要检查的四个矩阵的键名
    keys_to_check = [
        'angle_matrix', 
        'centroid_distance_matrix', 
        'shortest_distance_matrix', 
        'edge_path_matrix'
    ]

    for file_name in tqdm(bin_files, desc="Checking files"):
        file_path = os.path.join(data_folder, file_name)
        
        is_corrupt = False
        reason = ""

        try:
            # 加载图数据
            # dgl.load_graphs 返回 (graph_list, labels_dict)
            graph_list, labels_dict = dgl.load_graphs(file_path)
            
            # 1. 检查图本身是否为空 (有些生成失败的文件可能没有图)
            if len(graph_list) == 0:
                is_corrupt = True
                reason = "Graph list is empty"
            
            # 2. 检查节点特征 (node_data) 是否有 NaN (虽然你主要想查矩阵，但顺便查一下图特征更保险)
            elif torch.isnan(graph_list[0].ndata['x']).any():
                is_corrupt = True
                reason = "Graph node features contain NaN"
            
            # 3. 检查四个特征矩阵
            else:
                for key in keys_to_check:
                    if key not in labels_dict:
                        # 如果之前的脚本版本不同，可能缺少某些键，建议视为损坏或跳过
                        # 这里我们视为损坏，保证数据集一致性
                        is_corrupt = True
                        reason = f"Missing key: {key}"
                        break
                    
                    # 检查数值有效性
                    is_valid, msg = check_tensor_validity(labels_dict[key], key, file_name)
                    if not is_valid:
                        is_corrupt = True
                        reason = msg
                        break

        except Exception as e:
            # 加载过程报错（文件损坏）
            is_corrupt = True
            reason = f"Load Error: {str(e)}"

        # 处理结果
        if is_corrupt:
            corrupt_count += 1
            error_details.append(f"{file_name}: {reason}")
            
            if delete_files:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"删除失败 {file_name}: {e}")
        else:
            valid_count += 1

    # --- 打印报告 ---
    print("\n" + "="*40)
    print("扫描完成报告")
    print("="*40)
    print(f"总文件数: {total_files}")
    print(f"正常文件: {valid_count}")
    print(f"损坏文件: {corrupt_count}")
    
    if corrupt_count > 0:
        print("\n[问题文件列表及原因]:")
        # 只打印前20个，防止刷屏
        for err in error_details[:20]:
            print(f"  - {err}")
        if len(error_details) > 20:
            print(f"  - ... (还有 {len(error_details) - 20} 个)")

        if not delete_files:
            print(f"\n提示: 当前为预演模式。请运行带上 --delete 参数来执行删除操作。")
            print(f"建议命令: python clean_dataset.py --input-folder {data_folder} --delete")
        else:
            print(f"\n已删除 {corrupt_count} 个损坏的文件。")
    else:
        print("\n完美！数据集中没有发现 NaN 或 Inf。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for NaN/Inf in DGL bin files and optionally delete them.")
    
    parser.add_argument('--input-folder', type=str, default=r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology", 
                        help='Folder containing .bin files to check')
    parser.add_argument('--delete', action='store_true', 
                        help='If set, WILL DELETE corrupt files. If not set, only checks.')

    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Folder '{args.input_folder}' does not exist.")
    else:
        clean_dataset(args.input_folder, args.delete)