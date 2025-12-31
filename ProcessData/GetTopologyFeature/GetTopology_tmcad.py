import math
import os
import numpy as np
import torch
import dgl
import networkx as nx
from tqdm import tqdm
import argparse
import multiprocessing
import time

import shutup

shutup.please()

# ==========================================
# 核心特征提取函数（封装在一个类中或保持独立）
# ==========================================

def get_angle_matrix(file_path, device):
    g = dgl.load_graphs(file_path)[0][0].to(device)
    node_feature = g.ndata["x"].type(torch.float32).to(device)

    num_nodes = node_feature.shape[0]
    mean_normals_per_node = []

    for i in range(num_nodes):
        normals = node_feature[i, :, :, 3:6]
        hidden_status = node_feature[i, :, :, 6]
        mask = (hidden_status == 0)

        if torch.any(mask):
            mask_expanded = mask.unsqueeze(-1).expand_as(normals)
            filtered_normals = normals[mask_expanded].view(-1, 3)
            mean_normal = torch.mean(filtered_normals, dim=0)
        else:
            mean_normal = torch.zeros(3, device=device)
        mean_normals_per_node.append(mean_normal)

    mean_normals_per_node = torch.stack(mean_normals_per_node)
    num_nodes = mean_normals_per_node.shape[0]
    angle_matrix = torch.zeros((num_nodes, num_nodes), device=device)

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            dot_product = torch.dot(mean_normals_per_node[i], mean_normals_per_node[j])
            norm_i = torch.norm(mean_normals_per_node[i])
            norm_j = torch.norm(mean_normals_per_node[j])
            cos_theta = dot_product / (norm_i * norm_j + 1e-8)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            angle_radians = math.acos(cos_theta)
            angle_degrees = math.degrees(angle_radians)
            angle_matrix[i, j] = angle_degrees
            angle_matrix[j, i] = angle_degrees
    return angle_matrix

def get_centroid_distance_matrix(file_path, device):
    g = dgl.load_graphs(file_path)[0][0].to(device)
    node_centroid_matrix = g.ndata["c"].type(torch.float32).to(device)
    num_nodes = node_centroid_matrix.shape[0]
    expanded_a = node_centroid_matrix.unsqueeze(1).expand(-1, num_nodes, -1)
    expanded_b = node_centroid_matrix.unsqueeze(0).expand(num_nodes, -1, -1)
    distances = torch.norm(expanded_a - expanded_b, dim=2)
    return distances

def get_shortest_distance_matrix(file_path):
    # networkx 计算在 CPU 上进行，通常是内存占用的元凶
    g = dgl.load_graphs(file_path)[0][0]
    G = g.to_networkx()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    distance_matrix = [[lengths[i].get(j, float('inf')) for j in range(n)] for i in range(n)]
    max_distance = 0
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] != float('inf') and distance_matrix[i][j] > max_distance:
                max_distance = distance_matrix[i][j]
    return torch.tensor(distance_matrix, dtype=torch.float32), max_distance

def get_edge_path_matrix(file_path, max_distance, device):
    g = dgl.load_graphs(file_path)[0][0].to(device)
    num_nodes = g.num_nodes()
    max_dist_int = int(max_distance)
    edge_path_matrix = torch.full((num_nodes, num_nodes, max_dist_int), -1, dtype=torch.int32, device=device)

    for source in range(num_nodes):
        visited = {}
        queue = [(source, [], [])]  # (current_node, path_nodes, path_edges)
        while queue:
            curr, path, e_path = queue.pop(0)
            if curr not in visited or len(e_path) < visited[curr]:
                visited[curr] = len(e_path)
                if 0 < len(e_path) <= max_dist_int:
                    edge_path_matrix[source, curr, :len(e_path)] = torch.tensor(e_path, dtype=torch.int32, device=device)
                
                if len(e_path) < max_dist_int:
                    successors = g.successors(curr)
                    for succ in successors:
                        eid = g.edge_ids(curr, succ).item()
                        queue.append((succ.item(), path + [curr], e_path + [eid]))
    return edge_path_matrix

# ==========================================
# 隔离执行的子任务逻辑
# ==========================================

def worker_task(file_path, output_path):
    """
    该函数在独立的子进程中运行。
    退出后，该函数产生的所有内存占用会被操作系统强制释放。
    """
    try:
        # 子进程内部根据是否有可用 GPU 初始化设备
        local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算各项特征
        angle_mat = get_angle_matrix(file_path, local_device)
        centroid_dist_mat = get_centroid_distance_matrix(file_path, local_device)
        shortest_dist_mat, max_dist = get_shortest_distance_matrix(file_path)
        edge_path_mat = get_edge_path_matrix(file_path, max_dist, local_device)

        feature_dict = {
            'angle_matrix': angle_mat.cpu(), # 转到CPU存储以保持通用
            'centroid_distance_matrix': centroid_dist_mat.cpu(),
            'shortest_distance_matrix': shortest_dist_mat,
            'edge_path_matrix': edge_path_mat.cpu()
        }

        # 加载原始图并合并保存
        graphs, _ = dgl.load_graphs(file_path)
        dgl.save_graphs(output_path, graphs, feature_dict)
        
    except Exception as e:
        print(f"\n[Error] Processing {file_path} failed: {e}")

# ==========================================
# 主控制器
# ==========================================

def process_all_files(input_folder, output_folder, timeout_limit=600):
    """
    核心控制逻辑：
    1. 扫描二级目录
    2. 检查断点续传
    3. 启动带超时的子进程（彻底防止内存泄漏）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 扫描所有文件任务
    tasks = []
    # 查找子文件夹（bolt, nut, gear等）
    subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    if not subdirs: # 兼容一级目录
        subdirs = ["."]

    for subdir in subdirs:
        in_subdir_path = os.path.join(input_folder, subdir)
        out_subdir_path = os.path.join(output_folder, subdir)
        os.makedirs(out_subdir_path, exist_ok=True)

        for filename in os.listdir(in_subdir_path):
            if filename.endswith(".bin"):
                tasks.append((
                    os.path.join(in_subdir_path, filename),
                    os.path.join(out_subdir_path, filename)
                ))

    print(f"Total files to process: {len(tasks)}")

    # 2. 逐个处理任务
    for in_file, out_file in tqdm(tasks, desc="Overall Progress"):
        
        # 策略 1：检查文件是否存在（断点续传）
        if os.path.exists(out_file):
            continue

        # 策略 2：启动独立子进程执行，防止内存长期占用不释放
        p = multiprocessing.Process(target=worker_task, args=(in_file, out_file))
        p.start()

        # 等待子进程，设置超时时间
        p.join(timeout=timeout_limit)

        # 策略 3：处理超时挂死情况
        if p.is_alive():
            print(f"\n[Timeout Warning] File {in_file} exceeded {timeout_limit}s. Terminating process...")
            p.terminate()  # 强制杀掉进程，释放其占用的内存/显存
            p.join()       # 必须 join 以回收进程句柄资源
            
            # 清理可能产生的残余破损文件，防止下次被误判为已完成
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except:
                    pass

if __name__ == '__main__':
    # 针对 Windows 和 GPU 任务，必须使用 spawn 模式启动多进程
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Topology Feature Extractor with Multi-level folder support.')
    parser.add_argument('--input-folder', type=str, default=r"E:\CADdataset\TMCAD_Dataset\TMCAD_V2\BRepFormer\bin_attribute",
                        help='Input folder containing subfolders of .bin files.')
    parser.add_argument('--output-folder', type=str, default=r"E:\CADdataset\TMCAD_Dataset\TMCAD_V2\BRepFormer\bin_topology",
                        help='Output folder to store processed files with same structure.')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout for each file in seconds (default 10min)')

    args = parser.parse_args()

    # 开始处理
    process_all_files(args.input_folder, args.output_folder, args.timeout)