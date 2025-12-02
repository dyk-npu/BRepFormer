import math
import os

import numpy as np
import torch
import dgl
import networkx as nx
from tqdm import tqdm
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_angle_matrix(file_path):
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
            mean_normal = torch.zeros(3,device=device)

        mean_normals_per_node.append(mean_normal)

    mean_normals_per_node = torch.stack(mean_normals_per_node)
    num_nodes = mean_normals_per_node.shape[0]
    angle_matrix = torch.zeros((num_nodes, num_nodes),device=device)

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            dot_product = torch.dot(mean_normals_per_node[i], mean_normals_per_node[j])
            norm_i = torch.norm(mean_normals_per_node[i])
            norm_j = torch.norm(mean_normals_per_node[j])
            cos_theta = dot_product / (norm_i * norm_j)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            angle_radians = math.acos(cos_theta)
            angle_degrees = math.degrees(angle_radians)
            angle_matrix[i, j] = angle_degrees
            angle_matrix[j, i] = angle_degrees

    return angle_matrix


def get_centroid_distance_matrix(file_path):
    g = dgl.load_graphs(file_path)[0][0].to(device)
    node_centroid_matrix = g.ndata["c"].type(torch.float32).to(device)

    num_nodes = node_centroid_matrix.shape[0]
    expanded_a = node_centroid_matrix.unsqueeze(1).expand(-1, num_nodes, -1)
    expanded_b = node_centroid_matrix.unsqueeze(0).expand(num_nodes, -1, -1)
    distances = torch.norm(expanded_a - expanded_b, dim=2)

    return distances


def get_shortest_distance_matrix(file_path):
    g = dgl.load_graphs(file_path)[0][0]
    G = g.to_networkx()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    distance_matrix = [[lengths[i].get(j, float('inf')) for j in range(n)] for i in range(n)]

    max_distance = -np.inf
    # 遍历距离矩阵
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] > max_distance and distance_matrix[i][j] != float('inf'):

                max_distance = distance_matrix[i][j]

    return torch.tensor(distance_matrix, dtype=torch.float32),max_distance


def get_edge_path_matrix(file_path,max_distance):
    g = dgl.load_graphs(file_path)[0][0].to(device)
    num_nodes = g.num_nodes()


    max_distance = torch.tensor(max_distance, dtype=torch.int32,device=device)

    edge_path_matrix = torch.full((num_nodes, num_nodes, max_distance), -1, dtype=torch.int32,device=device)

    for source in range(num_nodes):
        visited = {}
        queue = [(source, [source], [])]  # (current node, path of nodes, path of edges)

        while queue:
            current_node, path, edge_path = queue.pop(0)

            if current_node not in visited:
                visited[current_node] = True

                if len(edge_path) < max_distance:
                    edge_path_matrix[source, current_node, :len(edge_path)] = torch.tensor(edge_path, dtype=torch.int32,device=device)

                neighbors = g.successors(current_node)
                for neighbor in neighbors:
                    edge_id = g.edge_ids(current_node, neighbor).item()
                    queue.append((neighbor.item(), path + [neighbor.item()], edge_path + [edge_id]))

    return edge_path_matrix




def process_files_in_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有 .bin 文件
    bin_files = [f for f in os.listdir(input_folder) if f.endswith('.bin')]

    # for file_name in bin_files:
    for file_name in tqdm(bin_files, desc="Processing files"):
        file_path = os.path.join(input_folder, file_name)

        output_file_path = os.path.join(output_folder, file_name)

        if os.path.exists(output_file_path):
            # 如果存在，则跳过当前文件
            print(f"{output_file_path} already exists")
            continue

        # 计算特征矩阵
        angle_matrix = get_angle_matrix(file_path)
        centroid_distance_matrix = get_centroid_distance_matrix(file_path)
        shortest_distance_matrix , max_distance= get_shortest_distance_matrix(file_path)
        edge_path_matrix = get_edge_path_matrix(file_path,max_distance)

        # 存储到字典中
        feature_dict = {
            'angle_matrix': angle_matrix,
            'centroid_distance_matrix': centroid_distance_matrix,
            'shortest_distance_matrix': shortest_distance_matrix,
            'edge_path_matrix': edge_path_matrix
        }

        # 加载原始图数据
        graphs, _ = dgl.load_graphs(file_path)

        # 保存到.bin文件的第二个对象
        # noinspection PyTypeChecker
        dgl.save_graphs(output_file_path, graphs, feature_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process .bin files and extract global features.')

    # 添加命令行参数
    parser.add_argument('--input-folder', type=str, default=r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin",
                        help='Input folder containing .bin files.')
    parser.add_argument('--output-folder', type=str, default=r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology",
                        help='Output folder to store processed files.')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理函数
    process_files_in_folder(args.input_folder, args.output_folder)

