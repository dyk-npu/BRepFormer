import gc
import json
import os

import dgl
import torch
import tqdm
import os.path as osp
from functools import partial
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import Pool

from dgl import graph

from Feature_extractor import GraphExtractor

import shutup


shutup.please()


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def save_json_data(path_name, data):
    """Export a data to a json file"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def save_dgl_graph(file_name ,graph_data : dict):

    # print("ty",graph_data.keys())
    # num_nodes = torch.tensor(graph_data['num_nodes'],dtype=torch.int32)

    # num_edges = torch.tensor(graph_data['graph']['edges'],dtype=torch.int32).shape[1]

    # 获取边的信息
    edges = torch.tensor(graph_data['graph']['edges'], dtype=torch.int32)
    src, dst = edges[0], edges[1]  #  edges 是 shape [2, num_edges] 的张量

    # adjacency_graph = torch.tensor(graph_data['graph']['edges'],dtype=torch.int32)

    # 创建DGL图
    g = dgl.graph((src, dst))

    # 检查节点数和边数
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()


    # Graph Face Feature
    g.ndata["x"] = torch.tensor(graph_data["graph_face_grid"],dtype=torch.float32).permute(0,2,3,1)
    graph_face_attr = torch.tensor(graph_data["graph_face_attr"], dtype=torch.float32) #[num_nodes,feature])

    node_type = graph_face_attr[: , 0:9]
    node_area = graph_face_attr[: , 9]
    node_rational = graph_face_attr[: , 10]
    node_centroid = graph_face_attr[: , 11:14]

    g.ndata["t"] = node_type
    g.ndata["a"] = node_area
    g.ndata["r"] = node_rational
    g.ndata["c"] = node_centroid

    #Graph Edge Feature
    g.edata["x"] = torch.tensor(graph_data["graph_edge_grid"], dtype=torch.float32).permute(0, 2, 1)
    graph_edge_attr = torch.tensor(graph_data["graph_edge_attr"], dtype=torch.float32)

    edge_convexity = graph_edge_attr[: , 0:3]
    edge_length = graph_edge_attr[: , 3]
    edge_type = graph_edge_attr[: , 4:15]

    g.edata["c"] = edge_convexity
    g.edata["l"] = edge_length
    g.edata["t"] = edge_type

    dgl.data.utils.save_graphs(file_name, g)





def process_one_file(args,output_path):

    file_path, config = args

    # 构建输出文件的完整路径
    output_file_path = os.path.join(output_path, file_path.stem + '.bin')

    # 检查输出路径下是否已存在同名.bin文件
    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists, skipping...")
        return [str(file_path.stem)]

    try:
        extractor = GraphExtractor(file_path, config, scale_body=True)
        out = extractor.process()
        graph_index = str(file_path.stem)
        # graph = [graph_index, out]


        save_dgl_graph(osp.join(output_path, graph_index + '.bin'), out)

        return [str(file_path.stem)]
    except Exception as e:
        print(e)
        return []


def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    step_path = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/step_files"

    output = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin"
    attribute_config_path = "/mnt/sdb/daiyongkang/Work/dyk/BRepFormer/ProcessData/GetAttributeFeature/attribute_config.json"
    num_workers = 16

    step_path = Path(step_path)
    output_path = Path(output)
    if not output_path.exists():
        output_path.mkdir()
    attribute_config_path = Path(attribute_config_path)

    attribute_config = load_json(attribute_config_path)
    step_files = list(step_path.glob("*.st*p"))

    # 创建一个partial函数，预设attribute_config参数
    partial_process_one_file = partial(process_one_file, output_path = output)

    pool = Pool(processes=num_workers, initializer=initializer,)
    try:
        results = list(tqdm.tqdm(
            pool.imap(partial_process_one_file, zip(step_files, repeat(attribute_config))),
            total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    pool.terminate()
    pool.join()

    graph_count = 0
    fail_count = 0
    graphs = []
    for res in results:
        if len(res) > 0:
            graph_count += 1
            graphs.append(res)
        else:
            fail_count += 1

    gc.collect()
    print(f"Process {len(results)} files. Generate {graph_count} graphs. Has {fail_count} failed files.")
