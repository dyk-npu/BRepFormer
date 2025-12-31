import os
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from .collator import collator
from .geometry_utils import get_random_rotation, rotate_uvgrid

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CADRecognition(Dataset):
    def __init__(self, root_dir, split="train", random_rotate=False, num_class=27):
        path = pathlib.Path(root_dir)
        self.split = split
        self.random_rotate = random_rotate
        self.file_paths = []
        self._get_filenames(path, f"{split}.txt")

    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        with open(root_dir / filelist, "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        
        root_dir = root_dir / 'bin_topology'
        for x in tqdm(root_dir.rglob("*.bin")):
            if x.stem in file_list:
                self.file_paths.append(x)
        print(f"Done loading {len(self.file_paths)} files")

    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]
        pyg_graph = PYGGraph()
        pyg_graph.graph = graph

        if self.random_rotate:
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"].type(torch.float32), rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"].type(torch.float32), rotation)

        pyg_graph.node_data = graph.ndata["x"].type(torch.float32)
        pyg_graph.edge_data = graph.edata["x"].type(torch.float32)
        pyg_graph.face_type = graph.ndata["t"].type(torch.float32)
        pyg_graph.face_area = graph.ndata["a"].type(torch.float32)
        pyg_graph.face_rational = graph.ndata["r"].type(torch.int32)
        pyg_graph.face_centroid = graph.ndata["c"].type(torch.float32)
        pyg_graph.label_feature = graph.ndata["l"].type(torch.float32)
        pyg_graph.edge_type = graph.edata["t"].type(torch.float32)
        pyg_graph.edge_length = graph.edata["l"].type(torch.float32)
        pyg_graph.edge_convexity = graph.edata["c"].type(torch.float32)

        dense_adj = graph.adj().to_dense().type(torch.float32)
        n_nodes = graph.num_nodes()
        pyg_graph.node_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

        pyg_graph.angle = graphfile[1]["angle_matrix"]
        pyg_graph.centroid_distance = graphfile[1]["centroid_distance_matrix"]
        pyg_graph.shortest_distance = graphfile[1]["shortest_distance_matrix"].to(torch.int32)
        pyg_graph.edge_path = graphfile[1]["edge_path_matrix"].to(torch.int32)

        basename = os.path.basename(file_path).replace(os.path.splitext(file_path)[1], "")
        pyg_graph.data_id = int([s for s in basename.split("_") if s.isdigit()][-1])

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.load_one_graph(self.file_paths[idx])

    def _collate(self, batch):
        return collator(batch, multi_hop_max_dist=16, spatial_pos_max=32)

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoaderX(
            dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate,
            num_workers=num_workers, drop_last=True, pin_memory=True, prefetch_factor=2, persistent_workers=False
        )