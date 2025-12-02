import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLP, EdgeConv

# --- CNN Helpers ---
def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )

def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )

def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
    )

class CurveEncoder(nn.Module):
    def __init__(self, in_channels=12, output_dims=128):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class SurfaceEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dims=128):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# --- Node & Bias Encoders ---

class NonLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = F.relu(self.bn2(self.linear2(x)))
        return x

def init_params_scaled(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeature(nn.Module):
    def __init__(self, num_heads, num_degree, hidden_dim, n_layers):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.surf_encoder = SurfaceEncoder(in_channels=7, output_dims=int(0.5*hidden_dim))
        self.face_area_encoder = NonLinear(1, int(0.125*hidden_dim))
        self.face_type_encoder = NonLinear(9, int(0.125*hidden_dim))
        self.face_centroid_encoder = NonLinear(3,int(0.125*hidden_dim))
        self.face_rational_encoder = nn.Embedding(2,int(0.125*hidden_dim))
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params_scaled(module, n_layers=n_layers))

    def forward(self, x, face_area, face_type, face_centroid, face_rational, padding_mask):
        n_graph, n_node = padding_mask.size()[:2]
        node_pos = torch.where(padding_mask == False)
        x = x.permute(0, 3, 1, 2)

        x_ = self.surf_encoder(x)
        
        # [修改] 数值保护：先 clamp 到最小正数，再做 log
        face_area_clamped = torch.clamp(face_area, min=1e-6)
        face_area_safe = torch.log(face_area_clamped.unsqueeze(dim=1)) 
        face_area_ = self.face_area_encoder(face_area_safe)

        face_type_ = self.face_type_encoder(face_type)
        face_centroid_ = self.face_centroid_encoder(face_centroid)
        face_rational_ = self.face_rational_encoder(face_rational)

        node_feature = torch.cat((x_, face_area_, face_type_, face_centroid_, face_rational_), dim=-1)
        face_feature = torch.zeros([n_graph, n_node, self.hidden_dim], device=x.device, dtype=x.dtype)
        face_feature[node_pos] = node_feature[:]

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, face_feature], dim=1)
        return graph_node_feature, node_feature

class GraphAttnBias(nn.Module):
    def __init__(self, dim_node, num_heads, num_distance, num_edge_dis, edge_type, multi_hop_max_dist, n_layers):
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.shortest_distance_encoder = nn.Embedding(num_distance, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.angle_encoder = NonLinear(1, num_heads)
        self.centroid_distance_encoder = NonLinear(1, num_heads)

        self.curv_encoder = CurveEncoder(in_channels=12, output_dims=num_heads)
        self.edge_type_encoder = NonLinear(11, num_heads)
        self.edge_length_encoder = NonLinear(1, num_heads)
        self.edge_convexity_encoder = NonLinear(3, num_heads)
        self.edge_type = edge_type
        
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
            self.node_cat = EdgeConv(edge_feats=num_heads, out_feats=num_heads, node_feats=dim_node)
        
        self.apply(lambda module: init_params_scaled(module, n_layers=n_layers))

    def forward(self, attn_bias, shortest_distance, angle, centroid_distance, edge_path, edge_data, edge_type, edge_length, edge_convexity, edge_padding_mask, graph, node_feat):
        n_graph, n_node = edge_path.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 1. Shortest Distance
        shortest_distance_bias = self.shortest_distance_encoder(shortest_distance)
        shortest_distance_bias = shortest_distance_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + shortest_distance_bias

        # 2. Virtual Token Distance
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 3. Angle
        angle = angle.reshape(-1, 1)
        angle_bias = self.angle_encoder(angle).reshape(n_graph, n_node, n_node, self.num_heads).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + angle_bias

        # 4. Centroid Distance
        centroid_distance = centroid_distance.reshape(-1, 1)
        centroid_distance = self.centroid_distance_encoder(centroid_distance).reshape(n_graph, n_node, n_node, self.num_heads).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + centroid_distance

        # 5. Edge Features (Multi-hop)
        if self.edge_type == "multi_hop":
            shortest_distance_ = shortest_distance.clone()
            shortest_distance_[shortest_distance_ == 0] = 1
            shortest_distance_ = torch.where(shortest_distance_ > 1, shortest_distance_ - 1, shortest_distance_)
            shortest_distance_ = shortest_distance_.clamp(0, self.multi_hop_max_dist)

            max_dist = self.multi_hop_max_dist
            edge_pos = torch.where(edge_padding_mask == False)
            
            edge_data = edge_data.permute(0, 2, 1)
            edge_data_ = self.curv_encoder(edge_data)
            
            edge_type_ = self.edge_type_encoder(edge_type)
            edge_length_ = self.edge_length_encoder(edge_length.unsqueeze(dim=1))
            edge_convexity_ = self.edge_convexity_encoder(edge_convexity)
            
            edge_feat = edge_data_ + edge_type_ + edge_length_ + edge_convexity_
            edge_feat_ = self.node_cat(graph, node_feat, edge_feat)

            n_edge = edge_padding_mask.size(1)
            edge_feature = torch.zeros([n_graph, (n_edge + 1), edge_feat_.size(-1)], device=edge_data.device, dtype=edge_data.dtype)
            edge_feature[edge_pos] = edge_feat_[:]

            edge_path = edge_path.reshape(n_graph, n_node * n_node * max_dist)
            dim_0 = torch.arange(n_graph, device=edge_path.device).reshape(n_graph, 1).long()
            edge_bias = edge_feature[dim_0, edge_path]
            edge_bias = edge_bias.reshape(n_graph, n_node, n_node, max_dist, self.num_heads)
            edge_bias = edge_bias.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)

            edge_bias = torch.bmm(
                edge_bias,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :]
            )
            edge_bias = edge_bias.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            
            # [Fix] 核心修复：防止除以0，以及防止 edge_bias 爆炸
            denom = shortest_distance_.float().unsqueeze(-1) + 1e-6
            edge_bias_sum = edge_bias.sum(-2)
            edge_bias = edge_bias_sum / denom
            
            # 最后的防线
            if torch.isnan(edge_bias).any():
                edge_bias = torch.nan_to_num(edge_bias, nan=0.0)

            edge_bias = edge_bias.permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_bias

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias