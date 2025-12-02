import torch
import dgl

def pad_mask_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_sq_matrix(x, padlen, val=0):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(val)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)

def collator(items, multi_hop_max_dist, spatial_pos_max):
    items = [
        (
            item.graph,
            item.node_data, item.face_area, item.face_type, item.face_centroid, item.face_rational,
            item.edge_data, item.edge_type, item.edge_length, item.edge_convexity,
            item.node_degree, item.attn_bias,
            item.angle, item.centroid_distance, item.shortest_distance,
            item.edge_path[:, :, :multi_hop_max_dist],
            item.label_feature, item.data_id
        ) for item in items
    ]

    (graphs, node_datas, face_areas, face_types, face_centroids, face_rationals,
     edge_datas, edge_types, edge_lengths, edge_convexitys,
     node_degrees, attn_biases, angles, centroid_distances, shortest_distances,
     edge_paths, label_features, data_ids) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][shortest_distances[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(max(i.size(-1) for i in edge_paths), multi_hop_max_dist)

    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in [torch.zeros(d.size(0), dtype=torch.bool) for d in node_datas]])
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in [torch.zeros(d.size(0), dtype=torch.bool) for d in edge_datas]])

    node_data = torch.cat(node_datas)
    face_area = torch.cat(face_areas)
    face_type = torch.cat(face_types)
    face_centroid = torch.cat(face_centroids)
    face_rational = torch.cat(face_rationals)

    edge_data = torch.cat(edge_datas)
    edge_type = torch.cat(edge_types)
    edge_length = torch.cat(edge_lengths)
    edge_convexity = torch.cat(edge_convexitys)

    edge_path = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]).long()
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    
    # Pad matrices
    shortest_distance = torch.cat([pad_sq_matrix(i + 1, max_node_num) for i in shortest_distances]) # +1 logic preserved
    angle = torch.cat([pad_sq_matrix(i + 1.0, max_node_num) for i in angles])
    centroid_distance = torch.cat([pad_sq_matrix(i, max_node_num) for i in centroid_distances])

    in_degree = torch.cat(node_degrees)
    batched_graph = dgl.batch(graphs)
    batched_label_feature = torch.cat(label_features)
    data_ids = torch.tensor(data_ids)

    return dict(
        padding_mask=padding_mask, edge_padding_mask=edge_padding_mask, graph=batched_graph,
        node_data=node_data, face_area=face_area, face_type=face_type, face_centroid=face_centroid, face_rational=face_rational,
        edge_data=edge_data, edge_type=edge_type, edge_length=edge_length, edge_convexity=edge_convexity,
        in_degree=in_degree, out_degree=in_degree, attn_bias=attn_bias,
        shortest_distance=shortest_distance, angle=angle, centroid_distance=centroid_distance, edge_path=edge_path,
        label_feature=batched_label_feature, id=data_ids
    )