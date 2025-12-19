import dgl
import torch

# 指定.bin文件的路径
bin_file_path = r"/mnt/sdb/daiyongkang/Work/dyk/CADDataSet/SolidLetters/bin_topology/a_ABeeZee_lower.bin"

# print("bin_file_path:",type(bin_file_path))d
# 加载图
graphfile = dgl.data.utils.load_graphs(bin_file_path)
print(type(graphfile) )
print("graphfile:",graphfile)

graphs, geo = dgl.data.utils.load_graphs(bin_file_path)

# print("graphs,geo",graphs,geo)
# 获取第一个图（如果有多个图存储在文件中）
graph = graphs[0]
print("graph:", graph)

print("type(geo):",type(geo))
keys = dict.keys(geo)
print("keys:", list(keys))


# angle = geo['angle_matrix']
# centroid_distance = geo['centroid_distance_matrix']
# shortest_distance = geo['shortest_distance_matrix']
# edge_path = geo['edge_path_matrix']

# adj_matrix = geo['adj_matrix']

print("show geo feature: ")



# print("angle.shape:",angle.shape)
# print("angle:",angle)
#
# print("centroid_distance.shape:" ,centroid_distance.shape)
# print("centroid_distance:",centroid_distance)
#
# edges_pathTemp = edge_path[3]
# print("edges_path.shape:",edge_path.shape)
# print("edges_path:",edge_path[0])
#
# print("shortest_distance.shape:" ,shortest_distance.shape)
# print("shortest_distance:",shortest_distance)


# print("adj_matrix.shape:",adj_matrix.shape)
# print("adj_matrix:",adj_matrix)


print("show node feature: ")
#--------------------------------

# node_feature_t = graph.ndata['t']
# print("node_feature_t.shape:", node_feature_t.shape)
# print("node_feature_t:", node_feature_t)
#
# node_feature_a = graph.ndata['a']
# print("node_feature_a.shape:", node_feature_a.shape)
# print("node_feature_a:", node_feature_a)
#
# node_feature_r = graph.ndata['r']
# print("node_feature_r.shape:", node_feature_r.shape)
# print("node_feature_r:", node_feature_r)
#
# node_feature_c = graph.ndata['c']
# print("node_feature_c.shape:", node_feature_c.shape)
# print("node_feature_c:", node_feature_c)
#
#
node_feature_l = graph.ndata['l']
print("node_feature_l.shape:", node_feature_l.shape)
print("node_feature_l:", node_feature_l)
#
#
#

node_feature_x = graph.ndata['x']
# print("graph.ndata['x'].type",type(graph.ndata['x']))
print("node_feature_x.shape:", node_feature_x.shape)

# print("node_feature_x:\n", node_feature_x[0][0])



print("show edge feature: ")
edge_feature_x = graph.edata['x']
print("edge_feature_x.shape:", edge_feature_x.shape)

# edge_feature_c = graph.edata['c']
# print("edge_feature_c.shape:", edge_feature_c.shape)
# print("edge_feature_c:", edge_feature_c)
#
# edge_feature_l = graph.edata['l']
# print("edge_feature_l.shape:", edge_feature_l.shape)
# print("edge_feature_l:", edge_feature_l)
#
# edge_feature_t = graph.edata['t']
# print("edge_feature_t.shape:", edge_feature_t.shape)
# print("edge_feature_t:", edge_feature_t)




