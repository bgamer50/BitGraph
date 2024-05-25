import numpy as np

from ogb.nodeproppred import NodePropPredDataset
from pybitgraph import BitGraph

dataset = NodePropPredDataset('ogbn-products', root='/mnt/bitgraph/data/ogb')
data = dataset[0]
dst, src = data[0]['edge_index']
node_feat = data[0]['node_feat'].T
num_nodes = data[0]['num_nodes']

graph = BitGraph(
    "int64",
    "int64",
    "DEVICE",
    "MANAGED",
    "MANAGED",
)

graph.add_vertices(num_nodes)
graph.add_edges(
    src,
    dst,
    "basic_edge",
)

for k in range(node_feat.shape[0]):
    graph.set_vertex_properties(
        f"emb{k}",
        np.arange(num_nodes, dtype='int64'),
        node_feat[k]
    )