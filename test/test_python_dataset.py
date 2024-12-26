import numpy as np

from ogb.nodeproppred import NodePropPredDataset
from pybitgraph import BitGraph

from pygremlinxx import GraphTraversal
__ = lambda : GraphTraversal()

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

g = graph.traversal()
a = g.V().sample(70).values("emb3").toArray()
print(a)
print(a.shape)

# random walk
batch_id = 0
g.V().sample(62).repeat(__().outE().sample(10).property("batch", batch_id).inV()).times(1).iterate()
b = g.E().has("batch", batch_id).toArray()

print(b)