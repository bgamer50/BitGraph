from pybitgraph import (
    BitGraph
)
from pygremlinxx import TraversalStrategy, GraphTraversal

import numpy as np

__ = lambda : GraphTraversal()

src = np.array([0,1,2,3,4,5])
dst = np.array([3,2,5,1,0,4])

graph = BitGraph(
    "int64",
    "int64",
    "DEVICE",
    "MANAGED",
    "MANAGED",
)

graph.add_vertices(6)
graph.add_edges(
    src,
    dst,
    "basic_edge",
)

print(f'num vertices: {graph.num_vertices()}')
print(f'num edges: {graph.num_edges()}')

g = graph.traversal()
print(g.addV("l").property("name", "joe").property("age", 3, "uint64").values("age").next())
print(g.V().has("name", "joe").id().next())
print(g.V().id().toArray())

print(f'num vertices: {graph.num_vertices()}')

print(g.V().has("name", "joe").repeat(__().out()).times(3).toArray())

print(GraphTraversal().V().explain())

print(g.E().sample(4).subgraph('sg').explain())
trv = g.E().sample(4).subgraph('sg')
trv.iterate()
sg = trv.getTraversalProperty('sg')

h = sg.traversal()
print(h)

print(g.V().sample(4).has('name','joe').has('age', 23).explain())