import pandas
import numpy as np

from pybitgraph import BitGraph

adf = pandas.read_parquet('/mnt/bitgraph/data/rag/articles.parquet')
edf = pandas.read_parquet('/mnt/bitgraph/data/rag/edgelist.parquet')

graph = BitGraph(
    "int64",
    "int64",
    "DEVICE",
    "MANAGED",
    "MANAGED",
)

graph.add_vertices(len(adf))
graph.add_edges(
    edf.src.values,
    edf.dst.values,
    'link'
)

g = graph.traversal()

g.V().has('')