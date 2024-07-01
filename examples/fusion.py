from time import perf_counter

import pandas
import numpy as np

from pybitgraph import BitGraph
from pygremlinxx import (
    TraversalStrategy,
    GraphTraversal,
    P
)

__ = lambda : GraphTraversal()

"""
Hall, B. H., A. B. Jaffe, and M. Trajtenberg (2001). "The NBER Patent Citation Data File: Lessons, Insights and Methodological Tools." NBER Working Paper 8498.
"""
pdf = pandas.read_csv('/mnt/bitgraph/data/patents/apat63_99.txt')
pdf.drop(['COUNTRY', 'POSTATE'],axis=1,inplace=True)

gdf = pandas.read_csv('/mnt/bitgraph/data/patents/cite75_99.txt')

num_nodes = max(
    gdf.CITING.max(),
    gdf.CITED.max(), 
    pdf.PATENT.max()
) + 1

graph = BitGraph(
    "int64",
    "int64",
    "DEVICE",
    "MANAGED",
    "MANAGED",
)

graph.add_vertices(num_nodes)
graph.add_edges(
    gdf.CITING.to_numpy(),
    gdf.CITED.to_numpy(),
    "citation",
)

g = graph.traversal()
print(g.V(5991572).out().count().toArray())

for col in pdf.columns:    
    fc = pdf[~pdf[col].isna()][['PATENT', col]]
    graph.declare_vertex_property(
        col,
        "MANAGED",
        str(pdf[col].dtype),
        len(fc),
    )
    print(col)
    graph.set_vertex_properties(
        col,
        fc.PATENT.to_numpy(),
        fc[col].to_numpy(),
    )

g = graph.traversal().withoutStrategy(TraversalStrategy.HasJoinStrategy).withoutStrategy(BitGraph.BitGraphSelectionStrategy)
print(g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).explain())
start_time = perf_counter()
result = g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).toArray()
end_time = perf_counter()
print("result:", result)
print(f"Time without optimization: {end_time - start_time} seconds")

g = graph.traversal().withoutStrategy(BitGraph.BitGraphSelectionStrategy)
print(g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).explain())
start_time = perf_counter()
result = g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).toArray()
end_time = perf_counter()
print("result:", result)
print(f"Time with HasJoin optimization: {end_time - start_time} seconds")

g = graph.traversal()
print(g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).explain())
start_time = perf_counter()
result = g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).toArray()
end_time = perf_counter()
print("result:", result)
print(f"Time with HasJoin and BitGraphSelection optimization: {end_time - start_time} seconds")


g = graph.traversal().withoutStrategy(TraversalStrategy.BasicPatternExtractionStrategy)
print(g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).order().by(__()._in().count()).explain())
start_time = perf_counter()
result = g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).order().by(__()._in().count()).toArray()
end_time = perf_counter()
print("result:", result)
print(f"Time without optimization: {end_time - start_time} seconds")

g = graph.traversal()
print(g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).order().by(__()._in().count()).explain())
start_time = perf_counter()
result = g.V().has('APPYEAR', P.gte(1970)).has("SECDLWBD", P.gte(0.6)).order().by(__()._in().count()).toArray()
end_time = perf_counter()
print("result:", result)
print(f"Time with BasicPatternExtraction optimization: {end_time - start_time} seconds")
