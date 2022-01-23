mkdir -p out
../../bin/lca_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt cpu 2>> out/results_cpu_ast.txt
../../bin/lca_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt gpu 2>> out/results_gpu_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./lca.groovy /root/bitgraph/data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt neo4j >> out/results_neo4j_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./lca.groovy /root/bitgraph/data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt tinkergraph >> out/results_tinkergraph_ast.txt