mkdir -p out
../../bin/lca_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt 1000 1500 cpu 1 2>> out/results_cpu_ast.txt
../../bin/lca_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt 1000 1500 gpu 1 2>> out/results_gpu_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./lca.groovy /root/bitgraph/data/cfg_lca_nodes.txt /root/bitgraph/data/cfg_lca_edges.txt 1000 1500 neo4j 1 2>> out/results_neo4j_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./lca.groovy /root/bitgraph/data/cfg_lca_nodes.txt /root/bitgraph/data/cfg_lca_edges.txt 1000 1500 tinkergraph 1 2>> out/results_tinkergraph_ast.txt
