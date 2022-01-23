mkdir -p out
../../bin/ast_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt cpu 2>> out/results_cpu_ast.txt
../../bin/ast_bchm.exe ../../data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt gpu 2>> out/results_gpu_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./ast.groovy /root/bitgraph/data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt neo4j 2>> out/results_neo4j_ast.txt
/opt/gremlin/bin/gremlin.sh -i ./ast.groovy /root/bitgraph/data/cfg_lca_nodes.txt ../../data/cfg_lca_edges.txt tinkergraph 2>> out/results_tinkergraph_ast.txt