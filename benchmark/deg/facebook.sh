mkdir -p out
../../bin/deg_bchm.exe ../../data/facebook.txt cpu 1 2>> out/results_cpu_facebook.txt
../../bin/deg_bchm.exe ../../data/facebook.txt gpu 1 2>> out/results_gpu_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./deg.groovy /root/bitgraph/data/facebook.txt neo4j 1 2>> out/results_neo4j_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./deg.groovy /root/bitgraph/data/facebook.txt tinkergraph 1 2>> out/results_tinkergraph_facebook.txt