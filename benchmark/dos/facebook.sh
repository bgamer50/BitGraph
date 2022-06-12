mkdir -p out
../../bin/dos_bchm.exe ../../data/facebook.txt 100 cpu 1 2>> out/results_cpu_facebook.txt
../../bin/dos_bchm.exe ../../data/facebook.txt 100 gpu 1 2>> out/results_gpu_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/facebook.txt 100 neo4j 1 2>> out/results_neo4j_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/facebook.txt 100 tinkergraph 1 2>> out/results_tinkergraph_facebook.txt