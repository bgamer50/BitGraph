mkdir -p out
../../bin/dos_bchm.exe ../../data/twitter.txt cpu 2>> out/results_cpu_twitter.txt
../../bin/dos_bchm.exe ../../data/twitter.txt gpu 2>> out/results_gpu_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/twitter.txt neo4j 2>> out/results_neo4j_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/twitter.txt tinkergraph 2>> out/results_tinkergraph_twitter.txt