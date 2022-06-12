mkdir -p out
../../bin/dos_bchm.exe ../../data/twitter.txt 99841247 cpu 1 2>> out/results_cpu_twitter.txt
../../bin/dos_bchm.exe ../../data/twitter.txt 99841247 gpu 1 2>> out/results_gpu_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/twitter.txt 99841247 neo4j 1 2>> out/results_neo4j_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./dos.groovy /root/bitgraph/data/twitter.txt 99841247 tinkergraph 1 2>> out/results_tinkergraph_twitter.txt