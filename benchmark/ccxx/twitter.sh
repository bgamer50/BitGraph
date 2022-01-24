mkdir -p out
../../bin/components_bchm.exe ../../data/twitter.txt cpu 2>> out/results_cpu_twitter.txt
../../bin/components_bchm.exe ../../data/twitter.txt gpu 2>> out/results_gpu_twitter.txt
python3 ccxx_cugraph.py ../../data/twitter.txt >> out/results_cugraph_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./ccxx.groovy /root/bitgraph/data/twitter.txt neo4j 2>> out/results_neo4j_twitter.txt
/opt/gremlin/bin/gremlin.sh -i ./ccxx.groovy /root/bitgraph/data/twitter.txt tinkergraph 2>> out/results_tinkergraph_twitter.txt