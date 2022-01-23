mkdir -p out
../../bin/components_bchm.exe ../../data/facebook.txt cpu 2>> out/results_cpu_facebook.txt
../../bin/components_bchm.exe ../../data/facebook.txt gpu 2>> out/results_gpu_facebook.txt
python3 ccxx_cugraph.py ../../data/facebook.txt >> out/results_cugraph_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./ccxx.groovy /root/bitgraph/data/facebook.txt neo4j 2>> out/results_neo4j_facebook.txt
/opt/gremlin/bin/gremlin.sh -i ./ccxx.groovy /root/bitgraph/data/facebook.txt tinkergraph 2>> out/results_tinkergraph_facebook.txt