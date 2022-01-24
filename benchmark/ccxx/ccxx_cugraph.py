import cudf,cugraph
from datetime import datetime
from sys import argv 

fname = argv[0]
tries = int(argv[1])

edge_df = cudf.read_csv(fname, names=['out','in'], sep=' ')
graph = cugraph.from_edgelist(edge_df,'out','in')

for r in range(tries):
    start_time = datetime.now()
    ccxx_results = cugraph.connected_components(graph)
    end_time = datetime.now()
    secs = (end_time - start_time).total_seconds()

    print(f'total ccxx time: {secs}')