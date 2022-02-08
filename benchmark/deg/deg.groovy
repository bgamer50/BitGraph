perror = System.err.&println
now = System.&currentTimeMillis

edges_file = args[0]
graph_type = args[1]
tries = Integer.parseInt(args[2])
EDGE_LABEL = 'basic_edge'
LABEL_V = 'basic_vertex'
NAME = 'name'

if(graph_type == 'tinkergraph') {
    graph = TinkerGraph.open()
    graph.createIndex(NAME, Vertex.class)
} else if(graph_type == 'neo4j') {
    graph = Neo4jGraph.open('data/neo4j_ce')
    graph.cypher(String.format('CREATE INDEX ON :%s(%s)', LABEL_V, NAME))
    graph.tx().commit()
} else {
  throw new IllegalArgumentException('invalid graph system')
}

g = graph.traversal()

s = new Scanner(new File(edges_file));
names = new HashSet<String>();

startTime = System.currentTimeMillis()
k=0
while(s.hasNextInt()) {
	++k;
	if(k % 1000 == 0) println(k);
	i = s.nextInt().toString();
	j = s.nextInt().toString();
	if(!names.contains(i)) v1 = g.addV(LABEL_V).property(NAME, i).next();
	else v1 = g.V().has(LABEL_V, NAME, i).next();

	if(!names.contains(j)) v2 = g.addV(LABEL_V).property(NAME, j).next();
	else v2 = g.V().has(LABEL_V, NAME, j).next();

	names.add(i);
	names.add(j);

	g.V(v1).addE(EDGE_LABEL).to(v2).iterate();
}

endTime = System.currentTimeMillis()
timeDiff = endTime - startTime

System.err.println('ingest time: ' + (timeDiff / 1000.0).toString() + 'seconds.')

r = 0
while(r < tries) {
    println('Calculating out-degree for all vertices')
    start = now()
    g.V().property("out_degree", __.out().count()).iterate()
    end = now()
    println(g.V().limit(10).values("out_degree").next())
    elapsed = end - start
    perror('dos time: ' + (elapsed/1000.0).toString())

    println('Calculating in-degree for all vertices')
    start = now()
    g.V().property("in_degree", __.in().count()).iterate()
    end = now()
    println(g.V().limit(10).values("in_degree").next())
    elapsed = end - start
    perror('dos time: ' + (elapsed/1000.0).toString())

    println('Calculating both-degree for all vertices')
    start = now()
    g.V().property("both_degree", __.both().count()).iterate()
    end = now()
    println(g.V().limit(10).values("both_degree").next())
    elapsed = end - start
    perror('dos time: ' + (elapsed/1000.0).toString())

    r++
}

:q