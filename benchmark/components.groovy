perror = System.err.&println
now = System.&currentTimeMillis

edges_file = args[0]
tries = Integer.parseInt(args[1])
EDGE_LABEL = 'basic_edge'
LABEL_V = 'basic_vertex'
NAME = 'name'

graph = TinkerGraph.open()
graph.createIndex(NAME, Vertex.class)

g = graph.traversal()

s = new Scanner(new File(edges_file));
names = new HashSet<String>(100000);

startTime = System.currentTimeMillis()
k=0
while(s.hasNextBigInteger()) {
	++k;
	if(k % 1000 == 0) println(k);
	i = s.nextBigInteger().toString();
	j = s.nextBigInteger().toString();
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
    println('Calculating connected components for all vertices')
    start = now()
    g.V().property("cc", __.id()).iterate()
    g.V().property("old_cc", __.values("cc")).iterate()

    diff = 1
    while(diff > 0) {
        diff = g.V().
            property("old_cc", __.values("cc")).
            barrier().
            property("cc", 
                __.union(__.both().values("old_cc"), __.values("old_cc")).min()
            ).
            valueMap("cc","old_cc").by(__.unfold()).
            where("cc", neq("old_cc")).
            count().
            next()
        println("diff: " + diff.toString());
    }
    end = now()
    println('done')
    elapsed = end - start
    perror('ccxx time: ' + (elapsed/1000.0).toString())

    r++
}

:q