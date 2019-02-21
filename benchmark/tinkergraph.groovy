LABEL_V = 'basic_vertex'
LABEL_E = 'basic_edge'
NAME = 'name'

graph = TinkerGraph.open()
g = graph.traversal()

s = new Scanner(new File('C:/Users/Alex/Downloads/facebook_combined.txt'))
names = new HashSet<String>();

startTime = System.currentTimeMillis()
k=0
while(s.hasNextInt()) {
	++k;
	if(k % 1000 == 0) println(k);
	i = s.nextInt().toString();
	j = s.nextInt().toString();
	if(!names.contains(i)) v1 = g.addV(LABEL_V).property(NAME, i).next();
	else v1 = g.V().has(NAME, i).next();

	if(!names.contains(j)) v2 = g.addV(LABEL_V).property(NAME, j).next();
	else v2 = g.V().has(NAME, j).next();

	names.add(i);
	names.add(j);

	g.V(v1).addE(LABEL_E).to(v2).iterate();
}

endTime = System.currentTimeMillis()
timeDiff = endTime - startTime

println('ingest time: ' + str(timeDiff / 1000) + 'seconds.')
