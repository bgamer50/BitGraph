LABEL_V = 'basic_vertex'
LABEL_E = 'basic_edge'
NAME = 'name'

graph = JanusGraphFactory.build().set('storage.backend', 'inmemory').open()

graph.tx().rollback()
mgmt = graph.openManagement()
name_key = mgmt.makePropertyKey(NAME).dataType(String.class).make()
mgmt.commit()

graph.tx().rollback()
mgmt = graph.openManagement()
name_key = mgmt.getPropertyKey(NAME)
idx = mgmt.buildIndex(NAME + '_index', Vertex.class).addKey(name_key).buildCompositeIndex()
mgmt.commit()
ManagementSystem.awaitGraphIndexStatus(graph, NAME + '_index').call()
graph.tx().rollback()

mgmt = graph.openManagement()
mgmt.updateIndex(mgmt.getGraphIndex(NAME + '_index'), SchemaAction.REINDEX).get()
mgmt.updateIndex(mgmt.getGraphIndex(NAME + '_index'), SchemaAction.REINDEX).get()
mgmt.commit()

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

	g.V(v1).addE(LABEL_E).to(__.V(v2)).iterate();
}

endTime = System.currentTimeMillis()
timeDiff = endTime - startTime

println('ingest time: ' + (timeDiff / 1000).toString() + 'seconds.')

startTime = System.currentTimeMillis()
g.V().property('cc', coalesce(both(), identity()).values('cc').min()).iterate()
endTime = System.currentTimeMillis()
timeDiff = endTime - startTime
println('cc1x time: ' + (timeDiff / 1000).toString() + 'seconds.')