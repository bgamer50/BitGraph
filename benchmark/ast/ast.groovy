perror = System.err.&println
now = System.&currentTimeMillis

nodes_file = args[0]
edges_file = args[1]
graph_type = args[2]
EDGE_LABEL = 'tree_edge'

if(graph_type == 'tinkergraph') {
  graph = TinkerGraph.open()
} else if(graph_type == 'neo4j') {
  graph = Neo4jGraph.open('data/neo4j_ce')
} else {
  throw new IllegalArgumentException('invalid graph system')
}

g = graph.traversal()

scn_nodes = new Scanner(new File(nodes_file));
scn_edges = new Scanner(new File(edges_file));

m = new HashMap<>()
k = 0
while(scn_nodes.hasNextLine()) {
  if(k % 1000 == 0) println(k)
  ln = scn_nodes.nextLine().split(',');
  v = g.addV().property('NAME',ln[0]).property('INFO',ln[1]).property('LEVEL',ln[2]).next()
  m[ln[0]] = v
  k++
}
scn_nodes.close()

k = 0
while(scn_edges.hasNextLine()) {
  if(k % 1000 == 0) println(k)
  ln = scn_edges.nextLine().split(' ');
  g.addE(EDGE_LABEL).from(m[ln[0]]).to(m[ln[1]]).iterate();
  k++
}
scn_edges.close()

// Traversal 1: Find nodes whose grandparent is a class template specialization and get their types.
start = now()
r = g.V().as('s').out().dedup().out().dedup().has('INFO', "ClassTemplateSpecializationDecl").select("s").values('INFO').toList()
end = now()
println(r)
elapsed = end - start
perror('Traversal 1 time: ' + (elapsed/1000.0).toString())

// Traversal 2: Find the unique types of nodes between a Class template specialization and a namespace declaration
start = now()
r = g.V().has('INFO','ClassTemplateSpecialization').out().as('t').dedup().out().dedup().has('INFO','NamespaceDecl').select('t').values('INFO').dedup().toList()
end = now()
println(r)
elapsed = end - start
perror('Traversal 2 time: ' + (elapsed/1000.0).toString())

// Traversal 3: For each while loop, count the number of if statements under the while loop
start = now()
r = g.V().has('INFO','WhileStmt').as('w').repeat(__.in()).emit(has('INFO','IfStmt')).select('w').values('NAME').groupCount().next()
end = now()
println(r)
elapsed = end - start
perror('Traversal 3 time: ' + (elapsed/1000.0).toString())