perror = System.err.&println
now = System.&currentTimeMillis

nodes_file = args[0]
edges_file = args[1]
voi1 = args[2]
voi2 = args[3]
graph_type = args[4]
tries = Integer.parseInt(args[5])
EDGE_LABEL = 'tree_edge'
LABEL_V = 'basic_vertex'
NAME = 'NAME'

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

g = graph.traversal().withComputer()

scn_nodes = new Scanner(new File(nodes_file));
scn_edges = new Scanner(new File(edges_file));

m = new HashMap<>()
k = 0
while(scn_nodes.hasNextLine()) {
  if(k % 1000 == 0) println(k)
  ln = scn_nodes.nextLine().split(',');
  v = g.addV(LABEL_V).property('NAME',ln[0]).property('INFO',ln[1]).property('LEVEL',ln[2]).next()
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

r = 0
while(r < tries) {
    // Find the lca
    start = now()
    lca = g.V().has(NAME, voi1).
        repeat(__.out()).emit().as("x").
        repeat(__.in()).emit(__.has(NAME, voi2)).
        select("x").limit(1).values(NAME).next()
    end = now()
    println(r)
    elapsed = end - start
    perror('lca time: ' + (elapsed/1000.0).toString())
    
    r++
}
:q