import java.time.LocalDate

perror = System.err.&println
now = System.&currentTimeMillis

vertices_file = args[0]
edges_file = args[1]
tries = Integer.parseInt(args[2])
EDGE_LABEL = 'basic_edge'
LABEL_V = 'basic_vertex'
NAME = 'name'

graph = TinkerGraph.open()
graph.createIndex(NAME, Vertex.class)

g = graph.traversal()

s = new Scanner(new File(vertices_file));
names = new HashMap<String, Integer>(100000);
startTime = System.currentTimeMillis()

header = s.nextLine().split(',')
k = 0
while(s.hasNextLine()) {
    ++k;
    if(k % 1000 == 0) println(k)
    line = s.nextLine().split(',');
    vname = line[0];
    println(vname)

    if(!names.containsKey(vname)) v = g.addV(LABEL_V).property(NAME, vname).next();
    names.put(vname, v.id());
    // don't care about the other props for now
}

s = new Scanner(new File(edges_file));
s.useDelimiter("\n")

k=0
//limit = 1e6
limit = 1e10
start_day = LocalDate.parse("2019-01-01").toEpochDay();
header = s.nextLine()
while(s.hasNextLine() && (k < limit)) {
	++k;
	if(k % 1000 == 0) println(k);

    line = s.nextLine().split(',');
    day = LocalDate.parse(line[0]).toEpochDay() - start_day;

    src = line[1];
    dst = line[2];
    // don't care about the other props for now 

	v_src = g.V(names[src]).next();
	v_dst = g.V(names[dst]).next();

	new_e = g.V(v_src).addE(EDGE_LABEL).to(v_dst).property("time", day).next();
}

endTime = System.currentTimeMillis()
timeDiff = endTime - startTime

System.err.println('ingest time: ' + (timeDiff / 1000.0).toString() + 'seconds.')

//g.io('/mnt/bitgraph/data/temporal/flight/tgbl-flight.kryo').write().iterate()

days = 4;
//time_start = 400;
time_start = 0;
time_end = time_start + days + 1;
println("Time window: [" + time_start.toString() + ", " + time_end.toString() + ")")

//start_v = "KHOU" // out-degree 3713 in flights-1M
//start_v = "KLAX" // out-degree 16679 in flights-1M
start_v = "YMML" // out-degree 6745 in flights-1M
start_v = g.V().has(NAME, start_v).next();

r = 0 
while(r < tries) {
    println('Calculating shortest path for ' + start_v)

    start = now()

    v_current = [start_v]
    v_total = 0
    g.V(v_current).property('last_time', time_start - 1).iterate()

    while(v_current.size() > 0) {
        v_current = g.V(v_current).
                        property('visited', 1).
                        outE().has('time', lte(time_end)).as('e').
                        project('time','last_time').
                            by(values('time')).
                            by(outV().values('last_time')).
                        select('time','last_time').
                            where('time', gt('last_time')).
                        select('e').
                        group().
                            by(inV()).
                        select(values).unfold().
                        map(
                            unfold().order().by('time').limit(1)
                        ).
                        as('ee').
                        values('time').as('last_time').
                        select('ee').
                        inV().
                        hasNot('visited').
                        property('last_time', select('last_time')).
                        toList()
        v_total += v_current.size()
    }
    
    println(v_total)

    end = now()
    println('done')
    elapsed = end - start
    perror('sp time: ' + (elapsed/1000.0).toString())

    r++
    g.V().properties('visited','last_time').drop().iterate()
}

