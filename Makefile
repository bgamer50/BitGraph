libbitgraph.dylib: CPUGraph.o CPUGraphTraversalSource.o BitVertex.o
	g++ -o libbitgraph.dylib --std=c++11 -lgremlin -shared CPUGraph.o CPUGraphTraversalSource.o BitVertex.o
	mv libbitgraph.dylib /usr/local/lib/libbitgraph.dylib
	chmod 755 /usr/local/lib/libbitgraph.dylib

CPUGraph.o: CPUGraph.cpp
	g++ -o CPUGraph.o -c -I../gremlin++ --std=c++11 CPUGraph.cpp

CPUGraphTraversalSource.o: CPUGraphTraversalSource.cpp
	g++ -o CPUGraphTraversalSource.o -c -I../gremlin++ --std=c++11 CPUGraphTraversalSource.cpp

BitVertex.o: BitVertex.cpp
	g++ -o BitVertex.o -c -I../gremlin++ --std=c++11 BitVertex.cpp

test.exe: test.cpp
	g++ -o test.exe -I../gremlin++ --std=c++11 test.cpp -lgremlin -lbitgraph

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe