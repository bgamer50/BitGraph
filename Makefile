CC := g++
CFLAGS := --std=c++14 -fPIC -g

ifeq ($(shell uname -s), Darwin)
	LIBBITGRAPH_PATH := /usr/local/lib/libbitgraph.dylib
	LIBBITGRAPH_NAME := libbitgraph.dylib
endif
ifeq ($(shell uname -s), CYGWIN_NT-10.0)
	LIBBITGRAPH_PATH := /usr/lib/libbitgraph.dll.a
	LIBBITGRAPH_NAME := libbitgraph.dll.a
endif
ifeq ($(shell uname -s), Linux)
	LIBBITGRAPH_PATH := /usr/lib/libbitgraph.so
	LIBBITGRAPH_NAME := libbitgraph.so
endif


lib: CPUGraphTraversal.o CPUGraph.o CPUGraphTraversalSource.o BitVertex.o BitEdge.o
	$(CC) $(CFLAGS) -o $(LIBBITGRAPH_NAME) -lgremlin -shared CPUGraph.o CPUGraphTraversalSource.o BitVertex.o BitEdge.o
	mv $(LIBBITGRAPH_NAME) $(LIBBITGRAPH_PATH)
	chmod 755 $(LIBBITGRAPH_PATH)

CPUGraphTraversal.o: CPUGraphTraversal.cpp CPUGraphTraversal.h
	$(CC) $(CFLAGS) -o CPUGraphTraversal.o -c -I../gremlin++ CPUGraphTraversal.cpp

CPUGraph.o: CPUGraph.cpp CPUGraph.h
	$(CC) $(CFLAGS) -o CPUGraph.o -c -I../gremlin++ CPUGraph.cpp

CPUGraphTraversalSource.o: CPUGraphTraversalSource.cpp CPUGraphTraversalSource.h
	$(CC) $(CFLAGS) -o CPUGraphTraversalSource.o -c -I../gremlin++ CPUGraphTraversalSource.cpp

BitVertex.o: BitVertex.cpp BitVertex.h
	$(CC) $(CFLAGS) -o BitVertex.o -c -I../gremlin++ BitVertex.cpp

BitEdge.o: BitEdge.cpp BitEdge.h
	$(CC) $(CFLAGS) -o BitEdge.o -c -I../gremlin++ BitEdge.cpp

test: test.exe
	./test.exe > test/test.stdout 2> test/test.stderr
	diff -b test/test.stdout test/expected.stdout
	diff -b test/test.stderr test/expected.stderr

test.exe: test.cpp
	$(CC) $(CFLAGS) -o test.exe -I../gremlin++ test.cpp -lgremlin -lbitgraph

ingest.exe: ingest_simple.cpp
	$(CC) $(CFLAGS) -o ingest.exe -I../gremlin++ ingest_simple.cpp -lgremlin -lbitgraph

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
