CC := g++
CFLAGS := --std=c++14 -fPIC

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


lib: CPUGraph.o CPUGraphTraversalSource.o BitVertex.o BitEdge.o
	$(CC) $(CFLAGS) -o $(LIBBITGRAPH_NAME) -lgremlin -shared CPUGraph.o CPUGraphTraversalSource.o BitVertex.o BitEdge.o
	mv $(LIBBITGRAPH_NAME) $(LIBBITGRAPH_PATH)
	chmod 755 $(LIBBITGRAPH_PATH)

CPUGraph.o: CPUGraph.cpp
	$(CC) $(CFLAGS) -o CPUGraph.o -c -I../gremlin++ CPUGraph.cpp

CPUGraphTraversalSource.o: CPUGraphTraversalSource.cpp
	$(CC) $(CFLAGS) -o CPUGraphTraversalSource.o -c -I../gremlin++ CPUGraphTraversalSource.cpp

BitVertex.o: BitVertex.cpp
	$(CC) $(CFLAGS) -o BitVertex.o -c -I../gremlin++ BitVertex.cpp

BitEdge.o: BitEdge.cpp
	$(CC) $(CFLAGS) -o BitEdge.o -c -I../gremlin++ BitEdge.cpp

test.exe: test.cpp
	$(CC) $(CFLAGS) -o test.exe -I../gremlin++ test.cpp -lgremlin -lbitgraph

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
