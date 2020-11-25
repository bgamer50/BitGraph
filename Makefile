CC := g++
CFLAGS := -Ofast -fopenmp --std=c++17 -floop-interchange -floop-strip-mine -funsafe-math-optimizations -frename-registers
IFLAGS := -I. -I../gremlin++/

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

lca.exe: lca.o
	$(CC) $(CFLAGS) lca.o -o lca.exe $(IFLAGS)

lca.o: lca.cpp
	$(CC) $(CFLAGS) lca.cpp -c -o lca.o $(IFLAGS)

valuemap.exe: valuemap.o
	$(CC) $(CFLAGS) valuemap.o -o valuemap.exe $(IFLAGS)

valuemap.o: valuemap.cpp
	$(CC) $(CFLAGS) valuemap.cpp -c -o valuemap.o $(IFLAGS)

components.exe: components.o
	$(CC) $(CFLAGS) components.o -o components.exe $(IFLAGS)

components.o: components.cpp
	$(CC) $(CFLAGS) components.cpp -c -o components.o $(IFLAGS)

repeat.exe: repeat.o
	$(CC) $(CFLAGS) repeat.o -o repeat.exe $(IFLAGS)

repeat.o: repeat.cpp
	$(CC) $(CFLAGS) repeat.cpp -c -o repeat.o $(IFLAGS)

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
