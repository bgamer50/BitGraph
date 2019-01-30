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

#test: test.exe
	#./test.exe > test/test.stdout 2> test/test.stderr
	#diff -b test/test.stdout test/expected.stdout
	#diff -b test/test.stderr test/expected.stderr

test.exe: test.cpp
	$(CC) $(CFLAGS) -o test.exe -I../gremlin++ test.cpp

ingest.exe: ingest_simple.cpp
	$(CC) $(CFLAGS) -o ingest.exe -I../gremlin++ ingest_simple.cpp

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
