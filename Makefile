CC := g++
CFLAGS := -O3 -fopenmp --std=c++14 -fno-default-inline 
IFLAGS := -I. -I../gremlin++

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

test.exe: test.o
	$(CC) $(CFLAGS) test.o -o test.exe $(IFLAGS) -L'C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64' -lOpenCL

test.o: test.cpp
	$(CC) $(CFLAGS) test.cpp -c -o test.o $(IFLAGS) -I'C:\Program Files (x86)\AMD APP SDK\3.0\include' -include alloca.h

ingest.exe: ingest.o
	$(CC) $(CFLAGS) ingest.o -o ingest.exe $(IFLAGS) -L'C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64' -lOpenCL

ingest.o: ingest_simple.cpp
	$(CC) $(CFLAGS) ingest_simple.cpp -c -o ingest.o $(IFLAGS) -I'C:\Program Files (x86)\AMD APP SDK\3.0\include' -include alloca.h

components.exe: components.o
	$(CC) $(CFLAGS) components.o -o components.exe $(IFLAGS) -L'C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64' -lOpenCL

components.o: components.cpp
	$(CC) $(CFLAGS) components.cpp -c -o components.o $(IFLAGS) -I'C:\Program Files (x86)\AMD APP SDK\3.0\include' -include alloca.h

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
