CC := g++
CFLAGS := -Ofast --std=c++17 -floop-interchange -floop-strip-mine -funsafe-math-optimizations -frename-registers

NVCC := /usr/local/cuda/bin/nvcc
NVCFLAGS := --forward-unknown-to-host-compiler -pg -O3 --std=c++17 -floop-strip-mine -funsafe-math-optimizations -frename-registers
NVLFLAGS := -lcusparse_static

IFLAGS := -I. -I../gremlin++/

GPULFLAGS := -L./lib/ -lbitgraph

test_gpu.exe: test_gpu.o lib/libbitgraph.so
	$(NVCC) $(NVCFLAGS) test_gpu.o -o test_gpu.exe $(IFLAGS) $(NVLFLAGS) $(GPULFLAGS)

test_gpu.o: test_gpu.cpp
	$(NVCC) -x cu $(NVCFLAGS) test_gpu.cpp -c -o test_gpu.o $(IFLAGS)

lib/libbitgraph.so: lib/GPUTraversalHelper.o
	$(NVCC) $(NVCFLAGS) -shared lib/GPUTraversalHelper.o -o lib/libbitgraph.so

lib/GPUTraversalHelper.o: step/gpu/impl/GPUTraversalHelper.cu
	$(NVCC) $(NVCFLAGS) step/gpu/impl/GPUTraversalHelper.cu -c -fpic -o lib/GPUTraversalHelper.o $(IFLAGS)

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

components_gpu.exe: components_gpu.o
	$(NVCC) $(NVCFLAGS) components_gpu.o -o components_gpu.exe $(IFLAGS) $(NVLFLAGS) $(GPULFLAGS)

components_gpu.o: components_gpu.cpp
	$(NVCC) -x cu $(NVCFLAGS) components_gpu.cpp -c -o components_gpu.o $(IFLAGS)

repeat.exe: repeat.o
	$(CC) $(CFLAGS) repeat.o -o repeat.exe $(IFLAGS)

repeat.o: repeat.cpp
	$(CC) $(CFLAGS) repeat.cpp -c -o repeat.o $(IFLAGS)

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
