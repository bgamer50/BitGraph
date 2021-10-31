CC :=  /opt/clang+llvm-13.0.0-aarch64-linux-gnu/bin/clang++ 
CFLAGS := -Ofast --std=c++17 -funsafe-math-optimizations -frename-registers -funroll-loops -freroll-loops -frewrite-imports -frewrite-includes

CUDAARCH := sm_53
CUDALIB := /usr/local/cuda/lib64

GPUCFLAGS := -Ofast --std=c++17 -xcuda --cuda-gpu-arch=$(CUDAARCH) -funsafe-math-optimizations -frename-registers -funroll-loops -freroll-loops -frewrite-imports -frewrite-includes
GPULFLAGS := -L$(CUDALIB) -lcusparse_static -lcudart_static -ldl -lrt -pthread

IFLAGS := -I. -I../gremlin++/

test_gpu.exe: test_gpu.o
	$(CC) $(CFLAGS) test_gpu.o -o test_gpu.exe $(GPULFLAGS)

test_gpu.o: test_gpu.cpp
	$(CC) $(GPUCFLAGS) test_gpu.cpp -c -o test_gpu.o $(IFLAGS)

lca.exe: lca.o
	$(CC) $(CFLAGS) lca.o -o lca.exe $(GPULFLAGS)

lca.o: lca.cpp
	$(CC) $(GPUCFLAGS) lca.cpp -c -o lca.o $(IFLAGS)

valuemap.exe: valuemap.o
	$(CC) $(CFLAGS) valuemap.o -o valuemap.exe $(IFLAGS)

valuemap.o: valuemap.cpp
	$(CC) $(CFLAGS) valuemap.cpp -c -o valuemap.o $(IFLAGS)

components.exe: components.o
	$(CC) $(CFLAGS) components.o -o components.exe $(IFLAGS)

components.o: components.cpp
	$(CC) $(CFLAGS) components.cpp -c -o components.o $(IFLAGS)

components_gpu.exe: components_gpu.o
	$(CC) $(CFLAGS) components_gpu.o -o components_gpu.exe $(GPULFLAGS)

components_gpu.o: components_gpu.cpp
	$(CC) $(GPUCFLAGS) components_gpu.cpp -c -o components_gpu.o $(IFLAGS)

repeat.exe: repeat.o
	$(CC) $(CFLAGS) repeat.o -o repeat.exe $(IFLAGS)

repeat.o: repeat.cpp
	$(CC) $(CFLAGS) repeat.cpp -c -o repeat.o $(IFLAGS)

clean:
	rm -rf *.o *.dylib *.lib *.so *.exe
