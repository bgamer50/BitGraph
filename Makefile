CC := g++
CFLAGS := -Ofast --std=c++17 -floop-interchange -floop-strip-mine -funsafe-math-optimizations -frename-registers

NVCC := /usr/local/cuda-11.1/bin/nvcc
NVCFLAGS := --forward-unknown-to-host-compiler -O3 --std=c++17 -floop-strip-mine -funsafe-math-optimizations -frename-registers
NVLFLAGS := -lcusparse_static

IFLAGS := -I. -I../gremlin++/

test_gpu.exe: test_gpu.o
	$(NVCC) $(NVCFLAGS) test_gpu.o -o test_gpu.exe $(IFLAGS) $(NVLFLAGS)

test_gpu.o: test_gpu.cpp
	$(NVCC) $(NVCFLAGS) test_gpu.cpp -c -o test_gpu.o $(IFLAGS)

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
