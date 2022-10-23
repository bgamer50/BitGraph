CC :=  g++
CFLAGS := -O3 --std=c++17 -funsafe-math-optimizations -frename-registers -funroll-loops -fsized-deallocation -fopenmp #-pg -no-pie

CUDALIB := /usr/local/cuda/lib64

GPUCC :=  /usr/local/cuda/bin/nvcc
GPUCFLAGS := -forward-unknown-to-host-compiler -O3 --std=c++17 -x cu -funsafe-math-optimizations -frename-registers -funroll-loops -fsized-deallocation -fopenmp #-g -pg -no-pie
GPULFLAGS := -L$(CUDALIB) -lcusparse_static -lcudart_static -ldl -lrt -pthread -ltbb

IFLAGS := -I. -I../gremlin++/ -I${CONDA_PREFIX}/include

bin/dos.exe: bin/dos.o
	$(CC) $(CFLAGS) bin/dos.o -o bin/dos.exe $(GPULFLAGS)

bin/dos.o: examples/dos.cpp
	$(GPUCC) $(GPUCFLAGS) examples/dos.cpp -c -o bin/dos.o $(IFLAGS)

bin/deg.exe: bin/deg.o
	$(CC) $(CFLAGS) bin/deg.o -o bin/deg.exe $(GPULFLAGS)

bin/deg.o: examples/deg.cpp
	$(GPUCC) $(GPUCFLAGS) examples/deg.cpp -c -o bin/deg.o $(IFLAGS)

bin/test_gpu.exe: bin/test_gpu.o
	$(CC) $(CFLAGS) bin/test_gpu.o -o bin/test_gpu.exe $(GPULFLAGS)

bin/test_gpu.o: examples/test_gpu.cpp
	$(GPUCC) $(GPUCFLAGS) examples/test_gpu.cpp -c -o bin/test_gpu.o $(IFLAGS)

bin/lca.exe: bin/lca.o
	$(CC) $(CFLAGS) bin/lca.o -o bin/lca.exe $(GPULFLAGS)

bin/lca.o: examples/lca.cpp
	$(GPUCC) $(GPUCFLAGS) examples/lca.cpp -c -o bin/lca.o $(IFLAGS)

bin/valuemap.exe: bin/valuemap.o
	$(CC) $(CFLAGS) bin/valuemap.o -o bin/valuemap.exe $(IFLAGS)

bin/valuemap.o: examples/valuemap.cpp
	$(GPUCC) $(CFLAGS) examples/valuemap.cpp -c -o bin/valuemap.o $(IFLAGS)

bin/components.exe: bin/components.o
	$(CC) $(CFLAGS) bin/components.o -o bin/components.exe $(IFLAGS) $(GPULFLAGS)

bin/components.o: examples/components.cpp
	$(GPUCC) $(GPUCFLAGS) examples/components.cpp -c -o bin/components.o $(IFLAGS)

bin/components_gpu.exe: bin/components_gpu.o
	$(CC) $(CFLAGS) bin/components_gpu.o -o bin/components_gpu.exe $(GPULFLAGS)

bin/components_gpu.o: examples/components_gpu.cpp
	$(GPUCC) $(GPUCFLAGS) examples/components_gpu.cpp -c -o bin/components_gpu.o $(IFLAGS)

bin/repeat.exe: bin/repeat.o
	$(CC) $(CFLAGS) bin/repeat.o -o bin/repeat.exe $(GPULFLAGS)

bin/repeat.o: examples/repeat.cpp
	$(GPUCC) $(GPUCFLAGS) examples/repeat.cpp -c -o bin/repeat.o $(IFLAGS)

clean:
	rm -rf bin/*
