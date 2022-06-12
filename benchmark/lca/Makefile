CC :=  /usr/local/cuda/bin/nvcc
CFLAGS := -O3 --std=c++17

GPUCFLAGS := -O3 --std=c++17 -x cu
GPULFLAGS := -lcusparse_static

IFLAGS := -I../../ -I../../../gremlin++/

../../bin/lca_bchm.exe: ../../bin/lca_bchm.o
	$(CC) $(CFLAGS) ../../bin/lca_bchm.o -o ../../bin/lca_bchm.exe $(GPULFLAGS)

../../bin/lca_bchm.o: ../../examples/lca.cpp
	$(CC) $(GPUCFLAGS) ../../examples/lca.cpp -c -o ../../bin/lca_bchm.o $(IFLAGS)

clean:
	rm ../../bin/lca_bchm.exe
	rm ../../bin/lca_bchm.o