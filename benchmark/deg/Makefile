CC :=  /usr/local/cuda/bin/nvcc
CFLAGS := -O3 --std=c++17

GPUCFLAGS := -O3 --std=c++17 -x cu
GPULFLAGS := -lcusparse_static

IFLAGS := -I../../ -I../../../gremlin++/

../../bin/deg_bchm.exe: ../../bin/deg_bchm.o
	$(CC) $(CFLAGS) ../../bin/deg_bchm.o -o ../../bin/deg_bchm.exe $(GPULFLAGS)

../../bin/deg_bchm.o: ../../examples/deg.cpp
	$(CC) $(GPUCFLAGS) ../../examples/deg.cpp -c -o ../../bin/deg_bchm.o $(IFLAGS)

clean:
	rm ../../bin/deg_bchm.exe
	rm ../../bin/deg_bchm.o