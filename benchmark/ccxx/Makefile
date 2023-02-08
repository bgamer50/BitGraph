CC :=  /usr/local/cuda/bin/nvcc
CFLAGS := -O3 --std=c++17

GPUCFLAGS := -O3 --std=c++17 -x cu
GPULFLAGS := 

IFLAGS := -I../../ -I../../../gremlin++/

../../bin/components_bchm.exe: ../../bin/components_bchm.o
	$(CC) $(CFLAGS) ../../bin/components_bchm.o -o ../../bin/components_bchm.exe $(GPULFLAGS)

../../bin/components_bchm.o: ../../examples/components_gpu.cpp
	$(CC) $(GPUCFLAGS) ../../examples/components_gpu.cpp -c -o ../../bin/components_bchm.o $(IFLAGS)

clean:
	rm ../../bin/components_bchm.exe
	rm ../../bin/components_bchm.o