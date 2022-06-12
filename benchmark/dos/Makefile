CC :=  /usr/local/cuda/bin/nvcc
CFLAGS := -O3 --std=c++17

GPUCFLAGS := -O3 --std=c++17 -x cu
GPULFLAGS := -lcusparse_static

IFLAGS := -I../../ -I../../../gremlin++/

../../bin/dos_bchm.exe: ../../bin/dos_bchm.o
	$(CC) $(CFLAGS) ../../bin/dos_bchm.o -o ../../bin/dos_bchm.exe $(GPULFLAGS)

../../bin/dos_bchm.o: ../../examples/dos.cpp
	$(CC) $(GPUCFLAGS) ../../examples/dos.cpp -c -o ../../bin/dos_bchm.o $(IFLAGS)

clean:
	rm ../../bin/dos_bchm.exe
	rm ../../bin/dos_bchm.o