CC :=  /usr/local/cuda/bin/nvcc
CFLAGS := -O3 --std=c++17

GPUCFLAGS := -O3 --std=c++17 -x cu
GPULFLAGS := -lcusparse_static

IFLAGS := -I../../ -I../../../gremlin++/

../../bin/ast_bchm.exe: ../../bin/ast_bchm.o
	$(CC) $(CFLAGS) ../../bin/ast_bchm.o -o ../../bin/ast_bchm.exe $(GPULFLAGS)

../../bin/ast_bchm.o: ../../examples/ast.cpp
	$(CC) $(GPUCFLAGS) ../../examples/ast.cpp -c -o ../../bin/ast_bchm.o $(IFLAGS)

clean:
	rm ../../bin/ast_bchm.exe
	rm ../../bin/ast_bchm.o