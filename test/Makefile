GA_DIR=/qfs/people/d3g293/tmp/ga-cuda/build_cuda/install

CFLAGS= -I${GA_DIR}/include/ga

FCFLAGS= -L${GA_DIR}/lib -L${GA_DIR}/lib64 -pthread -lga -larmci -lblas -llapack -lrt -lm

CUDAFLAGS=-L/share/apps/cuda/10.1/lib64 -L/share/apps/cuda/10.1/lib -lcudart

c_test: c_test.o
	mpif90 -g c_test.o -o c_test ${FCFLAGS} ${CUDAFLAGS}

c_test.o: c_test.c
	mpicc -g ${CFLAGS} c_test.c -c

clean:
	rm c_test.o c_test
