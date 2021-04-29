GA_DIR=/qfs/people/d3g293/tmp/ga-cuda/build_cuda/install

CUDA_DIR=/share/apps/cuda/10.1
#CUDA_DIR=/share/apps/cuda/10.1.243

CFLAGS= -I${GA_DIR}/include/ga

FCFLAGS= -L${GA_DIR}/lib -L${GA_DIR}/lib64 -pthread -lga -larmci -lblas -llapack -lrt -lm
#FCFLAGS= -L${GA_DIR}/lib -L${GA_DIR}/lib64 -pthread -lga -larmci -lrt -lm

CUDAFLAGS=-L${CUDA_DIR}/lib64 -L${CUDA_DIR}/lib -lcudart

c_test: c_test.o
	mpif90 -g c_test.o -o c_test ${FCFLAGS} ${CUDAFLAGS}

c_test.o: c_test.c
	mpicc -g ${CFLAGS} c_test.c -c

clean:
	rm c_test.o c_test
