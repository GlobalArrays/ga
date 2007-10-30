
#LAPACK := $(patsubst %.f,%.o,$(wildcard *.f))
ifdef GA_C_CORE
LAPACK = dummy.o
BLAS = 
else
LAPACK = dgetf2.o  dgetrs.o  dlaswp.o  ilaenv.o   \
         dgetrf.o  dlaset.o  dlacpy.o  lsame.o   xerbla.o
 
BLAS = dblas.o zaxpy.o zgemm.o cgemm.o
TESTBLAS = testblas.o
endif

include ../../config/makefile.h

ifdef HAS_BLAS
    OBJ = $(LAPACK) $(TESTBLAS)
else
    OBJ = $(LAPACK) $(BLAS) $(TESTBLAS)
endif

LIBRARY = liblinalg.a
LIBRARY_SHARED = liblinalg.so
LIB_DISTRIB = ../../lib

include ../../config/makelib.h
