
#LAPACK := $(patsubst %.f,%.o,$(wildcard *.f))
ifdef GA_C_CORE
LAPACK = dummy.o
BLAS = 
else
LAPACK = dgetf2.o  dgetrs.o  dlaswp.o  ilaenv.o   \
         dgetrf.o  dlaset.o  dlacpy.o  lsame.o   xerbla.o
 
BLAS = dblas.o zaxpy.o xerbla.o lsame.o zgemm.o
endif

include ../../config/makefile.h

ifdef HAS_BLAS
    OBJ = $(LAPACK) 
else
    OBJ = $(LAPACK) $(BLAS) 
endif

LIBRARY = liblinalg.a
LIB_DISTRIB = ../../lib

include ../../config/makelib.h
