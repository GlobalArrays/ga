# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#      "make test"        to create a test program for GAs 
#      "make testsolve"   to create a test program for linear equation solver 
#                         based on LU factorization
#      "make testeig"     to create a test program for GA eigensolver,
#                         matrix multiply, and symmetrization 
#      "make clean"       clean
#
#........................................................................


# default version and target (to be overwritten on command line)
CUR_VERSION = SHMEM
CUR_TARGET  = SUN 

# to be able to use parallel eigensolver and linear equation solver define 
# DIAG = PAR  &  LU_SOLVE = PAR

DIAG =
LU_SOLVE =

#do you want tracing ? [YES/NO]
#
GA_TRACE = 

ifndef VERSION
       VERSION = $(CUR_VERSION)
endif
ifndef TARGET 
       TARGET = $(CUR_TARGET)
endif


   LIBRARY = libglobal.a
   TESTS   = test test.p testeig testeig.p jacobix jacobix.p\
             testsolve testsolve.p testchol testchol.p\
	     testpatch testpatch.p 


export TARGET VERSION DIAG LU_SOLVE GA_TRACE

.PHONY: $(LIBRARY)

$(LIBRARY): 
	(echo TARGET is $(TARGET); cd ./src; $(MAKE) -j1 || exit 1;) 

$(TESTS):	$(LIBRARY)
	(cd ./testing; $(MAKE) $@  || exit 1;)

clean:
	$(RM) *.a events
	(cd ./src; $(MAKE) clean)
	(cd ./testing; $(MAKE) realclean)
	(cd ./trace; $(MAKE) clean)
	(cd ./X; $(MAKE) clean)
