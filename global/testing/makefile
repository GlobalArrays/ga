# Global Array library (shared and distributed memory version)
#
#      "make ipc.clean"   to create a utility to clean IPC resources 
#      "make test"        to create a test program for GAs 
#      "make testsolve"   to create a test program for linear equation solver 
#                         based on LU factorization
#      "make testeig"     to create a test program for GA eigensolver,
#                         matrix multiply, and symmetrization 
#      "make clean"       clean
#      "make realclean"   even cleaner    
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,IBM,SP1,IPSC,PARAGON,KSR] not defined "
	exit
endif

ifndef VERSION
error1:
	@echo " memory VERSION [SHMEM/DISMEM] not defined "
	exit
endif

ifeq (DIAG, PAR)
     DIAG_DEF = -DPAR_DIAG
endif
ifeq (LU_SOLVE, PAR)
     LU_SOLVE_DEF = -DPAR_LU
endif

LOC_INCLUDES = -I../src
LOC_DEFINES  = -D$(VERSION) $(DIAG_DEF) $(LU_SOLVE_DEF)

include ../Makefile.h
include ../Makelib.h
include ../MakeFiles.h

   LIBRARY = ../libglobal.a

ifeq ($(GA_TRACE),YES)
   LIBTRACE= ../trace/libtrace.a
endif

       OBJ =  $(GA_OBJ) $(GA_ALG) $(GA_UTIL) 


ipc.clean:	ipc.clean.o
	$(CC) -o $@ ipc.clean.o

#
# test programs

test:  stest.o  $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ stest.o $(LIBS) $(LIBCOM) 
	$(INSTALL) 

testpatch:  stestpatch.o  $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ stestpatch.o $(LIBS) $(LIBCOM) 
	$(INSTALL) 

jacobix:  jacobi.o  $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ jacobi.o $(LIBS) $(LIBCOM) $(LIBTRACE) 
	$(INSTALL) 

testeig: stesteig.o  $(LIBRARY) 
	$(FC) $(FLDOPT) -o $@ stesteig.o  $(LIBS) $(LIBCOM) 
	$(INSTALL) 

test.p:
	echo "`whoami` `hostname` 4 `pwd`/test /tmp" > test.p
testpatch.p:
	echo "`whoami` `hostname` 4 `pwd`/testpatch /tmp" > testpatch.p
testeig.p:
	echo "`whoami` `hostname` 4 `pwd`/testeig /tmp" > testeig.p
jacobix.p:
	echo "`whoami` `hostname` 4 `pwd`/jacobix /tmp" > jacobix.p

testsolve:  stestsolve.o  $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ stestsolve.o \
        $(LIBRARY) $(SCALAPACK) $(LIBS) $(LIBCOM)
	$(INSTALL) 

testsolve.p:
	echo "`whoami` `hostname` 4 `pwd`/testsolve /tmp" > testsolve.p

testget:  stestget.o  $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ testget.o $(LIBS) $(LIBCOM) 
	$(INSTALL) 

testget.p:
	echo "`whoami` `hostname` 4 `pwd`/testget /tmp" > test2.p

testchol: stestchol.o $(LIBRARY) 
	$(FC) $(FLDOPT) -o  $@   stestchol.o $(LIBS) $(LIBCOM)
	$(INSTALL) 

testchol.p:
	echo "`whoami` `hostname` 4 `pwd`/testchol /tmp" > $@

#

clean:   
	rm -f *.o *.a core *.p ./test*  ipc.clean event* jacobix

realclean:
	rm -f *.o *.a core *.p ./test* ipc.clean event* jacobix

