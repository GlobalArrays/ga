# Global Array library (shared and distributed memory version)
#
#      "make test.x"      to create a test program for GA 
#      "make patch.x"     to create a test program for GA patch operations
#      "make testsolve.x" to create a test program for linear equation solver 
#                         based on LU factorization
#      "make testeig.x"   to create a test program for GA eigensolver,
#                         matrix multiply, and symmetrization 
#      "make perform.x"   to create a program testing performance of GA
#                         primitives: get, put & accumulate,
#      "make testc.x"     to create a C test program for GA
#      "make ga-mpi.x"    to create a C test program for GA that demonstrates
#                         the interface to MPI
#      "make ipc.clean"   to create a utility to clean IPC resources 
#      "make clean"       clean
#      "make realclean"   even cleaner    
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine not defined "
	@echo "-- one of [SUN,SGI,SGITFP,IBM,DECOSF,SP1,IPSC,DELTA,PARAGON,KSR,CRAY-T3D]"
	exit
endif

ifeq ($(DIAG), PAR)
     DIAG_DEF = -DPAR_DIAG
endif

ifeq ($(LU_SOLVE), PAR)
     LU_SOLVE_DEF = -DPAR_LU
endif

LOC_INCLUDES = -I../src
LOC_DEFINES  = $(DIAG_DEF) $(LU_SOLVE_DEF)


include ../Makefile.h
include ../Makelib.h
include ../MakeFiles.h

   LIBRARY = ../libglobal.a

ifeq ($(GA_TRACE),YES)
   LIBTRACE= ../trace/libtrace.a
endif

#
# main program in C requires extra effort to provide Fortran libs
# Highly system dependent !!!
ifneq ($(MSG_COMMS), MPI)
  COPT_REN += -I../../tcgmsg/ipcv4.0
endif



FOPT=-g
COPT=-g

#
# build test programs and .p files if needed
#

.PRECIOUS: %.o

ifeq ($(P_FILE),YES)
    ifeq ($(EXPLICITF),FALSE)
        .SUFFIXES : 
        .SUFFIXES : .F .o .x

        %.o : %.F
	   $(FC) $(FFLAGS) -c $*.F 
        %.o : %.c
	   $(CC) $(CFLAGS) -DSYSV -c $*.c 
    endif

    %.x : %.o $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ $< $(LIBS) $(LIBCOM) 
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(INSTALL)
else
    %.x : %.o $(LIBRARY)
	$(FC) $(FLDOPT) -o $@ $< $(LIBS) $(LIBCOM) 
	$(INSTALL)
endif


ifeq ($(P_FILE),YES)
testc.x: testc.o $(LIBRARY)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(CLD) $(CLDOPT) -o $@ $< $(LIBS) $(LIBCOM) $(CLIB)
    ifdef USE_MPI
ga-mpi.x: ga-mpi.o $(LIBRARY)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(CLD) $(CLDOPT) -o $@ $< $(LIBS) $(LIBCOM) $(CLIB)
    else
ga-mpi.x:
	@echo "You need to define USE_MPI and recompile the package "
    endif
else
testc.x: testc.o $(LIBRARY)
	$(CLD) $(CLDOPT) -o $@ $< $(LIBS) $(LIBCOM) $(CLIB)
    ifdef USE_MPI
ga-mpi.x: ga-mpi.o $(LIBRARY)
	$(CLD) $(CLDOPT) -o $@ $< $(LIBS) $(LIBCOM) $(CLIB)
    else
ga-mpi.x:
	@echo "You need to define USE_MPI and recompile the package "
    endif
endif

ipc.clean:	ipc.clean.o
	$(CC) -o $@  ipc.clean.o


clean:   
	rm -f *.o *.a core mppcore mputil* *.p *.x  ipc.clean event* *.x.trace

realclean: clean
