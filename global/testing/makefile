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
#      "make clean"       clean
#      "make realclean"   even cleaner   
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine not defined "
	@echo "-- one of [SUN,SGI,SGITFP,IBM,DECOSF,SP1,SP,IPSC,DELTA,PARAGON,KSR,CRAY-T3D,CONVEX-SPP,HPUX]"
	exit
endif

LOC_INCLUDES = -I../src -I../../tcgmsg/ipcv4.0
ifeq ($(DIAG), PAR)
     LOC_DEFINES += -DPAR_DIAG
endif
ifdef USE_SCALAPACK
     LOC_DEFINES += -DSCALAPACK
endif

LIB_DISTRIB  = ../../lib
LIB_TARGETS = *.x *.p

include ../Makefile.h
include ../Makelib.h

ifndef LIBDIR
   LIBDIR = $(LIB_DISTRIB)/$(TARGET)
endif

ifeq ($(GA_TRACE),YES)
   LIBTRACE= ../trace/libtrace.a
endif
LIBS = $(CORE_LIBS) $(LINALG) $(LIBCOM) $(LIBTRACE)

#
# main program in C requires extra effort to provide Fortran libs
# Highly system dependent !!!

ifeq ($(MSG_COMMS), MPI)
  COPT_REN += -I../../tcgmsg/ipcv4.0
  P_FILE = no
endif

# no need to build TCGMSG file when GA runs with MPI
ifdef USE_MPI
  P_FILE = no
endif

FOPT= $(NOPT)
COPT= $(NOPT)

#
# build test programs and .p files if needed
#

#.PRECIOUS: %.o




ifeq ($(P_FILE),YES)
    testc.x: testc.o $(LIBDIR)/libglobal.a
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(CLD) $(CLDOPT) -o $@ $< -L$(LIBDIR) $(LIBS) $(CLIB)
    ifdef USE_MPI
      ga-mpi.x: ga-mpi.o $(LIBDIR)/libglobal.a
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(CLD) $(CLDOPT) -o $@ $< -L$(LIBDIR) $(LIBS) $(CLIB)
    else
      ga-mpi.x:
	@echo "You need to recompile the package to use MPI "
    endif
else
    testc.x: testc.o $(LIBDIR)/libglobal.a
	$(CLD) $(CLDOPT) -o $@ $< -L$(LIBDIR) $(LIBS) $(CLIB)
    ifdef USE_MPI
      ga-mpi.x: ga-mpi.o $(LIBDIR)/libglobal.a
	$(CLD) $(CLDOPT) -o $@ $< -L$(LIBDIR) $(LIBS) $(CLIB)
    else
      ga-mpi.x:
	@echo "You need to define USE_MPI and recompile the package "
    endif
endif



ifeq ($(P_FILE),YES)
    ifeq ($(EXPLICITF),FALSE)
        .SUFFIXES : 
        .SUFFIXES : .F .o .x

        %.o : %.F
	   $(FC) $(FFLAGS) -c $*.F 
        %.o : %.c
	   $(CC) $(CFLAGS) -DSYSV -c $*.c 
    endif

    %.x : %.o $(LIBDIR)/libglobal.a
	$(FC) $(FLDOPT) -o $@ $< -L$(LIBDIR) $(LIBS)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(INSTALL)
else
    %.x : %.o $(LIBDIR)/libglobal.a
	$(FC) $(FLDOPT) -o $@ $<  -L$(LIBDIR) $(LIBS)
	$(INSTALL)
endif

%.p:
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@

ifndef SCALAPACK
testspd.x:
	@echo testspd.x requires SCALAPACK installed: see global/README
endif

.PHONY: clean
clean:
	-$(RM) -f *.o *.p *core *stamp mputil.mp* *trace *.x *events*

.PHONY: realclean
realclean:      clean
	-$(RM) -rf *~ \#*\#

.PHONY: cleanstamp
cleanstamp:
	-$(RM) -rf *.stamp

