# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#      "make test.x"      to create a test program for GA 
#      "make patch.x"     to create a test program for patch ops in GA 
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
#
# All programs are built in ./testing subdirectory. 
# TARGET might be one of:
#      SUN, SGI, SGITFP, IBM, DECOSF, KSR, SP1, CRAY-T3D, IPSC, DELTA, PARAGON
#      SOLARIS, LINUX, LAPI, PGLINUX
#
#........................................................................


# default version and target (to be overwritten on command line)
CUR_TARGET  =
CUR_VERSION =

# to be able to use parallel eigensolver
# DIAG = PAR 
DIAG =

#do you want tracing ? [YES/NO]
#
GA_TRACE = 

SUBDIRS     = src testing trace X examples
MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE)  -C $$dir $@ || exit 1 ; done

ifndef TARGET 
       TARGET = $(CUR_TARGET)
endif
ifndef LIBDIR
     LIBDIR = ../lib/$(TARGET)
endif

   LIBRARY = $(LIBDIR)/libglobal.a
   TESTS   = test.x testeig.x jacobi.x testsolve.x patch.x perform.x testc.x

ifdef USE_MPI
   TESTS +=  ga-mpi.x 
else
ifeq ($(MSG_COMMS),MPI)
   TESTS +=  ga-mpi.x 
endif
endif
ifdef USE_SCALAPACK
   TESTS += testspd.x
endif
ifdef USE_SCALAPACK_I8
   TESTS += testspd.x
endif

export DIAG TARGET LU_SOLVE GA_TRACE MSG_COMMS

.PHONY: $(LIBRARY)

$(LIBRARY): 
	(echo TARGET is $(TARGET); cd ./src; $(MAKE) || exit 1;) 

$(TESTS) %.x:	$(LIBRARY)
	(cd ./testing; $(MAKE) $@  || exit 1;)

all: $(TESTS)

clean:
	$(MAKESUBDIRS) $^
	$(RM) -f $(LIBRARY)

cleanstamp:
	$(MAKESUBDIRS) $^
