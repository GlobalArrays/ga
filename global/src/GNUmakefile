#....................................................................
# $Id
# JN/June 9, 2000: cleaned makefile from old files 
#

ifndef TARGET
error:
	@echo "TARGET machine not defined" 
	exit
endif

LIB_DISTRIB = ../../lib
INCDIR = ../../include
HEADERS = ga.h global.h global.fh c.names.h

ifndef LIBRARY
  LIBRARY = libglobal.a
endif

#---------------------- file list ---------------------------------
# core of the library
GA_CORE = base.o onesided.o nbutil.o ghosts.o global.util.o collect.o decomp.o \
            capi.o global.nalg.o global.npatch.o ga_trace.o datatypes.o ga_malloc.o

# extra interfaces
GA_EXTRA = global.periodic.o sparse.o select.o ghosts.o DP.o hsort.scat.o \
		ga_solve_seq.o
ifdef GA_USE_VAMPIR
   GA_EXTRA += ga_vampir.o
endif

# Linear Algebra
  GA_LIN_ALG = matmul.o ga_symmetr.o  elem_alg.o matrix.o xgemm.o fort.o
ifndef GA_C_CORE
  GA_LIN_ALG += rsg.o ga_diag_seq.o
#  GA_EXTRA += fort.o
endif

# ga_transpose.o
ifeq ($(DIAG),PAR)
     GA_LIN_ALG += ga_diag.o
else
     GA_LIN_ALG += peigstubs.o
endif
ifdef USE_SCALAPACK
     GA_LIN_ALG += scalapack.o
else
     GA_LIN_ALG += sclstubs.o
endif

OBJ = $(GA_CORE) $(GA_EXTRA) $(GA_LIN_ALG) 

ifeq ($(TARGET),NEC)
OBJ_FRAGILE = global.armci.o
endif

#--------------------------- extra definitions ---------------------
  LIB_DEFINES += -DARMCI
  ifeq ($(GA_TRACE), YES)
    LIB_DEFINES += -DGA_TRACE
  endif
  LIB_INCLUDES += -I../../armci/src -I$(INCDIR)

#we USE_MPI requires us to acces MPI headers to handle ga_mpi_communicator
ifdef USE_MPI
  MSG_COMMS = MPI
endif
ifeq ($(MSG_COMMS),MPI)
include ../../armci/config/makemp.h
endif

# We can remove entire library when cleaning rather than individual objects
HARDCLEAN = yes

include ../../config/makefile.h

include ../../config/makelib.h

$(OBJ_FRAGILE): %.o : %.c
	$(CC) $(NOPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)  -c $<

$(patsubst %,$(FULL_LIBRARY_PATH)(%),$(GA_CORE))     : globalp.h global.h
$(patsubst %,$(FULL_LIBRARY_PATH)(%),global.alg.o)   : globalp.h global.h
$(patsubst %,$(FULL_LIBRARY_PATH)(%),ga_diag.o)      : global.fh diag.fh 
$(patsubst %,$(FULL_LIBRARY_PATH)(%),global.armci.o) : global.h global.armci.h 
global.h       : c.names.h

