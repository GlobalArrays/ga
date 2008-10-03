# GNU makefile to build the GA++ Library

ifndef TARGET
error:
	@echo "TARGET machine not defined" 
	exit
endif

ifndef GA_C_CORE
error:
	@echo "GA++ (optional) is not built, as GA_C_CORE is not defined [That is OK]"
	exit
endif

LIB_DISTRIB = ../../lib
INCDIR = ../../include
HEADERS = ga++.h init_term.h GAServices.h GlobalArray.h PGroup.h

ifndef LIBRARY
  LIBRARY = libga++.a
endif
  LIBRARY_SHARED = libga++.so

OBJ = init_term.o GAServices.o GlobalArray.o overload.o PGroup.o

#--------------------------- extra definitions ---------------------

LIB_INCLUDES += -I$(INCDIR)

# USE_MPI requires us to access MPI headers to handle ga_mpi_communicator.
# Here "__MPIPP" is defined. The reason is, in some platforms(I experienced in 
# Linux using MVICH - VIA) when mpiCC(C++ compiler) is used, name conflicts 
# arise between namespace MPI in mpi++.h and command line MACRO definition 
# -DMPI
__MPIPP=yes

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
LIB_INCLUDES += -I../../armci/src -I$(INCDIR)

%.o : %.cc
	$(CXX) $(NOPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)  -c $<

