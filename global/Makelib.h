# Makelib.h, 01.26.94
#

       CORE_LIBS = -lglobal -lma

ifndef LIBMPI
   LIBMPI = libmpi.a
endif

ifneq ($(MSG_COMMS),MPI)
  ifdef USE_MPI
       LIBCOM = -ltcgmsg-mpi
  else
       LIBCOM = -ltcgmsg
  endif
endif

#............................... LINUX .........................................
ifeq ($(TARGET),LINUX)
endif

#................................ SUN ..........................................
ifeq ($(TARGET),SUN)
endif
ifeq ($(TARGET),SOLARIS)
       EXTRA_LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif
#................................ DEC ..........................................
ifeq ($(TARGET),DECOSF)
endif
#................................ CRAY-T3D .....................................
#
ifeq ($(TARGET),CRAY-T3D)
#
endif
#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
# These are pointers to much faster (optimized for KSR) version of TCGMSG 
# (does not come with the GA distribution package)
#
#    LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a

       BLAS  = -lksrblas
 EXTRA_LIBS += -lrpc -para
endif
#................................ Intel .....................................
ifeq ($(INTEL),YES)
#
# all Intel machines
#
#................................ PARAGON ...................................
ifeq ($(TARGET),PARAGON)
       CLIB = -lm
#
       EXTRA_LIBS = -nx 
       MPI_DEV = paragon/ch_nx
else
       EXTRA_LIBS = -node 
endif
       BLAS  = -lkmath
endif
#................................   SGI ....................................
ifeq ($(TARGET),SGITFP)
endif
ifeq ($(TARGET),SGI)
       BLAS = -lblas
endif
ifeq ($(TARGET),SGI64)
       BLAS = -lblas
endif
#.................................. SP1 ....................................
ifeq ($(TARGET),SP1)
       BLAS = -lblas
endif
#...........................................................................
ifeq ($(TARGET),IBM)
       BLAS = -lblas
endif
#...........................................................................

#LIBS += $(BLAS) -llinalg $(BLAS)


ifdef USE_MPI
   ifndef MPI_LIB
      ERRMSG = "YOU MUST DEFINE MPI LIBRARY LOCATION - MPI_LIB\\n"
   endif
   LIBCOM += $(MPI_LIB)/$(LIBMPI)
endif

ifdef IWAY
  LIBCOM += -lserver 
endif

LIBCOM += $(EXTRA_LIBS)

ifeq (LU_SOLVE, PAR)
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a

  LINALG = $(SCALAPACK)
endif

LINALG += $(BLAS) -llinalg $(BLAS)
