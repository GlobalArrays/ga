# Makelib.h, 01.26.94
#
# TARGET is one of (SUN, SGI, SGITFP, IBM, KSR, SP1, T3D)
#
#
# If you want to build test programs for GA, you also need to provide
# the following libraries:
#       program:               libraries:
#        test                    TCGMSG
#        testeig                 TCGMSG, MA, PEIGS, BLAS, LAPACK
#
#

       LIBS = ../libglobal.a \
 	      ../../ma/libma.a
       BLAS = -lblas

ifneq ($(MSG_COMMS),MPI)
       LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
endif

#............................... LINUX .........................................
ifeq ($(TARGET),LINUX)
       BLAS = ../../lapack_blas/libblas.a
       MPI_DEV = linux/ch_p4
endif

#................................ SUN ..........................................
ifeq ($(TARGET),SUN)
       BLAS = ../../lapack_blas/libblas.a
       MPI_DEV = sun4/ch_p4
endif
ifeq ($(TARGET),SOLARIS)
       BLAS = ../../lapack_blas/libblas.a
       EXTRA_LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
       MPI_DEV = solaris/ch_shmem
endif
#................................ DEC ..........................................
ifeq ($(TARGET),DECOSF)
       BLAS = ../../lapack_blas/libblas.a
endif
#................................ CRAY-T3D .....................................
#
ifeq ($(TARGET),CRAY-T3D)
#
# 
#      LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a /mpp/lib/old/libsma.a
       LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a
       MPI_DEV = cray_t3d/t3d
       BLAS=
endif
#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
#
# These are pointers to much faster (optimized for KSR) version of TCGMSG 
# (does not come with the GA distribution package)
#
#       SRC = /home5/d3h325
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
#
       CLIB = -lm
ifeq ($(TARGET),PARAGON)
#
       EXTRA_LIBS = -nx 
       MPI_DEV = paragon/ch_nx
else
       EXTRA_LIBS = -node 
endif
       BLAS  = -lkmath
endif
#................................ SGITFP ...................................
#
ifeq ($(TARGET),SGITFP)
#
       BLAS = ../../lapack_blas/libblas.a
       MPI_DEV = IRIX/ch_shmem
endif
#.................................. SP1 ....................................
#
ifeq ($(TARGET),SP1)
       MPI_DEV = rs6000/ch_eui
       BLAS += ../../lapack_blas/libblas.a
endif
#...........................................................................
ifeq ($(TARGET),IBM)
       MPI_DEV = rs6000/ch_p4
       BLAS += ../../lapack_blas/libblas.a
endif
#...........................................................................

LIBS += ../../lapack_blas/liblapack.a $(BLAS)

ifdef USE_MPI
   ifndef MPI_LOC
      MPI_LOC='YOU MUST DEFINE MPI_LOC'
   endif
       LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
   LIBCOM = ../../tcgmsg-mpi/libtcgmsg.a $(MPI_LOC)/lib/$(MPI_DEV)/libmpi.a
endif

LIBCOM += $(EXTRA_LIBS)

ifeq (LU_SOLVE, PAR)
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
endif
