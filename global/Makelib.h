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
       LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
       BLAS = -lblas

#................................ SUN ..........................................
ifeq ($(TARGET),SUN)
       BLAS = ../../lapack_blas/libblas.a
endif
#................................ DEC ..........................................
ifeq ($(TARGET),DECOSF)
       BLAS = ../../lapack_blas/libblas.a
endif
#................................ CRAY-T3D .....................................
#
ifeq ($(TARGET),CRAY-T3D)
#

       LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a /mpp/lib/old/libsma.a
#      LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a
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
     LIBCOM += -lrpc -para
endif
#................................ Intel .....................................
ifeq ($(INTEL),YES)
#
# all Intel machines
#
#................................ PARAGON ...................................
#
ifeq ($(TARGET),PARAGON)
#
       LIBS += -nx 
else
       LIBS += -node 
endif
       BLAS  = -lkmath
endif
#................................ SGITFP ...................................
#
ifeq ($(TARGET),SGITFP)
#
       BLAS = ../../lapack_blas/libblas.a
endif
#.................................. SP1 ....................................
#
ifeq ($(TARGET),SP1)
#
# IBM SP1 under EUIH or MPL 
#
ifdef EUIH
       LIBS += -bnso -bI:/lib/syscalls.exp -bI:$(EUIH)/eui.exp -e main
endif
endif

#...........................................................................

LIBS += $(BLAS) ../../lapack_blas/liblapack.a

ifeq (LU_SOLVE, PAR)
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
endif
