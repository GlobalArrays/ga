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
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a
       LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 

#................................ CRAY-T3D .....................................
#
ifeq ($(TARGET),CRAY-T3D)
#

       LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a
endif

#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
        SRC = /home5/d3h325

       LIBS +=  -lksrblas
     LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a
     LIBCOM += -lrpc -para
endif

#................................ IPSC ......................................
#
ifeq ($(TARGET),IPSC)
#
       LIBS += -lkmath -node 
endif

#................................ DELTA .....................................
#
ifeq ($(TARGET),DELTA)
#
       LIBS += -lkmath -node 
endif

#................................ PARAGON ...................................
#
ifeq ($(TARGET),PARAGON)
#
       LIBS += -lkmath -nx 
endif

#.............................. SP1 .........................................
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
ifeq (LU_SOLVE, PAR)
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
endif
