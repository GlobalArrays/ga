# Makelib.h, 01.26.94
#
# TARGET is one of (SUN, SGI, IBM, KSR, SP1)
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
#................................ SUN ......................................
#

ifeq ($(TARGET),SUN)
#
# Sun running SunOS
#
         LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
              /msrc/apps/lib/gcc-lib/sparc-sun-sunos4.1.3/2.4.3/libgcc.a

endif


#................................ CRAY-T3D .....................................
#
ifeq ($(TARGET),CRAY-T3D)
#
#

       LIBCOM = ../../tcgmsg/ipcv5.0/libtcgmsg.a 
endif


#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
       SRC1 = /home5/d3h325
        SRC = $(SRC1)/scf/src

     LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a  -lrpc -para 
endif
#................................ SGI ......................................
#
ifeq ($(TARGET),SGI)
#
# SGI running IRIX
#
        SRC = $(SRC1)/scf/src
       SRC1 = /usr/people/jaroslaw
#

    FLD_REN = -v -Wl,-U 
       LIBS = ../libglobal.a \
              ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
              ../../ma/libma.a\
              -lblas
endif
#................................ IPSC ......................................
#
ifeq ($(TARGET),IPSC)
#
# DELTA/IPSC running NX
#

      LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
 	      ../../ma/libma.a\
             -lkmath -node 
endif
#................................ PARAGON ...................................
#
ifeq ($(TARGET),PARAGON)
#
#
      LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
             -lkmath -nx
endif
#.............................. SP1 .........................................
#
ifeq ($(TARGET),SP1)
#
# IBM SP1 under EUIH 
#
       LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
              -bnso -bI:/lib/syscalls.exp -bI:$(EUIH)/eui.exp -e main
endif
#.............................. IBM ........................................
#
ifeq ($(TARGET),IBM)
#
       SRC =
endif
#...........................................................................
ifeq (LU_SOLVE, PAR)
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
endif
