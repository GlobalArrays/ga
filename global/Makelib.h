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
#................................ SUN ......................................
#

ifeq ($(TARGET),SUN)
#
# Sun running SunOS
#
          SRC = /msrc/files/home/d3h325/extra/scf/src
         LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
              /msrc/apps/lib/gcc-lib/sparc-sun-sunos4.1.3/2.4.3/libgcc.a

       LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a 
    SCALAPACK = /msrc/proj/scalapack/scalapack.a /msrc/proj/scalapack/pbblas.a\
                /msrc/proj/scalapack/blacs.a /msrc/proj/scalapack/SLtools.a

#               $(SRC)/peigs1.0/libpeigs.a \
#              /msrc/apps/lib/liblapack.a\
#      	      /msrc/files/home/d3g270/spare2/peigs1.0/libpeigs.a\
#      	      /msrc/files/home/d3g270/spare2/peigs1.0/liblapack.a\

endif


#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
       SRC1 = /home2/d3h325
        SRC = $(SRC1)/scf/src

       LIBS = ../libglobal.a \
              $(SRC)/peigs1.0/libpeigs.a \
              $(SRC)/peigs1.0/liblapack.a \
 	      ../../ma/libma.a\
              $(SRC)/peigs1.0/blas.a
     LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a  -lrpc -para 
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
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

     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
endif


#................................ SGITFP ......................................
#
ifeq ($(TARGET),SGITFP)
#
# SGI running IRIX
#
    FLD_REN = -d8 -i8 -v -Wl,-U 
       LIBS = ../libglobal.a \
              ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
              ../../ma/libma.a\
              -lblas

     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
  SCALAPACK = $(SRC1)/scalapack/scalapack.a $(SRC1)/scalapack/pbblas.a\
              $(SRC1)/scalapack/blacs.a $(SRC1)/scalapack/SLtools.a
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
     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
endif


#................................ PARAGON ...................................
#
ifeq ($(TARGET),PARAGON)
#
#
#

      LIBS = ../libglobal.a \
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a\
             -lkmath -nx

     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
#              $(SRC1)/peigs1.0/libpeigs.a \
#              $(SRC1)/peigs1.0/liblapack.a \

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
     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
endif

#.............................. IBM .........................................
#
ifeq ($(TARGET),IBM)
#
# IBM RS/6000 under AIX  
#
       LIBS = ../libglobal.a\
 	      ../../ma/libma.a\
              ../../lapack_blas/liblapack_blas.a

     LIBCOM = ../../tcgmsg/ipcv4.0/libtcgmsg.a 
endif

