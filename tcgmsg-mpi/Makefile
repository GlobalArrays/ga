#
# Makefile for TCGMSG library on top of MPI
#
# Author: Jarek Nieplocha
# August 1995

#Either MACHINE or TARGET platform must be defined

ifndef MACHINE
ifdef TARGET
   MACHINE = $(TARGET)
else
        ERRMSG = "You must define MACHINE\\n"
endif
endif


ifdef MPI_INCLUDE
       GLOB_INCLUDES = -I$(MPI_INCLUDE)
endif


# On some platforms NXVAL can be implemented using non-portable "native"
# services. To enable use of such an implementation, you need to uncomment
# the following line: 
USE_NATIVE_NXTVAL=1

# You need to specify MACHINE target and location of MPI include files and
# library -- MPI_LOC
#

LIBRARY = libtcgmsg-mpi.a
LIB_DISTRIB = ../lib

# name of MPI library
ifndef LIBMPI
  LIBMPI  = -lmpi
endif



           FC = f77
           CC = cc
         FOPT = -O
         COPT = -O
           AR = ar
           AS = as
       RANLIB = @echo
          CPP = /usr/lib/cpp
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
      ARFLAGS = r
    MAKEFLAGS = -j 1
        MKDIR = mkdir
 GLOB_DEFINES = -D$(MACHINE)

  
#
#.................................. SUN ....................................
#
ifeq ($(MACHINE),SUN)
     FOPT_REN = -Nl100
       RANLIB = ranlib
endif

#................................ SOLARIS ..................................
#
ifeq ($(MACHINE),SOLARIS)
        CDEFS = -DSUNF77_2
   EXTRA_LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif

#................................ LINUX ....................................
#
ifeq ($(MACHINE),LINUX)
     FOPT_REN = -Nl100
       RANLIB = ranlib
endif

#................................ DEC ALPHA ................................
#
ifeq ($(MACHINE),DECOSF)
#
# DEC ALPHA running OSF/1
#
       RANLIB = ranlib
        CDEFS = -DEXT_INT
     FOPT_REN = -i8
    EXTRA_OBJ = memalign.o
endif

#................................ CRAY-T3D ..................................
#
ifeq ($(MACHINE),CRAY-T3D)
#
           FC = cf77
         FOPT = -g
          CPP = /mpp/lib/mppcpp
     FOPT_REN = -Ccray-t3d -Wf-dp
 GLOB_DEFINES = -DCRAY_T3D
    EXTRA_OBJ = memalign.o
   NXTVAL_OBJ = nxtval.t3d.o
 NXTVAL_AVAIL = yes
endif

#................................ KSR ......................................
#
ifeq ($(MACHINE),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
     FOPT_REN = -r8
     COPT_REN = -qdiv
        CDEFS = -DEXT_INT
   EXTRA_LIBS = -lrpc -para
endif

#................................ HP ......................................
ifeq ($(TARGET),HPUX)
           CC = gcc
           FC = fort77
          CPP = /lib/cpp
     FOPT_REN = +ppu
 GLOB_DEFINES = -DHPUX -DEXTNAME
    EXTRA_OBJ = memalign.o
endif

#................................ CONVEX ...................................
ifeq ($(TARGET),CONVEX-SPP)
           FC = fc
          CPP = /lib/cpp
    ifeq ($(FOPT),-O)
        FOPT = -O1
    endif
    ifeq ($(FOPT),-g)
        FOPT = -no
    endif
    ifeq ($(COPT),-g)
        COPT = -no
    endif
     COPT_REN = -or none
     FOPT_REN = -ppu -or none
 GLOB_DEFINES = -DCONVEX -DHPUX -DEXTNAME
    EXTRA_OBJ = memalign.o
endif

#................................ SGI ......................................
#
ifeq ($(MACHINE),SGI)
#
# SGI running IRIX 5.XX
#
 GLOB_DEFINES = -DSGI
endif

#............................. SGI Power Challenge ..........................
#
# SGI running IRIX6.XX 
# (Fortran integer size equivalent to C long)
#
ifeq ($(MACHINE),SGITFP)
        CDEFS = -DEXT_INT
     FOPT_REN = -i8 -64
 GLOB_DEFINES = -DSGI64 -DSGI
endif

ifeq ($(MACHINE),SGI64)
#
# SGI running IRIX6.XX with native Fortran integer sizes
#
 GLOB_DEFINES = -DSGI64 -DSGI
endif

#................................ Intel Paragon .............................
#
ifeq ($(MACHINE),PARAGON)
            CC = icc
            FC = if77
            AR = ar860
     FOPT_REN += -Knoieee -Mreentrant
     COPT_REN += -Knoieee -Mreentrant

     EXTRA_OBJ = memalign.o
    NXTVAL_OBJ = nxtval.nx.o
    EXTRA_LIBS = -nx
  NXTVAL_AVAIL = yes
endif

#.............................. IBM SP .....................................
#
ifeq ($(MACHINE),SP1)
# IBM SP-x 
#
           CC = mpcc
           FC = mpxlf
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX
#   mpxlf fails with parallel make
    MAKEFLAGS = -j 1
       RANLIB = ranlib
     FOPT_REN = -qEXTNAME
   NXTVAL_OBJ = nxtval.sp.o
    EXTRA_OBJ = memalign.o
 NXTVAL_AVAIL = yes
endif

#.............................. IBM .........................................
#
ifeq ($(MACHINE),IBM)
#
# IBM RS/6000 under AIX
#
           FC = xlf
       RANLIB = ranlib
 GLOB_DEFINES = -DEXTNAME -DAIX
     FOPT_REN = -qEXTNAME
    EXTRA_OBJ = memalign.o
endif

#...........................................................................


ifneq ($(NXTVAL_AVAIL),yes)
 GLOB_DEFINES += -DNXTVAL_SERVER
    NXTVAL_OBJ = nxtval.o
else
 ifndef USE_NATIVE_NXTVAL
   GLOB_DEFINES += -DNXTVAL_SERVER
      NXTVAL_OBJ = nxtval.o
 endif
endif

      DEFINES = $(GLOB_DEFINES) $(LOC_DEFINES)
     INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)
       FFLAGS = $(FOPT) $(FOPT_REN) $(INCLUDES)
       CFLAGS = $(COPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)
       FLDOPT = $(FOPT) $(FOPT_REN) $(FLD_REN)
       CLDOPT = $(COPT) $(COPT_REN) $(CLD_REN)

OBJ = checkbyte.o	drand48.o	pbeginf.o	sizeof.o\
      collect.o		misc.o		p2p.o		pfilecopy.o\
      evlog.o		evon.o		random.o

      OBJ += $(EXTRA_OBJ) $(NXTVAL_OBJ) 

# This include file specifies how and where the library should be built
# LIBRARY_PATH  is the library location

include ../config/makelib.h

ifdef MPI_LIB
      LIBS = -L$(MPI_LIB)
endif

LIBS += $(LIBMPI) $(EXTRA_LIBS)

all:	test.x testf.x


test.x:	test.o $(LIBRARY_PATH)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

testf.x: testf.o $(LIBRARY_PATH)
	$(FC)  $(FLDOPT) -o $@ $^ $(LIBS)

hello.x: hello.o $(LIBRARY_PATH)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

test_arcv.x: test_arcv.o $(LIBRARY_PATH)
	$(FC)  $(FLDOPT) -o $@ $^ $(LIBS)

test_asyn.x: test_asyn.o $(LIBRARY_PATH)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

$(patsubst %,$(LIBRARY_PATH)(%),$(OBJ))      : tcgmsgP.h
$(patsubst %,$(LIBRARY_PATH)(%),evon.o)      : evlog.h
$(patsubst %,$(LIBRARY_PATH)(%),evlog.o)     : evlog.h
$(patsubst %,$(LIBRARY_PATH)(%),pfilecopy.o) : msgtypesc.h
$(patsubst %,$(LIBRARY_PATH)(%),pbeginf.o))  : farg.h
tcgmsgP.h: srftoc.h
sndrcv.h: msgtypesc.h
srftoc.h: types.h
srftoc.h: sndrcv.h
test.o hello.o: sndrcv.h
