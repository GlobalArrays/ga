#
# Makefile for TCGMSG library on top of MPI
#

ifndef MACHINE
error:
	@echo "You must define MACHINE"
	exit
endif

# On some platforms NXVAL can be implemented using non-portable "native"
# services. To enable use of such an implementation, you need to uncomment
# the following line: 
USE_NATIVE_NXTVAL=1


# You need to specify MACHINE target and location of MPI include files and
# library -- MPI_LOC
#

ifndef MPI_LOC
  MPI_LOC = /usr/local/mpi/mpich
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
      ARFLAGS = rcv
GLOB_INCLUDES = -I$(MPI_LOC)/include
 GLOB_DEFINES = -D$(MACHINE)

  
OBJ_CORE =\
    checkbyte.o	drand48.o	pbeginf.o	sizeof.o\
    collect.o	misc.o		p2p.o		pfilecopy.o\
    evlog.o	evon.o		random.o

#
#.................................. SUN ....................................
#
ifeq ($(MACHINE),SUN)
     FOPT_REN = -Nl100
       RANLIB = ranlib
      MPI_LIB = $(MPI_LOC)/lib/sun4/ch_p4
endif
#................................ SOLARIS ..................................
#
ifeq ($(MACHINE),SOLARIS)
        CDEFS = -DSUNF77_2
      MPI_LIB = $(MPI_LOC)/lib/solaris/ch_shmem
   EXTRA_LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
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
      MPI_LIB = $(MPI_LOC)/lib/dec/ch_p4
endif
#................................ CRAY-T3D ..................................
#
ifeq ($(MACHINE),CRAY-T3D)
#
           FC = cf77
          CPP = /mpp/lib/mppcpp
     FOPT_REN = -Ccray-t3d -Wf-dp
      FLD_REN = -Wl"-Drdahead=on -Ddalign=64"
      CLD_REN = -Wl"-Drdahead=on -Ddalign=64"
 GLOB_DEFINES = -DCRAY_T3D
      MPI_LIB = $(MPI_LOC)/lib/cray_t3d/t3d
    EXTRA_OBJ = memalign.o
         FOPT = -g
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
#................................ SGI ......................................
#
ifeq ($(MACHINE),SGI)
#
# SGI running IRIX 5.XX
#
 GLOB_DEFINES = -DSGI
      MPI_LIB = $(MPI_LOC)/lib/IRIX/ch_shmem
endif
#................................ SGI Power Challenge .......................
#
ifeq ($(MACHINE),SGITFP)
        CDEFS = -DEXT_INT
     FOPT_REN = -i8 -64
 GLOB_DEFINES = -DSGI64 -DSGI
      MPI_LIB = $(MPI_LOC)/lib/IRIX/ch_shmem
endif

ifeq ($(MACHINE),SGI64)
#
# SGI running IRIX6.XX
#
#       CDEFS = -DEXT_INT
#    FOPT_REN = -i8 -64 
 GLOB_DEFINES = -DSGI64 -DSGI
      MPI_LIB = $(MPI_LOC)/lib/IRIX/ch_shmem
endif
#................................ Intel Paragon .............................
#
ifeq ($(MACHINE),PARAGON)
            CC = icc
            FC = if77
            AR = ar860
     FOPT_REN += -Knoieee -Mreentrant
     COPT_REN += -Knoieee -Mreentrant

       MPI_LIB = $(MPI_LOC)/lib/paragon/ch_nx
     EXTRA_OBJ = memalign.o
    NXTVAL_OBJ = nxtval.nx.o
    EXTRA_LIBS = -nx
  NXTVAL_AVAIL = yes
endif

#.............................. IBM SP .....................................
#
ifeq ($(MACHINE),SP1)

           CC = mpcc
           FC = mpxlf
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX
#   mpxlf fails with parallel make
    MAKEFLAGS = -j 1
       RANLIB = ranlib
     FOPT_REN = -qEXTNAME
      MPI_LIB = $(MPI_LOC)/lib/rs6000/ch_eui
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
      MPI_LIB = $(MPI_LOC)/lib/rs6000/ch_p4
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

      DEFINES = $(GLOB_DEFINES) $(LOC_DEFINES) $(DEF_TRACE)
     INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)
       FFLAGS = $(FOPT) $(FOPT_REN) $(INCLUDES) $(DEFINES)
       CFLAGS = $(COPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)
       FLDOPT = $(FOPT) $(FLD_REN)
       CLDOPT = $(COPT) $(CLD_REN)

      OBJ = $(OBJ_CORE) $(EXTRA_OBJ) $(NXTVAL_OBJ) 

     LIBS = $(MPI_LIB)/libmpi.a $(EXTRA_LIBS)
     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

  EXECS = test.x testf.x hello

LIBRARY=libtcgmsg.a

all:	test.x testf.x

libtcgmsg.a:	$(OBJ) 
	/bin/rm -f $@ 
	$(AR) $(ARFLAGS) $@ $(OBJ) 
	$(RANLIB) $@ 
	@echo
	@echo TCGMSG-MPI is built for $(MACHINE)
	@echo

clean:
	/bin/rm -f *.o $(EXECS) *.x core $(LIBRARY) PI* *core  mputil.*

realclean:	clean
	/bin/rm -f *~ \#* 

$(OBJ): tcgmsgP.h 

.f.o:
	$(FC) -c $(FFLAGS)  $*.f

.c.o:
	$(CC) $(CFLAGS) -c $*.c

test.x:	test.o $(LIBRARY)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

testf.x: testf.o $(LIBRARY)
	$(FC)  $(FLDOPT) -o $@ $^ $(LIBS)

hello.x: hello.o $(LIBRARY)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

test_arcv.x: test_arcv.o $(LIBRARY)
	$(FC)  $(FLDOPT) -o $@ $^ $(LIBS)

test_asyn.x: test_asyn.o $(LIBRARY)
	$(CC)  $(CLDOPT) -o $@ $^ $(LIBS)

