# Makefile.h, Wed Jan 25 13:01:15 PST 1995 
#
# Define TARGET to be the machine you wish to build for
# (one of SUN,SOLARIS,SGI,SGITFP,IBM,KSR,SP1,CRAY-T3D,IPSC,DELTA,PARAGON,DECOSF)
#
# Specify message-passing library to be used with GA. The current choices
# are: TCGMSG (default) or MPI. For MPI, please refer to global.doc for 
# configuration info.
# MSG_COMMS = MPI
#
# common definitions (overwritten later if required)
#
           FC = f77
           CC = cc
          FLD = $(FC)
          CLD = $(FLD)
          CXX = CC
         FOPT = -O
         COPT = -O
GLOB_INCLUDES = -I../../ma
           AR = ar
           AS = as
       RANLIB = @echo
          CPP = /usr/lib/cpp
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
      INSTALL = @echo 
       P_FILE = YES
      ARFLAGS = rcv
    EXPLICITF = FALSE
    MAKEFLAGS = -j 4
  CUR_VERSION = SHMEM


ifeq ($(GA_TRACE), YES)
    DEF_TRACE = -DGA_TRACE
endif

GLOB_DEFINES = -D$(TARGET)

#
#................................ LINUX ....................................
# IBM ThinkPad running Linux 1.2.13
#
ifeq ($(TARGET),LINUX)
    MAKEFLAGS = -j 1
    EXPLICITF = TRUE
endif
#
#................................ SUN ......................................
#
ifeq ($(TARGET),SUN)
#
# Sun running SunOS
#
#          CC = gcc
     FOPT_REN = -Nl100 -fast -dalign
       RANLIB = ranlib
     WARNINGS = -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual \
		-Wwrite-strings
 GLOB_DEFINES = -DSUN
endif

#
#.............................. SOLARIS ....................................
#
ifeq ($(TARGET),SOLARIS)
#
# Sun running Solaris
#
     WARNINGS = -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual \
                -Wwrite-strings
 GLOB_DEFINES = -DSOLARIS
     FLD_REN = -xs
endif

#
#................................ DEC ALPHA ................................
#
ifeq ($(TARGET),DECOSF)
#
# DEC ALPHA running OSF/1
#
       RANLIB = ranlib
        CDEFS = -DEXT_INT
     FOPT_REN = -i8
 GLOB_DEFINES = -DDECOSF
endif

#
#................................ CRAY-T3D ..................................
#
ifeq ($(TARGET),CRAY-T3D)
#
#
#
       LIBSMA = ../../../libsma
           FC = cf77
          CPP = /mpp/lib/mppcpp
       P_FILE = NO
 ifeq ($(FOPT),-O)
         FOPT = -O1
 endif
 ifeq ($(COPT),-O)
         COPT = -O2
 endif
     FOPT_REN = -Ccray-t3d -Wf-dp
     COPT_REN = -h inline3 
      FLD_REN = -Wl"-Drdahead=on -Ddalign=64"
      CLD_REN = -Wl"-Drdahead=on -Ddalign=64"
 GLOB_DEFINES = -DCRAY_T3D
    EXPLICITF = TRUE
endif

#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
     FOPT_REN = -r8
     COPT_REN = -qdiv
 GLOB_DEFINES = -DKSR
        CDEFS = -DEXT_INT
endif

#................................ SGI ......................................
#
ifeq ($(TARGET),SGI)
#
# SGI running IRIX
#
#
      FLD_REN = -v -Wl,-U
 GLOB_DEFINES = -DSGI
endif

#................................ SGI Power Challenge .......................
#
ifeq ($(TARGET),SGITFP)
#
# SGI running IRIX6.0
#
#
 ifeq ($(FOPT),-O)
         FOPT = -O3
 endif
        CDEFS = -DEXT_INT
     FOPT_REN = -i8 -align64 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000 
 GLOB_DEFINES = -DSGI -DSGI64 -DSGIUS
endif

#............................. IPSC/DELTA/PARAGON .............................
#
ifeq ($(TARGET),IPSC)
#
# IPSC running NX
#
        INTEL = YES
     FOPT_REN = -node
     COPT_REN = -node
      INSTALL = @echo "See TCGMSG README file on how to run program "
endif
#
#....................
#
ifeq ($(TARGET),DELTA)
#
# DELTA running NX
#
        INTEL = YES
     FOPT_REN = -node
     COPT_REN = -node
 GLOB_DEFINES = -DDELTA
      INSTALL = rcp $@ delta2: 
endif
#
#....................
#
ifeq ($(TARGET),PARAGON)
#
# PARAGON running OS>=1.2 with NX (crosscompilation on Sun)
#
        INTEL = YES
     FOPT_REN = -nx
     COPT_REN = -nx -Msafeptr
 GLOB_DEFINES = -DPARAGON
endif
#
ifeq ($(INTEL),YES)
#
# all Intel machines
#
           FC = if77
           CC = icc
           AR = ar860
           AS = as860
          CLD = $(CC)
       P_FILE = NO
 ifeq ($(FOPT),-O)
         FOPT = -O2
 endif
     FOPT_REN += -Knoieee -Mquad -Mreentrant -Mrecursive
     COPT_REN += -Knoieee -Mquad -Mreentrant
 GLOB_DEFINES += -DNX
  CUR_VERSION = DISMEM
    EXPLICITF = TRUE
endif
 
#.............................. SP1 .........................................
#
ifeq ($(TARGET),SP1)
#
# IBM SP-1 and SP-2 under EUIH or MPL 

       P_FILE = NO
ifdef EUIH
         EUIH = /usr/lpp/euih/eui
           FC = xlf
GLOB_INCLUDES = -I. -I../../ma -I$(EUIH)
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX -DEUIH
      FLD_REN = -b  rename:.lockrnc_,.lockrnc
else
           CC = mpcc
           FC = mpxlf
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX
      FLD_REN = -b rename:.daxpy_,.daxpy -b rename:.dgemm_,.dgemm
endif

#   mpxlf fails with parallel make
    MAKEFLAGS = -j 1
       RANLIB = ranlib
     FOPT_REN = -qEXTNAME
  CUR_VERSION = DISMEM
    EXPLICITF = TRUE
endif
 
#.............................. IBM .........................................
#
ifeq ($(TARGET),IBM)
#
# IBM RS/6000 under AIX  
#
           FC = xlf
       RANLIB = ranlib
 GLOB_DEFINES = -DEXTNAME -DAIX
     FOPT_REN = -qEXTNAME 
      FLD_REN = -b rename:.daxpy_,.daxpy -b rename:.dgemm_,.dgemm
    EXPLICITF = TRUE
endif

#
#.......................... other common defs ............................
#

ifndef VERSION
       VERSION = $(CUR_VERSION)
endif
ifeq ($(VERSION),SHMEM)
 GLOB_DEFINES += -DSHMEM
endif
ifdef USE_MPI
 GLOB_INCLUDES += -I$(MPI_LOC)/include 
 ifeq ($(MSG_COMMS),MPI)
    GLOB_DEFINES += -DMPI
 endif
endif

      DEFINES = $(GLOB_DEFINES) $(LOC_DEFINES) $(DEF_TRACE)
     INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)
       FFLAGS = $(FOPT) $(FOPT_REN) $(INCLUDES) $(DEFINES)
       CFLAGS = $(COPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)
       FLDOPT = $(FOPT) $(FOPT_REN) $(FLD_REN)
       CLDOPT = $(COPT) $(COPT_REN) $(CLD_REN)
     CXXFLAGS = $(CFLAGS)

ifeq ($(EXPLICITF),TRUE)
#
# Needed on machines where FCC does not preprocess .F files
# with CPP to get .f files
#
.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c

.F.o:	
	$(MAKE) $*.f
	$(FC) $(FOPT) $(FOPT_REN) -c $*.f
	$(RM) -f $*.f

.f.o:
	$(FC) $(FOPT) $(FOPT_REN) -c $*.f

.F.f:	
	$(CPP) $(INCLUDES) $(DEFINES) < $*.F | sed '/^#/D' > $*.f

.c.o:
	$(CC) $(CFLAGS) -c $*.c
endif

