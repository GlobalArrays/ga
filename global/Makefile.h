#$Id: Makefile.h,v 1.22 1995-02-02 23:12:51 d3g681 Exp $
# Makefile.h, Wed Jan 25 13:01:15 PST 1995 
#
# Define TARGET to be the machine you wish to build for
# (one of SUN, SGI, SGITFP, IBM, KSR, SP1, CRAY-T3D, IPSC, DELTA, PARAGON)
#
# Define VERSION of memory (SHMEM/DISMEM) - or accept machine default
#
# common definitions (overwritten later if required)
#
           FC = f77
           CC = cc
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

#
#................................ SUN ......................................
#
ifeq ($(TARGET),SUN)
#
# Sun running SunOS
#
#          CC = gcc
     FOPT_REN = -Nl100
       RANLIB = ranlib
     WARNINGS = -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual \
		-Wwrite-strings
 GLOB_DEFINES = -DSUN
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
         COPT = -O3
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
#        COPT = -g
        CDEFS = -DEXT_INT
     FOPT_REN = -d8 -i8 -64 -mips4 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000 
 GLOB_DEFINES = -DSGI -DSGI64
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
       P_FILE = NO
 ifeq ($(COPT),-O)
         COPT = -O3
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
# IBM SP1 under EUIH or MPL 

       P_FILE = NO
ifdef EUIH
         EUIH = /usr/lpp/euih/eui
           FC = xlf
GLOB_INCLUDES = -I. -I../../ma -I$(EUIH)
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX -DEUIH
      FLD_REN = -b  rename:.lockrnc_,.lockrnc
else
           CC = mpcc_rnc
           FC = mpxlf_rnc
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX
endif

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

      DEFINES = $(GLOB_DEFINES) $(LOC_DEFINES) $(DEF_TRACE)
     INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)
       FFLAGS = $(FOPT) $(FOPT_REN) $(INCLUDES) $(DEFINES)
       CFLAGS = $(COPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)
       FLDOPT = $(FOPT) $(FLD_REN)
       CLDOPT = $(COPT) $(CLD_REN)
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

