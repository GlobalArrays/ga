# Makefile.h, Thu May 26 15:01:41 PDT 1994
#
# Define TARGET to be the machine you wish to build for
# (one of SUN, SGI, SGITFP, IBM, KSR, SP1, CRAY-T3D, IPSC, DELTA, PARAGON)
#
# Define VERSION of memory 
# (SHMEM/DISMEM) - on some machines you can have either
#
#
# common definitions (overwritten later if required)
#
           FC = f77
           CC = cc
         FOPT = -O
         COPT = -O
GLOB_INCLUDES = -I../../ma -I.
           AR = ar
       RANLIB = echo
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
#
#          CC = gcc
     FOPT_REN = -Nl100
     COPT_REN = 
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
         FOPT = -g 
       P_FILE = NO
#GLOB_INCLUDES = -I../../ma -I$(LIBSMA)
     FOPT_REN = -Ccray-t3d -Wf-dp -Wl"-Drdahead=on" 
     COPT_REN = -Wl"-Drdahead=on" 
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
         COPT = -g
        CDEFS = -DEXT_INT
     FOPT_REN = -d8 -i8 -64 -mips4 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000 
 GLOB_DEFINES = -DSGI -DSGITFP
endif


#............................. IPSC/DELTA/PARAGON .............................
#
ifeq ($(TARGET),IPSC)
#
# IPSC running NX
#

     FOPT_REN = -node
     COPT_REN = -node
 GLOB_DEFINES = -DNX
        INTEL = YES
      INSTALL = @echo "See TCGMSG README file on how to run program"
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
 GLOB_DEFINES = -DNX
      INSTALL = rcp $* delta1: 
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
     COPT_REN = -nx
 GLOB_DEFINES = -DPARAGON -DNX
endif
#
ifeq ($(INTEL),YES)
#
# all Intel machines
#
           FC = if77
           CC = icc
           AR = ar860
       P_FILE = NO
     FOPT_REN += -Knoieee -Mquad -Mreentrant -Mrecursive
     COPT_REN += -Knoieee
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

      DEFINES = $(GLOB_DEFINES)  $(LOC_DEFINES) $(DEF_TRACE)
     INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)
       FFLAGS = $(FOPT) $(FOPT_REN) $(INCLUDES) $(DEFINES)
       CFLAGS = $(COPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS)
       FLDOPT = $(FOPT) $(FLD_REN)
       CLDOPT = $(COPT) $(CLD_REN)


ifeq ($(EXPLICITF),TRUE)
#
# Needed on machines where FCC does not preprocess .F files
# with CPP to get .f files
#
.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c

.F.o:	
	$(MAKE) $*.f
	$(FC) -c $(FOPT) $(FOPT_REN)  $*.f
	$(RM) -f $*.f

.f.o:
	$(FC) -c $(FOPT) $(FOPT_REN)  $*.f

.F.f:	
	$(CPP) $(INCLUDES) $(DEFINES) < $*.F | sed '/^#/D' > $*.f

.c.o:
	$(CC) $(CFLAGS) -c $*.c
endif

