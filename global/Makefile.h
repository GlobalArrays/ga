# Makefile.h, Thu May 26 15:01:41 PDT 1994
#
# Define TARGET to be the machine you wish to build for
# (one of SUN, SGI, IBM, KSR)
#
# Define VERSION of memory 
# (SHMEM/DISMEM) - on some machines you can have either
#
#
# common definitions (overwritten later if required)
#
           FC = f77
           CC = cc
         FOPT = -g 
         COPT = -g
           AR = ar
       RANLIB = ranlib
          CPP = /usr/lib/cpp
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
         MAKE = make
      INSTALL =
      ARFLAGS = rcv
    EXPLICITF = FALSE


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
GLOB_INCLUDES = -I../../ma
     WARNINGS = -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual \
		-Wwrite-strings
 GLOB_DEFINES = -DSUN
endif

#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
       RANLIB = echo
     FOPT_REN = -r8
GLOB_INCLUDES = -I../../ma
 GLOB_DEFINES = -DKSR
        CDEFS = -DEXT_INT
endif

#................................ SGI ......................................
#
ifeq ($(TARGET),SGI)
#
# SGI running IRIX
#
         HOME = /usr/people/jaroslaw
          SRC = $(HOME)/scf/src
#
       RANLIB = echo
      FLD_REN = -v -Wl,-U
GLOB_INCLUDES = -I../../ma
 GLOB_DEFINES = -DSGI
endif

#................................ IPSC ......................................
#
ifeq ($(TARGET),IPSC)
#
# DELTA/IPSC running NX
#
#
           FC = if77
           CC = icc
           AR = ar860
       RANLIB = echo
      INSTALL = rcp $@ delta1:

     FOPT_REN = -Knoieee -Mquad -Mreentrant -Mrecursive -node
     COPT_REN = -Knoieee -node
GLOB_INCLUDES = -I. -I../../ma
 GLOB_DEFINES = -DNX -DIPSC -DNO_BCOPY
    EXPLICITF = TRUE
endif

#.............................. PARAGON ......................................
#
ifeq ($(TARGET),PARAGON)
#
# PARAGON running OS>=1.2 with NX (crosscompilation on Sun)
#
           FC = if77
           CC = icc
           AR = ar860
       RANLIB = echo

     FOPT_REN = -Knoieee -Mquad -Mreentrant -Mrecursive -nx
     COPT_REN = -Knoieee -nx
GLOB_INCLUDES = -I. -I../../ma
 GLOB_DEFINES = -DPARAGON -DNX -DIPSC -DNO_BCOPY
    EXPLICITF = TRUE
endif
 
#.............................. SP1 .........................................
#
ifeq ($(TARGET),SP1)
#
# IBM SP1 under EUIH 

         EUIH = /usr/lpp/euih/eui
           FC = xlf
         MAKE = gnumake

GLOB_INCLUDES = -I. -I../../ma -I$(EUIH)
 GLOB_DEFINES = -DSP1 -DEXTNAME -DAIX
      FLD_REN = -b  rename:.lockrnc_,.lockrnc
     FOPT_REN = -qEXTNAME
    EXPLICITF = TRUE
endif
 
#.............................. IBM .........................................
#
ifeq ($(TARGET),IBM)
#
# IBM RS/6000 under AIX  
#
          FC = xlf
GLOB_INCLUDES = -I../../ma
GLOB_DEFINES = -DIBM -DEXTNAME -DAIX
    FOPT_REN = -qEXTNAME 
#    FLD_REN = -b rename:.dscal_,.dscal
   EXPLICITF = TRUE
endif

#
#.......................... other common defs ............................
#

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
	/bin/rm -f $*.f

.f.o:
	$(FC) -c $(FOPT) $(FOPT_REN)  $*.f

.F.f:	
	$(CPP) $(INCLUDES) $(DEFINES) < $*.F | sed '/^#/D' > $*.f

.c.o:
	$(CC) $(CFLAGS) -c $*.c
endif
