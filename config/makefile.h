# $Id: makefile.h,v 1.63 2002-01-30 01:15:26 d3h325 Exp $
# This is the main include file for GNU make. It is included by makefiles
# in most subdirectories of the package.
# It includes compiler flags, preprocessor and library definitions
#
# JN 03/31/2000
# 
# A note on the compiler optimization flags:
# The most aggressive flags should be set for ARMCI elsewhere.
# The code compiled with the flags set below is not floating point intensive.
# The only exception are a few lapack/blas calls used by some
# GA test programs but this should not be an issue here since
# real GA apps should use their own version of blas/lapack for best performance.
#

           FC = f77
           CC = cc
          FLD = $(FC)
          CLD = $(FLD)
           M4 = /usr/bin/m4
          CXX = CC
         FOPT = -O
         COPT = -O
         NOPT = -g
           AR = ar
           AS = as
       RANLIB = @echo
          CPP = /usr/lib/cpp -P
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
      RMFLAGS = -r
      INSTALL = @echo
      ARFLAGS = rcv
    EXPLICITF = FALSE
        MKDIR = mkdir
    MAKEFLAGS = -j 1
       LINK.f = $(FLD)
       LINK.c = $(CLD)
      LIBBLAS = -lblas
       P_FILE = YES
        CLIBS = -lm
          _FC = $(notdir $(FC))
          _CC = $(notdir $(CC))


 GLOB_DEFINES = -D$(TARGET)
     FCONVERT = $(CPP) $(CPP_FLAGS) $< > $*.f

ifdef OPTIMIZE
         FOPT = -O
         COPT = -O
endif

# to enable two underscores in fortran names, please define environment variable
# F2C_TWO_UNDERSCORES or uncomment the following line
#F2C_TWO_UNDERSCORES=1
#
#........................ SUN and Fujitsu Sparc/solaris ........................
#
ifeq ($(TARGET),SOLARIS)
          M4 = /usr/ccs/bin/m4
 ifeq ($(_CC),mpifcc)
       _CC = fcc
 endif
 ifeq ($(_FC),mpifrt)
       _FC = frt
 endif
 ifeq ($(_CC),cc)
     COPT_REN = -dalign
 endif
 ifeq ($(_FC),f77)
      FLD_REN = -xs
     FOPT_REN = -dalign
 endif
 ifeq ($(_FC),frt)
     FOPT_REN += -fw -Kfast -KV8PFMADD
     CMAIN = -Dmain=MAIN__
 endif
 ifeq ($(_CC),fcc)
      COPT_REN += -Kfast -KV8PFMADD
 endif
     ifdef LARGE_FILES
        LOC_LIBS += $(shell getconf LFS_LIBS)
     endif
endif
#
#    64-bit version
ifeq ($(TARGET),SOLARIS64)
        M4 = /usr/ccs/bin/m4
  ifeq ($(_CC),mpifcc)
       _CC = fcc
  endif
  ifeq ($(_FC),mpifrt)
       _FC = frt
  endif
  ifeq ($(_CC),fcc)
     COPT_REN = -Kfast -KV9FMADD
  else
     COPT_REN = -xarch=v9 -dalign
  endif
  ifeq ($(_FC),frt)
     FOPT_REN = -Kfast -KV9FMADD -CcdII8 -CcdLL8
     CMAIN = -Dmain=MAIN__
  else
     FOPT_REN = -xarch=v9 -dalign
ifdef USE_INTEGER4
     FOPT_REN += -xtypemap=real:64,double:64,integer:32
else
     FOPT_REN += -xtypemap=real:64,double:64,integer:64
endif
     FLD_REN = -xs
  endif
  ifdef LARGE_FILES
        LOC_LIBS += $(shell getconf LFS_LIBS)
  endif
 GLOB_DEFINES = -DSOLARIS -DSOLARIS64
ifdef USE_INTEGER4
else
        CDEFS = -DEXT_INT
endif
endif
#
#obsolete: SunOS 4.X
ifeq ($(TARGET),SUN)
           CC = gcc
     FOPT_REN = -Nl100 -dalign
       RANLIB = ranlib
endif
#
#................................ FUJITSU ..................................
#
#32-bit VPP5000
ifeq ($(TARGET),FUJITSU-VPP)
           FC = frt
     FOPT_REN = -Sw -KA32
     COPT_REN = -KA32
 GLOB_DEFINES = -DFUJITSU
        CMAIN = -Dmain=MAIN__
endif

#64-bit VPP5000
ifeq ($(TARGET),FUJITSU-VPP64)
           FC = frt
     FOPT_REN = -Sw -CcdII8 -CcdLL8
 GLOB_DEFINES = -DFUJITSU
        CDEFS = -DEXT_INT
        CMAIN = -Dmain=MAIN__
endif
#

#32-bit AP3000
ifeq ($(TARGET),FUJITSU-AP)
           CC = fcc
           FC = frt
     FOPT_REN = -fw
 GLOB_DEFINES = -DFUJITSU
endif
#
#................................ LINUX ....................................
# IBM PC running Linux
#
ifeq ($(TARGET),LINUX)
           CC = gcc
           FC = g77
          CPP = gcc -E -nostdinc -undef -P
       RANLIB = ranlib
         _CPU = $(shell uname -m |\
                 awk ' /sparc/ { print "sparc" }; /i*86/ { print "x86" } ' )

ifneq (,$(findstring mpif,$(_FC)))
         _FC = $(shell $(FC) -v 2>&1 | awk ' /g77 version/ { print "g77"; exit }; /pgf/ { print "pgf77" ; exit } ' )
endif
ifneq (,$(findstring mpicc,$(_CC)))
         _CC = $(shell $(CC) -v 2>&1 | awk ' /gcc version/ { print "gcc" ; exit  } ' )
endif
#
#              GNU compilers
ifeq ($(_CPU),x86)
     OPT_ALIGN = -malign-double
endif
ifeq ($(_CPU),786)
     OPT_ALIGN = -malign-double
endif
ifeq ($(_CC),gcc)
   ifeq ($(COPT),-O)
          COPT = -O2
     COPT_REN += -funroll-loops $(OPT_ALIGN)
   endif
endif
#
#           g77
ifeq ($(_FC),g77)
   ifeq ($(FOPT),-O)
           FOPT = -O2
      FOPT_REN += -funroll-loops -fomit-frame-pointer $(OPT_ALIGN)
   endif
else
#
#             PGI fortran compiler on intel
   ifneq (,$(findstring pgf,$(_FC)))
       CMAIN = -Dmain=MAIN_
       FOPT_REN = -Mdalign -Minform,warn -Mnolist -Minfo=loop -Munixlogical
       GLOB_DEFINES += -DPGLINUX
   endif
   ifneq (,$(findstring ifc,$(_FC)))
       FOPT_REN = -O3 -prefetch 
       GLOB_DEFINES += -DIFCLINUX
   endif
   ifneq (,$(findstring icc,$(_CC)))
       FOPT_REN = -O3 -prefetch 
       GLOB_DEFINES += -DIFCLINUX
   endif
endif

endif
#
#................................ LINUX64 ....................................
# Linux 64-bit
# Alphas running Linux
# using DEC compilers
# ia64 using Intel Compiler
# to cross compile on x86 type: make _CPU=ia64
ifeq ($(TARGET),LINUX64)
       RANLIB = echo
GLOB_DEFINES += -DLINUX 
ifdef USE_INTEGER4
    FOPT_REN += -i4  
else
GLOB_DEFINES += -DEXT_INT
    FOPT_REN +=-i8
endif
         _CPU = $(shell uname -m)
#
# IA64 --- only Intel fortran compiler supported
FOPT_REN += -cm -w90 -w95
ifeq  ($(_CPU),ia64)
           CC = ecc
           FC = efc
ifneq ($(FC),efc)
     FLD_REN =   -Wl,--relax  -Wl,-Bstatic 
     CLD_REN =   -Wl,--relax  -Wl,-Bstatic 
endif
ifneq (,$(findstring efc,$(_FC)))
      FLD_REN = -Vaxlib
    GLOB_DEFINES += -DIFCLINUX
endif
ifneq (,$(findstring sgif90,$(_FC)))
# FOPT and COPT = -O breaks in global.armci.c with sgi pro64 0.13
        FOPT= -O0
     FOPT_REN =  -macro_expand 
   GLOB_DEFINES += -DSGILINUX
endif
ifneq (,$(findstring sgicc,$(_CC)))
        COPT = -O0
endif
endif
#
# Alpha
ifeq  ($(_CPU),alpha)
           CC = ccc
           FC = fort
    FOPT_REN +=-align_dcommons -fpe3 -check nooverflow 
    FOPT_REN +=-assume accuracy_sensitive -check nopower -check nounderflow
ifndef F2C_TWO_UNDERSCORES
    FOPT_REN +=-assume no2underscore
endif
ifdef USE_INTEGER4
    FLD_REN +=  -Wl,-taso
    CLD_REN+= -Wl,-taso 
endif
        CLIBS = -lfor
endif
          CLD = $(CC)
endif

#............................. CYGNUS on Windows ..........................
#
ifeq ($(TARGET),CYGWIN)
           FC = g77
           CC = gcc
 GLOB_DEFINES = -DCYGWIN
     COPT_REN = -malign-double
       RANLIB = ranlib
endif
ifeq ($(TARGET),CYGNUS)
           FC = g77
           CC = gcc
 GLOB_DEFINES = -DLINUX -DCYGNUS
     COPT_REN = -malign-double
       RANLIB = ranlib
endif
#
ifeq ($(TARGET),INTERIX)
           FC = g77
           CC = gcc
     COPT_REN = -malign-double
endif
#
#
#................................ HP  ....................................
ifeq ($(TARGET),HPUX)
# free HP cc compiler is not up to the job: use gcc if no commercial version
#          CC = gcc
#          FC = fort77
           FC = f90

          CPP = /lib/cpp
    ifeq ($(FOPT),-O)
         FOPT = -O1
    endif
      FOPT_REN = +ppu
      COPT_REN = -Ae
        FLIBS = -lU77
 GLOB_DEFINES = -DHPUX -DEXTNAME
#   EXPLICITF = TRUE
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
endif
#
ifeq ($(TARGET),HPUX64)
# 64-bit version
         _CPU = $(shell uname -m)
           FC = f90
    ifeq ($(FOPT),-O)
         FOPT = -O1
    endif
     FOPT_REN = +ppu 
     COPT_REN = -Ae
ifeq  ($(_CPU),ia64)
     FOPT_REN = +DD64
     COPT_REN = +DD64
else
     FOPT_REN += +DA2.0W
     COPT_REN += +DA2.0W
endif
        FLIBS = -lU77
 GLOB_DEFINES+= -DHPUX -DEXTNAME
ifdef USE_INTEGER4
#     COPT_REN +=+u1 # this is to fix alignment problems
else
     FOPT_REN += +i8
        CDEFS = -DEXT_INT
endif
endif
#
#................................ Compaq/DEC ALPHA .............................
# we use a historical name
#
ifeq ($(TARGET),DECOSF)
     FOPT_REN = -fpe2 -check nounderflow -check nopower -check nooverflow
ifdef USE_INTEGER4
     FOPT_REN += -i4 
#    COPT_REN += -misalign # alignment fix
else
     FOPT_REN += -i8 
        CDEFS = -DEXT_INT
endif
       RANLIB = ranlib
        CLIBS = -lfor -lots -lm
          CLD = $(CC)
endif
#
#................................ SGI ......................................
#
ifeq ($(TARGET),SGI)
       RANLIB = echo
     COPT_REN = -32 
     FOPT_REN = -32 
     HAS_BLAS = yes
endif

ifeq ($(TARGET),SGI_N32)
       RANLIB = echo
 GLOB_DEFINES = -DSGI -DSGI_N32
     COPT_REN = -n32 -mips4
     FOPT_REN = -n32 -mips4
     HAS_BLAS = yes
endif

ifeq ($(TARGET),SGITFP)
       RANLIB = echo
        CDEFS = -DEXT_INT
     COPT_REN = -64 -mips4 
 GLOB_DEFINES = -DSGI -DSGITFP
     FOPT_REN = -i8 -align64 -64 -mips4 
endif

ifeq ($(TARGET),SGI64)
       RANLIB = echo
 GLOB_DEFINES = -DSGI -DSGI64
     COPT_REN = -64 -mips4 
     FOPT_REN = -align64 -64 -mips4
endif
#
#................................ CRAY ..................................
# covers also J90 and SV1
#
ifeq ($(TARGET),CRAY-YMP)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
     COPT_REN = -htaskprivate 
           FC = f90
          CPP = cpp -P -N
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
 GLOB_DEFINES = -DCRAY_YMP -D_MULTIP_
     FOPT_REN = -dp -ataskcommon
     HAS_BLAS = yes
      LIBBLAS = 
    EXPLICITF = TRUE
endif
#
ifeq ($(TARGET),CRAY-T3D)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
           FC = cf77
          CPP = /mpp/lib/cpp -P -N
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
     FOPT_REN = -Ccray-t3d -Wf-dp
    EXPLICITF = TRUE
endif
#
ifeq ($(TARGET),CRAY-T3E)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
           FC = f90
          CPP = cpp -P -N
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
     FOPT_REN = -dp
 GLOB_DEFINES = -DCRAY_T3D -DCRAY_T3E
    EXPLICITF = TRUE
endif
#................................. NEC SX-5 ..................................
ifeq ($(TARGET),NEC)
#
     FC = f90
     ifeq ($(FOPT), -O)
         FOPT = -Cvopt -Wf"-pvctl nomsg"
     endif
     ifeq ($(COPT), -O)
         COPT = -O nomsg -pvctl,nomsg
     endif
     FOPT_REN = -ew
     CDEFS    = -DEXT_INT
     CLIBS    = -li90sx
endif
#
#.............................. IBM .........................................
# LAPI is the primary target for SP
#
ifeq ($(TARGET),LAPI)
         IBM_ = 1
         FLD  = mpcc_r -lxlf -lxlf90 -lm
           CC = mpcc_r
GLOB_DEFINES += -DSP
endif

ifeq ($(TARGET),LAPI64)
         IBM_ = 1
         FLD  = mpcc_r -lxlf -lxlf90 -lm
           CC = mpcc_r
     FOPT_REN = -q64 
     COPT_REN = -q64
ifdef USE_INTEGER4
   FOPT_REN += -qintsize=4
else
   FOPT_REN += -qintsize=8
        CDEFS = -DEXT_INT
endif
      ARFLAGS = -rcv 
      AR = ar -X 64
GLOB_DEFINES += -DSP -DLAPI
endif

#....................
ifeq ($(TARGET),SP1)
#
         IBM_ = 1
         FLD  = mpxlf
           CC = mpcc
endif
#....................
ifeq ($(TARGET),SP)
#
         IBM_ = 1
         FLD  = mpxlf
           CC = mpcc

# need to strip symbol table to alleviate a bug in AIX 4.1 ld
define AIX4_RANLIB
  ranlib $@
  strip
endef
       RANLIB = $(AIX4_RANLIB)
endif

ifeq ($(TARGET),IBM)
# IBM RS/6000 under AIX  
#
         IBM_ = 1
GLOB_DEFINES =
endif

ifeq ($(TARGET),IBM64)
# 64-bit port, 8-byte fortran integers
         IBM_ = 1
     FOPT_REN = -q64 
     COPT_REN = -q64
ifdef USE_INTEGER4
   FOPT_REN += -qintsize=4
else
   FOPT_REN += -qintsize=8
        CDEFS = -DEXT_INT
endif
      ARFLAGS = -rcv -X 64
endif


ifdef IBM_
           FC = xlf
     FOPT_REN += -qEXTNAME -qarch=auto
GLOB_DEFINES += -DIBM -DAIX
       CDEFS += -DEXTNAME
    EXPLICITF = TRUE
# we compile blas to avoid headache with missing underscores in the IBM library
# testsolve.x uses several blas routines
#     HAS_BLAS = yes
endif

#
#.............................. final flags ....................................
#

#get rid of 2nd underscore under g77
ifeq ($(_FC),g77)
ifndef F2C_TWO_UNDERSCORES
     FOPT_REN += -fno-second-underscore
endif
     ifndef OLD_G77
        FOPT_REN += -Wno-globals
     endif
endif

#add 2nd underscore under linux/cygwin to match g77 names
ifdef F2C_TWO_UNDERSCORES
     CDEFS += -DF2C2_
endif

       DEFINES = $(GLOB_DEFINES) $(LIB_DEFINES)

ifeq ($(MSG_COMMS),MPI)
  INCLUDES += $(MP_INCLUDES)
  DEFINES += -DMPI
endif

#Fujitsu fortran compiler requires -Wp prefix for cpp symbols
ifeq ($(TARGET),FUJITSU-VPP)
       comma:= ,
       empty:=
       space:= $(empty) $(empty)
       FDEFINES_0 = $(strip  $(DEFINES))
       FDEFINES = -Wp,$(subst $(space),$(comma),$(FDEFINES_0))
else
       FDEFINES = $(DEFINES)
endif


       INCLUDES += $(LIB_INCLUDES)
       CPP_FLAGS += $(INCLUDES) $(FDEFINES)

       FFLAGS = $(FOPT) $(FOPT_REN) 
       CFLAGS = $(INCLUDES) $(DEFINES) $(COPT) $(COPT_REN) $(CDEFS) $(LIB_CDEFS)
       CFLAGS := $(strip $(CFLAGS))
       FFLAGS := $(strip $(FFLAGS))
       FLDOPT =  $(FLD_REN)
       CLDOPT =  $(CLD_REN)

ifeq ($(LINK.f),$(FC))
       FLDOPT += $(FOPT_REN)
else
       FLDOPT += $(COPT_REN)
endif
ifeq ($(LINK.c),$(FC))
       CLDOPT += $(FOPT_REN)
else
       CLDOPT += $(COPT_REN)
endif

#
# Define known suffixes mostly so that .p files don't cause pc to be invoked
#
.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c .m4

ifeq ($(EXPLICITF), TRUE)
#
# Needed on machines where FCC does not preprocess .F files
# with CPP to get .f files
#
.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c .m4

.m4.o:
	$(M4) $*.m4 > $*.F
	$(MAKE) $*.f
	$(FC) $(FOPT_REN) -c $*.f
	$(RM) -f $*.F $*.f

.F.o:	
	@echo Converting $*.F '->' $*.f
	@$(FCONVERT)
	$(FC) -c $(FFLAGS) $*.f
	@$(RM) $*.f

.F.f:
	@echo Converting $*.F '->' $*.f
	$(FCONVERT)
else

.SUFFIXES:      .m4

.m4.o:
	$(M4) $*.m4 > $*.F
	$(FC) $(CPP_FLAGS) $(FOPT_REN) -c $*.F -o $*.o
	$(RM) $*.F

endif

# 
# More explicit rules to avoid infinite recursion, to get dependencies, and
# for efficiency.  CRAY does not like -o with -c.

%.o:	%.F
ifeq ($(EXPLICITF),TRUE)
	@echo Converting $< '->' $*.f
	$(FCONVERT)
ifeq (CRAY,$(findstring CRAY,$(TARGET)))
	$(FC) -c $(FFLAGS) $*.f
else
	$(FC) -c $(FFLAGS) -o $@ $*.f
endif
	@/bin/rm -f $*.f
else
	$(FC) -c $(FFLAGS) $(CPP_FLAGS) $<
endif

ifeq (CRAY,$(findstring CRAY,$(TARGET)))
%.o:	%.f
	$(FC) -c $(FFLAGS) $*.f
endif

#
#.................. libraries for test programs ...............................
# Almost every library in the package contains its test programs.
# LIBS contains definitions of libraries used by these programs.
# LOC_LIBS defines extra libraries required by test programs for each library 
# This is rather complicated because of all different configurations and 
# options supported:
# We create list of libs needed by test programs in each of
# the subdirectories by concatenating library definitions for
# linear algebra, ARMCI, message-passing library, and any lower level libs
#
# core libs
LIBS = -L$(LIB_DISTRIB)/$(TARGET) -lglobal -lma 
#
#linear algebra
ifdef USE_SCALAPACK
  LIBS += $(SCALAPACK)
endif
LIBS += -llinalg $(LOC_LIBS)
ifeq ($(HAS_BLAS),yes)
  LIBS += $(LIBBLAS)
endif
#
#communication libs
LIBS += -larmci

ifdef USE_MPI
  ifndef LIBMPI
      LIBMPI = -lmpi
  endif
  ifdef MPI_LIB
      LIBS += -L$(MPI_LIB)
  endif
  LIBS += -ltcgmsg-mpi $(LIBMPI)
else
  ifeq ($(MSG_COMMS),MPI)
    LIBS += $(MP_LIBS)
  else
    LIBS += -ltcgmsg 
  endif
endif

# lower level libs used by communication libraries
ifdef COMM_LIBS
  LIBS += $(COMM_LIBS)
endif

LIBS += -lm
#........................... End ..............................................
