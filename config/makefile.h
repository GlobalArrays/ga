# $Id: makefile.h,v 1.24 2000-05-10 23:07:24 d3h325 Exp $
# This is the main include file for GNU make. It is included by makefiles
# in most subdirectories of the package.
# It includes compiler flags, preprocessor and library definitions
#
# JN 03/31/2000

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


 GLOB_DEFINES = -D$(TARGET)
     FCONVERT = $(CPP) $(CPP_FLAGS) $< > $*.f

ifdef OPTIMIZE
         FOPT = -O
         COPT = -O
endif
#
#................................ SUN ......................................
#
ifeq ($(TARGET),SOLARIS)
          M4 = /usr/ccs/bin/m4
     FLD_REN = -xs
     COPT_REN = -dalign
     FOPT_REN = -dalign
     ifdef LARGE_FILES
        LOC_LIBS += $(shell getconf LFS_LIBS)
     endif
endif
ifeq ($(TARGET),SOLARIS64)
           M4 = /usr/ccs/bin/m4
      FLD_REN = -xs
     COPT_REN = -xarch=v9 -dalign
     FOPT_REN = -xarch=v9 -dalign -xtypemap=real:64,double:64,integer:64
     ifdef LARGE_FILES
        LOC_LIBS += $(shell getconf LFS_LIBS)
     endif
 GLOB_DEFINES = -DSOLARIS -DSOLARIS64
        CDEFS = -DEXT_INT
endif
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
endif

#64-bit VPP5000
ifeq ($(TARGET),FUJITSU-VPP64)
           FC = frt
     FOPT_REN = -Sw -CcdII8
 GLOB_DEFINES = -DFUJITSU
        CDEFS = -DEXT_INT
endif
#
#................................ LINUX ....................................
# IBM PC running Linux
#
ifeq ($(TARGET),LINUX)
           CC = gcc
          CPP = gcc -E -nostdinc -undef -P
       RANLIB = ranlib
 GLOB_DEFINES = -DLINUX
ifndef USE_F77
#    Linux with g77
     FOPT_REN = -fno-second-underscore
           FC = g77
else
    EXPLICITF = TRUE
     FCONVERT = @(/bin/cp $< .tmp.$$$$.c; \
		$(CPP) $(CPP_FLAGS) .tmp.$$$$.c  | sed '/^$$/d' > $*.f; \
	 	/bin/rm -f .tmp.$$$$.c ) || exit 1
endif

ifndef TARGET_CPU
  ifeq ($(FC),g77)
       FOPT_REN += -malign-double
  endif
  ifeq ($(CC),gcc)
       COPT_REN += -malign-double
  endif
endif
#
#                GNU compilers
ifeq ($(CC),gcc)
   ifeq ($(COPT),-O)
#        COPT = -O2
    COPT_REN += -funroll-loops
#   COPT_REN += -finline-functions -funroll-loops
   endif
endif
ifeq ($(FC),g77)
   ifeq ($(FOPT),-O)
#        FOPT = -O3
    FOPT_REN += -funroll-loops -fomit-frame-pointer
   endif
   #for 2.7.2 and earlier
   ifndef OLD_G77
      FOPT_REN += -Wno-globals
   endif
endif     
#
# Portland Group compilers
# for pentium
# FOPT_REN  += -tp p5  
# for Pentium Pro or Pentium II
# FOPT_REN  += -tp p6
#
ifeq ($(FC),pgf77)
       PGLINUX = 1 
endif
ifeq ($(FC),pgf90)
       PGLINUX = 1 
endif
ifdef PGLINUX
       FOPT_REN = -Mdalign -Minform,warn -Mnolist -Minfo=loop -Munixlogical
       GLOB_DEFINES += -DPGLINUX
endif

endif
#
#................................ LINUX64 ....................................
# Alphas running Linux
# using DEC compilers
#
ifeq ($(TARGET),LINUX64)
           CC = ccc
           FC = fort
       RANLIB = echo
 GLOB_DEFINES = -DLINUX -DLINUX64 -DEXT_INT -DNOAIO
FOPT_REN=-i8 -assume no2underscore -align_dcommons 
#COPT_REN= 
endif

#............................. CYGNUS on Windows ..........................
#
ifeq ($(TARGET),CYGNUS)
           FC = g77
           CC = gcc
 GLOB_DEFINES = -DLINUX -DCYGNUS
     FOPT_REN = -fno-second-underscore
     COPT_REN = -malign-double
       RANLIB = ranlib
endif
#
#................................ HP  ....................................
ifeq ($(TARGET),HPUX)
# free HP cc compiler is not up to the job: use gcc if no commercial version
#          CC = gcc
           FC = fort77
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
           FC = f90
    ifeq ($(FOPT),-O)
         FOPT = -O1
    endif
     FOPT_REN = +DA2.0W +ppu +i8
     COPT_REN = +DA2.0W -Ae
        FLIBS = -lU77
 GLOB_DEFINES+= -DHPUX -DEXTNAME
        CDEFS = -DEXT_INT
endif
#
#................................ Compaq/DEC ALPHA .............................
# we use historical name
#
ifeq ($(TARGET),DECOSF)
     FOPT_REN = -i8
        CDEFS = -DEXT_INT
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
     COPT_REN = -htaskprivate $(LIBCM) 
           FC = f90
          CPP = cpp -P -N
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
 GLOB_DEFINES = -DCRAY_YMP
     FOPT_REN = -dp -ataskcommon $(LIBCM)
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
     FOPT_REN = -q64 -qintsize=8
     COPT_REN = -q64
        CDEFS = -DEXT_INT
      ARFLAGS = -rcv -X 64
endif

ifdef IBM_
           FC = xlf
     FOPT_REN += -qEXTNAME -qarch=com
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
       DEFINES = $(GLOB_DEFINES) $(LIB_DEFINES)

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
	$(FC) $(CPP_FLAGS) -c $*.F -o $*.o
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
  LIBS += -ltcgmsg
endif
# lower level libs used by communication libraries
ifdef COMM_LIBS
  LIBS += $(COMM_LIBS)
endif

LIBS += -lm
#........................... End ..............................................
