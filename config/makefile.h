
           FC = f77
           CC = cc
          FLD = $(FC)
          CLD = $(FLD)
          CXX = CC
         FOPT = -g
         COPT = -g
           AR = ar
           AS = as
       RANLIB = @echo
          CPP = /usr/lib/cpp -P
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
      RMFLAGS = -f
      INSTALL = @echo
      ARFLAGS = rcv
    EXPLICITF = FALSE
        MKDIR = mkdir
    MAKEFLAGS = -j 1
       LINK.f = $(FLD)
       LINK.c = $(CLD)
      LIBBLAS = -lblas


 GLOB_DEFINES = -D$(TARGET)
     FCONVERT = $(CPP) $(CPP_FLAGS) $< > $*.f

ifdef OPTIMIZE
         FOPT = -O
         COPT = -O
endif


ifeq ($(TARGET),SUN)
           CC = gcc
     FOPT_REN = -Nl100 -dalign
       RANLIB = ranlib
endif

ifeq ($(TARGET),FUJITSU-VPP)
           FC = frt
     FOPT_REN = -Sw
 GLOB_DEFINES = -DFUJITSU
     HAS_BLAS = yes
     LIBBLAS = /usr/lang/lib/libblasvp.a
endif


ifeq ($(TARGET),SOLARIS)
     FLD_REN = -xs
endif


ifeq ($(TARGET),LINUX)
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
 GLOB_DEFINES = -DLINUX
          CPP = gcc -E -nostdinc -undef -P
       RANLIB = ranlib
endif


ifeq ($(TARGET),CYGNUS)
           FC = g77
           CC = gcc
 GLOB_DEFINES = -DLINUX -DCYGNUS
     FOPT_REN = -fno-second-underscore
       RANLIB = ranlib
endif


ifeq ($(TARGET),HPUX)
#          CC = gcc
           FC = fort77
          CPP = /lib/cpp
    ifeq ($(FOPT),-O)
         FOPT = -O1
    endif
     FOPT_REN = +ppu
     COPT_REN = -Ae
#   EXPLICITF = TRUE
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
endif


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
    COPT_REN  = -or none
     FOPT_REN = -ppu -or none
    EXPLICITF = TRUE
     FCONVERT = $(CPP) $(CPP_FLAGS)  $< | sed '/^\#/D'  > $*.f
 GLOB_DEFINES = -DCONVEX
endif


ifeq ($(TARGET),KSR)
       RANLIB = echo
     FOPT_REN = -r8
        CDEFS = -DEXT_INT
     HAS_BLAS = yes
endif

ifeq ($(TARGET),DECOSF)
     FOPT_REN = -i8
        CDEFS = -DEXT_INT
       RANLIB = ranlib
endif

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
#    COPT_REN = -fullwarn -64 
     COPT_REN = -64 -mips4 
     FOPT_REN = -i8 -align64 -64 -mips4 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000 
endif

ifeq ($(TARGET),SGI64)
       RANLIB = echo
     COPT_REN = -64 -mips4 
     FOPT_REN = -align64 -64 -mips4 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000 
endif


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


ifeq ($(TARGET),IPSC)
#
# IPSC running NX
#
        INTEL = YES
     FOPT_REN = -node
     COPT_REN = -node
      INSTALL = @echo "See TCGMSG README file on how to run program "
endif

ifeq ($(TARGET),DELTA)
#
# Delta
#
        INTEL = YES
     FOPT_REN = -node
     COPT_REN = -node
      INSTALL = @echo 
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
      INSTALL = rcp $@ delta2:
endif
#
#....................
#
ifeq ($(TARGET),PARAGON)
#
# PARAGON running OS>=1.2 with NX
#
        INTEL = YES
     FOPT_REN = -nx
     COPT_REN = -nx -Msafeptr
endif
#
ifeq ($(INTEL), YES)
#
# all Intel machines
#
           FC = if77
           CC = icc
           AR = ar860
           AS = as860
          CLD = $(CC)
 ifeq ($(FOPT),-O)
         FOPT = -O2
 endif
     FOPT_REN += -Knoieee -Mquad -Mreentrant -Mrecursive
     COPT_REN += -Knoieee -Mquad -Mreentrant
 GLOB_DEFINES += -DNX
  CUR_VERSION = DISMEM
     HAS_BLAS = yes
endif
 
ifeq ($(TARGET),LAPI)
         IBM  = 1
         FLD  = mpcc_r -lxlf -lxlf90 -lm
GLOB_DEFINES += -DSP
endif

#....................
ifeq ($(TARGET),SP1)
#
         IBM  = 1
         FLD  = mpxlf
endif
 
#....................
ifeq ($(TARGET),SP)
#
         IBM  = 1
         FLD  = mpxlf

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
         IBM = 1
GLOB_DEFINES =
endif

ifdef IBM
           FC = xlf
     FOPT_REN = -qEXTNAME -qarch=com
GLOB_DEFINES += -DIBM -DAIX
    EXPLICITF = TRUE
# we compile blas to avoid headache with missing underscores in the IBM library
# testsolve.x uses several blas routines
#     HAS_BLAS = yes
endif

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


#
# Define known suffixes mostly so that .p files don't cause pc to be invoked
#

.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c

ifeq ($(EXPLICITF), TRUE)
#
# Needed on machines where FCC does not preprocess .F files
# with CPP to get .f files
#
.SUFFIXES:	
.SUFFIXES:	.o .s .F .f .c

.F.o:	
	@echo Converting $*.F '->' $*.f
	@$(FCONVERT)
	$(FC) -c $(FFLAGS) $*.f
	@$(RM) $*.f

.F.f:
	@echo Converting $*.F '->' $*.f
	$(FCONVERT)
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

# LOC_LIBS defines extra libraries required to build test programs 

LIBS = -L$(LIB_DISTRIB)/$(TARGET) -lchemio -lglobal -lma -llinalg $(LOC_LIBS)
ifeq ($(HAS_BLAS),yes)
      LIBS += $(LIBBLAS)
endif
ifdef MPI_LIB
      LIBS += -L$(MPI_LIB)
endif

ifdef USE_ARMCI
LIBS += -larmci
endif

ifdef USE_MPI
LIBS += -ltcgmsg-mpi -lmpi
else
LIBS += -ltcgmsg
endif
