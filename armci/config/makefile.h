           FC = f77
           CC = cc
           AR = ar
           AS = as
       RANLIB = echo
          CPP = /usr/lib/cpp
        SHELL = /bin/sh
           MV = /bin/mv
           RM = /bin/rm
      RMFLAGS = -r
      INSTALL = @echo
      ARFLAGS = rcv
        MKDIR = mkdir
       LINK.f = $(FLD)
       LINK.c = $(CLD)
 GLOB_DEFINES = -D$(TARGET)


ifeq ($(TARGET),LINUX)
     FOPT_REN = -fno-second-underscore
           FC = g77
           CC = gcc
       RANLIB = ranlib
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
endif

ifeq ($(TARGET),SOLARIS)
     FLD_REN  = -xs
endif

ifeq ($(TARGET),HPUX)
# HP compiler can generate bad code for shared memory data
# use gcc if cc breaks
#          CC = gcc
           FC = fort77
          CPP = /lib/cpp
    ifeq ($(FOPT),-O)
         FOPT = -O1
    endif
     FOPT_REN = +ppu
     COPT_REN = -Ae
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
 GLOB_DEFINES = -DCONVEX
endif

ifeq ($(TARGET),SGI)
     COPT_REN = -n32 -mips4
     FOPT_REN = -n32 -mips4
endif

ifeq ($(TARGET),SGI64)
       RANLIB = echo
     COPT_REN = -64 -mips4
     FOPT_REN = -align64 -64 -mips4 -OPT:IEEE_arithmetic=2:fold_arith_limit=4000
 GLOB_DEFINES +=-DSGI
endif


ifeq ($(TARGET),CRAY-YMP)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
     COPT_REN = -htaskprivate $(LIBCM)
           FC = f90
 GLOB_DEFINES = -DCRAY_YMP
     FOPT_REN = -dp -ataskcommon $(LIBCM)
endif


ifeq ($(TARGET),CRAY-T3D)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
           FC = cf77
 GLOB_DEFINES = -DCRAY_T3D
endif


ifeq ($(TARGET),CRAY-T3E)
     ifeq ($(FOPT), -O)
         FOPT = -O1
     endif
           FC = f90
     FOPT_REN = -dp
 GLOB_DEFINES = -DCRAY_T3E
endif


ifeq ($(TARGET),PARAGON)
     FOPT_REN = -nx
     COPT_REN = -nx -Msafeptr
           FC = if77
           CC = icc
           AR = ar860
           AS = as860
     FOPT_REN += -Knoieee -Mquad -Mreentrant -Mrecursive
     COPT_REN += -Knoieee -Mquad -Mreentrant
 GLOB_DEFINES += -DNX
endif


ifeq ($(TARGET),LAPI)
         IBM  = 1
          CC  = mpcc_r
      LINK.f  = mpcc_r -lc_r -lxlf -lxlf90 -lm
GLOB_DEFINES += -DSP
endif

ifeq ($(TARGET),IBM)
# IBM RS/6000 under AIX
#
         IBM  = 1
endif

ifdef IBM
           FC = xlf
GLOB_DEFINES  += -DAIX
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

ifeq (CRAY,$(findstring CRAY,$(TARGET)))
%.o:    %.f
	$(FC) -c $(FFLAGS) $*.f
endif
