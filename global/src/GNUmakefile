#........................................................................
# Lower-level makefile for building libglobal.a. 
# 
# JN
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,SGITFP,IBM,DECOSF,SP1,IPSC,PARAGON,KSR,CRAY-T3D,CONVEX-SPP,LAPI] not defined "
	exit
endif

INTERRUPT_AVAILABLE = SP1 SP IPSC DELTA PARAGON LAPI
DATA-SERVER = SUN SGI SGI_N32 SGITFP SGI64 HPUX SOLARIS IBM DECOSF LINUX
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP LAPI FUJITSU-VPP CYGNUS

include ../Makefile.h

ifdef IWAY
       LOC_DEFINES = -DIWAY -DSOCKCONNECT
endif



ifndef USE_ARMCI
ifeq ($(TARGET),$(findstring $(TARGET),$(DATA-SERVER)))
    ifndef IWAY
       LOC_DEFINES += -DDATA_SERVER
    endif
endif

ifeq ($(VERSION),SHMEM)
  ifneq ($(TARGET),$(findstring $(TARGET),$(ONESIDED_AVAILABLE)))
    LOC_DEFINES += -DSYSV
  endif
endif
endif



# We need to remove entire library when changing TARGET rather
# than individual objects 
#
HARDCLEAN = yes

LIB_DISTRIB = ../../lib

ifdef USE_ARMCI
	LOC_DEFINES = -DARMCI
	LOC_INCLUDES = -I../../armci/src
ifndef USE_MPI
	LOC_DEFINES = -DARMCI -DTCGMSG
endif
endif

INCDIR = ../../include
HEADERS = ga.h global.h global.fh c.names.h cray.names.h

#upper level might want to override library name
ifndef LIBRARY
  LIBRARY = libglobal.a
endif

include ../MakeFiles.h
include ../../config/makelib.h

$(OBJ_FRAGILE): %.o : %.c
	$(CC) $(NOPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS) -c $<

$(patsubst %,$(LIBRARY_PATH)(%),$(GA_CORE))     : globalp.h global.h
$(patsubst %,$(LIBRARY_PATH)(%),global.core.o)  : global.core.h message.h mem.ops.h lapidefs.h
$(patsubst %,$(LIBRARY_PATH)(%),global.serv.o)  : message.h mem.ops.h
$(patsubst %,$(LIBRARY_PATH)(%),global.msg.o)   : message.h
$(patsubst %,$(LIBRARY_PATH)(%),global.alg.o)   : globalp.h global.h
$(patsubst %,$(LIBRARY_PATH)(%),ga_handler.o)   : interrupt.h message.h
$(patsubst %,$(LIBRARY_PATH)(%),lapi.o)         : lapidefs.h
$(patsubst %,$(LIBRARY_PATH)(%),semaphores.o)   : semaphores.h
$(patsubst %,$(LIBRARY_PATH)(%),shmalloc.o)     : shmalloc.h
$(patsubst %,$(LIBRARY_PATH)(%),iway.o)         : global.h message.h farg.h
$(patsubst %,$(LIBRARY_PATH)(%),net.o)          : global.h message.h farg.h
$(patsubst %,$(LIBRARY_PATH)(%),ga_diag.o)      : global.fh diag.fh 
global.h       : cray.names.h
global.core.h  : interrupt.h mem.ops.h semaphores.h locks.h lapidefs.h
