#........................................................................
# Lower-level makefile for building libglobal.a. 
# 
# JN
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine not defined (see README file for list of valid targets" 
	exit
endif

# We can remove entire library when cleaning rather than individual objects
#
HARDCLEAN = yes

LIB_DISTRIB = ../../lib
INCDIR = ../../include
HEADERS = ga.h global.h global.fh c.names.h cray.names.h

#upper level might want to override library name
ifndef LIBRARY
  LIBRARY = libglobal.a
endif

ifeq ($(GA_TRACE), YES)
    DEF_TRACE = -DGA_TRACE
endif

include ../MakeFiles.h


ifdef OLD_GA
#............................ old version of GA (pre 3.0) ...................
INTERRUPT_AVAILABLE = SP1 SP IPSC DELTA PARAGON LAPI
DATA-SERVER = SUN SGI SGI_N32 SGITFP SGI64 HPUX SOLARIS IBM DECOSF LINUX
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP LAPI FUJITSU-VPP CYGNUS
#
ifdef IWAY
       LOC_DEFINES = -DIWAY -DSOCKCONNECT
endif
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
LOC_DEFINES += $(DEF_TRACE)
include ../Makefile.h

else

# ............................... ARMCI based version ........................
# you can restore the old threshold for default decomposition in ga_create
# by uncommenting the following statement
# LIB_DEFINES += -DOLD_DEFAULT_BLK

  LIB_DEFINES += -DARMCI $(DEF_TRACE)
  LIB_INCLUDES += -I../../armci/src -I$(INCDIR)

#we USE_MPI requires us to acces MPI headers to handle ga_mpi_communicator
ifdef USE_MPI
  MSG_COMMS = MPI
endif
ifeq ($(MSG_COMMS),MPI)
include ../../armci/config/makemp.h
LIB_INCLUDES += $(MP_INCLUDES)
LIB_DEFINES += -DMPI
endif
include ../../config/makefile.h
endif



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
$(patsubst %,$(LIBRARY_PATH)(%),global.armci.o) : global.h global.armci.h 
global.h       : cray.names.h
global.core.h  : interrupt.h mem.ops.h semaphores.h locks.h lapidefs.h
