# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,SGITFP,IBM,DECOSF,SP1,IPSC,PARAGON,KSR,CRAY-T3D,CONVEX-SPP] not defined "
	exit
endif

INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON

include ../Makefile.h

ifdef IWAY
#       LOC_DEFINES = -DIWAY -DSOCKCONNECT
endif

ifeq ($(VERSION),SHMEM)
  ifneq ($(TARGET),CRAY-T3D)
    LOC_DEFINES += -DSYSV
    ifndef IWAY
      ifneq ($(TARGET),KSR)
        ifneq ($(TARGET),CONVEX-SPP)
            LOC_DEFINES += -DDATA_SERVER
        endif
      endif
    endif
  endif
endif

LIB_DISTRIB = ../../lib
LIBRARY = libglobal.a

include ../MakeFiles.h
include ../../config/makelib.h

$(OBJ_FRAGILE): %.o : %.c
	$(CC) $(NOPT) $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS) -c $<

$(patsubst %,$(LIBRARY_PATH)(%),$(GA_CORE))     : globalp.h global.h
$(patsubst %,$(LIBRARY_PATH)(%),global.core.o)  : global.core.h message.h
$(patsubst %,$(LIBRARY_PATH)(%),global.serv.o)  : message.h mem.ops.h
$(patsubst %,$(LIBRARY_PATH)(%),global.msg.o)   : message.h
$(patsubst %,$(LIBRARY_PATH)(%),global.alg.o)   : globalp.h global.h
$(patsubst %,$(LIBRARY_PATH)(%),ga_handler.o)   : interrupt.h message.h
$(patsubst %,$(LIBRARY_PATH)(%),hsort.scat.o)   : types.f2c.h
$(patsubst %,$(LIBRARY_PATH)(%),semaphores.o)   : semaphores.h types.f2c.h
$(patsubst %,$(LIBRARY_PATH)(%),shmalloc.o)     : shmalloc.h
$(patsubst %,$(LIBRARY_PATH)(%),iway.o)         : global.h message.h farg.h
$(patsubst %,$(LIBRARY_PATH)(%),net.o)          : global.h message.h farg.h
$(patsubst %,$(LIBRARY_PATH)(%),ga_diag.o)      : global.fh diag.fh 
global.h       : types.f2c.h cray.names.h
global.core.h  : interrupt.h mem.ops.h semaphores.h locks.h
