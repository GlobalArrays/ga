# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,SGITFP,IBM,DECOSF,SP1,IPSC,PARAGON,KSR,CRAY-T3D] not defined "
	exit
endif

INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON

include ../Makefile.h

ifeq ($(VERSION),SHMEM)
  ifneq ($(TARGET),CRAY-T3D)
    LOC_DEFINES = -DSYSV
    ifneq ($(TARGET),KSR)
       LOC_DEFINES += -DDATA_SERVER
    endif
  endif
else
  ifneq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
       LOC_DEFINES = -DDATA_SERVER
  endif
endif

include ../Makelib.h
include ../MakeFiles.h

   LIBRARY = ../libglobal.a


OBJ = $(GA_OBJ) $(GA_ALG) $(GA_UTIL)


all: $(GA_SYNC) $(LIBRARY)

$(LIBRARY):  $(OBJ) $(EXTRA)
	$(RM) -f $@
	$(AR) $(ARFLAGS) $@ $(OBJ) $(EXTRA) 
	$(RANLIB) $@

$(GA_SYNC): $(GA_SYNC:.o=.c)
	cc -g $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS) -c  $(GA_SYNC:.o=.c)

$(OBJ): $(TARGET).$(VERSION).stamp 

$(TARGET).$(VERSION).stamp: 
	$(RM) -f *.o *.a core *.*.stamp
	echo "" > $(TARGET).$(VERSION).stamp

#clean *.o  and crap left by MPL on SP1
clean:
	$(RM) -f *.o *.a core *.*.stamp mputil.mp*

