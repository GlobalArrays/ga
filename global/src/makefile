# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,IBM,SP1,IPSC,PARAGON,KSR,CRAY-T3D] not defined "
	exit
endif

ifndef VERSION
error1:
	@echo " memory VERSION [SHMEM/DISMEM] not defined "
	exit
endif

INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON

ifeq ($(VERSION),SHMEM)
  ifneq ($(TARGET),CRAY-T3D)
    LOC_DEFINES = -DSHMEM -DSYSV -DTCGMSG
  else
    LOC_DEFINES = -DSHMEM
  endif
else
  ifeq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
       LOC_DEFINES = -DTCGMSG 
  else
       LOC_DEFINES = -DTCGMSG -DDATA_SERVER
  endif
endif

include ../Makefile.h
include ../Makelib.h
include ../MakeFiles.h

   LIBRARY = ../libglobal.a


OBJ = $(GA_OBJ) $(GA_ALG) $(GA_UTIL)


$(LIBRARY):  $(OBJ) $(EXTRA)
	$(RM) -f $@
	$(AR) $(ARFLAGS) $@ $(OBJ) $(EXTRA) 
	$(RANLIB) $@

$(GA_SYNC): 
	cc -g $(COPT_REN) $(INCLUDES) $(DEFINES) $(CDEFS) -c  $(GA_SYNC:.o=.c)

$(OBJ):  $(TARGET).$(VERSION).stamp 

$(TARGET).$(VERSION).stamp: 
	$(RM) -f *.o *.a core *.*.stamp
	echo "" > $(TARGET).$(VERSION).stamp

clean:
	$(RM) -f *.o *.a core *.*.stamp

