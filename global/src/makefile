# Global Array library (shared and distributed memory version)
#
# Type "make libglobal.a" to create the GA library
#
#........................................................................

ifndef TARGET
error:
	@echo "TARGET machine [SUN,SGI,IBM,SP1,IPSC,PARAGON,KSR] not defined "
	exit
endif

ifndef VERSION
error1:
	@echo " memory VERSION [SHMEM/DISMEM] not defined "
	exit
endif

INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON

ifeq ($(VERSION),SHMEM)
  LOC_DEFINES = -DSHMEM -DSYSV -DTCGMSG
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

$(OBJ):  $(TARGET).$(VERSION).stamp 

$(TARGET).$(VERSION).stamp: 
	$(RM) -f *.o *.a core *.*.stamp
	echo "" > $(TARGET).$(VERSION).stamp

clean:
	$(RM) -f *.o *.a core *.*.stamp

