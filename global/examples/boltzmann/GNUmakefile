 LIB_DISTRIB = ../../../lib
 LIB_TARGETS = *.x *.p
      INCDIR = ../../../include
LIB_INCLUDES = -I$(INCDIR)

include ../../../armci/config/makecoms.h
include ../../../armci/config/makemp.h
include ../../../config/makefile.h

ifndef LIBDIR
   LIBDIR = $(LIB_DISTRIB)/$(TARGET)
endif

ifeq ($(MSG_COMMS), MPI)
  P_FILE = no
endif
ifdef USE_MPI
  P_FILE = no
endif

all: boltz.x

BINARIES = main.o boltzmann.o setup.o initpar.o properties.o\
           equil.o get_patch.o timestep.o printdat.o vorticity.o

%.x :  $(BINARIES) $(LIBDIR)/libglobal.a $(LIBDIR)/libarmci.a  $(EXTRA)
ifeq ($(P_FILE),YES)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
endif
	$(LINK.f) $(NOPT) $(FLDOPT) -o $@ $(BINARIES) $(LIBS) $(FLIBS);

%.p:
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@


.PHONY: clean
clean:
	-$(RM) -f *.o *.p *.exe *.obj *core *stamp *trace *.x *events* bltz.gmv
	-$(RM) -rf ./obj

.PHONY: realclean
realclean:      clean
	-$(RM) -rf *~ \#*\#

.PHONY: cleanstamp
cleanstamp:
	-$(RM) -rf *.stamp
