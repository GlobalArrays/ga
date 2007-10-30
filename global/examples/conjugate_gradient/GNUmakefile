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

%.x :  %.o matvecmul.o computeloops.o read_input.o $(LIBDIR)/libglobal.a $(LIBDIR)/libarmci.a  $(EXTRA)
ifeq ($(P_FILE),YES)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
endif
#	$(FC)  -o $@ computeloops.o matvecmul.o read_input.o $< $(LIBS) $(FLIBS);
	$(LINK.c) $(NOPT) $(CLDOPT) -o $@ computeloops.o matvecmul.o read_input.o $< $(LIBS) $(CLIBS);

%.p:
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@


.PHONY: clean
clean:
	-$(RM) -f *.o *.p *.exe *.obj *core *stamp *trace *.x *events* *~
	-$(RM) -rf ./obj

.PHONY: realclean
realclean:      clean
	-$(RM) -rf *~ \#*\#

.PHONY: cleanstamp
cleanstamp:
	-$(RM) -rf *.stamp
