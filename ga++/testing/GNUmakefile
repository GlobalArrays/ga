

ifndef TARGET
error:
	@echo "TARGET machine not defined "
	@echo "-- one of [SUN,SGI,SGITFP,IBM,DECOSF,SP1,SP,IPSC,DELTA,PARAGON,KSR,CRAY-T3D,CONVEX-SPP,HPUX]"
	exit
endif

ifndef GA_C_CORE
error:
	@echo "GA_C_CORE not defined "
	exit
endif

ifeq ($(DIAG), PAR)
     LIB_DEFINES += -DPAR_DIAG
endif
ifdef USE_SCALAPACK
     LIB_DEFINES += -DSCALAPACK
endif
ifdef USE_SCALAPACK_I8
     LIB_DEFINES += -DSCALAPACK
endif

LIB_DISTRIB  = ../../lib
LIB_TARGETS  = *.x *.p
INCDIR       = ../../include
LIB_INCLUDES = -I$(INCDIR)

GXXLIBS = -lga++

include ../../armci/config/makecoms.h
include ../../armci/config/makemp.h
include ../../config/makefile.h

ifndef LIBDIR
   LIBDIR = $(LIB_DISTRIB)/$(TARGET)
endif

ifeq ($(MSG_COMMS), MPI)
  P_FILE = no
endif
ifdef USE_MPI
  P_FILE = no
endif

TESTUTIL = util.o

FOPT= $(NOPT)
COPT= $(NOPT)

#
# build test programs and .p files if needed
#
#.PRECIOUS: %.o



ifeq ($(P_FILE),YES)
%.x : %.o $(TESTUTIL) $(LIBDIR)/libga++.a $(LIBDIR)/libglobal.a \
		$(LIBDIR)/libarmci.a  $(EXTRA)
	echo "`whoami` `hostname` 4 `pwd`/$@ /tmp" > $@.p
	$(LINK.cc) $(NOPT) $(CXXLDOPT) -o $@ $< $(TESTUTIL) -L$(LIBDIR) \
		$(GXXLIBS) $(LIBS) $(CLIBS)

else
%.x : %.o $(TESTUTIL) $(LIBDIR)/libga++.a $(LIBDIR)/libglobal.a  \
		$(LIBDIR)/libarmci.a $(EXTRA)
	echo $(CXXLDOPT)
	$(LINK.cc) $(NOPT) $(CXXLDOPT) -o $@ $< $(TESTUTIL) -L$(LIBDIR) \
		$(GXXLIBS) $(LIBS) $(CLIBS)
endif


%.p:
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@



.PHONY: clean
clean:
	-$(RM) -f *.o *.p *.exe *.obj *core *stamp mputil.* *trace *.x *events*
	-$(RM) -rf ./obj

.PHONY: realclean
realclean:      clean
	-$(RM) -rf *~ \#*\#

.PHONY: cleanstamp
cleanstamp:
	-$(RM) -rf *.stamp


