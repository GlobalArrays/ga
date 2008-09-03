#$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:42:13 vinod Exp $
include ../../../examples_config.h

#default message-passing library for ARMCI is MPI
ifndef MSG_COMMS
  MSG_COMMS=MPI
endif
LIBRARY = libarmci.a
LIB_DISTRIB = $(ARMCI_HOME)/../lib
#defines communication libs and symbols
include $(ARMCI_HOME)/../config/makemp.h
include $(ARMCI_HOME)/../config/makecoms.h
EXTRA_LIBS += $(MP_LIBS) $(COMM_LIBS)
LIBS += $(EXTRA_LIBS)
MP_DEFINES += -DARMCI_ENABLE_GPC_CALLS
#......................................................................
LIBRARY = libarmci.a
HARDCLEAN = yes
#......................................................................

.PHONY : all
all: computation_impact.x

%.x:%.c $(FULL_LIBRARY_PATH)
	$(CC) -g -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIB_DISTRIB)/$(TARGET)/$(LIBRARY) $(LIBS) -lm -lblas

.PHONY : clean
clean: 
	$(RM) *.x
