#$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:41:23 vinod Exp $
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
include $(ARMCI_HOME)/../config/makefile.h
include $(ARMCI_HOME)/../config/makelib.h

EXTRA_LIBS += $(MP_LIBS) $(COMM_LIBS)
LIBS += $(EXTRA_LIBS)
#......................................................................
LIBRARY = libarmci.a
HARDCLEAN = yes
#......................................................................

SRCS = ../timing.c
OBJS = timing.o
.PHONY : all
all: lu.x lu-block.x lu-b-bc.x

timing.o:
	$(CC) -g -c ../timing.c

%.x:%.c $(FULL_LIBRARY_PATH) $(OBJS)
	$(CC) -g -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIB_DISTRIB)/$(TARGET)/$(LIBRARY) $(LIBS) -lm

.PHONY : clean
clean: 
	$(RM) *.x *.o
