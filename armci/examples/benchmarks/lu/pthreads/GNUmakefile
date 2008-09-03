#$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:41:32 vinod Exp $
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
#......................................................................
LIBRARY = libarmci.a
HARDCLEAN = yes
#......................................................................

SRCS = ../timing.c barrier.c 
OBJS = timing.o barrier.o
.PHONY : all
all: lu-thread.x

timing.o:
	$(CC) -g -c ../timing.c

barrier.o:
	$(CC) -g -c barrier.c

%.x:%.c $(FULL_LIBRARY_PATH) $(OBJS)
	$(CC) -g -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIB_DISTRIB)/$(TARGET)/$(LIBRARY) $(LIBS) -lm -lpthread

.PHONY : clean
clean: 
	$(RM) *.x *.o
