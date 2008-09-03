#$Id: GNUmakefile,v 1.1.2.2 2007-07-05 15:29:56 vinod Exp $
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

SRCS = ../timing.c read_input.c compute.c
OBJS = timing.o read_input.o compute.o
.PHONY : all
all: cg.x

timing.o:
	$(CC) -g -c ../timing.c
compute.o:
	$(CC) -g -c -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) compute.c
read_input.o:
	$(CC) -g -c -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) read_input.c

%.x:%.c $(FULL_LIBRARY_PATH) $(OBJS)
	$(CC) -g -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIB_DISTRIB)/$(TARGET)/$(LIBRARY) $(LIBS) -lm

.PHONY : clean
clean: 
	$(RM) *.x *.o
