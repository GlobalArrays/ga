#$id$
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


SRCS = DistHashmap.cc GPCHashmap.cc GPCHashmapHandler.cc HashFunctions.cc \
	HashUtil.cc

OBJS = DistHashmap.o GPCHashmap.o GPCHashmapHandler.o HashFunctions.o \
	HashUtil.o test_hashtable.o

CXXFLAGS=-g -Wall -DMPI $(INCS) -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES)

.PHONY : all
all: test_hashtable

#%.o:%.cc
#	g++ -g -Wall -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -c $(SRCS)

test_hashtable:$(OBJS)
	g++ -g -Wall -I$(ARMCI_HOME) $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIB_DISTRIB)/$(TARGET)/$(LIBRARY) $(OBJS) $(LIBS) -lm

.PHONY : clean
clean:
	$(RM) -f *~ *.o test_hashtable.x
