#
# Makefile for TCGMSG library on top of MPI
#
# Author: Jarek Nieplocha
# August 1995

#Either MACHINE or TARGET platform must be defined

ifndef MACHINE
ifdef TARGET
   MACHINE = $(TARGET)
else
        ERRMSG = "You must define MACHINE\\n"
endif
endif

MSG_COMMS=MPI
#defines communication libs and symbols
include ../armci/config/makemp.h
include ../armci/config/makecoms.h

LIBRARY = libtcgmsg-mpi.a
LIB_DISTRIB = ../lib
INCDIR = ../include
LIB_INCLUDES += -I$(INCDIR) -I../armci/src $(MP_INCLUDES)
LIB_DEFINES += -DARMCI
LIB_TARGETS = *.x

NEED_MEMALIGN = DECOSF CRAY-T3E CRAY-YMP HPUX HPUX64 LAPI IBM IBM64 NEC 

ifeq ($(MACHINE),$(findstring $(MACHINE),$(NEED_MEMALIGN)))
        EXTRA_OBJ= memalign.o
        LIB_DEFINES += -DNEED_MEMALIGN
endif
ifeq ($(TARGET),HPUX)
        EXTRA_OBJ += hpuxargs.o
endif
ifeq ($(TARGET),HPUX64)
        EXTRA_OBJ += hpuxargs.o
endif


OBJ = checkbyte.o	drand48.o	pbeginf.o	sizeof.o\
      collect.o		misc.o		p2p.o		pfilecopy.o\
      evlog.o		evon.o		random.o 	nxtval-armci.o wrap.o

      OBJ += $(EXTRA_OBJ) 

HEADERS = sndrcv.h srftoc.h msgtypesc.h msgtypesf.h tcgmsg.fh
ifdef GA_USE_VAMPIR
  HEADERS += tcgmsg_vampir.h ../ga_vt.h
  EXTRA_OBJ += tcgmsg_vampir.o ga_vt.o
endif

# This include file specifies how and where the library should be built
# LIBRARY_PATH  is the library location

include ../config/makefile.h
include ../config/makelib.h


ifeq ($(TARGET),CRAY-YMP)
pbeginf.o:
	$(CC) $(COPT) $(INCLUDES) $(DEFINES) -g -c pbeginf.c

endif

ifdef GA_USE_VAMPIR
ga_vt.o:	../ga_vt.c
	ln -s ../ga_vt.c ga_vt.c
	$(CC) $(CFLAGS) -c ga_vt.c
	rm ga_vt.c
endif

LIBS = -L$(LIBDIR) -larmci $(MP_LIBS) $(COMM_LIBS)

all:	test.x testf.x


test.x: test.o	$(LIBRARY_PATH)
	$(CC)  $(COPT_REN) -o $@ $^ $(LIBS) $(CLIBS)

testf.x: testf.o $(LIBRARY_PATH)
	$(FC) $(FLD_REN) $(FOPT_REN) -o $@ $^ $(LIBS)

hello.x: hello.o $(LIBRARY_PATH)
	$(CC)  $(COPT_REN) -o $@ $^ $(LIBS) $(CLIBS)

test_arcv.x: test_arcv.o $(LIBRARY_PATH)
	$(FC)  $(FOPT_REN) -o $@ $^ $(LIBS) $(CLIBS)

test_asyn.x: test_asyn.o $(LIBRARY_PATH)
	$(CC)  $(COPT_REN) -o $@ $^ $(LIBS) $(CLIBS)

$(patsubst %,$(LIBRARY_PATH)(%),$(OBJ))      : tcgmsgP.h
$(patsubst %,$(LIBRARY_PATH)(%),evon.o)      : evlog.h
$(patsubst %,$(LIBRARY_PATH)(%),evlog.o)     : evlog.h
$(patsubst %,$(LIBRARY_PATH)(%),pfilecopy.o) : msgtypesc.h
$(patsubst %,$(LIBRARY_PATH)(%),pbeginf.o))  : farg.h
tcgmsgP.h: srftoc.h
sndrcv.h: msgtypesc.h
srftoc.h: sndrcv.h
test.o hello.o: sndrcv.h
