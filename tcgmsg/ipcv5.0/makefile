ifdef TARGET
    MACHINE = $(TARGET)
endif

ifndef MACHINE
error:
	@echo "You must define MACHINE"
	exit
endif

       CC = cc
   RANLIB = echo
       RM = rm
MAKEFLAGS = -j1


      OBJ_CORE = pbegin.o queues.o shmem.o usleep.o nnodes.o nodeid.o \
                 signals.o synch.o error.o async_send.o waitall.o busy.o\
                 mitob.o mdtob.o

ifeq ($(MACHINE),SUN)
       CC = gcc
   CFLAGS = -g -Wall -DSHMEM -DSYSV -DSUN
   RANLIB = ranlib
endif

ifeq ($(MACHINE),KSR)
   CFLAGS = -g -DSHMEM -DSYSV -DKSR
    EXTRA = copyall.o
endif

ifeq ($(MACHINE),SGITFP)
   CFLAGS = -O3 -fullwarn -DSHMEM -DSYSV -DSGI -DSGITFP
endif


ifeq ($(MACHINE),CRAY-T3D)
     LINT = cclint
       FC = cf77
 INCLUDES = -I.
  DEFINES = -DCRAY_T3D -DCRAY -DSHMEM
# -DFLUSHCACHE
   FFLAGS =  -Ccray-t3d -g -Wf"-dp -ojump"
#  CFLAGS = -g $(DEFINES) $(INCLUDES) -Wl"-Drdahead=on"
   CFLAGS = -O3 -h inline3 $(DEFINES) $(INCLUDES) -Tcray-t3d -hjump
     LOPT = -Wl"-Drdahead=on"
# -happrentice
#   LIBS  = -lapp
    EXTRA = t3d.o snd_rcv_probe.o

 OBJ_CORE = pbegin.o queues.o  usleep.o mitob.o mdtob.o\
            async_send.o mtime.o memalign.o drand48.o busy.o \
            random.o
endif

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o
#..........................................................................

OBJ = $(OBJ_CORE) $(EXTRA)  
LIB_TARGETS = *.x pvm.ring
LIB_DISTRIB = ../../lib
LIBRARY = libtcgmsg.a
include ../../config/makelib.h

#......................... make targets ...................................
#

all:	 $(LIBRARY_PATH) test.x

test.x:	test.isend.o checkbyte.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) $(LOPT) -o $@ $^ $(LIBS)

hello.x: hello.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) $(LOPT) -o $@ $^ $(LIBS)

pvm.ring: pvm.o checkbyte.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) $(LOPT) -o $@ $^ $(LIBS)

jacobi.x: jacobi.o $(JOBJ) $(LIBRARY_PATH)
	$(FC) $(FFLAGS) $(LOPT) -o $@ $^ $(LIBS)


lint:
	$(LINT) $(DEFINES) $(INCLUDES) test.isend.c t3d.c queues.c async_send.c\
                                       snd_rcv_probe.c

.f.o:
	$(FC) -c $(FFLAGS)  $*.f

$(patsubst %,$(LIBRARY_PATH)(%),drand48.o): srftoc.h
$(patsubst %,$(LIBRARY_PATH)(%),t3d.o): srftoc.h evlog.h
$(patsubst %,$(LIBRARY_PATH)(%),usleep.o): tcgmsgP.h
$(patsubst %,$(LIBRARY_PATH)(%),error.o): tcgmsgP.h
$(patsubst %,$(LIBRARY_PATH)(%),waitall.o): tcgmsg.h
$(patsubst %,$(LIBRARY_PATH)(%),snd_rcv_probe.o): tcgmsg.h tcgmsgP.h srftoc.h
$(patsubst %,$(LIBRARY_PATH)(%),mdtob.o): sndrcv.h
$(patsubst %,$(LIBRARY_PATH)(%),evlog.o): sndrcv.h evlog.h

test.isend.o: sndrcv.h
hello.o: sndrcv.h
pvm.o: sndrcv.h




