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
  TESTOBJ = test.isend.o


      OBJ_CORE = pbegin.o queues.o globalop.o usleep.o nnodes.o nodeid.o \
                 synch.o  waitall.o drand48.o random.o\
                 mitob.o mdtob.o mitod.o mdtoi.o misc.o snd_rcv_probe.o mtime.o

ifeq ($(MACHINE),SUN)
       CC = gcc
   CFLAGS = -g -Wall -DSHMEM -DSYSV -DSUN
   RANLIB = ranlib
    EXTRA = shmem.o error.o nxtval.shm.o busy.o async_send.o signals.o
endif


ifeq ($(MACHINE),SGITFP)
   CFLAGS = -O3 -fullwarn -DSHMEM -DSYSV -DSGI -DSGITFP
    EXTRA = shmem.o error.o nxtval.shm.o busy.o async_send.o signals.o
endif

ifeq ($(MACHINE),SGI)
   CFLAGS = -O -DSHMEM -DSYSV -DSGI -DNOSPIN -DNOTIFY_SENDER
    EXTRA = shmem.o error.o nxtval.shm.o busy.o async_send.o signals.o
endif


ifeq ($(MACHINE),LINUX)
       CC = gcc
   CFLAGS = -Wall -O -DSHMEM -DSYSV -DLINUX -DNOSPIN -DNOTIFY_SENDER
    EXTRA = shmem.o error.o nxtval.shm.o busy.o async_send.o signals.o
   RANLIB = ranlib
endif

ifeq ($(MACHINE),CYGNUS)
       CC = gcc
       FC = g77
   CFLAGS = -Wall -O -DSHMEM -DLINUX -DCYGNUS
    EXTRA =  error.o busy.o async_send.o nxtval.shm.o pbeginf.o memalign.o
endif

ifeq ($(MACHINE),LAPI)
       FC = mpxlf
       CC = mpcc_r
      FLD = $(CC)
   FFLAGS = -qEXTNAME
   CFLAGS = -O -DEXTNAME -DSHMEM -DAIX -DIBM -DLAPI -DNOTIFY_SENDER
    EXTRA = lapi_putget.o memalign.o async_send_lapi.o signals.o
  TESTOBJ = test.o
    FLIBS =  -lxlf90 -lxlf
LIBLAPIDIR = /usr/lpp/ssp/css/lib
   LIBHAL = -lhal_r
  LIBLAPI = -llapi_r

  LDFLAGS = -L$(LIBLAPIDIR) $(LIBHAL) $(LIBLAPI) -bnso \
          -bI:/usr/lib/threads.exp -lppe_r \
          -bI:/usr/lpp/ssp/css/libus/fs_ext.exp \
          -bI:/usr/lib/syscalls.exp  -lm \
          -lppe -bl:new.map
endif


ifeq ($(MACHINE),CRAY-T3E)
     LINT = cclint
       FC = f90
 INCLUDES = -I.
  DEFINES = -DCRAY_T3D   -DCRAY_T3E -DCRAY -DSHMEM
   FFLAGS = -g -Wf"-dp -ojump"
   CFLAGS = -O3 -h inline3 $(DEFINES) $(INCLUDES)
    EXTRA = t3d.o snd_rcv_probe.o

 OBJ_CORE = pbegin.o queues.o  usleep.o mitob.o mdtob.o\
            async_send.o mtime.o memalign.o drand48.o busy.o \
            random.o mitod.o mdtoi.o
endif

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

#..........................................................................

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
            random.o mitod.o mdtoi.o
endif

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o


OBJ = $(OBJ_CORE) $(EXTRA)  
LIB_TARGETS = *.x pvm.ring
LIB_DISTRIB = ../../lib
LIBRARY = libtcgmsg.a
HEADERS = sndrcv.h srftoc.h msgtypesc.h msgtypesf.h ../tcgmsg.fh
INCDIR  = ../../include
include ../../config/makelib.h

#......................... make targets ...................................
#

all:	 $(LIBRARY_PATH) test.x

testf:  testf.o $(LIBRARY_PATH)
	$(FC) $(LOPT) -o $@ $^ $(LIBS) $(FLIBS)

test.x:	$(TESTOBJ) checkbyte.o $(LIBRARY_PATH)
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
test.x: $(LIBRARY_PATH)



