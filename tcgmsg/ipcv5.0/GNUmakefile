# $Id: GNUmakefile,v 1.2 2001-05-08 17:42:12 edo Exp $
#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libtcgmsg.a
  LIB_DISTRIB = ../../lib
      OBJ_CORE = pbegin.o queues.o globalop.o usleep.o nnodes.o nodeid.o \
                 waitall.o drand48.o random.o pfilecopy.o\
                 mitob.o mdtob.o mitod.o mdtoi.o misc.o snd_rcv_probe.o\
                 mtime.o wrap.o
      OBJ_SYSV = shmem.o error.o nxtval.shm.o busy.o async_send.o signals.o pbeginf.o synch.o

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

HEADERS = sndrcv.h srftoc.h msgtypesc.h msgtypesf.h ../tcgmsg.fh

OBJ = $(OBJ_CORE) $(EXTRA)  
LIB_TARGETS = *.x pvm.ring

  LIB_TARGETS = *.x *.o
  INCDIR      = ../../include

ifeq ($(TARGET),SUN)
   LIB_DEFINES =  -DSHMEM -DSYSV -DSUN
    EXTRA = $(OBJ_SYSV)
endif

ifeq ($(TARGET),SGITFP)
   LIB_DEFINES = -DSHMEM -DSYSV -DSGI -DSGITFP
    EXTRA = $(OBJ_SYSV)
endif

ifeq ($(TARGET),SGI)
   LIB_DEFINES= -DSHMEM -DSYSV -DSGI -DNOSPIN -DNOTIFY_SENDER
    EXTRA = $(OBJ_SYSV)
endif


ifeq ($(TARGET),LINUX)
   LIB_DEFINES = -DSHMEM -DSYSV -DLINUX -DNOSPIN -DNOTIFY_SENDER
    EXTRA = $(OBJ_SYSV)
endif

ifeq ($(TARGET),CYGNUS)
   LIB_DEFINES = -DLINUX -DCYGNUS -DSHMEM
    EXTRA =  error.o busy.o async_send.o nxtval.shm.o memalign.o pbeginf.o synch.o
endif

ifeq ($(TARGET),INTERIX)
   LIB_DEFINES =   -DINTERIX -DSHMEM -DSYSV
    EXTRA = $(OBJ_SYSV) 
endif

ifeq ($(TARGET),LAPI)
   LIB_DEFINES =  -DSHMEM  -DNOTIFY_SENDER
    EXTRA = lapi_putget.o memalign.o async_send_lapi.o signals.o
  TESTOBJ = test.o
LIBLAPIDIR = /usr/lpp/ssp/css/lib
   LIBHAL = -lhal_r
  LIBLAPI = -llapi_r

  LDFLAGS = -L$(LIBLAPIDIR) $(LIBHAL) $(LIBLAPI) -bnso \
          -bI:/usr/lib/threads.exp -lppe_r \
          -bI:/usr/lpp/ssp/css/libus/fs_ext.exp \
          -bI:/usr/lib/syscalls.exp  -lm \
          -lppe -bl:new.map
endif

ifeq ($(TARGET),LAPI64)
   LIB_DEFINES = -O -DSHMEM  -DNOTIFY_SENDER
    EXTRA = lapi_putget.o memalign.o async_send_lapi.o signals.o
  TESTOBJ = test.o
    FLIBS = -lxlf90 -lxlf
     LIBS = $(LAPI64LIBS) #-llapi_r -lhal_r
endif


ifeq ($(TARGET),CRAY-T3E)
  LIB_DEFINES = -DCRAY -DSHMEM
    EXTRA = t3d.o snd_rcv_probe.o

 OBJ_CORE = pbegin.o queues.o  usleep.o mitob.o mdtob.o\
            async_send.o mtime.o memalign.o drand48.o busy.o \
            random.o mitod.o mdtoi.o
endif

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

#..........................................................................

ifeq ($(TARGET),CRAY-T3D)
  LIB_DEFINES = -DCRAY_T3D -DCRAY -DSHMEM
# -DFLUSHCACHE
    EXTRA = t3d.o snd_rcv_probe.o

 OBJ_CORE = pbegin.o queues.o  usleep.o mitob.o mdtob.o\
            async_send.o mtime.o memalign.o drand48.o busy.o \
            random.o mitod.o mdtoi.o
endif

     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

         

include ../../armci/config/makecoms.h
include ../../armci/config/makemp.h
include ../../config/makefile.h
include ../../config/makelib.h




#.......................... general ................................
ifndef BINDIR
       BINDIR = .
endif


LOADLIBES = $(LIBRARY_PATH) $(COMM_LIBS) 
#......................... make targets ........................
#
      all:    $(LIBRARY_PATH) test.x testf.x hello.x msgtypesf.h


hello.x: $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ hello.c  $(LOADLIBES)

test.x:	$(LIBRARY_PATH)  checkbyte.o test.c
	$(CC) $(CFLAGS) -o $@ test.c $(LOADLIBES) checkbyte.o
	$(RM) checkbyte.o

pvm.ring:	$(LIBRARY_PATH)  checkbyte.o  pvm.c
	$(CC) $(CFLAGS) -o $@ pvm.c $(LOADLIBES) checkbyte.o
	$(RM) checkbyte.o

testf.x: msgtypesf.h $(LIBRARY_PATH) testf.o
	$(FLD) $(FFLAGS) -o $@ testf.o $(LOADLIBES)

jacobi.x: $(LIBRARY_PATH) jacobi.o $(JOBJ)
	$(FLD) $(FFLAGS) -o $@ jacobi.o $(JOBJ) $(LOADLIBES)


$(LIBOBJ)  test.o testf.o:	$(INCLUDES)

testf.o:	msgtypesf.h
$(patsubst %,$(LIBRARY_PATH)(%),globalop.o):	msgtypesc.h


machdep: msgtypesf.h
