
ifndef MACHINE
error:
	@echo "You must define MACHINE"
	exit
endif

       CC = cc
   RANLIB = echo


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
   CFLAGS = -O3 -user_readonly_const -fullwarn -DSHMEM -DSYSV -DSGI -DSGITFP
endif

ifeq ($(MACHINE),CRAY-T3D)
     LINT = cclint
       FC = cf77
  DEFINES = -DCRAY_T3D -DCRAY -DSHMEM
 INCLUDES = -I.
   FFLAGS =  -Ccray-t3d -g -Wf"-dp -ojump"
#  CFLAGS = -g $(DEFINES) $(INCLUDES) -Wl"-Drdahead=on"
   CFLAGS = -O3 -h inline3 $(DEFINES) $(INCLUDES) -Tcray-t3d -hjump
     LOPT = -Wl"-Drdahead=on"
# -happrentice
    EXTRA = t3d.o snd_rcv_probe.o
#   LIBS  = -lapp

 OBJ_CORE = pbegin.o queues.o  usleep.o mitob.o mdtob.o\
            async_send.o mtime.o memalign.o drand48.o checkbyte.o busy.o \
            random.o copyall.o
endif

      OBJ = $(OBJ_CORE) $(EXTRA)  
     JOBJ = fexit.o timer.o getmem.o mxv_fortran.o

  TARGETS = test.x test2 hello jacobi.x libtcgmsg.a pvm.ring

all:	test.x

test.x:	test.isend.o libtcgmsg.a
	$(CC) $(CFLAGS) $(LOPT) -o test.x test.isend.o  libtcgmsg.a $(LIBS)

test2:	test2.o libtcgmsg.a
	$(CC) $(CFLAGS)  $(LOPT) -o $@ $^ $(LIBS)

hello:	hello.o libtcgmsg.a
	$(CC) $(CFLAGS) $(LOPT) -o $@ $^ $(LIBS)

pvm.ring: pvm.o checkbyte.o mtime.o libtcgmsg.a	
	$(CC) $(CFLAGS) $(LOPT) -o pvm.ring pvm.o checkbyte.o mtime.o libtcgmsg.a $(LIBS)
jacobi.x: jacobi.o $(JOBJ) libtcgmsg.a
	$(FC) $(FFLAGS) $(LOPT) -o jacobi.x jacobi.o $(JOBJ) libtcgmsg.a $(LIBS)

libtcgmsg.a:	$(OBJ) 
	/bin/rm -f $@ 
	$(AR) $(ARFLAGS) $@ $(OBJ) 
	$(RANLIB) $@ 

clean:
	/bin/rm -f *.o $(TARGETS) core

realclean:	clean
	/bin/rm -f *~ \#*

# DO NOT DELETE THIS LINE -- make depend depends on it.

$(OBJ): tcgmsgP.h tcgmsg.h

testsimple.o:	tcgmsg.h

lint:
	$(LINT) $(DEFINES) $(INCLUDES) test.c t3d.c queues.c async_send.c snd_rcv_probe.c

.f.o:
	$(FC) -c $(FFLAGS)  $*.f

