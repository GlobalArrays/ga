# $Id: GNUmakefile,v 1.1 2001-04-24 01:37:19 edo Exp $
#
# Makefile for a portable dynamic memory allocator.

      LIBRARY = libtcgmsg.a
  LIB_DISTRIB = ../../lib
OBJ =  brdcst.o    nodeid.o    setdbg.o                          \
          checkbyte.o llog.o      nxtval.o    shmem.o     synch.o   \
          cluster.o   memalign.o  signals.o                         \
          usleep.o    srmover.o   pbegin.o    snd.o       waitall.o \
          drand48.o   mtime.o     pbeginf.o   sockets.o             \
          error.o     nnodes.o    sema.o      stats.o               \
          xdrstuff.o  waitcom.o   random.o    niceftn.o \
	  mdtob.o     mitob.o     mitod.o   \
	  mdtoi.o     globalop.o  evlog.o   \
          evon.o      pfilecopy.o $(EXTRA)

 HEADERS =  msgtypesf.h msgtypesc.h sndrcv.h srftoc.h ../tcgmsg.fh


  LIB_TARGETS = *.x *.o
  INCDIR      = ../../include

ifeq ($(TARGET),IBM)
   LIB_DEFINES =  -DAIX -DEXTNAME -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),IBM64)
   LIB_DEFINES = -DAIX -DEXTNAME -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
endif
ifeq ($(TARGET),LINUX)
   LIB_DEFINES =   -DLINUX -DSYSV -DSHMEM -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),LINUX64)
   LIB_DEFINES =  -DLINUX -DLINUX64 -DCOMPAQ_FORT -DSYSV -DSHMEM -DTIMINGS -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),SGI_N32)
   LIB_DEFINES = -DSGI -DSYSV -DSHMEM  -DGOTXDR -DPARTIALSPIN -DSGIUS
   EXTRA = strdup.o  wrap.o
endif
ifeq ($(TARGET),SGITFP)
   LIB_DEFINES = -DSGI -DSYSV -DSHMEM  -DPARTIALSPIN -DSGIUS
   EXTRA = strdup.o  wrap.o
endif
ifeq ($(TARGET),HPUX)
   LIB_DEFINES = -DHPUX -DEXTNAME -DSYSV -DSHMEM  -DNOSPIN
     EXTRA = hpuxargs.o wrap.o
endif
ifeq ($(TARGET),HPUX64)
   LIB_DEFINES = -DHPUX64 -DHPUX -DEXTNAME -DSYSV -DSHMEM  -DNOSPIN
     EXTRA = hpuxargs.o wrap.o
endif
ifeq ($(TARGET),SOLARIS)
   LIB_DEFINES = -DSUN -DSOLARIS -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN -DSUNF77_2
     EXTRA = wrap.o
endif
ifeq ($(TARGET),SOLARIS64)
   LIB_DEFINES = -DSUN -DSOLARIS -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN -DSUNF77_2
     EXTRA = wrap.o
endif
ifeq ($(TARGET),DECOSF)
   LIB_DEFINES =  -DDECFORT -DSYSV -DSHMEM  -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),DECOSF32)
   LIB_DEFINES = -DDECOSF -DDECFORT -DSYSV -DSHMEM  -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif

include ../../config/makefile.h
include ../../config/makelib.h


#.......................... general ................................
LIBS=
ifndef BINDIR
       BINDIR = .
endif

INTERRUPT_AVAILABLE = SP1 
NO_TOPLOT = ARDENT CRAY SGI SGI_N32 SGITFP NEXT NEXTNOEXT HPUX HPUXNOEXT KSR
NO_PARALLEL = SP1 SP


LOADLIBES = $(LIBRARY_PATH) $(LIBS)
#......................... make targets ........................
#
ifeq ($(TARGET),$(findstring $(TARGET),$(NO_PARALLEL)))
      all:    $(LIBRARY_PATH) test.x testf.x hello.x msgtypesf.h
else
      all:    $(LIBRARY_PATH) test.x testf.x hello.x msgtypesf.h hello.p test.p testf.p
endif

everything:     all $(BINDIR)/parallel parse toplot

hello.x: $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ hello.c  $(LOADLIBES)



ifneq ($(TARGET),$(findstring $(TARGET),$(NO_PARALLEL)))
  ifeq ($(TARGET),IPSC)
#       IPSC ... parallel must be built on the front end machine!
	$(BINDIR)/parallel:
		echo TARGET BINDIR
		cc -g -o $@ para.ipsc.c -host
		echo Don't forget to copy parallel into bin directory
  else
        $(BINDIR)/parallel:	 $(LIBRARY_PATH)
		$(CC) $(CFLAGS) -o $@ parallel.c $(LOADLIBES)
		@echo ">>>" $@ "is used to run TCGMSG programs"
		
  endif
else
.PHONY:
        $(BINDIR)/parallel:
		@echo "No parallel on this system"
endif

test.x:	$(LIBRARY_PATH) $(BINDIR)/parallel test.c
	$(CC) $(CFLAGS) -c checkbyte.c
	$(CC) $(CFLAGS) -o $@ test.c $(LOADLIBES)
	$(RM) checkbyte.o

testf.x: msgtypesf.h testf.f.proto $(LIBRARY_PATH) $(BINDIR)/parallel
	./port "$(TARGET)" testf.f.proto
	$(FC) $(FFLAGS) -o $@ testf.f $(LOADLIBES)

parse:	parse.o 
	$(CC) -O -c parse.c
	$(CC) -O -o $@ parse.c strdup.o
	$(RM) strdup.o


ifeq ($(TARGET),$(findstring $(TARGET),$(NO_TOPLOT)))
.PHONY: toplot
       toplot:
		@echo "No toplot on this system"
else
       toplot:	strdup.o
		$(CC) -g -o $@ toplot.c strdup.o -lplot
endif

testpf.x: $(LIBRARY_PATH)
	$(FC) $(FFLAGS) -o $@ testpf.f $(LIBRARY_PATH) $(LIBS)

$(LIBOBJ) parallel.o test.o testf.o:	$(INCLUDES)

testf.o:	msgtypesf.h
$(patsubst %,$(LIBRARY_PATH)(%),globalop.o):	msgtypesc.h

msgtypesf.h:	msgtf.h
	cp msgtf.h msgtypesf.h

msgtf.h:   msgtf.h.proto
	./port "$(TARGET)" $^

testf.f:	testf.f.proto
	./port "$(TARGET)" $^

lint:	
	lint $(LINTFLAGS) \
	brdcst.c    error.c    mtime.c    pbegin.c signals.c synch.c \
	checkbyte.c nnodes.c   pbeginf.c  snd.c    test.c \
	cluster.c   llog.c     nodeid.c   sema.c   sockets.c waitall.c \
	usleep.c    memalign.c nxtval.c   setdbg.c stats.c \
	drand48.c   srmover.c  parallel.c shmem.c  strdup.c  strtok.c \
	mdtob.c     mitob.c    mitod.c    mdtoi.c  random.c   \
	xdrstuff.c  evon.c     evlog.c    globalop.c niceftn.c

hello.p test.p testf.p:
ifeq ($(TARGET),CRAY)
	echo "$$LOGNAME `hostname` 1 `pwd`/hello.x /tmp" > hello.p
	echo "$$LOGNAME `hostname` 1 `pwd`/hello.x /tmp" >> hello.p
	echo "$$LOGNAME `hostname` 1 `pwd`/hello.x /tmp" >> hello.p
	echo "$$LOGNAME `hostname` 1 `pwd`/hello.x /tmp" >> hello.p
	echo "$$LOGNAME `hostname` 1 `pwd`/test.x /tmp" > test.p
	echo "$$LOGNAME `hostname` 1 `pwd`/test.x /tmp" >> test.p
	echo "$$LOGNAME `hostname` 1 `pwd`/test.x /tmp" >> test.p
	echo "$$LOGNAME `hostname` 1 `pwd`/test.x /tmp" >> test.p
	echo "$$LOGNAME `hostname` 1 `pwd`/testf.x /tmp" > testf.p
	echo "$$LOGNAME `hostname` 1 `pwd`/testf.x /tmp" >> testf.p
	echo "$$LOGNAME `hostname` 1 `pwd`/testf.x /tmp" >> testf.p
	echo "$$LOGNAME `hostname` 1 `pwd`/testf.x /tmp" >> testf.p
else
	echo "`whoami` `hostname` 4 `pwd`/hello.x /tmp" > hello.p
	echo "`whoami` `hostname` 4 `pwd`/test.x /tmp" > test.p
	echo "`whoami` `hostname` 4 `pwd`/testf.x /tmp" > testf.p
endif


ifeq ($(TARGET),CONVEX)
# In their infinite wisdom the version of sprintf differs between
# the C and FORTRAN environments on the CONVEX and it's not possible
# to just link against -lc due to other conflicts. Need to drag out
# the one that we need by hand. Ugh.
sprintf.o:      /usr/lib/libc.a
	ar x /usr/lib/libc.a sprintf.o
endif 

machdep: msgtypesf.h
