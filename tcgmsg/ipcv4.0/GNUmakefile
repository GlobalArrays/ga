# $Id: GNUmakefile,v 1.15 2003-07-17 22:45:56 vinod Exp $
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

  LIB_TARGETS = *.x *.o parallel msgtypesf.h msgtf.h testf.f
  INCDIR      = ../../include

ifeq ($(TARGET),IBM)
   LIB_DEFINES = -DGOTXDR -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),IBM64)
   LIB_DEFINES = -DGOTXDR -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),LINUX)
   LIB_DEFINES = -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),LINUX64)
   LIB_DEFINES = -DTIMINGS -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),SGI_N32)
   LIB_DEFINES = -DGOTXDR -DPARTIALSPIN -DSGIUS
   EXTRA = strdup.o  wrap.o
endif
ifeq ($(TARGET),SGITFP)
   LIB_DEFINES = -DPARTIALSPIN -DSGIUS
   EXTRA = strdup.o  wrap.o
endif
ifeq ($(TARGET),MACX)
   LIB_DEFINES = -DNOSPIN
endif
ifeq ($(TARGET),HPUX)
   LIB_DEFINES = -DNOSPIN
     EXTRA = hpuxargs.o wrap.o
endif
ifeq ($(TARGET),HPUX64)
   LIB_DEFINES = -DNOSPIN
     EXTRA = hpuxargs.o wrap.o
endif
ifeq ($(TARGET),SOLARIS)
   LIB_DEFINES = -DSUN -DGOTXDR -DNOSPIN -DSUNF77_2
     EXTRA = wrap.o
endif
ifeq ($(TARGET),SOLARIS64)
   LIB_DEFINES = -DSUN -DGOTXDR -DNOSPIN -DSUNF77_2
     EXTRA = wrap.o
endif
ifeq ($(TARGET),DECOSF)
   LIB_DEFINES =  -DDECFORT -DNOSPIN
   EXTRA = strdup.o strtok.o wrap.o
endif
ifeq ($(TARGET),CRAY-YMP)
   LIB_DEFINES = -DCRAY  -DGOTXDR
 LINTFLAGS = $(DEFINES) -u
endif                                                                         
ifeq ($(TARGET),SP)
    FFLAGS = -g -qEXTNAME
   LIB_DEFINES = -DSP1 -DEXTNAME
#enable workaround for a bug in MPL under PSSP 3.1 on SMP nodes
   LIB_DEFINES += -DMPL_SMP_BUG
 LINTFLAGS = -u -a -wkD $(LIB_DEFINES)
     EXTRA = strdup.o strtok.o
   OBJ  = sp1.o evon.o evlog.o checkbyte.o mtime.o \
             memalign.o random.o drand48.o mitob.o mdtob.o \
             mdtoi.o mitod.o llog.o $(EXTRA)
endif
ifeq ($(TARGET),SP1)
    FFLAGS = -g -qEXTNAME
   LIB_DEFINES = -DSP1 -DEXTNAME
 LINTFLAGS = -u -a -wkD $(LIB_DEFINES)
     EXTRA = strdup.o strtok.o
   OBJ  = sp1.o evon.o evlog.o checkbyte.o mtime.o \
             memalign.o random.o drand48.o mitob.o mdtob.o \
             mdtoi.o mitod.o llog.o $(EXTRA)
endif
                                                                           
# the next targets have no entries in the main tools config files

ifeq ($(TARGET),CRAY)
        FC = cf77
   LIB_DEFINES = -DCRAY  -DGOTXDR
    CFLAGS = -g $(DEFINES)
 LINTFLAGS = $(DEFINES) -u
    RANLIB = echo $*
#This for older versions of UNICOS
#     COMM_LIBS = -lbsd -lnet -lrpc
endif 
 
  
ifeq ($(TARGET),CONVEX-SPP)
        FC = fc
    FFLAGS = -O1 -ppu -or none
   LIB_DEFINES = -DHPUX -DCONVEX -DEXTNAME -DSPPLOCKS
    CFLAGS = -O1 -or none $(DEFINES)
    RANLIB = echo $*
 LINTFLAGS = $(DEFINES) -bhu
endif

SYSTEM_V =  IBM IBM64 LINUX SGI SGI_N32 SGITFP HPUX HPUX64 SOLARIS SOLARIS64 
SYSTEM_V += DECOSF CONVEX-SPP LINUX64 MACX

ifeq ($(TARGET),$(findstring $(TARGET),$(SYSTEM_V)))
     LIB_DEFINES += -DSYSV -DSHMEM
endif

ifdef GA_USE_VAMPIR
   EXTRA += tcgmsg_vampir.o ga_vt.o
endif
         
include ../../armci/config/makecoms.h
include ../../armci/config/makemp.h
include ../../config/makefile.h
include ../../config/makelib.h


#.......................... general ................................
ifndef BINDIR
       BINDIR = .
endif

INTERRUPT_AVAILABLE = SP1 
NO_TOPLOT = ARDENT CRAY SGI SGI_N32 SGITFP NEXT NEXTNOEXT HPUX HPUXNOEXT KSR
NO_PARALLEL = SP1 SP


LOADLIBES = $(FULL_LIBRARY_PATH) $(COMM_LIBS) 
ifdef GA_USE_VAMPIR
   ifdef GA_USE_VAMPIR
      ifdef VT_LIB
       ifdef LIBVT
          LIBS += -ltcgmsg -L$(VT_LIB) $(LIBVT)
       else
          LIBS += -ltcgmsg -L$(VT_LIB) -lVT
       endif
      endif
   endif
   ifdef VT_INCLUDE
      INCLUDES += -I$(VT_INCLUDE)
   endif
endif
#......................... make targets ........................
#
ifeq ($(TARGET),$(findstring $(TARGET),$(NO_PARALLEL)))
      all:    $(FULL_LIBRARY_PATH) test.x testf.x hello.x msgtypesf.h
else
      all:    $(FULL_LIBRARY_PATH) test.x testf.x hello.x msgtypesf.h hello.p test.p testf.p
endif

everything:     all $(BINDIR)/parallel parse toplot

hello.x: $(FULL_LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ hello.c  $(LOADLIBES)



ifneq ($(TARGET),$(findstring $(TARGET),$(NO_PARALLEL)))
  ifeq ($(TARGET),IPSC)
#       IPSC ... parallel must be built on the front end machine!
	$(BINDIR)/parallel:
		echo TARGET BINDIR
		cc -g -o $@ para.ipsc.c -host
		echo Don't forget to copy parallel into bin directory
  else
        $(BINDIR)/parallel:	 $(FULL_LIBRARY_PATH)
		$(CC) $(CFLAGS) -o $@ parallel.c $(LOADLIBES)
		@echo ">>>" $@ "is used to run TCGMSG programs"
  endif
else
.PHONY:
        $(BINDIR)/parallel:
		@echo "No parallel on this system"
endif

test.x:	$(FULL_LIBRARY_PATH) $(BINDIR)/parallel test.c
	$(CC) $(CFLAGS) $(CLD_REN) -o $@ test.c $(LOADLIBES)

testf.x: msgtypesf.h testf.f.proto $(FULL_LIBRARY_PATH) $(BINDIR)/parallel
	./port "$(TARGET)" testf.f.proto
	$(FC) $(FFLAGS) $(FLD_REN) -o $@ testf.f $(LOADLIBES)

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

testpf.x: $(FULL_LIBRARY_PATH)
	$(FC) $(FFLAGS) -o $@ testpf.f $(LOADLIBES) 

$(LIBOBJ) parallel.o test.o testf.o:	$(INCLUDES)

testf.o:	msgtypesf.h
$(patsubst %,$(FULL_LIBRARY_PATH)(%),globalop.o):	msgtypesc.h

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

ifdef GA_USE_VAMPIR
tcgmsg_vampir.o:	../tcgmsg_vampir.c
	ln -s ../tcgmsg_vampir.c tcgmsg_vampir.c
	$(CC) $(CFLAGS) -c tcgmsg_vampir.c
	rm tcgmsg_vampir.c
 
ga_vt.o:	../../global/ga_vt.c
	ln -s ../../global/ga_vt.c ga_vt.c
	$(CC) $(CFLAGS) -c ga_vt.c
	rm ga_vt.c
endif

machdep: msgtypesf.h
