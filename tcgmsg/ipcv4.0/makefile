ifdef MACHINE
    TARGET = $(MACHINE)
    CC = kwas
endif

ifndef TARGET
error:
	@echo "TARGET/MACHINE machine not defined"
	exit
endif


     SHELL = /bin/sh
        FC = f77
        CC = cc
   INSTALL = echo "$@" is built
    RANLIB = ranlib
    CFLAGS = -O $(DEFINES)
    FFLAGS = -O
 LINTFLAGS = $(DEFINES) -bhzu
 MAKEFLAGS = -j1
        RM = rm



#................................. system specific ..................

ifeq ($(TARGET),SP)
        FC = mpxlf
        CC = mpcc
    FFLAGS = -g -qEXTNAME
   DEFINES = -DSP1 -DAIX -DEXTNAME
#enable workaround for a bug in MPL under PSSP 3.1 on SMP nodes
   DEFINES += -DMPL_SMP_BUG
    CFLAGS = -g $(DEFINES)
 LINTFLAGS = -u -a -wkD $(DEFINES)
     EXTRA = strdup.o strtok.o
   LIBOBJ  = sp1.o evon.o evlog.o checkbyte.o mtime.o \
             memalign.o random.o drand48.o mitob.o mdtob.o \
             mdtoi.o mitod.o llog.o $(EXTRA)
endif

ifeq ($(TARGET),SP1)
        FC = mpxlf
        CC = mpcc
    FFLAGS = -g -qEXTNAME
   DEFINES = -DSP1 -DAIX -DEXTNAME  
    CFLAGS = -g $(DEFINES)
 LINTFLAGS = -u -a -wkD $(DEFINES)
     EXTRA = strdup.o strtok.o
   LIBOBJ  = sp1.o evon.o evlog.o checkbyte.o mtime.o \
             memalign.o random.o drand48.o mitob.o mdtob.o \
             mdtoi.o mitod.o llog.o $(EXTRA)
endif


ifeq ($(TARGET),IPSC)
   DEFINES = -DIPSC
   CFLAGS  = -i860 -g -Knoieee -nx $(DEFINES)
   FFLAGS  = -i860 -O2 -Knoieee -nx
    INTEL  = yes
endif

ifeq ($(TARGET),DELTA)
   DEFINES = -DIPSC -DDELTA
   INSTALL = rcp "$@" delta1: &
   CFLAGS  = -i860 -g -Knoieee -node $(DEFINES)
   FFLAGS  = -i860 -O2 -Knoieee -node
    INTEL  = yes
endif

ifeq ($(TARGET),PARAGON)
   DEFINES = -DIPSC -DPARAGON
   CFLAGS  = -nx -g -Knoieee $(DEFINES)
   FFLAGS  = -nx -O2 -Knoieee 
    INTEL  = yes
endif


ifeq ($(INTEL),yes)
    CC     = icc
    FC     = if77
    AR     = ar860
    RANLIB = echo $*
      LIBOBJ  = ipsc.o globalop.o evon.o evlog.o checkbyte.o \
             memalign.o random.o drand48.o mitob.o mdtob.o \
             mdtoi.o mitod.o llog.o
endif


ifeq ($(TARGET),CONVEX-SPP)
        FC = fc
    FFLAGS = -O1 -ppu -or none
   DEFINES = -DHPUX -DCONVEX -DEXTNAME -DSYSV -DSHMEM -DSPPLOCKS 
    CFLAGS = -O1 -or none $(DEFINES)
    RANLIB = echo $*
 LINTFLAGS = $(DEFINES) -bhu
endif


ifeq ($(TARGET),CONVEX)
# Convex C series
        FC = fc
    FFLAGS =  -O1
   DEFINES = -DCONVEX  -DSHMEM -DGOTXDR -DEVENTLOG -DNOSPIN
    CFLAGS = -O1 -or none $(DEFINES)
 LINTFLAGS = $(DEFINES) -bhu
     EXTRA = strdup.o strtok.o sprintf.o
endif


ifeq ($(TARGET),IBM)
#IBM RISC 6000 ... AIX  with -qEXTNAME ... recommended choice.
## You MUST Compile ALL FORTRAN with -qEXTNAME to use this option.
##
## This is the first of two possible configurations for AIX.
## -qEXTNAME appends an underscore to all FORTRAN global symbols
## so that there are NO clashes with any system or C or TCGMSG routines.
## However, certain useful libraries (e.g. the BLAS!) do not export
## the appropriate symbols.

        FC = xlf
    FFLAGS = -O -qEXTNAME
   DEFINES = -DAIX -DEXTNAME -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
 LINTFLAGS = -u -a -wkD $(DEFINES)
      LIBS = -lc
     EXTRA = strdup.o strtok.o
endif

ifeq ($(TARGET),IBM64)
        FC = xlf
    FFLAGS = -O -q64 -qintsize=8 -qEXTNAME
    CFLAGS+= -q64
   DEFINES = -DAIX -DEXTNAME -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
 LINTFLAGS = -u -a -wkD $(DEFINES)
     EXTRA = strdup.o strtok.o
   ARFLAGS = -rcv -X 64
endif

ifeq ($(TARGET),IBMNOEXT)
#IBM RISC 6000 ... AIX 3.1
## You MUST append an underscore to ALL FORTRAN calls to system routines
## if you use this option.
## e.g. call getenv('data',env) -> call getenv_('data',env).
##
## This is the second of two possible configurations for AIX
## By default all FORTRAN global symbols are lowercase and CLASH with
## corresponding C and system routines.  TCGMSG has been set up to avoid
## generating any clashes BUT it needs access to the C system interface.
## Thus ALL FORTRAN system calls must explicitly be changed by appending
## an underscore.  For some reason there is an unavoidable clash in
## the event logging which must therefore be disabled.
##
        FC = xlf
   DEFINES = -DAIX -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
 LINTFLAGS = -u -a -wkD $(DEFINES)
      LIBS = -lc
     EXTRA = strdup.o strtok.o
endif


ifeq ($(TARGET),DECOSF)
    FFLAGS = -O -integer_size 64
   DEFINES = -DDECOSF -DDECFORT -DSYSV -DSHMEM  -DNOSPIN
 LINTFLAGS = $(DEFINES) -u
     EXTRA = strdup.o strtok.o
endif


ifeq ($(TARGET),LINUX)
# PC80x86 Linux pl10.
# f77 usage of argc&argv inside f2c is (xargc & xargv).
# -DPLOT option removed for linux. 
#
   DEFINES = -DLINUX -DSYSV -DSHMEM -DNOSPIN
ifeq ($(FC),pgf77)
       PGLINUX = 1
endif
ifeq ($(FC),pgf90)
       PGLINUX = 1
endif
ifdef PGLINUX
       FOPT_REN = -Mdalign -Minform,warn -Mnolist -Minfo=loop -Munixlogical
       DEFINES += -DPGLINUX
endif
 LINTFLAGS = $(DEFINES) -u
      LIBS = -lc
     EXTRA = strdup.o strtok.o
endif
ifeq ($(TARGET),LINUX64)
# LINUX 64-bit on Alpha using DEC compilers
# LINUX64 define used in pbeginf and shmem
       FC = fort
   FFLAGS = -i8
       CC = ccc
   DEFINES = -DLINUX -DLINUX64 -DSYSV -DSHMEM -DTIMINGS -DNOSPIN -DEXT_INT
    CFLAGS = -O $(DEFINES)
     EXTRA = strdup.o strtok.o
 LINTFLAGS = $(DEFINES) -u
endif

ifeq ($(TARGET),SGI-OLD)
# 32-bit systems running IRIX 5.3 and older
    FFLAGS = -O2 -32
   DEFINES = -DSGI -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
    CFLAGS = -O -32 -I/usr/include/bsd -I/usr/include/sun $(DEFINES)
 LINTFLAGS = -I/usr/include/bsd -I/usr/include/sun $(DEFINES) -u
      LIBS = -lmalloc -lbsd -lrpcsvc -lsun
    RANLIB = echo $*
     EXTRA = strdup.o strtok.o
endif


ifeq ($(TARGET),SGI)
# 32-bit systems
    FFLAGS = -O2 -32
    CFLAGS = -O -32 $(DEFINES)
   DEFINES = -DSGI -DSYSV -DSHMEM  -DGOTXDR -DPARTIALSPIN
 LINTFLAGS =  $(DEFINES) -u
    RANLIB = echo $*
     EXTRA = strdup.o
endif

ifeq ($(TARGET),SGI_N32)
# 64-bit processors running IRIX > 6.0
    FFLAGS = -O2 -n32
   DEFINES = -DSGI -DSYSV -DSHMEM  -DGOTXDR -DPARTIALSPIN -DSGIUS
    CFLAGS = -O -n32 $(DEFINES)
 LINTFLAGS =  $(DEFINES) -u
    RANLIB = echo $*
     EXTRA = strdup.o
endif


ifeq ($(TARGET),SGITFP)
    FFLAGS = -O2 -i8 -64
    CFLAGS = -O -64 $(DEFINES)
   DEFINES = -DSGI -DSYSV -DSHMEM  -DPARTIALSPIN -DSGIUS
 LINTFLAGS = $(DEFINES) -u
    RANLIB = @echo $*
     EXTRA = strdup.o
endif


ifeq ($(TARGET),KSR)
    FFLAGS = -para -O2 -r8
   DEFINES = -DKSR -DSYSV -DSHMEM  -DGOTXDR
    CFLAGS = -para -g $(DEFINES)
 LINTFLAGS = $(DEFINES) -u
    RANLIB = @echo $*
      LIBS = -lrpc
     EXTRA = ksr.o
endif

ifeq ($(TARGET),HPUX)
    FFLAGS = -O +ppu
   DEFINES = -DHPUX -DEXTNAME -DSYSV -DSHMEM  -DNOSPIN
 LINTFLAGS = $(DEFINES) -bhu
    RANLIB = @echo $*
     EXTRA = hpuxargs.o
endif

ifeq ($(TARGET),HPUXNOEXT)
   DEFINES = -DHPUX -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
 LINTFLAGS = $(DEFINES) -bhu
    RANLIB = @echo $*
     EXTRA = hpuxargs.o
endif


ifeq ($(TARGET),SUN)
   DEFINES = -DSUN -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN
    FFLAGS = -O3
endif

ifeq ($(TARGET),SUN77_2)
   DEFINES = -DSUN -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN -DSUNF77_2
    FFLAGS = -O3
endif

ifeq ($(TARGET),SOLARIS)
   DEFINES = -DSUN -DSOLARIS -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN\
             -DSUNF77_2
      LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
    RANLIB = echo $*
endif
ifeq ($(TARGET),SOLARIS64)
    FFLAGS = -xarch=v9 -xtypemap=real:64,double:64,integer:64
   DEFINES = -DSUN -DSOLARIS -DSYSV -DSHMEM  -DGOTXDR -DNOSPIN\
             -DSUNF77_2
    CFLAGS = -O -xarch=v9 -dalign $(DEFINES)
      LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
    RANLIB = echo $*
endif

ifeq ($(TARGET),CRAY)
        FC = cf77
   DEFINES = -DCRAY  -DGOTXDR
    CFLAGS = -g $(DEFINES)
 LINTFLAGS = $(DEFINES) -u
    RANLIB = echo $*
#This for older versions of UNICOS
#     LIBS = -lbsd -lnet -lrpc
endif


ifeq ($(TARGET),ENCORE)
#ENCORE ... note that testf.f compiles incorrectly with -O
    FFLAGS = -O -q loops
        CC = gcc
   DEFINES = -DENCORE -DSYSV -DSHMEM 
    CFLAGS = -g $(DEFINES)
      LIBS = -lpp
     EXTRA = strdup.o strtok.o
endif


ifeq ($(TARGET),SEQUENT)
        FC = fortran
    FFLAGS = -O -Z064000
   DEFINES = -DSEQUENT -DSYSV -DSHMEM  -DGOTXDR
    CFLAGS = -O -Z064000 $(DEFINES)
      LIBS = -lpps
     EXTRA = strdup.o strtok.o
endif

#.......................... general ................................


INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON
NO_TOPLOT = ARDENT CRAY SGI SGI_N32 SGITFP NEXT NEXTNOEXT HPUX HPUXNOEXT KSR PARAGON
NO_PARALLEL = DELTA PARAGON SP1 SP



ifndef LIBOBJ
LIBOBJ =  brdcst.o    nodeid.o    setdbg.o                          \
          checkbyte.o llog.o      nxtval.o    shmem.o     synch.o   \
          cluster.o   memalign.o  signals.o                         \
          usleep.o    srmover.o   pbegin.o    snd.o       waitall.o \
          drand48.o   mtime.o     pbeginf.o   sockets.o             \
          error.o     nnodes.o    sema.o      stats.o               \
          xdrstuff.o  waitcom.o   random.o    niceftn.o \
	  mdtob.o     mitob.o     mitod.o   \
	  mdtoi.o     globalop.o  evlog.o   \
          evon.o      pfilecopy.o $(EXTRA)

 INCLUDES = cluster.h  sema.h     signals.h  sndrcvP.h  srftoc.h    \
            farg.h     shmem.h    sndrcv.h   sockets.h  xdrstuff.h  \
            msgtypesc.h           evlog.h    defglobals.h

endif


ifeq ($(TARGET),KSR)
# need optimization for this but everything else breaks with it
ksr.o:	ksr.c ksr.h sndrcv.h sndrcvP.h srftoc.h
	$(CC) -O -c $(DEFINES) ksr.c
endif 

#...................................................................

INCDIR = ../../include

ifndef BINDIR
       BINDIR = .
endif

#ifndef LIBDIR
#       LIBDIR = .
#endif

ifndef CNFDIR
       CNFDIR = .
endif

OBJ := $(LIBOBJ)
LIB_DISTRIB = ../../lib
LIBRARY = libtcgmsg.a
LIB_TARGETS = $(BINDIR)/parallel *.x *.x testf.? core events events.??? log.???\
               parse toplot TAGS *.log *~ *.o *.p *.log msgtf.h msgtypesf.h #*#
HEADERS = msgtypesf.h msgtypesc.h sndrcv.h srftoc.h ../tcgmsg.fh
include ../../config/makelib.h


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
	$(FC) $(FFLAGS) -o $@ testpf.c $(LIBRARY_PATH) $(LIBS)

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
