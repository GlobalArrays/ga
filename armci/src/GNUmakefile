         FOPT = -O
         COPT = -O

OBJ= strided.o armci.o memory.o fcopy.o fence.o rmw.o faccumulate.o acc.o\
 memlock.o pack.o vector.o clusterinfo.o message.o mutex.o

#reference to the MPI library name, overwritten by LIBMPI environment variable
MPI_LIB_NAME = -lmpi

#reference to the PVM library name, overwritten by LIBPVM environment variable
ifeq ($(TARGET),CRAY-T3E)
     PVM_LIB_NAME = -lpvm3
else
     PVM_LIB_NAME = -lgpvm3 -lpvm3
endif

INTERRUPT_AVAILABLE = SP1 SP PARAGON LAPI
SYSTEM_V = SUN SGI SGI_N32 SGI64 SGITFP HPUX SOLARIS IBM DECOSF CONVEX LINUX
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP LAPI FUJITSU-VPP
NATIVE_LOCKS = SGI SGI_N32 SGI64 SGITFP CONVEX WIN32 CRAY-YMP
#USE_POSIX = IBM

ifeq ($(TARGET),$(findstring $(TARGET),$(SYSTEM_V)))
     LIB_DEFINES += -DSYSV
     OBJ += shmalloc.o signaltrap.o dataserv.o request.o sockets.o shmlimit.o
 ifeq ($(TARGET),$(findstring $(TARGET),$(USE_POSIX)))
     OBJ += winshmem.o
 else
     OBJ += shmem.o
 endif
 ifeq ($(TARGET),$(findstring $(TARGET),$(NATIVE_LOCKS)))
     OBJ += locks.o
 else
     OBJ += semaphores.o
 endif
else
 ifeq ($(TARGET),$(findstring $(TARGET),$(NATIVE_LOCKS)))
     OBJ += locks.o
 endif
endif

#let compiler to choose the appropriate MPI lib
ifeq ($(TARGET),LAPI)
     OBJ += lapi.o request.o
     MPI_LIB_NAME =
ifdef LAPI2
     LIB_DEFINES = -DLAPI2
     EXTRA_LIBS += /u2/d3h325/lapi_vector_beta/liblapi_r_dbg.a
else
     OBJ += shmalloc.o signaltrap.o shmem.o shmlimit.o
     LIB_DEFINES = -DSYSV
endif
     
endif

ifneq ($(TARGET),$(findstring $(TARGET),$(ONESIDED_AVAILABLE)))
     LIB_DEFINES += -DSHMEM
endif


#
# settings dependent on selection of the message-passing library
#
ifeq ($(MSG_COMMS),PVM)
#........................... settings for PVM ....................
ifdef PVM_INCLUDE
  LIB_INCLUDES += -I$(PVM_INCLUDE)
endif
ifdef PVM_LIB
  LIBS += -L$(PVM_LIB)
endif
ifdef LIBPVM
  PVM_LIB_NAME = $(LIBPVM)
endif
LIBS += $(PVM_LIB_NAME)
LIB_DEFINES += -DPVM
#........................... end of PVM section ..................
else
  ifeq ($(MSG_COMMS),TCGMSG)
    ifdef TCG_INCLUDE
      LIB_INCLUDES += -I$(TCG_INCLUDE)
    endif
    ifdef TCG_LIB
      LIBS += -L$(TCG_LIB) -ltcgmsg
    endif
    LIB_DEFINES += -DTCG
  else
#.............. MPI is the default library for ARMCI ..................
    ifdef MPI_INCLUDE
      LIB_INCLUDES += -I$(MPI_INCLUDE)
    endif
    ifdef MPI_LIB
      LIBS += -L$(MPI_LIB)
    endif
    ifdef LIBMPI
      MPI_LIB_NAME = $(LIBMPI)
    endif
    LIBS += $(MPI_LIB_NAME)
    LIB_DEFINES += -DMPI
  endif
#........................... end of MPI section ..................
endif


LIBRARY = libarmci.a
HARDCLEAN = yes

#add some extra Microsoft VC files to clean if any left 
NTSTUFF = *.ilk *.exe *.ncb
LIB_TARGETS = *.x $(NTSTUFF)
LIB_DISTRIB = ../lib

include ../config/makefile.h
include ../config/makelib.h

LIBS += $(EXTRA_LIBS)

%.x:%.c  $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

test0.x: test.o $(LIBRARY_PATH)
	$(LINK.c) $(CFLAGS) -o $@ $^ $(LIBS)

ipctest.x: $(LIBRARY_PATH)
	$(CC) $(CFLAGS) ipctest.c -o $@ $^ $(LIBS)

$(LIBRARY_PATH)(winshmem.o):    ./shmem.h ./shmalloc.h 
$(LIBRARY_PATH)(vector.o):      ./memlock.h ./copy.h ./armcip.h ./acc.h 
$(LIBRARY_PATH)(strided.o):     ./memlock.h ./copy.h ./armcip.h ./acc.h 
$(LIBRARY_PATH)(shmem.o):       ./shmem.h ./shmalloc.h 
$(LIBRARY_PATH)(semaphores.o):  ./semaphores.h 
$(LIBRARY_PATH)(rmw.o): 	./locks.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(request.o):     ./request.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(pack.o):        ./armcip.h 
$(LIBRARY_PATH)(mutex.o):       ./request.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(message.o):     ./message.h ./armcip.h 
$(LIBRARY_PATH)(memory.o):      ./shmem.h ./message.h ./armcip.h 
$(LIBRARY_PATH)(memlock.o):     ./memlock.h ./locks.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(locks.o):       ./locks.h ./armcip.h 
$(LIBRARY_PATH)(lapi.o):        ./copy.h ./armcip.h ./lapidefs.h
$(LIBRARY_PATH)(ipctest.o):     ./shmem.h ./locks.h 
$(LIBRARY_PATH)(fence.o):       ./copy.h ./armcip.h ./armci.h 
$(LIBRARY_PATH)(dataserv.o):    ./sockets.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(clusterinfo.o): ./message.h ./armcip.h 
$(LIBRARY_PATH)(armci.o):       ./signaltrap.h ./shmem.h ./memlock.h ./locks.h ./lapidefs.h ./copy.h 

locks.h: semaphores.h
armcip.h: message.h lapidefs.h memlock.h
%.x: %.c
