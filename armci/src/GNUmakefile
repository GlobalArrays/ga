         FOPT = -O
         COPT = -O

OBJ= strided.o armci.o memory.o fence.o rmw.o acc.o\
 memlock.o pack.o vector.o clusterinfo.o message.o mutex.o locks.o

#ARMCI can be built without using fortran code when NOFORT is defined
#NOFORT =1
ifndef NOFORT
OBJ += faccumulate.o fcopy.o
endif


#default message-passing library for ARMCI is MPI
ifndef MSG_COMMS
  MSG_COMMS=MPI
endif

#defines communication libs and symbols
include ../config/makemp.h
include ../config/makecoms.h

SYSTEM_V = SUN SOLARIS SOLARIS64 SGI SGI_N32 SGI64 SGITFP 
SYSTEM_V += HPUX HPUX64 IBM IBM64 DECOSF LINUX LINUX64 INTERIX NEC
SHMEM    = $(SYSTEM_V) LAPI LAPI64
SOCKETS  = $(SYSTEM_V)
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP LAPI FUJITSU-VPP FUJITSU-AP
#USE_PTHREADS = LINUX SGI_N32 SOLARIS
#USE_POSIX = IBM

ifeq ($(TARGET),$(findstring $(TARGET),$(USE_PTHREADS)))
     LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT
     EXTRA_LIBS += -lpthread
endif

ifeq ($(TARGET),HITACHI)
     LIB_DEFINES += -DMMAP -DCLUSTER
#     OBJ += shmalloc.o signaltrap.o winshmem.o spawn.o ds-shared.o request.o
#     LIB_DEFINES += -DDATA_SERVER -DMMAP
     OBJ += shmalloc.o signaltrap.o winshmem.o ds-shared.o request.o 
endif

ifeq ($(TARGET),$(findstring $(TARGET),$(SHMEM)))
     LIB_DEFINES += -DSYSV
     OBJ += shmalloc.o signaltrap.o shmlimit.o shmem.o 
endif

ifeq ($(TARGET),LAPI)
     OBJ += async.o
endif

ifeq ($(TARGET),LAPI64)
     OBJ += async.o
endif

ifeq ($(TARGET),$(findstring $(TARGET),$(SYSTEM_V)))
 OBJ += semaphores.o 
endif

ifeq ($(ARMCI_NETWORK),GM)
#  OBJ += dataserv-myri.o myrinet.o spawn.o request.o 
  OBJ += dataserv-gm.o ds-shared.o myrinet.o spawn.o request.o bufalloc.o async.o buffers.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DSERVER_THREAD -D_REENTRANT
endif

ifeq ($(ARMCI_NETWORK),VIA)
  OBJ += spawn.o  request.o via.o ds-shared.o buffers.o async.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DSERVER_THREAD -D_REENTRANT
endif

ifeq ($(ARMCI_NETWORK),QUADRICS)
# shared memory causes problems for shmem: NO_SHM disables use of shared memory
# only one of the two lines below can be uncommented
#   LIB_DEFINES += -DNO_SHM
    LIB_DEFINES += -DALLOC_MUNMAP -DACC_SMP
    OBJ += elan.o
    LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT -DELAN_ACC
    OBJ += spawn.o ds-shared.o request.o
    EXTRA_LIBS += -lpthread
endif

ifdef ARMCI_NETWORK
  LIBRARY_STAMP = $(ARMCI_NETWORK).$(MSG_COMMS)
else
  # sockets
  LIBRARY_STAMP = $(MSG_COMMS)
  ifeq ($(TARGET),$(findstring $(TARGET),$(SOCKETS)))
     OBJ += spawn.o dataserv.o sockets.o request.o ds-shared.o buffers.o async.o
     LIB_DEFINES += -DDATA_SERVER -DSOCKETS
     # to use pthread version uncomment the following two lines
     # LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT
     # EXTRA_LIBS += -lpthread
  endif
endif

ifdef GA_USE_VAMPIR
   OBJ += armci_vampir.o
   LIB_INCLUDES += -I../../include
endif

LIB_DEFINES += $(MP_DEFINES) $(COMM_DEFINES)
LIB_INCLUDES += $(COMM_INCLUDES) $(MP_INCLUDES)
EXTRA_LIBS += $(MP_LIBS) $(COMM_LIBS)

ifdef NOFORT
LIB_DEFINES += -DNOFORT
endif

ifdef REPORT_SHMMAX
LIB_DEFINES += -DREPORT_SHMMAX
endif

#......................................................................
LIBRARY = libarmci.a
HARDCLEAN = yes

#add some extra Microsoft VC files to clean if any left 
NTSTUFF = *.ilk *.exe *.ncb
LIB_TARGETS = *.x shmtest $(NTSTUFF)
LIB_DISTRIB = ../lib

include ../config/makefile.h
OBJ += $(EXTRA_OBJ)

include ../config/makelib.h
LIBS += $(EXTRA_LIBS)

#......................................................................
%.x.p:%.x
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@

%.x:%.c  $(LIBRARY_PATH)
	$(CC) -g -I. $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIBS) -lm

ipctest.x: $(LIBRARY_PATH)
	$(CC) $(CFLAGS) $(MP_DEFINES) ipctest.c -o $@ $^ $(LIBS)


$(LIBRARY_PATH)(winshmem.o):    ./shmem.h ./shmalloc.h 
$(LIBRARY_PATH)(vector.o):      ./memlock.h ./copy.h ./armcip.h ./acc.h 
$(LIBRARY_PATH)(strided.o):     ./memlock.h ./copy.h ./armcip.h ./acc.h 
$(LIBRARY_PATH)(shmem.o):       ./shmem.h ./shmalloc.h 
$(LIBRARY_PATH)(semaphores.o):  ./semaphores.h 
$(LIBRARY_PATH)(rmw.o): 	./locks.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(request.o):     ./request.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(pack.o):        ./armcip.h ./request.h
$(LIBRARY_PATH)(mutex.o):       ./request.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(message.o):     ./message.h ./armcip.h  ./copy.h
$(LIBRARY_PATH)(memory.o):      ./shmem.h ./message.h ./armcip.h 
$(LIBRARY_PATH)(memlock.o):     ./memlock.h ./locks.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(locks.o):       ./locks.h ./armcip.h 
$(LIBRARY_PATH)(lapi.o):        ./copy.h ./armcip.h ./lapidefs.h
$(LIBRARY_PATH)(ipctest.o):     ./shmem.h ./locks.h 
$(LIBRARY_PATH)(fence.o):       ./copy.h ./armcip.h ./armci.h 
$(LIBRARY_PATH)(ds-shared.o):   ./myrinet.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(dataserv.o):    ./sockets.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(dataserv-gm.o): ./myrinet.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
$(LIBRARY_PATH)(clusterinfo.o): ./message.h ./armcip.h 
$(LIBRARY_PATH)(armci.o):       ./signaltrap.h ./shmem.h ./memlock.h ./locks.h ./lapidefs.h ./copy.h 

locks.h: semaphores.h
armcip.h: message.h lapidefs.h memlock.h
%.x: %.c

