#$Id: GNUmakefile,v 1.103.2.18 2007-08-30 22:31:04 manoj Exp $

         FOPT = -O
         COPT = -O

OBJ= strided.o armci.o memory.o fence.o rmw.o acc.o kr_malloc.o\
 memlock.o pack.o vector.o clusterinfo.o mutex.o locks.o aggregate.o\
 rtinfo.o iterator.o

OBJ_FRAGILE = message.o

#ARMCI can be built without using fortran code when NOFORT is defined
#NOFORT =1
#ifdef GA_C_CORE
#  NOFORT =1
#endif
ifndef NOFORT
OBJ += faccumulate.o fcopy.o
endif

# ARMCI can use non-blocking calls for strided operations (where appropriate)
#NB_NONCONT = 1

# if enabled procedures are thread safe; one should specify the maximum number
# of threads (environment ARMCI_MAX_THREADS) and thread library (below)
# note, supported for the following networks: SOCKETS ELAN4 OPENIB LAPI64
#THREAD_SAFE     = 1
#THREAD_LIBRARY  = POSIX_THREADS

#default message-passing library for ARMCI is MPI
ifndef MSG_COMMS
  MSG_COMMS=MPI
endif

#defines communication libs and symbols
include ../config/makemp.h
include ../config/makecoms.h

ifeq ($(MSG_COMMS),MPI)
  OBJ += groups.o
endif

ifeq ($(TARGET),BGL) 
ifeq ($(MSG_COMMS), BGMLMPI)
  OBJ+= message.o bgml.o groups.o
else
  OBJ+= bgml.o
endif
endif 

ifeq ($(TARGET),BGP)
ifeq ($(MSG_COMMS), DCMFMPI)
   OBJ += message.o groups.o
endif
   OBJ += request.o
   LIB_DEFINES += -DARMCIX
   DCMF_VER = dcmf2
   OBJ += armcix/$(DCMF_VER)/armcix_impl.o armcix/$(DCMF_VER)/armcix_wait.o \
	  armcix/$(DCMF_VER)/armcix_fence.o armcix/$(DCMF_VER)/armcix_rmw.o \
	  armcix/$(DCMF_VER)/armcix_get.o armcix/$(DCMF_VER)/armcix_put.o   \
	  armcix/$(DCMF_VER)/armcix_lock.o armcix/$(DCMF_VER)/armcix_acc.o
endif


SYSTEM_V = SUN SOLARIS SOLARIS64 SGI SGI_N32 SGI64 SGITFP 
SYSTEM_V += HPUX HPUX64 IBM IBM64 DECOSF LINUX LINUX64 INTERIX NEC
SHMEM    = $(SYSTEM_V) LAPI LAPI64
CKPT = LINUX LINUX64 MACX MACX64
SHMMMAP  = NEC
SOCKETS  = $(SYSTEM_V) MACX MACX64
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP CRAY-SV2  LAPI FUJITSU-VPP FUJITSU-AP BGML BGP
NB_NONCONT_NET = CRAY-SHMEM #QUADRICS ELAN4 PORTALS
#USE_PTHREADS = LINUX SGI_N32 SOLARIS
#USE_POSIX = IBM

ifeq ($(TARGET),$(findstring $(TARGET),$(USE_PTHREADS)))
     LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT
     EXTRA_LIBS += -lpthread
endif

ifeq ($(TARGET),$(findstring $(TARGET),$(CKPT)))
ifdef ENABLE_CKPT
     OBJ += armci_chkpt.o armci_storage.o
endif
endif

ifeq ($(TARGET),BGML)
     LIB_DEFINES += -DNO_SHM -DUSE_MALLOC
endif

ifeq ($(TARGET), BGP)
   LIB_DEFINES += -DNO_SHM
endif

ifeq ($(TARGET),CATAMOUNT)
     LIB_DEFINES += -DCLUSTER
     ifndef ARMCI_NETWORK
        ARMCI_NETWORK = CRAY_SHMEM
     endif
endif
#uncomment 3 lines below to use GPC in armci
#LIB_DEFINES += -DARMCI_ENABLE_GPC_CALLS
#OBJ += gpc.o
#MP_DEFINES += -DARMCI_ENABLE_GPC_CALLS
#LIB_DEFINES += -DCHANGE_SERVER_AFFINITY

ifeq ($(TARGET),HITACHI)
     LIB_DEFINES += -DMMAP -DCLUSTER
#     OBJ += shmalloc.o signaltrap.o winshmem.o spawn.o ds-shared.o request.o
#     LIB_DEFINES += -DDATA_SERVER -DMMAP
     OBJ +=  signaltrap.o winshmem.o ds-shared.o request.o buffers.o
endif

ifeq ($(TARGET),$(findstring $(TARGET),$(SHMEM)))
     LIB_DEFINES += -DSYSV
     OBJ += signaltrap.o 
  ifeq ($(TARGET),$(findstring $(TARGET),$(SHMMMAP)))
     OBJ += winshmem.o 
  else
     OBJ += shmlimit.o shmem.o 
  endif
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

ifeq ($(TARGET),MACX)
 OBJ += semaphores.o 
endif

ifeq ($(TARGET),MACX64)
 OBJ += semaphores.o 
endif

ifeq ($(ARMCI_NETWORK),GM)
#  OBJ += dataserv-myri.o myrinet.o spawn.o request.o 
  OBJ += dataserv-gm.o ds-shared.o myrinet.o spawn.o request.o bufalloc.o async.o buffers.o
  OBJ += regions.o gpc.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DALLOW_PIN -DSERVER_THREAD -D_REENTRANT
endif

ifeq ($(ARMCI_NETWORK),VIA)
  OBJ += spawn.o  request.o via.o ds-shared.o buffers.o async.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DSERVER_THREAD -D_REENTRANT
endif

ifeq ($(ARMCI_NETWORK),PORTALS)
  OBJ += portals.o regions.o
  OBJ += signaltrap.o 
ifeq ($(TARGET),LINUX64)
  LIB_DEFINES += -DACC_SMP #-DPUT_LOCAL_ONLY_COMPLETION
endif
  LIB_DEFINES += -DXT3
  EXTRA_LIBS += -lpthread
endif

ifeq ($(ARMCI_NETWORK),OPENIB)
  OBJ += spawn.o  request.o ds-shared.o buffers.o async.o openib.o
  OBJ += regions.o gpc.o pendbufs.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DSERVER_THREAD -D_REENTRANT -DOPENIB -DVAPI -DALLOW_PIN -DPEND_BUFS
  EXTRA_LIBS += -lpthread
endif

ifeq ($(ARMCI_NETWORK),MELLANOX)
  OBJ += spawn.o  request.o vapi.o ds-shared.o buffers.o async.o
  OBJ += regions.o gpc.o
  LIB_DEFINES += -DPTHREADS -DDATA_SERVER -DSERVER_THREAD -D_REENTRANT -DMELLANOX -DVAPI -DALLOW_PIN
  EXTRA_LIBS += -lpthread
endif

ifeq ($(ARMCI_NETWORK),LAPI)
  OBJ += lapi.o request.o buffers.o async.o
endif
ifdef LAPI_RDMA
  OBJ += regions.o
  LIB_DEFINES += -DALLOW_PIN -DLAPI_RDMA
endif

ifeq ($(ARMCI_NETWORK),QUADRICS)
# works for Elan-3 network
# shared memory causes problems for shmem: NO_SHM disables use of shared memory
#   LIB_DEFINES += -DNO_SHM
ifdef LIBELAN_NATTACH
    LIB_DEFINES +=  -DMULTI_CTX -DACC_SMP
else
    LIB_DEFINES += -DALLOC_MUNMAP -DACC_SMP
endif
    OBJ += elan.o
    LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT # -DELAN_ACC
    OBJ += spawn.o ds-shared.o request.o buffers.o
    EXTRA_LIBS += -lpthread
endif

ifeq ($(ARMCI_NETWORK),ELAN4)
# works for Quadrics Elan-4 network
    OBJ += elan4.o pgs.o spawn.o ds-shared.o request.o buffers.o gpc.o
    LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT -DACC_SMP
    EXTRA_LIBS += -lpthread
endif

ifeq ($(ARMCI_NETWORK),MPI-SPAWN)
# works for ARMCi on top of MPI-2
    OBJ += request.o ds-shared.o buffers.o mpi2_client.o mpi2_server.o
    LIB_DEFINES += -DDATA_SERVER
    MP_DEFINES += -DMPI_SPAWN
ifdef ARMCI_SPAWN_CRAY_XT
      MP_DEFINES += -DSPAWN_CRAY_XT
endif
    EXTRA_LIBS += -lpthread
endif


ifeq ($(ARMCI_NETWORK), CRAY-SHMEM)
# works for Cray XT (Shmem network protocol)
    OBJ += shmalloc.o
    LIB_DEFINES += -DXT3
endif

ifdef ARMCI_NETWORK
  LIBRARY_STAMP = $(ARMCI_NETWORK).$(MSG_COMMS)
else
  # sockets
  ARMCI_NETWORK = SOCKETS
endif

ifeq ($(ARMCI_NETWORK),SOCKETS)
  LIBRARY_STAMP = $(MSG_COMMS)
  ifeq ($(TARGET),$(findstring $(TARGET),$(SOCKETS)))
     OBJ += spawn.o dataserv.o sockets.o request.o ds-shared.o buffers.o async.o
     LIB_DEFINES += -DDATA_SERVER -DSOCKETS
     # to use pthread version uncomment the following two lines
     # LIB_DEFINES += -DPTHREADS -DSERVER_THREAD -D_REENTRANT
     # EXTRA_LIBS += -lpthread
  endif
endif

ifdef NB_NONCONT
	ifeq ($(ARMCI_NETWORK),$(findstring $(ARMCI_NETWORK),$(NB_NONCONT_NET)))
		LIB_DEFINES += -DNB_NONCONT
		OBJ += noncont.o
	endif
endif


ifdef GA_USE_VAMPIR
   OBJ += armci_vampir.o
   LIB_INCLUDES += -I../../include
endif
ifeq ($(ARMCI_PROFILE),YES) 
   OBJ += armci_profile.o
   LIB_DEFINES += -DARMCI_PROFILE
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

ifdef ENABLE_CKPT
MP_DEFINES+=-DDO_CKPT 
endif

ifdef THREAD_SAFE
    LIB_DEFINES += -DTHREAD_SAFE
    OBJ += threads.o utils.o
    THREAD_SAFE_NET = SOCKETS LAPI64 ELAN OPENIB
    ifeq ($(ARMCI_NETWORK),$(findstring $(ARMCI_NETWORK),$(THREAD_SAFE_NET)))
        ifeq ($(THREAD_LIBRARY),POSIX_THREADS)
            LIB_DEFINES	+= -DPOSIX_THREADS -D_REENTRANT
            MP_DEFINES  += -DTHREAD_SAFE -DPOSIX_THREADS -D_REENTRANT
            EXTRA_LIBS	+= -lpthread
        endif
    endif

    ifneq ($(THREAD_LIBRARY),POSIX_THREADS)
error:
	@echo "Error: THREAD LIBS OTHER THAN PTHREADS ARE NOT SUPPORTED !!!"
	exit 1
    endif
endif # end: ifdef THREAD_SAFE

ifeq ($(ARMCI_GROUP),YES)
   MP_DEFINES += -DARMCI_GROUP
   LIB_DEFINES += -DARMCI_GROUP
endif

#......................................................................
LIBRARY = libarmci.a
LIBRARY_SHARED = libarmci.so
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

%.x:%.c  $(FULL_LIBRARY_PATH)
	$(CC) -g -I. $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) $(LIB_DEFINES) -o $@ $^ $(LIBS) -lm

ipctest.x: $(FULL_LIBRARY_PATH)
	$(CC) $(CFLAGS) $(MP_DEFINES) ipctest.c -o $@ $^ $(LIBS)


# $(FULL_LIBRARY_PATH)(winshmem.o):    ./shmem.h ./kr_malloc.h ./armcip.h
# $(FULL_LIBRARY_PATH)(vector.o):      ./memlock.h ./copy.h ./armcip.h ./acc.h 
# $(FULL_LIBRARY_PATH)(strided.o):     ./memlock.h ./copy.h ./armcip.h ./acc.h ./mpi2.h
# $(FULL_LIBRARY_PATH)(shmem.o):       ./shmem.h ./kr_malloc.h ./shmlimit.h ./message.h ./armcip.h
# $(FULL_LIBRARY_PATH)(semaphores.o):  ./semaphores.h 
# $(FULL_LIBRARY_PATH)(rmw.o): 	./locks.h ./copy.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(request.o):     ./request.h ./memlock.h ./copy.h ./armcip.h ./shmem.h ./mpi2.h
# $(FULL_LIBRARY_PATH)(pack.o):        ./armcip.h ./request.h
# $(FULL_LIBRARY_PATH)(mutex.o):       ./request.h ./copy.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(message.o):     ./message.h ./armcip.h  ./copy.h
# $(FULL_LIBRARY_PATH)(memory.o):      ./shmem.h ./message.h ./armcip.h ./kr_malloc.h
# $(FULL_LIBRARY_PATH)(memlock.o):     ./memlock.h ./locks.h ./copy.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(locks.o):       ./locks.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(lapi.o):        ./copy.h ./armcip.h ./lapidefs.h
# $(FULL_LIBRARY_PATH)(myrinet.o):     ./copy.h ./armcip.h ./request.h
# $(FULL_LIBRARY_PATH)(ipctest.o):     ./shmem.h ./locks.h 
# $(FULL_LIBRARY_PATH)(fence.o):       ./copy.h ./armcip.h ./armci.h 
# $(FULL_LIBRARY_PATH)(ds-shared.o):   ./myrinet.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h ./mpi2.h
# $(FULL_LIBRARY_PATH)(dataserv.o):    ./sockets.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(dataserv-gm.o): ./myrinet.h ./request.h ./message.h ./memlock.h ./copy.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(clusterinfo.o): ./message.h ./armcip.h 
# $(FULL_LIBRARY_PATH)(armci.o):       ./signaltrap.h ./shmem.h ./memlock.h ./locks.h ./lapidefs.h ./copy.h 
# $(FULL_LIBRARY_PATH)(kr_malloc.o): ./kr_malloc.h ./armcip.h ./shmem.h ./locks.h
# $(FULL_LIBRARY_PATH)(mpi2_server.o): ./mpi2.h ./kr_malloc.h ./armcip.h ./locks.h
# $(FULL_LIBRARY_PATH)(mpi2_client.o): ./mpi2.h ./kr_malloc.h ./armcip.h ./locks.h request.h

# locks.h: semaphores.h
# armcip.h: message.h lapidefs.h memlock.h
# %.x: %.c

#Hack to derive fortran & .gcc file names from object file names
#SRCS = $(subst x86copy.c,x86copy.gcc,$(subst tas.c,tas.gcc,$(subst fcopy.c,fcopy.f,$(subst faccumulate.c,faccumulate.f,$(OBJ:.o=.c)))))

PHONY: depend

depend: 
	./depend.py *.f *.c *.gcc > deps.h

include deps.h


