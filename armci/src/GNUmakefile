         FOPT = -O
         COPT = -O

OBJ= armci.o copy.o memory.o fcopy.o fence.o rmw.o acc.o faccumulate.o memlock.o

MPI_LIB_NAME = -lmpi
INTERRUPT_AVAILABLE = SP1 SP PARAGON LAPI
SYSTEM_V = SUN SGI SGI_N32 SGI64 HPUX SOLARIS IBM DECOSF CONVEX LINUX
ONESIDED_AVAILABLE = CRAY-T3D CRAY-T3E CRAY-YMP LAPI FUJITSU-VPP
NATIVE_LOCKS = SGI SGI_N32 SGI64 CONVEX WIN32 CRAY-YMP
USE_POSIX = IBM

ifeq ($(TARGET),$(findstring $(TARGET),$(SYSTEM_V)))
     LIB_DEFINES += -DSYSV
     OBJ += shmalloc.o signaltrap.o
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

ifeq ($(TARGET),LAPI)
     OBJ += lapi.o
     MPI_LIB_NAME =
endif


ifneq ($(TARGET),$(findstring $(TARGET),$(ONESIDED_AVAILABLE)))
     LIB_DEFINES += -DSHMEM
endif

ifdef MPI_INCLUDE
LIB_INCLUDES += -I$(MPI_INCLUDE)
endif
ifdef MPI_LIB
LIBS += -L$(MPI_LIB)
endif

LIBS +=$(MPI_LIB_NAME)

LIBRARY = libarmci.a
HARDCLEAN = yes

# include NT files to clean when needed
NTSTUFF = *.ilk *.exe *.ncb ./Debug/*.obj ./Debug/vc* ./Debug/*.pdb ./Debug/*.pch
LIB_TARGETS = *.x $(NTSTUFF)
LIB_DISTRIB = ../lib

include ../config/makefile.h
include ../config/makelib.h


test.x: test.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

ipctest.x: ipctest.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

test.o: $(LIBRARY_PATH)
ipctest.o: $(LIBRARY_PATH)

$(patsubst %,$(LIBRARY_PATH)(%),copy.o): copy.h armcip.h
$(patsubst %,$(LIBRARY_PATH)(%),armci.o): armci.h armcip.h
$(patsubst %,$(LIBRARY_PATH)(%),memory.o): armci.h armcip.h locks.h shmem.h
$(patsubst %,$(LIBRARY_PATH)(%),shmem.o):  shmem.h shmalloc.h
$(patsubst %,$(LIBRARY_PATH)(%),shmalloc.o):   shmalloc.h
$(patsubst %,$(LIBRARY_PATH)(%),locks.o):   locks.h
$(patsubst %,$(LIBRARY_PATH)(%),rmw.o):   locks.h
$(patsubst %,$(LIBRARY_PATH)(%),acc.o):   acc.h
$(patsubst %,$(LIBRARY_PATH)(%),copy.o):   acc.h
$(patsubst %,$(LIBRARY_PATH)(%),semaphores.o):   semaphores.h
locks.h: semaphores.h
