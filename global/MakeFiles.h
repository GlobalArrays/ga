# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# MakeFiles.h -- specifies files for the GA library depending 
#                     on TARGET and VERSION 
#
# components: GA_OBJ GA_ALG GA_UTIL
#........................................................................
#
#
NATIVE LOCKS = SGITFP SGI64 SGI_N32 CONVEX-SPP

#                  Synchronization
#
ifeq ($(VERSION),SHMEM)
     ifeq ($(TARGET),KSR)
          GA_SYNC = barrier.KSR.o
          EXTRA = ksrcopy.o
     else
       ifneq ($(TARGET),$(findstring $(TARGET),$(ONESIDED_AVAILABLE)))
            GA_SYNC =  semaphores.o
       endif
     endif
     ifeq ($(TARGET),$(findstring $(TARGET),$(NATIVE LOCKS)))
            GA_SYNC +=  locks.o
     endif
else
     ifeq ($(INTEL),YES)
            EXTRA = memcpy.i860.o fops.2d.i860.o bcopy.i860.o
     endif
endif


#
#
#                Core Routines of GAs
#
ifeq ($(VERSION),SHMEM)
     ifneq ($(TARGET),$(findstring $(TARGET),$(ONESIDED_AVAILABLE)))
          IPC = shmem.o shmalloc.o signal.o
     endif
endif
ifeq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
          GA_HANDLER = ga_handler.o
endif

GA_CORE := global.core.o global.util.o global.patch.o global.msg.o \
           global.serv.o ga_lock.o fops.2d.o global.ma.o global.alg.o

ifeq ($(TARGET),LAPI)
  GA_CORE += lapi.o
endif

ifdef USE_ARMCI
  GA_CORE = global.armci.o global.util.o global.patch.o \
	    collect.o decomp.o capi.o global.nalg.o global.npatch.o
else
ifdef IWAY
  GA_CORE += iway.o net.o
else
  ifdef USE_MPI
    GA_CORE += mpi.o
  else
    GA_CORE += tcgmsg.o
  endif
endif
endif

  GA_OBJ = $(GA_CORE) $(GA_SYNC) $(GA_HANDLER) $(IPC)

#
#
#                  Linear Algebra
#
GA_ALG_BLAS = ga_dgemm.o ga_symmetr.o ga_diag_seq.o rsg.o\
              rs-mod.o ga_solve_seq.o ga_transpose.o 

ifeq ($(DIAG),PAR)
     GA_ALG_DIAG = ga_diag.o 
else
     GA_ALG_DIAG = peigstubs.o
endif

ifdef USE_SCALAPACK
     GA_ALG_SOLVE= SLface.o ga_solve.o ga_spd.o
else
     GA_ALG_SOLVE= sclstubs.o
endif

GA_ALG = $(GA_ALG_BLAS) $(GA_ALG_DIAG) $(GA_ALG_SOLVE)
#
#
#                 Utility Routines
#
GA_UTIL = ffflush.o fill.o ga_summarize.o hsort.scat.o\
          DP.o fort.o fops.2d.o

ifndef USE_ARMCI
  OBJ_FRAGILE = $(GA_SYNC) $(GA_HANDLER) $(IPC)
endif

OBJ = $(GA_CORE) $(GA_ALG) $(GA_UTIL) $(EXTRA) $(OBJ_FRAGILE)
