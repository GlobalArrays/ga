# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# MakeFiles.h -- specifies files for the GA library depending 
#                     on TARGET and VERSION 
#
# components: GA_OBJ GA_ALG GA_UTIL
#........................................................................
#
#
INTERRUPT_AVAILABLE = SP1 SP IPSC DELTA PARAGON
NATIVE LOCKS = SGITFP SGI64 CONVEX-SPP

#                  Synchronization
#
ifeq ($(VERSION),SHMEM)
     ifeq ($(TARGET),KSR)
          GA_SYNC = barrier.KSR.o
          EXTRA = ksrcopy.o
     else
       ifneq ($(TARGET),CRAY-T3D)
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
     ifneq ($(TARGET),CRAY-T3D)
          IPC = shmem.o shmalloc.o signal.o
     endif
else
     ifeq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
          GA_HANDLER = ga_handler.o
     endif
endif
GA_CORE := global.core.o global.util.o global.patch.o global.msg.o \
           global.serv.o

ifdef IWAY
  GA_CORE += iway.o net.o
else
  ifdef USE_MPI
    GA_CORE += mpi.o
  else
    GA_CORE += tcgmsg.o
  endif
endif

GA_OBJ = $(GA_CORE) $(GA_SYNC) $(GA_HANDLER) $(IPC)

#
#
#                  Linear Algebra
#
GA_ALG_BLAS = global.alg.o ga_dgemm.o ga_symmetr.o ga_diag_seq.o rsg.o\
              rs-mod.o ga_solve_seq.o ga_transpose.o ga_cholesky.o 
#
#ifeq ($(DIAG),PAR)
     GA_ALG_DIAG = ga_diag.o 
#endif
#
ifeq ($(LU_SOLVE),PAR)
     GA_ALG_SOLVE= SLface.o ga_solve.o 
endif
GA_ALG = $(GA_ALG_BLAS) $(GA_ALG_DIAG) $(GA_ALG_SOLVE)
#
#
#                 Utility Routines
#
GA_UTIL = ffflush.o fill.o ga_summarize.o hsort.scat.o global.ma.o\
          DP.o fops.2d.o

OBJ_FRAGILE = $(GA_SYNC) $(GA_HANDLER) $(IPC)
OBJ = $(GA_CORE) $(GA_ALG) $(GA_UTIL) $(EXTRA) $(OBJ_FRAGILE)
