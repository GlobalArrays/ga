# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# makefile.files.h -- specifies files for the GA library depending 
#                     on TARGET and VERSION 
#
# components: GA_OBJ GA_ALG GA_UTIL
#........................................................................
#
#
INTERRUPT_AVAILABLE = SP1 IPSC DELTA PARAGON

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
     ifeq ($(TARGET),SGITFP)
            GA_SYNC =  sgi.locks.o
     endif
else
     ifeq ($(INTEL),YES)
            EXTRA = memcpy.i860.o
        ifeq ($(TARGET),PARAGON)
            EXTRA += fops.2d.parag.o bcopy.o
        endif
     endif
endif

ifeq ($(TARGET),SGITFP)
#       EXTRA = daxpy.o.sgi
endif
    

#
#
#                Core Routines of GAs
#
ifeq ($(VERSION),SHMEM)
     ifneq ($(TARGET),CRAY-T3D)
          IPC = shmem.o signal.o shmalloc.o
     endif
else
     ifeq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
          GA_HANDLER = ga_handler.o
     endif
endif
GA_CORE = global.core.o global.util.o global.patch.o global.msg.o \
          global.server.o

ifdef USE_MPI
  GA_CORE += mpi.o
else
  GA_CORE += tcgmsg.o
endif

ifdef USE_SUMMA
  SUMMA_OBJ = ga_summa_layout.o    summa_abt2.o \
              ga_summa_layout2.o   ga_summa_to_ga.o \
              summa_atb.o          summa_ab.o \
              summa_atb2.o         ga_create3.o \
              ga_summa.o           summa_ab2.o \
              ga_summa_alloc.o     ga_summa_c.o \
              summa_abt.o          ga_summa_cc.o

  VPATH += summa
  GLOB_INCLUDES += -I../../tcgmsg/ipcv4.0
   GLOB_DEFINES += -DUSE_SUMMA
endif

GA_OBJ = $(GA_CORE) $(GA_SYNC) $(GA_HANDLER) $(IPC)

#
#
#                  Linear Algebra
#
GA_ALG_BLAS = global.alg.o ga_dgemm.o ga_symmetrize.o ga_diag_seq.o rsg.o\
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
GA_UTIL = ffflush.o ifill.o dfill.o ga_summarize.o hsort.scat.o global.ma.o\
          DP.o fops.2d.o


$(GA_CORE)     : globalp.h global.h
global.core.o  : global.core.h message.h interrupt.h mem.ops.h
global.server.o: globalp.h message.h interrupt.h
global.msg.o   : message.h globalp.h
global.alg.o   : globalp.h global.h
ga_handler.o   : interrupt.h message.h
hsort.scat.o   : types.f2c.h
semaphores.o   : semaphores.h
shmalloc.o     : shmalloc.h
global.h       : types.f2c.h cray.names.h
