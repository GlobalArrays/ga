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
          GA_SYNC =  semaphores.o 
     endif
endif
#
#
#                Core Routines of GAs
#
ifeq ($(VERSION),SHMEM)
     GA_CORE = global.common.o global.shm.o global.ma.o shmem.o signal.o \
               global.patch.o shmalloc.o
else
     GA_CORE = global.common.o global.tcgmsg.o ma_addressing.o hsort.scat.o\
               global.patch.o  
     ifeq ($(TARGET),$(findstring $(TARGET),$(INTERRUPT_AVAILABLE)))
          GA_HANDLER = ga_handler.o
     endif
endif
GA_OBJ = $(GA_CORE) $(GA_SYNC) $(GA_HANDLER)
#
#
#                  Linear Algebra
#
GA_ALG_BLAS = global.alg.o ga_dgemm.o ga_symmetrize.o ga_diag_seq.o rsg.o\
              rs-mod.o ga_solve_seq.o ga_transpose.o ga_cholesky.o 
#
ifeq ($(DIAG),PAR)
     GA_ALG_DIAG = ga_diag.o rsg.o
endif
#
ifeq ($(LU_SOLVE),PAR)
     GA_ALG_SOLVE= SLface.o ga_solve.o 
endif
GA_ALG = $(GA_ALG_BLAS) $(GA_ALG_DIAG) $(GA_ALG_SOLVE)
#
#
#                 Utility Routines
#
GA_EXTRA_COMMON = ffflush.o ifill.o dfill.o 
ifneq ($(VERSION),SHMEM)
     GA_EXTRA = lenstr.o icopy.o dcopy.o 
endif
GA_UTIL = $(GA_EXTRA) $(GA_EXTRA_COMMON)

