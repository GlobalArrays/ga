#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "armcip.h"
#include "armci.h"
#include "copy.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if defined(TCGMSG)
#   include <sndrcv.h>
static void tcg_synch(long type)
{
    long atype = type;

    SYNCH_(&atype);
}
#else
#   include <mpi.h>
#endif

char *_armci_fence_arr;

void armci_init_fence()
{
#if defined (DATA_SERVER)
#if defined(THREAD_SAFE)
     _armci_fence_arr = calloc(armci_nproc*armci_user_threads.max,1);
#else
     _armci_fence_arr=calloc(armci_nproc,1);
#endif
     if(!_armci_fence_arr)
         armci_die("armci_init_fence: calloc failed",0);
#endif
}

void armci_finalize_fence()
{
#if defined (DATA_SERVER)
     free(_armci_fence_arr);
     _armci_fence_arr = NULL;
#endif
}

void PARMCI_Fence(int proc)
{
#if defined(DATA_SERVER)
     if(FENCE_ARR(proc) && (armci_nclus >1)){

           int cluster = armci_clus_id(proc);
           int master = armci_clus_info[cluster].master;

           armci_rem_ack(cluster);

           bzero(&FENCE_ARR(master),
                   armci_clus_info[cluster].nslave);
     }
#else
     FENCE_NODE(proc);
     MEM_FENCE;
#endif
}

void PARMCI_GroupFence(ARMCI_Group *group)
{
  /* Stub to prevent compilation problems with Comex build */
}


void PARMCI_AllFence()
{
#if defined(CLUSTER)
    int p;

    for(p = 0;p < armci_nproc; p++) {
        PARMCI_Fence(p); 
    }
#endif
    MEM_FENCE;
}

void PARMCI_Barrier()
{
    if (armci_nproc==1) return;
    PARMCI_AllFence();
#ifdef MSG_COMMS_MPI
    MPI_Barrier(ARMCI_COMM_WORLD);
#else
    {
       long type=ARMCI_TAG;
       tcg_synch(type);
    }
#endif
    MEM_FENCE;
}
