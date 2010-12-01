#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "armcip.h"
#include "armci.h"
#include "copy.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if defined(PVM)
#   include <pvm3.h>
#elif defined(TCGMSG)
#   include <tcgmsg.h>
#elif defined(BGML)
#   include "bgml.h"
#else
#   include <mpi.h>
#endif

char *_armci_fence_arr;

#ifdef GA_USE_VAMPIR
#include "armci_vampir.h"
#endif
#ifdef ARMCI_PROFILE
#include "armci_profile.h"
#endif
void armci_init_fence()
{
#if defined (DATA_SERVER) || defined(PORTALS)
#if defined(THREAD_SAFE)
     _armci_fence_arr = calloc(armci_nproc*armci_user_threads.max,1);
#else
     _armci_fence_arr=calloc(armci_nproc,1);
#endif
     if(!_armci_fence_arr)
         armci_die("armci_init_fence: calloc failed",0);
#endif
}

#ifdef PORTALS
void armci_update_fence_array(int proc, int inc)
{
    if (inc)
        FENCE_ARR(proc)++;
    else
        FENCE_ARR(proc)--;
}
#endif


void PARMCI_Fence(int proc)
{
#ifdef GA_USE_VAMPIR
     vampir_begin(ARMCI_FENCE,__FILE__,__LINE__);
 if (armci_me != proc)
        vampir_start_comm(proc,armci_me,0,ARMCI_FENCE);
#endif
#ifdef ARMCI_PROFILE
 if (!SAMECLUSNODE(proc))
 armci_profile_start(ARMCI_PROF_FENCE);
#endif

#if defined(DATA_SERVER) && !(defined(GM) && defined(ACK_FENCE))
     if(FENCE_ARR(proc) && (armci_nclus >1)){

           int cluster = armci_clus_id(proc);
           int master = armci_clus_info[cluster].master;

           armci_rem_ack(cluster);

           bzero(&FENCE_ARR(master),
                   armci_clus_info[cluster].nslave);
     }
#elif defined(ARMCIX)
     ARMCIX_Fence (proc);
#elif defined(BGML)
     BGML_WaitProc(proc);
     MEM_FENCE;
#else
     FENCE_NODE(proc);
     MEM_FENCE;
#endif
#ifdef ARMCI_PROFILE
 if (!SAMECLUSNODE(proc))
 armci_profile_stop(ARMCI_PROF_FENCE);
#endif
#ifdef GA_USE_VAMPIR
     if (armci_me != proc)
        vampir_end_comm(proc,armci_me,0,ARMCI_FENCE);
     vampir_end(ARMCI_FENCE,__FILE__,__LINE__);
#endif
}


void PARMCI_AllFence()
{
#ifdef GA_USE_VAMPIR
    vampir_begin(ARMCI_ALLFENCE,__FILE__,__LINE__);
#endif
#ifdef ARMCI_PROFILE
    armci_profile_start(ARMCI_PROF_ALLFENCE);
#endif
#if defined(ARMCIX)
    ARMCIX_AllFence ();
#elif defined(BGML)
    BGML_WaitAll();
#elif defined(LAPI) || defined(CLUSTER)
#if defined(GM) && !defined(ACK_FENCE)
    _armci_amina_allfence(); 
#else
    { int p; for(p=0;p<armci_nproc;p++)PARMCI_Fence(p); }
#endif
#endif
#ifdef ARMCI_PROFILE
    armci_profile_stop(ARMCI_PROF_ALLFENCE);
#endif
#ifdef GA_USE_VAMPIR
    vampir_end(ARMCI_ALLFENCE,__FILE__,__LINE__);
#endif
    MEM_FENCE;
}

void PARMCI_Barrier()
{
    if(armci_nproc==1)return;
#ifdef ARMCI_PROFILE
    armci_profile_start(ARMCI_PROF_BARRIER);
#endif
#if defined(BGML)
    BGML_WaitAll();
    bgml_barrier(3);
#else
    PARMCI_AllFence();
#  ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#  else
    {
       long type=ARMCI_TAG;
       tcg_synch(type);
    }
#  endif
#endif
    MEM_FENCE;
#ifdef ARMCI_PROFILE
    armci_profile_stop(ARMCI_PROF_BARRIER);
#endif

}
