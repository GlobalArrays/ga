/* $Id: fence.c,v 1.10 2003-03-10 18:51:06 vinod Exp $ */
#include "armcip.h"
#include "armci.h"
#include "copy.h"

#ifdef CLUSTER
   char *_armci_fence_arr;
#endif

#ifdef GA_USE_VAMPIR
#include "armci_vampir.h"
#endif

void armci_init_fence()
{
#ifdef DATA_SERVER
# ifdef GM /*when fence moves to ds-shared, fence_init would become common*/
     armci_gm_fence_init();
#endif
     _armci_fence_arr=calloc(armci_nproc,1);
     if(!_armci_fence_arr)armci_die("armci_init_fence: calloc failed",0);
#endif
}

void ARMCI_Fence(int proc)
{
#ifdef GA_USE_VAMPIR
     vampir_begin(ARMCI_FENCE,__FILE__,__LINE__);
 if (armci_me != proc)
        vampir_start_comm(proc,armci_me,0,ARMCI_FENCE);
#endif
#if defined(DATA_SERVER) && !(defined(GM) && defined(ACK_FENCE))
     if(_armci_fence_arr[proc] && (armci_nclus >1)){
         
           int cluster = armci_clus_id(proc);
           int master=armci_clus_info[cluster].master;

           armci_rem_ack(cluster);

           /* one ack per cluster node suffices */
           bzero(_armci_fence_arr+master, armci_clus_info[cluster].nslave); 

     }
#else
     FENCE_NODE(proc);
#endif
#ifdef GA_USE_VAMPIR
     if (armci_me != proc) 
        vampir_end_comm(proc,armci_me,0,ARMCI_FENCE);
     vampir_end(ARMCI_FENCE,__FILE__,__LINE__);
#endif
}


void ARMCI_AllFence()
{
#if defined(LAPI) || defined(CLUSTER)
     int p;
#endif
#ifdef GA_USE_VAMPIR
     vampir_begin(ARMCI_ALLFENCE,__FILE__,__LINE__);
#endif
#ifdef _CRAYMPP
     if(cmpl_proc != -1) FENCE_NODE(cmpl_proc);
#elif defined(LAPI) || defined(CLUSTER)
     for(p=0;p<armci_nproc;p++)ARMCI_Fence(p);
#endif
#ifdef GA_USE_VAMPIR
     vampir_end(ARMCI_ALLFENCE,__FILE__,__LINE__);
#endif
}
