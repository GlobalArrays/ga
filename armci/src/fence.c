/* $Id: fence.c,v 1.6 2002-07-17 18:05:33 vinod Exp $ */
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
     _armci_fence_arr=calloc(armci_nproc,1);
     if(!_armci_fence_arr)armci_die("armci_init_fence: calloc failed",0);
#endif
}

void ARMCI_Fence(int proc)
{
#ifdef GA_USE_VAMPIR
     vampir_begin(ARMCI_FENCE,__FILE__,__LINE__);
#endif
#ifdef DATA_SERVER
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
     if (armci_me != proc) {
        (void) VT_log_sendmsg(proc,armci_me,0,ARMCI_FENCE,0);
        (void) VT_log_recvmsg(armci_me,proc,0,ARMCI_FENCE,0);
     };
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
