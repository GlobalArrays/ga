#include "armcip.h"
#include "armci.h"
#include "copy.h"

#ifdef CRAY
#include <mpp/shmem.h>
#endif 


#ifdef CLUSTER
   char *_armci_fence_arr;
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
}


void ARMCI_AllFence()
{
#ifdef CRAY
     if(cmpl_proc != -1) FENCE_NODE(cmpl_proc);
#elif defined(LAPI) || defined(CLUSTER)
     int p;
     for(p=0;p<armci_nproc;p++)ARMCI_Fence(p);
#endif
}
