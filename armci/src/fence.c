#include "armcip.h"
#include "armci.h"
#include "copy.h"

#ifdef CRAY
#include <mpp/shmem.h>
#endif 



void ARMCI_Fence(int proc)
{
     FENCE_NODE(proc);
}


void ARMCI_AllFence()
{
#ifdef CRAY
     if(cmpl_proc != -1) FENCE_NODE(cmpl_proc);
#elif defined(LAPI)
     int p;
     for(p=0;p<armci_nproc;p++)ARMCI_Fence(p);
#endif
}
