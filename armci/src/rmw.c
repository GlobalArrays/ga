#include "armcip.h"
#include "locks.h"
#include "copy.h"
#include <stdio.h>

void armci_generic_rmw(int op, int *ploc, int *prem, int extra, int proc)
{
    int lock = proc%NUM_LOCKS;
    NATIVE_LOCK(lock);
      if(op ==ARMCI_FETCH_AND_ADD){
                volatile int temp;
                armci_get(prem,ploc,sizeof(int),proc);
                temp = *ploc + extra;
                armci_put((int*)&temp,prem,sizeof(int),proc);
      }else{
                volatile long temp;
                armci_get(prem,ploc,sizeof(long),proc);
                temp = *(long*)ploc + extra;
                armci_put((long*)&temp,prem,sizeof(long),proc);
      }

      ARMCI_Fence(proc); /* we need fence before unlocking */

    NATIVE_UNLOCK(lock);
}


int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc)
{

    if(op != ARMCI_FETCH_AND_ADD && op != ARMCI_FETCH_AND_ADD_LONG)
                              armci_die("rmw: op type not supported",op);

#if defined(CLUSTER) && !defined(LAPI)
     if(!SAMECLUSNODE(proc)){
       armci_rem_rmw(op, ploc, prem,  extra, proc);
       return 0;
     }
#endif

#  ifdef _CRAYMPP
        /* here sizeof(long)= sizeof(int) */
        {
#         include <limits.h>
          long lval;
          while ( (lval = shmem_swap((long*)prem, INVALID, proc) ) == INVALID);
          *(int*)ploc   = lval;
          (void) shmem_swap((long*)prem, (lval + extra), proc);
        }
#  elif defined(LAPI)
   {      int rc, local;
          lapi_cntr_t req_id;
          if( rc = LAPI_Setcntr(lapi_handle,&req_id,0))
                        armci_die("setcntr failed",rc);
          if( rc = LAPI_Rmw(lapi_handle, FETCH_AND_ADD, proc, prem,
                        &extra, &local, &req_id)) armci_die("rmw failed",rc);
          if( rc = LAPI_Waitcntr(lapi_handle, &req_id, 1, NULL))
                        armci_die("wait failed",rc);
          *ploc  = local;
   }
#else
    armci_generic_rmw(op, ploc, prem,  extra, proc);
#endif

    return 0;
}

