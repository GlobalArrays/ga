#include "armcip.h"
#include "locks.h"
#include "copy.h"
#include <stdio.h>


int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc)
{
    int lock = proc%NUM_LOCKS;

    if(op != ARMCI_FETCH_AND_ADD && op != ARMCI_FETCH_AND_ADD_LONG) 
                              armci_die("rmw: op type not supported",op);
#  ifdef _CRAYMPP
        /* here sizeof(long)= sizeof(int) */
        { 
#         include <limits.h>
#         define INVALID (long)(_INT_MIN_64 +1)
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
#endif

    return 0;
}
