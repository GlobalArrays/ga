/* $Id: rmw.c,v 1.8 2000-08-01 22:37:04 d3h325 Exp $ */
#include "armcip.h"
#include "locks.h"
#include "copy.h"
#include <stdio.h>

/* enable use of newer interfaces in SHMEM */
#define SHMEM_FADD 

/* global scope to prevent compiler optimization of volatile code */
int  _a_temp;
long _a_ltemp;

void armci_generic_rmw(int op, void *ploc, void *prem, int extra, int proc)
{
    int lock = proc%NUM_LOCKS;

    NATIVE_LOCK(lock);
    switch (op) {
      case ARMCI_FETCH_AND_ADD:
                armci_get(prem,ploc,sizeof(int),proc);
                _a_temp = *(int*)ploc + extra;
                armci_put(&_a_temp,prem,sizeof(int),proc);
           break;
      case ARMCI_FETCH_AND_ADD_LONG:
                armci_get(prem,ploc,sizeof(long),proc);
                _a_ltemp = *(long*)ploc + extra;
                armci_put(&_a_ltemp,prem,sizeof(long),proc);
           break;
      case ARMCI_SWAP:
                armci_get(prem,&_a_temp,sizeof(int),proc);
                armci_put(ploc,prem,sizeof(int),proc);
                *(int*)ploc = _a_temp; 
           break;
      case ARMCI_SWAP_LONG:
                armci_get(prem,&_a_ltemp,sizeof(long),proc);
                armci_put(ploc,prem,sizeof(long),proc);
                *(int*)ploc = _a_ltemp;
           break;
      default: armci_die("rmw: operation not supported",op);
    }

    ARMCI_Fence(proc); /* we need fence before unlocking */
    NATIVE_UNLOCK(lock);
}


int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc)
{
#ifdef LAPI
    int  ival, rc, opcode=0;
    lapi_cntr_t req_id;
#elif defined(_CRAYMPP) || defined(QUADRICS)
    int  ival;
    long lval;
#endif

#if defined(CLUSTER) && !defined(LAPI) && !defined(QUADRICS)
     if(!SAMECLUSNODE(proc)){
       armci_rem_rmw(op, ploc, prem,  extra, proc);
       return 0;
     }
#endif

    switch (op) {
#   if defined(QUADRICS) || defined(_CRAYMPP)
      case ARMCI_FETCH_AND_ADD:
#ifdef SHMEM_FADD
          *(int*) ploc = shmem_int_fadd(prem, extra, proc);
#else
          while ( (ival = shmem_int_swap(prem, INT_MAX, proc) ) == INT_MAX);
          (void) shmem_int_swap(prem, ival +extra, proc);
          *(int*) ploc = ival;
#endif
        break;
      case ARMCI_FETCH_AND_ADD_LONG:
#ifdef SHMEM_FADD
          *(long*) ploc = shmem_long_fadd( (long*)prem, (long) extra, proc);
#else
          while ((lval=shmem_long_swap((long*)prem,LONG_MAX,proc)) == LONG_MAX);
          (void) shmem_long_swap((long*)prem, (lval + extra), proc);
          *(long*)ploc   = lval;
#endif
        break;
      case ARMCI_SWAP:
          *(int*)ploc = shmem_int_swap((int*)prem, *(int*)ploc,  proc); 
        break;
      case ARMCI_SWAP_LONG:
          *(long*)ploc = shmem_swap((long*)prem, *(long*)ploc,  proc); 
        break;
#   elif defined(LAPI)
      /************** here sizeof(long)= sizeof(int) **************/
      case ARMCI_FETCH_AND_ADD:
      case ARMCI_FETCH_AND_ADD_LONG:
           opcode = FETCH_AND_ADD;
      case ARMCI_SWAP:
      case ARMCI_SWAP_LONG:
          if(opcode!=FETCH_AND_ADD)opcode = SWAP;
          if( rc = LAPI_Setcntr(lapi_handle,&req_id,0))
                        armci_die("rmw setcntr failed",rc);
          if( rc = LAPI_Rmw(lapi_handle, opcode, proc, prem,
                        &extra, &ival, &req_id)) armci_die("rmw failed",rc);
          if( rc = LAPI_Waitcntr(lapi_handle, &req_id, 1, NULL))
                        armci_die("rmw wait failed",rc);
          *ploc  = ival;
        break;
#   else
      case ARMCI_FETCH_AND_ADD:
      case ARMCI_FETCH_AND_ADD_LONG:
      case ARMCI_SWAP:
      case ARMCI_SWAP_LONG:
           armci_generic_rmw(op, ploc, prem,  extra, proc);
        break;
#   endif
      default: armci_die("rmw: operation not supported",op);
    }

    return 0;
}

