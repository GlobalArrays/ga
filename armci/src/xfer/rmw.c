#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "armcip.h"
#include "locks.h"
#include "copy.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if (defined(__i386__) || defined(__x86_64__)) && !defined(NO_I386ASM)
#  include "atomics-i386.h"
#endif


/* enable use of newer interfaces in SHMEM */
#ifndef CRAY
#ifndef LIBELAN_ATOMICS
/* manpages for shmem_fadd exist on the T3E but library code does not */
#define SHMEM_FADD 
#endif
#endif


/* global scope to prevent compiler optimization of volatile code */
int  _a_temp;
long _a_ltemp;

void armci_generic_rmw(int op, void *ploc, void *prem, int extra, int proc)
{
#if defined(CLUSTER) && !defined(SGIALTIX)
    int lock = (proc-armci_clus_info[armci_clus_id(proc)].master)%NUM_LOCKS;
#else
    int lock = 0;
#endif

    NATIVE_LOCK(lock,proc);

    switch (op) {
      case ARMCI_FETCH_AND_ADD:
#if (defined(__i386__) || defined(__x86_64__)) && !defined(NO_I386ASM)
#if (defined(__GNUC__) || defined(__INTEL_COMPILER__) ||defined(__PGIC__)) && !defined(NO_I386ASM)
        if(SERVER_CONTEXT || armci_nclus == 1){
/* 	  *(int*)ploc = __sync_fetch_and_add((int*)prem, extra); */
	  atomic_fetch_and_add(prem, ploc, extra, sizeof(int));
	}
	else 
#endif
#endif
	  {
                armci_get(prem,ploc,sizeof(int),proc);
                _a_temp = *(int*)ploc + extra;
                armci_put(&_a_temp,prem,sizeof(int),proc);
	  }
           break;
      case ARMCI_FETCH_AND_ADD_LONG:
                armci_get(prem,ploc,sizeof(long),proc);
                _a_ltemp = *(long*)ploc + extra;
                armci_put(&_a_ltemp,prem,sizeof(long),proc);
           break;
      case ARMCI_SWAP:
#if (defined(__i386__) || defined(__x86_64__)) && !defined(NO_I386ASM)
        if(SERVER_CONTEXT || armci_nclus==1){
	  atomic_exchange(ploc, prem, sizeof(int));
        }
        else 
#endif
        {
	  armci_get(prem,&_a_temp,sizeof(int),proc);
	  armci_put(ploc,prem,sizeof(int),proc);
	  *(int*)ploc = _a_temp; 
        }
	break;
      case ARMCI_SWAP_LONG:
                armci_get(prem,&_a_ltemp,sizeof(long),proc);
                armci_put(ploc,prem,sizeof(long),proc);
                *(long*)ploc = _a_ltemp;
           break;
      default: armci_die("rmw: operation not supported",op);
    }
#ifdef VAPI
    if(!SERVER_CONTEXT)
#endif
      PARMCI_Fence(proc); 
    NATIVE_UNLOCK(lock,proc);
}


int PARMCI_Rmw(int op, void *ploc, void *prem, int extra, int proc)
{
#if defined(_CRAYMPP) || defined(QUADRICS) || defined(CRAY_SHMEM)
    int  ival;
    long lval;
#endif

#if defined(CLUSTER) && !defined(QUADRICS) &&!defined(CYGWIN)\
    && !defined(HITACHI) && !defined(CRAY_SHMEM) 
     if(!SAMECLUSNODE(proc)){
       armci_rem_rmw(op, ploc, prem,  extra, proc);
       return 0;
     }
#endif

#ifdef REGION_ALLOC
     if(SAMECLUSNODE(proc)) (void)armci_region_fixup(proc,&prem);
#endif
    switch (op) {
#   if defined(QUADRICS) || defined(_CRAYMPP) || defined(CRAY_SHMEM)
      case ARMCI_FETCH_AND_ADD:
#ifdef SHMEM_FADD
         /* printf(" calling intfdd arg %x %ld \n", prem, *prem); */
          *(int*) ploc = shmem_int_fadd(prem, extra, proc);
#elif defined(LIBELAN_ATOMICS)
          *(int*) ploc = elan_int_fadd(prem, extra, proc);
#else
          while ( (ival = shmem_int_swap(prem, INT_MAX, proc) ) == INT_MAX);
          (void) shmem_int_swap(prem, ival +extra, proc);
          *(int*) ploc = ival;
#endif
        break;
      case ARMCI_FETCH_AND_ADD_LONG:
#ifdef SHMEM_FADD
          *(long*) ploc = shmem_long_fadd( (long*)prem, (long) extra, proc);
#elif defined(LIBELAN_ATOMICS)
          *(long*) ploc = elan_long_fadd( (long*)prem, (long) extra, proc);
#else
          while ((lval=shmem_long_swap((long*)prem,LONG_MAX,proc)) == LONG_MAX);
          (void) shmem_long_swap((long*)prem, (lval + extra), proc);
          *(long*)ploc   = lval;
#endif
        break;
      case ARMCI_SWAP:
#ifdef LIBELAN_ATOMICS
          *(int*)ploc = elan_int_swap((int*)prem, *(int*)ploc,  proc); 
#else
          *(int*)ploc = shmem_int_swap((int*)prem, *(int*)ploc,  proc); 
#endif
        break;
      case ARMCI_SWAP_LONG:
#ifdef LIBELAN_ATOMICS
          *(long*)ploc = elan_long_swap((long*)prem, *(long*)ploc,  proc); 
#else
          *(long*)ploc = shmem_long_swap((long*)prem, *(long*)ploc,  proc); 
#endif
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

