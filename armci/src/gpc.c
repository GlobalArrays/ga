/* $Id: gpc.c,v 1.2 2004-08-04 22:42:46 manoj Exp $ *****************************************************
  Prototype of Global Procedure Calls.
  July/03 JN - shared memory version  
  
*************************************************************/ 

#include <stdio.h>
#include "armcip.h"
#include "locks.h"

#define GPC_SLOTS 32 
#define GPC_OFFSET -100
static void *_table[GPC_SLOTS]={
(void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0,
(void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0,
(void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0,
(void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0, (void*)0};

/*\ callback functions must be registered -- user gets int handle back
\*/
int ARMCI_Gpc_register( void (*func) ())
{
int handle =-1, candidate = 0;

    do{
      if(!_table[candidate]){
           handle = candidate;
          _table[candidate]=func;
      }
      candidate++;
    }while(candidate < GPC_SLOTS && handle == -1);
    return(GPC_OFFSET-handle);
}

/*\ release/deassociate handle with previously registered callback function
\*/
void ARMCI_Gpc_release(int handle)
{
     int h = -handle + GPC_OFFSET;
     if(h<0 || h >= GPC_SLOTS) armci_die("ARMCI_Gpc_release: bad handle",h);
     _table[h] = (void*)0;
}



/*\ Send Request to Execute callback function in a global address space 
 *  Arguments:
 *  f     - handle to the callback function
 *  p     - remote processor
 *  hdr   - header data - used to pack extra args for callback (local buffer) 
 *  hlen  - size of header data < ARMCI_GPC_HLEN
 *  data  - bulk data passed to callback (local buffer)
 *  dlen  - length of bulk data
 *  rhdr  - ptr to reply header (return args from callback)
 *  rhlen - length of buffer to store reply header < ARMCI_GPC_HLEN  
 *  rdata - ptr to where reply data from callback should be stored (local buf)
 *  rdlen - size of the buffer to store reply data  
 *  nbh   - nonblocking handle
 *  
\*/
int ARMCI_Gpc_exec(int h, int p, void  *hdr, int hlen,  void *data,  int dlen,
                                 void *rhdr, int rhlen, void *rdata, int rdlen, 
                                                              armci_hdl_t* nbh)
{
int rhsize, rdsize;
void (*func)();
int hnd = -h + GPC_OFFSET;

    if(hnd <0 || hnd>= GPC_SLOTS) 
       armci_die2("ARMCI_Gpc_exec: bad callback handle",hnd,GPC_SLOTS);
    if(!_table[hnd]) armci_die("ARMCI_Gpc_exec: NULL function",hnd);

    func = _table[hnd];
    func(p, armci_me, hdr, hlen, data, dlen, rhdr, rhlen, &rhsize,
                                             rdata, rdlen, &rdsize); 
    return 0;
}

/*\
 *   This is a template for the callback function
 *   The arguments are passed as specified in ARMCI_Gpc_exec
 *   In addition,
 *      rhsize specifies the actual size of reply header data returned
 *      rdsize specifies the actual size of reply data returned
\*/
void example_func(int to, int from, void *hdr,   int hlen,
                                    void *data,  int dlen,
                                    void *rhdr,  int rhlen, int *rhsize,
                                    void *rdata, int rdlen, int *rdsize);
     
     
/*\
 *  Translate pointer to memory on processor "proc"
 *  to be used in a callback function send by processor "from"
\*/
void * ARMCI_Gpc_translate(void *ptr, int proc, int from)
{
return ptr;
}


/*\ acquire lock in a callback function executed in context of processor "proc"
\*/
void ARMCI_Gpc_lock(int proc)
{
#if defined(CLUSTER) && !defined(SGIALTIX)
    int lock = (proc-armci_clus_info[armci_clus_id(proc)].master)%NUM_LOCKS;
#else
    int lock = 0;
#endif
    NATIVE_LOCK(lock,proc);
}

/*\ try acquire lock in a callback function to be executed in context of
 *  processor "proc"
 *  return value: 1 - success
 *                0 - failure (already locked by another thread)
\*/
int ARMCI_Gpc_trylock(int proc)
{
armci_die("ARMCI_Gpc_trylock: not yet implemented",0);
return 0;
}

/*\ release lock in a callback function executed in context of processor "proc"
\*/
void ARMCI_Gpc_unlock(int proc)
{
#if defined(CLUSTER) && !defined(SGIALTIX)
    int lock = (proc-armci_clus_info[armci_clus_id(proc)].master)%NUM_LOCKS;
#else
    int lock = 0;
#endif
    NATIVE_UNLOCK(lock,proc);
}


