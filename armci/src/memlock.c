#include "armcip.h"
#include "locks.h"
#include "copy.h"
#include "memlock.h"
#include <stdio.h>

#define INVALID_VAL -9999999
static int locked_slot=INVALID_VAL;
static int locked_proc=INVALID_VAL;

volatile double armci_dummy_work=0.;
void **memlock_table_array;

/* constants for cache line alignment */
#ifdef SOLARIS
#  define CALGN 32
#  define LOG_CALGN 5
#else
#  define CALGN 64
#  define LOG_CALGN 6
#endif

#define ALIGN_ADDRESS(x)  (char*)( ( ((unsigned long)x) >> LOG_CALGN ) << LOG_CALGN ) 

/*\ idle for a time proportional to factor 
\*/
void armci_waitsome(int factor)
{
int i=factor*100000;

   if(factor <= 1) armci_dummy_work =0.;
   if(factor < 1) return;
   while(--i){
      armci_dummy_work = armci_dummy_work + 1./(double)i;  
   }
}
   

#ifdef CRAY_T3E
#pragma _CRI cache_align table
#endif
static memlock_t table[MAX_SLOTS];

/*\ acquire exclusive LOCK to MEMORY area <pstart,pend> owned by process "proc"
 *   . only one area can be locked at a time by the calling process
 *   . must unlock it with armci_unlockmem
\*/
void armci_lockmem(void *start, void *end, int proc)
{
     register void* pstart, *pend;
     register  int slot, avail=0;
     int turn=0, conflict=0;
     memlock_t *memlock_table = (memlock_t*)memlock_table_array[proc];
     register int lock = proc%NUM_LOCKS;
     locked_proc = proc;

#ifdef ALIGN_ADDRESS
     /* align address range on cache line boundary to avoid false sharing */
     pstart = ALIGN_ADDRESS(start);
     pend = CALGN -1 + ALIGN_ADDRESS(end);
#else
     pstart=start;
     pend =end;
#endif

     while(1){

        NATIVE_LOCK(lock);

/*        armci_get(memlock_table, table, sizeof(table), proc);*/
        armci_copy(memlock_table, table, sizeof(table));
        
        /* inspect the table */
        conflict = 0; avail =-1;
        for(slot = 0; slot < MAX_SLOTS; slot ++){

            /* nonzero starting address means the slot is occupied */ 
            if(table[slot].start == NULL){

              /* remember a free slot to store address range */
              avail = slot;  

            }else{
           
              /*check for conflict: overlap between stored and current range*/
              if(  (pstart >= table[slot].start && pstart <= table[slot].end)
                 || (pend >= table[slot].start && pend <= table[slot].end) ){

                  conflict = 1;
                  break;

              }
              /*
              printf("%d: locking %ld-%ld (%d) conflict\n",
                  armci_me,  */
            }
       }
        
       if(avail != -1 && !conflict)break;

       NATIVE_UNLOCK(lock);
       armci_waitsome( ++turn );

     }

     /* we got the memory lock: enter address into the table */
     table[avail].start = pstart;
     table[avail].end = pend;
     armci_put(table+avail,memlock_table+avail,sizeof(memlock_t),proc);

     FENCE_NODE(proc);

     NATIVE_UNLOCK(lock);
     locked_slot = avail;

}
        

/*\ release lock to the memory area locked by previous call to armci_lockemem
\*/
void armci_unlockmem()
{
     void *null[2] = {NULL,NULL};
     memlock_t *memlock_table;

#ifdef DEBUG
     if(locked_slot == INVALID_VAL) armci_die("armci_unlock: empty",0);
     if(locked_slot >= MAX_SLOTS || locked_slot <0) 
        armci_die("armci_unlock: corrupted slot?",locked_slot);
     if(locked_proc >= MAX_PROC || locked_proc <0) 
        armci_die("armci_unlock: corrupted proc?",locked_proc);
#endif

     memlock_table = (memlock_t*)memlock_table_array[locked_proc];
     armci_put(null,&memlock_table[locked_slot].start,2*sizeof(void*),locked_proc);

}
    
void armci_lockmem_(void *pstart, void *pend, int proc)
{
    locked_proc =proc;
    NATIVE_LOCK(proc);
#   ifdef LAPI
    {
       extern int kevin_ok;
       kevin_ok=0;
    }
#   endif
}

void armci_unlockmem_()
{
    NATIVE_UNLOCK(locked_proc);
#   ifdef LAPI
    {
       extern int kevin_ok;
       kevin_ok=1;
    }
#   endif
}
