#ifndef _MEMLOCK_H_
#define _MEMLOCK_H_ 


/* data structure for locking memory areas */
#define MAX_SLOTS 8
typedef struct{
    void *start;
    void *end;
} memlock_t;

extern void** memlock_table_array;
extern int *armci_use_memlock_table;

#if defined(LAPI) || defined(FUJITSU) || defined(PTHREADS) || defined(QUADRICS)\
                  || defined(HITACHI) || (defined(LINUX64)&&defined(__GNUC__)&&defined(__alpha__))\
                  || defined(CYGWIN) || defined(__crayx1)
#  define ARMCI_LOCKMEM armci_lockmem_
#  define ARMCI_UNLOCKMEM armci_unlockmem_
#else
#  define ARMCI_LOCKMEM armci_lockmem
#  define ARMCI_UNLOCKMEM armci_unlockmem
#endif

extern void ARMCI_LOCKMEM(void *pstart, void *pend, int proc);
extern void ARMCI_UNLOCKMEM(int proc);
#define MEMLOCK_SHMEM_FLAG 
#endif
