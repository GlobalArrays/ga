#ifndef _MEMLOCK_H_
#define _MEMLOCK_H_ 


/* data structure for locking memory areas */
#define MAX_SLOTS 16
typedef struct{
    void *start;
    void *end;
} memlock_t;

extern void** memlock_table_array;
extern int *armci_use_memlock_table;

#if defined(LAPI) || defined(FUJITSU) || defined(PTHREADS) || defined(QUADRICS)\
                  || (defined(LINUX64)&&defined(__GNUC__)&&defined(__alpha__))
#  define ARMCI_LOCKMEM armci_lockmem_
#  define ARMCI_UNLOCKMEM armci_unlockmem_
#else
#  define ARMCI_LOCKMEM armci_lockmem
#  define ARMCI_UNLOCKMEM armci_unlockmem
#endif

#endif
