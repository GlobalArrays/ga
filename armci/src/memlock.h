#ifndef _MEMLOCK_H_
#define _MEMLOCK_H_ 

#define MAX_SLOTS 16
typedef struct{
    void *start;
    void *end;
} memlock_t;

extern void** memlock_table_array;


/* DATA_SERVER code requires more work to use memlock table */

#if defined(LAPI) || defined(FUJITSU) || defined(DATA_SERVER_)
#  define ARMCI_LOCKMEM armci_lockmem_
#  define ARMCI_UNLOCKMEM armci_unlockmem_
#else
#  define ARMCI_LOCKMEM armci_lockmem
#  define ARMCI_UNLOCKMEM armci_unlockmem
#endif

#endif
