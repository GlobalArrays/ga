#ifndef _MEMLOCK_H_
#define _MEMLOCK_H_ 

#define MAX_SLOTS 16
typedef struct{
    void *start;
    void *end;
} memlock_t;

extern void** memlock_table_array;

#endif
