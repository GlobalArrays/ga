/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#ifndef MUTEX_H_
#define MUTEX_H_

#define MUTEX_NATIVE_ONLY

#ifndef MUTEX_NATIVE_ONLY
#include <mpi.h>
#endif /* MUTEX_NATIVE_ONLY */

/* mutex infrastructure/implementation, based on MCS mutexes */

#define MUTEX_TAGMASK (777L << 32)
#define LOCAL_MUTEX_TAGMASK (778L << 32)

typedef struct fid_mr* mutex_mr_t;
typedef uint64_t mutex_key_t;

typedef struct mcs_mutex_t
{
    int         proc;
    int         count;
    mutex_key_t key;
    int*        tail;     /* pointer to tail elements on remote host */
    int*        elem; /* index of element where 'elem' begins in 'tail' array */
} mcs_mutex_t;

typedef struct mutex_t
{
    mutex_mr_t          mr;
    struct mcs_mutex_t* mcs_mutex;
    int*                elem_offset; /* list of start indexes for 'elem's */
    void*               data; /* used to store locally allocated data */
    uint64_t            tagmask;
} mutex_t;

/* every node must have 'elem' for every mutex used in cluster.
 * to optimize resource usage all such 'elem' are located in solid
 * array which registered as memory region in OFI. to get index of
 * required element based on proc and mutex is used macto below */

#define ELEM_INDEX(mtx, proc, id) ((mtx)->elem_offset[proc] + (id))
#define MUTEX_ELEM_IDX(mtx, index) ((mtx).elem_idx + (index))
#define MUTEX_ELEM(mtx, index) ((mtx).tail[MUTEX_ELEM_IDX(mtx, index)])

/* original implementation of mcs mutexes use 'tail' element on 'root'
 * node and 'elem' element on every node which may lock mutex.
 * usually it is implemented using atomic access to remote memory,
 * for example MPI window (MPI based implementation), or registered
 * memory/atomic API of OFI.
 *
 * core idea of MCS mutex is in manipulation of pair elements: 'tail'
 * on node which locked mutex, and 'elem' element on node which waiting
 * for mutex unlocked. when one of node is going to lock mutex on remote
 * (or local) node, it atomically swap values of 'tail' element on target
 * node, and in case if on target node previously was non-default value
 * then current node writes own proc to 'elem' of node which locked mutex.
 * using this mechanism connected list/queue is created.
 *
 * here implemented a bit modified algorithm: every node may have
 * any number of mutexes, to optimize usage of 'tail's and 'elem's
 * common arrays are used to store required objects */

 #endif /* MUTEX_H_ */

