/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#ifndef COMEX_IMPL_H_
#define COMEX_IMPL_H_

/* C and/or system headers */
#include <stdarg.h>
#include <stdio.h>
#include <execinfo.h>

/* 3rd party headers */
#include <mpi.h>
#include "rdma/fabric.h"

/* our headers */
#include "request.h"


#define HANDLE_UNDEFINED -1

#ifndef min
#  define min(a,b)              \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a < _b ? _a : _b; })
#endif /* min */

#ifndef max
#  define max(a,b)              \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a > _b ? _a : _b; })
#endif /* max */

#define unlikely(x_) __builtin_expect(!!(x_),0)
#define likely(x_)   __builtin_expect(!!(x_),1)

#define INSERT_TO_LIST(root, item) \
    do                             \
    {                              \
        item->next = root;         \
        root = item;               \
    } while (0)

#define REMOVE_FROM_LIST(root, item, type)  \
    do                                      \
    {                                       \
        if (root == item)                   \
            root = item->next;              \
        else                                \
        {                                   \
            type* ptr = root;               \
            while (ptr)                     \
            {                               \
                if (ptr->next == item)      \
                {                           \
                    ptr->next = item->next; \
                    break;                  \
                }                           \
                else                        \
                    ptr = ptr->next;        \
            }                               \
        }                                   \
    } while (0)


#define OFI_LOCK_INIT()                \
do                                     \
{                                      \
    pthread_spin_init(&poll_spin, 0);  \
    pthread_spin_init(&mutex_spin, 0); \
    pthread_spin_init(&acc_spin, 0);   \
} while(0)

#define OFI_LOCK_DESTROY()             \
do                                     \
{                                      \
    pthread_spin_destroy(&poll_spin);  \
    pthread_spin_destroy(&mutex_spin); \
    pthread_spin_destroy(&acc_spin);   \
} while(0)

#define OFI_LOCK() pthread_spin_lock(&poll_spin)
#define OFI_TRYLOCK() (!pthread_spin_trylock(&poll_spin))
#define OFI_UNLOCK() pthread_spin_unlock(&poll_spin);
#define OFI_CALL(ret, func) \
do                          \
{                           \
    OFI_LOCK();             \
    ret = func;             \
    OFI_UNLOCK();           \
} while(0)

#define OFI_VCALL(func) \
do                      \
{                       \
    OFI_LOCK();         \
    func;               \
    OFI_UNLOCK();       \
} while(0)

/* Logging macroses */
#define EXPR_CHKANDJUMP(ret, fmt, ...)  \
  do                                    \
  {                                     \
      if (unlikely(!(ret)))             \
      {                                 \
          err_printf("%s: " fmt,        \
          __FUNCTION__, ##__VA_ARGS__); \
          goto fn_fail;                 \
      }                                 \
  } while (0)

#define OFI_FAILED() assert(0)

#define COMEX_CHKANDJUMP(ret, fmt, ...)   \
  do                                      \
  {                                       \
      if (unlikely(ret != COMEX_SUCCESS)) \
      {                                   \
          err_printf("(%u) %s: " fmt,     \
          (unsigned)getpid(),             \
          __FUNCTION__, ##__VA_ARGS__);   \
          goto fn_fail;                   \
      }                                   \
  } while (0)

#define OFI_CHKANDJUMP(func, fmt, ...)                       \
  do                                                         \
  {                                                          \
      ssize_t __ret;                                         \
      OFI_CALL(__ret, func);                                 \
      if (unlikely(__ret < 0))                               \
      {                                                      \
          err_printf("(%u) %s: " fmt ": ret %d, error %s, ", \
          (unsigned)getpid(), __FUNCTION__,                  \
          ##__VA_ARGS__, __ret,                              \
          STR_ERROR(&ld_table, __ret));                      \
          OFI_FAILED();                                      \
          goto fn_fail;                                      \
      }                                                      \
  } while (0)

#define OFI_RETRY(func, ...)                       \
    do{                                            \
        ssize_t _ret;                              \
        do {                                       \
            OFI_CALL(_ret, func);                  \
            if (likely(_ret == 0)) break;          \
            if (_ret != -FI_EAGAIN)                \
                OFI_CHKANDJUMP(_ret, __VA_ARGS__); \
            poll(0);                               \
        } while (_ret == -FI_EAGAIN);              \
    } while (0)

#define MPI_CHKANDJUMP(ret, fmt, ...)       \
  do                                        \
  {                                         \
      if (unlikely(ret != MPI_SUCCESS))     \
      {                                     \
          err_printf("(%u) %s: " fmt,       \
          (unsigned)getpid(), __FUNCTION__, \
          ##__VA_ARGS__);                   \
          goto fn_fail;                     \
      }                                     \
  } while (0)

#define PAUSE()                                               \
do                                                            \
{                                                             \
    if((async_progress || async_progress_thread) &&           \
        !pthread_equal(pthread_self(), tid))                  \
        sched_yield();                                        \
} while(0)
/*#define PAUSE() sched_yield()*/

static int comex_var_bool(const char* var)
{
    const char* val = getenv(var);
    if(val &&
      (val[0] == 'y' || val[0] == 'Y' ||
       val[0] == 't' || val[0] == 'T' ||
      (val[0] >= '1' && val[0] <= '9')))
        return 1;
    return 0;
}

static void print_backtrace(void)
{
    int j, nptrs;
    void *buffer[100];
    char **strings;

    nptrs = backtrace(buffer, 100);
    printf("backtrace() returned %d addresses\n", nptrs);
    fflush(stdout);

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL)
    {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < nptrs; j++)
    {
        printf("%s\n", strings[j]);
        fflush(stdout);
    }

    free(strings);
}

static void err_printf(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
    vfprintf(stderr, fmt, list);
    va_end(list);
    fprintf(stderr, "\n");
    fflush(stderr);
    //print_backtrace();
}

/* Struct declaration */
typedef struct {
    MPI_Comm world_comm;
    int proc;
    int size;
} local_state;
extern local_state l_state;

typedef enum flush_type_t
{
    flush_default  = 0,
    flush_transmit = 1,
    flush_receive  = 2,
    flush_all      = 3
} flush_type_t;

typedef struct local_window_t
{
    struct fid_mr*         mr_rma;
    struct fid_mr*         mr_atomics;
    void*                  ptr;
    struct local_window_t* next;
} local_window_t;

typedef struct ofi_window_t
{
    int                    world_proc;
    uint64_t               key_rma;
    uint64_t               key_atomics;
    uint64_t               ptr;  /* remote address */
    size_t                 size; /* size of remote buffer */
    struct peer_t*         peer_rma;
    struct peer_t*         peer_atomics;
    struct local_window_t* local;
    struct ofi_window_t*   next;
} ofi_window_t;

typedef struct strided_context_t
{
    int                datatype;
    void*              scale;
    void*              src;
    void*              dst;
    int*               count;
    int                proc;
    comex_group_t      group;
    struct request_t*  request;
    int                ops;
    int                is_get_op;
    struct fi_msg_rma* msg;
    ofi_window_t*      wnd;
    int cur_iov_idx;
} strided_context_t;

#define ATOMICS_PROTO_TAGMASK    (779L << 32)
#define ATOMICS_DATA_TAGMASK     (780L << 32)
#define ATOMICS_ACC_DATA_TAGMASK (781L << 32)
#define ATOMICS_ACC_CMPL_TAGMASK (782L << 32)
#define ATOMICS_MUTEX_TAGMASK    (783L << 32)
#define ATOMICS_PROTO_IGNOREMASK (0)

/* struct used to store data required for fi_atomicmsg, used to free resources */
typedef struct acc_data_t
{
    void*                 middle;
    struct fi_msg_atomic* msg;
    int                   proc;
} acc_data_t;

typedef enum ofi_proto_e
{
    ofi_proto_rmw_add,
    ofi_proto_rmw_add_long,
    ofi_proto_rmw_swap,
    ofi_proto_rmw_swap_long
} ofi_proto_e;

typedef struct ofi_proto_t
{
    int      proc;
    int      op;
    uint64_t tag;
} ofi_proto_t;

typedef struct ofi_rmw_t
{
    ofi_proto_t proto;
    uint64_t    src;
    int         extra;
    uint64_t    addr;
} ofi_rmw_t;

typedef struct ofi_acc_t
{
    ofi_proto_t proto;
    int         count;  /* number of IOV elements expected */
    uint64_t    addr;   /* target address to acc */
    int         len;    /* length of every IOV element */
    int         posted; /* length of data in this packet */
    char        data[];
} ofi_acc_t;

typedef struct ofi_mutex_t
{
    ofi_proto_t proto;
    int         num;
} ofi_mutex_t;

typedef union ofi_atomics_t
{
    ofi_proto_t proto;
    ofi_rmw_t   rmw;
    ofi_acc_t   acc;
    ofi_mutex_t mutex;
} ofi_atomics_t;

#define OFI_MUTEX_AM_LOCK   1024
#define OFI_MUTEX_AM_UNLOCK 1025

#endif /* COMEX_IMPL_H_ */
