#ifndef _COMEX_STRUCTS
#define _COMEX_STRUCTS
/* data structures */

typedef enum {
    OP_PUT = 0,
    OP_PUT_PACKED,
    OP_PUT_DATATYPE,
    OP_PUT_IOV,
    OP_GET,
    OP_GET_PACKED,
    OP_GET_DATATYPE,
    OP_GET_IOV,
    OP_ACC_INT,
    OP_ACC_DBL,
    OP_ACC_FLT,
    OP_ACC_CPL,
    OP_ACC_DCP,
    OP_ACC_LNG,
    OP_ACC_INT_PACKED,
    OP_ACC_DBL_PACKED,
    OP_ACC_FLT_PACKED,
    OP_ACC_CPL_PACKED,
    OP_ACC_DCP_PACKED,
    OP_ACC_LNG_PACKED,
    OP_ACC_INT_IOV,
    OP_ACC_DBL_IOV,
    OP_ACC_FLT_IOV,
    OP_ACC_CPL_IOV,
    OP_ACC_DCP_IOV,
    OP_ACC_LNG_IOV,
    OP_FENCE,
    OP_FETCH_AND_ADD,
    OP_SWAP,
    OP_CREATE_MUTEXES,
    OP_DESTROY_MUTEXES,
    OP_LOCK,
    OP_UNLOCK,
    OP_QUIT,
    OP_MALLOC,
    OP_MALLOC_DEV,
    OP_FREE,
    OP_NULL
} op_t;


typedef struct {
    op_t operation;
    void *remote_address;
    void *local_address;
    int rank; /**< rank of target (rank of sender is iprobe_status.MPI_SOURCE) */
    int length; /**< length of message/payload not including header */
#ifdef ENABLE_DEVICE
    int use_dev;
    int dev_id;
#endif
} header_t;


/* keep track of all mutex requests */
typedef struct lock_link {
    struct lock_link *next;
    int rank;
} lock_t;

typedef struct {
    void *ptr;
    int stride_levels;
    int stride[COMEX_MAX_STRIDE_LEVEL];
    int count[COMEX_MAX_STRIDE_LEVEL+1];
} stride_t;


typedef struct message_link {
    struct message_link *next;
    void *message;
    MPI_Request request;
    MPI_Datatype datatype;
    int need_free;
    stride_t *stride;
    comex_giov_t *iov;
} message_t;

typedef struct {
  int in_use;
  int send_size;
  message_t *send_head;
  message_t *send_tail;
  int recv_size;
  message_t *recv_head;
  message_t *recv_tail;
} nb_t;

typedef struct {
    int rank;
    void *ptr;
    int dev_id;
} rank_ptr_t;

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define PAUSE_ON_ERROR 0
#define STATIC static inline

#if USE_MEMSET_AFTER_MALLOC
#define MAYBE_MEMSET(a,b,c) (void)memset(a,b,c)
#else
#define MAYBE_MEMSET(a,b,c) ((void)0)
#endif

#define XSTR(x) #x
#define STR(x) XSTR(x)

#endif
