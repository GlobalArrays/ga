/* C and/or system headers */
#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <unistd.h>

/* 3rd party headers */
#include <mpi.h>
#if USE_SICM
#include <sicm_low.h>
//#include <sicm_impl.h>
sicm_device_list nill;

#endif

/* our headers */
#include "cmx.h"
#include "cmx_impl.h"
#include "groups.h"
#include "reg_cache.h"
#include "acc.h"

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
    OP_FREE,
    OP_NULL
} op_t;

typedef struct {
    op_t operation;
    void *remote_address;
    void *local_address;
    int rank; /**< rank of target (rank of sender is iprobe_status.MPI_SOURCE */
    int length; /**< length of message/payload not including header */
} header_t;


/* keep track of all mutex requests */
typedef struct lock_link {
    struct lock_link *next;
    int rank;
} lock_t;


#if 0
typedef struct message_link {
    struct message_link *next;
    void *message;
    MPI_Request request;
    MPI_Datatype datatype;
    int need_free;
    stride_t *stride;
    cmx_giov_t *iov;
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
} rank_ptr_t;
#endif

int _cmx_me;

/* static state */
static int *num_mutexes = NULL;     /**< (all) how many mutexes on each process */
static int **mutexes = NULL;        /**< (masters) value is rank of lock holder */
static lock_t ***lq_heads = NULL;   /**< array of lock queues */
static char *sem_name = NULL;       /* local semaphore name */
static sem_t **semaphores = NULL;   /* semaphores for locking within SMP node */
static int initialized = 0;         /* for cmx_initialized(), 0=false */
static char *fence_array = NULL;

static int nb_max_outstanding = CMX_MAX_NB_OUTSTANDING;
static int nb_last_request = 0;
static int nb_index = 0;
static int nb_count_event = 0;
static int nb_count_event_processed = 0;
static int nb_count_send = 0;
static int nb_count_send_processed = 0;
static int nb_count_recv = 0;
static int nb_count_recv_processed = 0;

static char *static_server_buffer = NULL;
static int static_server_buffer_size = 0;
static int eager_threshold = -1;
static int max_message_size = -1;

static int CMX_ENABLE_PUT_SELF = ENABLE_PUT_SELF;
static int CMX_ENABLE_GET_SELF = ENABLE_GET_SELF;
static int CMX_ENABLE_ACC_SELF = ENABLE_ACC_SELF;
static int CMX_ENABLE_PUT_SMP = ENABLE_PUT_SMP;
static int CMX_ENABLE_GET_SMP = ENABLE_GET_SMP;
static int CMX_ENABLE_ACC_SMP = ENABLE_ACC_SMP;
static int CMX_ENABLE_PUT_PACKED = ENABLE_PUT_PACKED;
static int CMX_ENABLE_GET_PACKED = ENABLE_GET_PACKED;
static int CMX_ENABLE_ACC_PACKED = ENABLE_ACC_PACKED;
static int CMX_ENABLE_PUT_DATATYPE = ENABLE_PUT_DATATYPE;
static int CMX_ENABLE_GET_DATATYPE = ENABLE_GET_DATATYPE;
static int CMX_PUT_DATATYPE_THRESHOLD = 8192;
static int CMX_GET_DATATYPE_THRESHOLD = 8192;
static int CMX_ENABLE_PUT_IOV = ENABLE_PUT_IOV;
static int CMX_ENABLE_GET_IOV = ENABLE_GET_IOV;
static int CMX_ENABLE_ACC_IOV = ENABLE_ACC_IOV;

#if USE_SICM
static sicm_device_list devices = {0};
#if SICM_OLD
static sicm_device *device_dram = NULL;
static sicm_device *device_knl_hbm = NULL;
static sicm_device *device_ppc_hbm = NULL;
#else
static sicm_device_list device_dram = {0};
static sicm_device_list device_knl_hbm = {0};
static sicm_device_list device_ppc_hbm = {0};
#endif
#endif

/* Non-blocking handle list */
static _cmx_request **nb_list;

#if PAUSE_ON_ERROR
static int AR_caught_sig=0;
static int AR_caught_sigsegv=0;
void (*SigSegvOrig)(int);
void SigSegvHandler(int sig)
{
    char name[256];
    AR_caught_sig= sig;
    AR_caught_sigsegv=1;
    if (-1 == gethostname(name, 256)) {
        perror("gethostname");
        cmx_error("gethostname failed", errno);
    }
    fprintf(stderr,"%d(%s:%d): Segmentation Violation ... pausing\n",
            g_state.rank, name, getpid());
    pause();

    cmx_error("Segmentation Violation error, status=",(int) sig);
}
#endif

/* static function declarations */

/* error checking */
#define CHECK_MPI_RETVAL(retval) check_mpi_retval((retval), __FILE__, __LINE__)
STATIC void check_mpi_retval(int retval, const char *file, int line);
STATIC const char *str_mpi_retval(int retval);

/* server fuctions */
STATIC void server_send(void *buf, int count, int dest);
STATIC void server_send_datatype(void *buf, MPI_Datatype dt, int dest);
STATIC void server_recv(void *buf, int count, int source);
STATIC void server_recv_datatype(void *buf, MPI_Datatype dt, int source);
STATIC void _progress_server();
STATIC void _put_handler(header_t *header, char *payload, int proc);
STATIC void _put_packed_handler(header_t *header, char *payload, int proc);
STATIC void _put_datatype_handler(header_t *header, char *payload, int proc);
STATIC void _put_iov_handler(header_t *header, int proc);
STATIC void _get_handler(header_t *header, int proc);
STATIC void _get_packed_handler(header_t *header, char *payload, int proc);
STATIC void _get_datatype_handler(header_t *header, char *payload, int proc);
STATIC void _get_iov_handler(header_t *header, int proc);
STATIC void _acc_handler(header_t *header, char *scale, int proc);
STATIC void _acc_packed_handler(header_t *header, char *payload, int proc);
STATIC void _acc_iov_handler(header_t *header, char *scale, int proc);
STATIC void _fence_handler(header_t *header, int proc);
STATIC void _fetch_and_add_handler(header_t *header, char *payload, int proc);
STATIC void _swap_handler(header_t *header, char *payload, int proc);
STATIC void _mutex_create_handler(header_t *header, int proc);
STATIC void _mutex_destroy_handler(header_t *header, int proc);
STATIC void _lock_handler(header_t *header, int proc);
STATIC void _unlock_handler(header_t *header, int proc);
STATIC void _malloc_handler(header_t *header, char *payload, int proc);
STATIC void _free_handler(header_t *header, char *payload, int proc);

/* worker functions */
STATIC void nb_send_common(void *buf, int count, int dest, _cmx_request *nb, int need_free);
STATIC void nb_send_datatype(void *buf, MPI_Datatype dt, int dest, _cmx_request *nb);
STATIC void nb_send_header(void *buf, int count, int dest, _cmx_request *nb);
STATIC void nb_send_buffer(void *buf, int count, int dest, _cmx_request *nb);
STATIC void nb_recv_packed(void *buf, int count, int source, _cmx_request *nb, stride_t *stride);
STATIC void nb_recv_datatype(void *buf, MPI_Datatype dt, int source, _cmx_request *nb);
STATIC void nb_recv_iov(void *buf, int count, int source, _cmx_request *nb, _cmx_giov_t *iov);
STATIC void nb_recv(void *buf, int count, int source, _cmx_request *nb);
STATIC void nb_wait_for_send1(_cmx_request *nb);
STATIC void nb_wait_for_recv1(_cmx_request *nb);
STATIC void nb_wait_for_all(_cmx_request *nb);
STATIC int nb_test_for_all(_cmx_request *nb);
STATIC void nb_register_request(_cmx_request *nb);
STATIC void nb_unregister_request(_cmx_request *nb);
STATIC void nb_handle_init(_cmx_request *nb);
STATIC void nb_put(void *src, void *dst, int bytes, int proc, _cmx_request *nb);
STATIC void nb_get(void *src, void *dst, int bytes, int proc, _cmx_request *nb);
STATIC void nb_acc(int datatype, void *scale, void *src, void *dst, int bytes, int proc, _cmx_request *nb);
STATIC void nb_puts(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_puts_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_puts_datatype(
        void *src_ptr, int *src_stride_ar,
        void *dst_ptr, int *dst_stride_ar,
        int *count, int stride_levels,
        int proc, _cmx_request *nb);
STATIC void nb_gets(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_gets_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_gets_datatype(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_accs(
        int datatype, void *scale,
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_accs_packed(
        int datatype, void *scale,
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb);
STATIC void nb_putv(_cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
STATIC void nb_putv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
STATIC void nb_getv(_cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
STATIC void nb_getv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
STATIC void nb_accv(int datatype, void *scale,
        _cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
STATIC void nb_accv_packed(int datatype, void *scale,
        _cmx_giov_t *iov, int proc, _cmx_request *nb);
STATIC void _fence_master(int master_rank);
STATIC int _eager_check(int extra_bytes);

/* other functions */
STATIC int _packed_size(int *src_stride, int *count, int stride_levels);
STATIC char* pack(char *src, int *src_stride,
                int *count, int stride_levels, int *size);
STATIC void unpack(char *packed_buffer,
                char *dst, int *dst_stride, int *count, int stride_levels);
STATIC char* _generate_shm_name(int rank);
STATIC reg_entry_t* _cmx_malloc_local(size_t size);
#if USE_SICM
#if SICM_OLD
STATIC reg_entry_t* _cmx_malloc_local_memdev(size_t size, sicm_device *device);
#else
STATIC reg_entry_t* _cmx_malloc_local_memdev(size_t size, sicm_device_list device);
#endif
int _cmx_free_local_memdev(void *ptr);
#endif
STATIC void* _get_offset_memory(reg_entry_t *reg_entry, void *memory);
STATIC int _is_master(void);
STATIC int _get_world_rank(cmx_igroup_t *igroup, int rank);
STATIC int* _get_world_ranks(cmx_igroup_t *igroup);
STATIC int _smallest_world_rank_with_same_hostid(cmx_igroup_t *group);
STATIC int _largest_world_rank_with_same_hostid(cmx_igroup_t *igroup);
STATIC void _malloc_semaphore(void);
STATIC void _free_semaphore(void);
STATIC void* _shm_create(const char *name, size_t size);
STATIC void* _shm_attach(const char *name, size_t size);
STATIC void* _shm_map(int fd, size_t size);
#if USE_SICM
#if SICM_OLD
STATIC void* _shm_create_memdev(const char *name, size_t size, sicm_device *device);
STATIC void* _shm_attach_memdev(const char *name, size_t size, sicm_device *device);
#else
STATIC void* _shm_create_memdev(const char *name, size_t size, sicm_device_list device);
STATIC void* _shm_attach_memdev(const char *name, size_t size, sicm_device_list device);
#endif
STATIC void* _shm_map_arena(int fd, size_t size, sicm_arena arena);
#endif
STATIC int _set_affinity(int cpu);
STATIC void _translate_mpi_error(int ierr, const char* location);
STATIC void strided_to_subarray_dtype(int *stride_array, int *count, int levels, MPI_Datatype base_type, MPI_Datatype *type);

int cmx_init()
{
    int status = 0;
    int init_flag = 0;
    int i = 0;
    
    if (initialized) {
        return 0;
    }
    initialized = 1;

    /* Assert MPI has been initialized */
    status = MPI_Initialized(&init_flag);
    _translate_mpi_error(status,"cmx_init");
    CHECK_MPI_RETVAL(status);
    assert(init_flag);

    /*MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);*/


    /* groups */
    cmx_group_init();

    /* env vars */
    {
        char *value = NULL;
        nb_max_outstanding = CMX_MAX_NB_OUTSTANDING/2; /* default */
        value = getenv("CMX_MAX_NB_OUTSTANDING");
        if (NULL != value) {
            nb_max_outstanding = atoi(value);
        }
        CMX_ASSERT(nb_max_outstanding > 0);

        static_server_buffer_size = CMX_STATIC_BUFFER_SIZE; /* default */
        value = getenv("CMX_STATIC_BUFFER_SIZE");
        if (NULL != value) {
            static_server_buffer_size = atoi(value);
        }
        CMX_ASSERT(static_server_buffer_size > 0);

        eager_threshold = -1; /* default */
        value = getenv("CMX_EAGER_THRESHOLD");
        if (NULL != value) {
            eager_threshold = atoi(value);
        }

        CMX_ENABLE_PUT_SELF = ENABLE_PUT_SELF; /* default */
        value = getenv("CMX_ENABLE_PUT_SELF");
        if (NULL != value) {
            CMX_ENABLE_PUT_SELF = atoi(value);
        }

        CMX_ENABLE_GET_SELF = ENABLE_GET_SELF; /* default */
        value = getenv("CMX_ENABLE_GET_SELF");
        if (NULL != value) {
            CMX_ENABLE_GET_SELF = atoi(value);
        }

        CMX_ENABLE_ACC_SELF = ENABLE_ACC_SELF; /* default */
        value = getenv("CMX_ENABLE_ACC_SELF");
        if (NULL != value) {
            CMX_ENABLE_ACC_SELF = atoi(value);
        }

        CMX_ENABLE_PUT_SMP = ENABLE_PUT_SMP; /* default */
        value = getenv("CMX_ENABLE_PUT_SMP");
        if (NULL != value) {
            CMX_ENABLE_PUT_SMP = atoi(value);
        }

        CMX_ENABLE_GET_SMP = ENABLE_GET_SMP; /* default */
        value = getenv("CMX_ENABLE_GET_SMP");
        if (NULL != value) {
            CMX_ENABLE_GET_SMP = atoi(value);
        }

        CMX_ENABLE_ACC_SMP = ENABLE_ACC_SMP; /* default */
        value = getenv("CMX_ENABLE_ACC_SMP");
        if (NULL != value) {
            CMX_ENABLE_ACC_SMP = atoi(value);
        }

        CMX_ENABLE_PUT_PACKED = ENABLE_PUT_PACKED; /* default */
        value = getenv("CMX_ENABLE_PUT_PACKED");
        if (NULL != value) {
            CMX_ENABLE_PUT_PACKED = atoi(value);
        }

        CMX_ENABLE_GET_PACKED = ENABLE_GET_PACKED; /* default */
        value = getenv("CMX_ENABLE_GET_PACKED");
        if (NULL != value) {
            CMX_ENABLE_GET_PACKED = atoi(value);
        }

        CMX_ENABLE_ACC_PACKED = ENABLE_ACC_PACKED; /* default */
        value = getenv("CMX_ENABLE_ACC_PACKED");
        if (NULL != value) {
            CMX_ENABLE_ACC_PACKED = atoi(value);
        }

        CMX_ENABLE_PUT_DATATYPE = ENABLE_PUT_DATATYPE; /* default */
        value = getenv("CMX_ENABLE_PUT_DATATYPE");
        if (NULL != value) {
            CMX_ENABLE_PUT_DATATYPE = atoi(value);
        }

        CMX_ENABLE_GET_DATATYPE = ENABLE_GET_DATATYPE; /* default */
        value = getenv("CMX_ENABLE_GET_DATATYPE");
        if (NULL != value) {
            CMX_ENABLE_GET_DATATYPE = atoi(value);
        }

        CMX_PUT_DATATYPE_THRESHOLD = 8192; /* default */
        value = getenv("CMX_PUT_DATATYPE_THRESHOLD");
        if (NULL != value) {
            CMX_PUT_DATATYPE_THRESHOLD = atoi(value);
        }

        CMX_GET_DATATYPE_THRESHOLD = 8192; /* default */
        value = getenv("CMX_GET_DATATYPE_THRESHOLD");
        if (NULL != value) {
            CMX_GET_DATATYPE_THRESHOLD = atoi(value);
        }

        CMX_ENABLE_PUT_IOV = ENABLE_PUT_IOV; /* default */
        value = getenv("CMX_ENABLE_PUT_IOV");
        if (NULL != value) {
            CMX_ENABLE_PUT_IOV = atoi(value);
        }

        CMX_ENABLE_GET_IOV = ENABLE_GET_IOV; /* default */
        value = getenv("CMX_ENABLE_GET_IOV");
        if (NULL != value) {
            CMX_ENABLE_GET_IOV = atoi(value);
        }

        CMX_ENABLE_ACC_IOV = ENABLE_ACC_IOV; /* default */
        value = getenv("CMX_ENABLE_ACC_IOV");
        if (NULL != value) {
            CMX_ENABLE_ACC_IOV = atoi(value);
        }

        max_message_size = INT_MAX; /* default */
        value = getenv("CMX_MAX_MESSAGE_SIZE");
        if (NULL != value) {
            max_message_size = atoi(value);
        }

#if DEBUG
        if (0 == g_state.rank) {
            printf("CMX_MAX_NB_OUTSTANDING=%d\n", nb_max_outstanding);
            printf("CMX_STATIC_BUFFER_SIZE=%d\n", static_server_buffer_size);
            printf("CMX_MAX_MESSAGE_SIZE=%d\n", max_message_size);
            printf("CMX_EAGER_THRESHOLD=%d\n", eager_threshold);
            printf("CMX_PUT_DATATYPE_THRESHOLD=%d\n", CMX_PUT_DATATYPE_THRESHOLD);
            printf("CMX_GET_DATATYPE_THRESHOLD=%d\n", CMX_GET_DATATYPE_THRESHOLD);
            printf("CMX_ENABLE_PUT_SELF=%d\n", CMX_ENABLE_PUT_SELF);
            printf("CMX_ENABLE_GET_SELF=%d\n", CMX_ENABLE_GET_SELF);
            printf("CMX_ENABLE_ACC_SELF=%d\n", CMX_ENABLE_ACC_SELF);
            printf("CMX_ENABLE_PUT_SMP=%d\n", CMX_ENABLE_PUT_SMP);
            printf("CMX_ENABLE_GET_SMP=%d\n", CMX_ENABLE_GET_SMP);
            printf("CMX_ENABLE_ACC_SMP=%d\n", CMX_ENABLE_ACC_SMP);
            printf("CMX_ENABLE_PUT_PACKED=%d\n", CMX_ENABLE_PUT_PACKED);
            printf("CMX_ENABLE_GET_PACKED=%d\n", CMX_ENABLE_GET_PACKED);
            printf("CMX_ENABLE_ACC_PACKED=%d\n", CMX_ENABLE_ACC_PACKED);
            printf("CMX_ENABLE_PUT_DATATYPE=%d\n", CMX_ENABLE_PUT_DATATYPE);
            printf("CMX_ENABLE_GET_DATATYPE=%d\n", CMX_ENABLE_GET_DATATYPE);
            printf("CMX_ENABLE_PUT_IOV=%d\n", CMX_ENABLE_PUT_IOV);
            printf("CMX_ENABLE_GET_IOV=%d\n", CMX_ENABLE_GET_IOV);
            printf("CMX_ENABLE_ACC_IOV=%d\n", CMX_ENABLE_ACC_IOV);
            fflush(stdout);
        }
#endif
    }

    /* mutexes */
    mutexes = NULL;
    num_mutexes = NULL;
    lq_heads = NULL;

#if USE_SICM
    devices = sicm_init();
#if SICM_OLD
    for(i = 0; i < devices.count; i++) {
        if (devices.devices[i].tag == SICM_DRAM) {
            device_dram = &devices.devices[i];
        }
        if (devices.devices[i].tag == SICM_KNL_HBM) {
            device_knl_hbm = &devices.devices[i];
        }
        if (devices.devices[i].tag == SICM_POWERPC_HBM) {
            device_ppc_hbm = &devices.devices[i];
        }
    }
    if (!device_dram) {
      printf("Device DRAM not found\n");
      exit(18);
    }
#else
   for(i =0; i < devices.count; ++i){
      sicm_device *curr = devices.devices[i];
      if(curr->tag == SICM_DRAM){
         device_dram.count = 1;
         device_dram.devices = &devices.devices[i];
      }
      if (curr->tag == SICM_KNL_HBM) {
         device_knl_hbm.count = 1;
         device_knl_hbm.devices = &devices.devices[i];
      }
      if (curr->tag == SICM_POWERPC_HBM) {
         device_ppc_hbm.count = 1;
         device_ppc_hbm.devices = &devices.devices[i];
      }
   }
   if(device_dram.devices == NULL){
      printf("Device DRAM not found\n");
      exit(18);
   }
#endif
#endif

    /* reg_cache */
    /* note: every process needs a reg cache and it's always based on the
     * world rank and size */
    reg_cache_init(g_state.size);

    _malloc_semaphore();

#if DEBUG
    fprintf(stderr, "[%d] cmx_init() before progress server\n", g_state.rank);
#endif

#if PAUSE_ON_ERROR
    if ((SigSegvOrig=signal(SIGSEGV, SigSegvHandler)) == SIG_ERR) {
        cmx_error("signal(SIGSEGV, ...) error", -1);
    }
#endif

    status = _set_affinity(g_state.node_rank);
    CMX_ASSERT(0 == status);

    if (_is_master()) {
        /* TODO: wasteful O(p) storage... */
        mutexes = (int**)malloc(sizeof(int*) * g_state.size);
        CMX_ASSERT(mutexes);
        /* create one lock queue for each proc for each mutex */
        lq_heads = (lock_t***)malloc(sizeof(lock_t**) * g_state.size);
        CMX_ASSERT(lq_heads);
        /* start the server */
        _progress_server();
    }
    nb_list = (_cmx_request**)malloc(sizeof(_cmx_request*)*nb_max_outstanding);
    for (i=0; i<nb_max_outstanding; i++) nb_list[i] = NULL;

    cmx_group_rank(CMX_GROUP_WORLD, &_cmx_me);

    /* Synch - Sanity Check */
    /* This barrier is on the world worker group */
    status = MPI_Barrier(CMX_GROUP_WORLD->comm);
    _translate_mpi_error(status, "cmx_init:MPI_Barrier");

    /* static state */
    fence_array = malloc(sizeof(char) * g_state.size);
    CMX_ASSERT(fence_array);
    for (i = 0; i < g_state.size; ++i) {
        fence_array[i] = 0;
    }

#if DEBUG
    fprintf(stderr, "[%d] cmx_init() before barrier\n", g_state.rank);
#endif

    /* Synch - Sanity Check */
    /* This barrier is on the world worker group */
    status = MPI_Barrier(CMX_GROUP_WORLD->comm);
    _translate_mpi_error(status, "cmx_init:MPI_Barrier");

#if DEBUG
    fprintf(stderr, "[%d] cmx_init() success\n", g_state.rank);
#endif

    return CMX_SUCCESS;
}


int cmx_init_args(int *argc, char ***argv)
{
    int init_flag;
    int status;
    
    status = MPI_Initialized(&init_flag);
    CHECK_MPI_RETVAL(status);
    
    if(!init_flag) {
        status = MPI_Init(argc, argv);
        _translate_mpi_error(status, "cmx_init_args:MPI_Init");
        CHECK_MPI_RETVAL(status);
    }
    
    return cmx_init();
}


int cmx_initialized()
{
#if DEBUG
    if (initialized) {
        fprintf(stderr, "[%d] cmx_initialized()\n", g_state.rank);
    }
#endif

    return initialized;
}


int cmx_finalize()
{
    int i, ierr;
#if DEBUG
    fprintf(stderr, "[%d] cmx_finalize()\n", g_state.rank);
#endif

    /* it's okay to call multiple times -- extra calls are no-ops */
    if (!initialized) {
        return CMX_SUCCESS;
    }

    cmx_barrier(CMX_GROUP_WORLD);

    initialized = 0;

    _free_semaphore();

    /* Make sure that all outstanding operations are done */
    cmx_wait_all(CMX_GROUP_WORLD);
    
    cmx_barrier(CMX_GROUP_WORLD);

    /* clean up non-blocking calls */
    for (i=0; i<nb_max_outstanding; i++) {
      if (nb_list[i]) {
        nb_wait_for_all(nb_list[i]);
      }
    }
    free(nb_list);

    /* send quit message to thread */
    int smallest_rank_with_same_hostid, largest_rank_with_same_hostid; 
    int num_progress_ranks_per_node, is_node_ranks_packed;
    int my_rank_to_free;
    int is_notifier = 0; 

    num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    is_node_ranks_packed = get_progress_rank_distribution_on_node();
    smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(group_list);
    largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(group_list);
    my_rank_to_free = get_my_rank_to_free(g_state.rank,
        g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if DEBUG
    fprintf(stderr, "[%d] cmx_finalize() sr_in_group %d\n", g_state.rank, 
        my_rank_to_free);
    //     smallest_rank_with_same_hostid + g_state.node_size*
    //  ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size));
#endif
    // is_notifier = g_state.rank == smallest_rank_with_same_hostid + g_state.node_size*
    //   ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size);
    // if (_smallest_world_rank_with_same_hostid(group_list) == g_state.rank) 
    if(is_notifier = my_rank_to_free == g_state.rank)
    {
        int my_master = -1;
        header_t *header = NULL;
        _cmx_request nb;
        nb_handle_init(&nb);

        my_master = g_state.master[g_state.rank];
        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_QUIT;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = 0;
        nb_send_header(header, sizeof(header_t), my_master, &nb);
        nb_wait_for_all(&nb);
    }

    free(fence_array);

    ierr = MPI_Barrier(g_state.comm);
    _translate_mpi_error(ierr, "cmx_finalize:MPI_Barrier");

    /* reg_cache */
    reg_cache_destroy();

    /* destroy the groups */
#if DEBUG
    fprintf(stderr, "[%d] before cmx_group_finalize()\n", g_state.rank);
#endif
    cmx_group_finalize();
#if DEBUG
    fprintf(stderr, "[%d] after cmx_group_finalize()\n", g_state.rank);

#endif
#if USE_SICM
    sicm_fini();
#endif

#if DEBUG_TO_FILE
    fclose(cmx_trace_file);
#endif

    return CMX_SUCCESS;
}


void cmx_error(char *msg, int code)
{
#if DEBUG
    fprintf(stderr, "[%d] Received an Error in Communication: (%d) %s\n",
            g_state.rank, code, msg);
#if DEBUG_TO_FILE
    fclose(cmx_trace_file);
#endif
#endif
    fprintf(stderr,"[%d] Received an Error in Communication: (%d) %s\n",
            g_state.rank, code, msg);
    
    MPI_Abort(g_state.comm, code);
}

/* return the allocated pointer corresponding to proc from the cmx_handle
 * object*/
void* find_alloc_ptr(cmx_handle_t cmx_hdl, int proc)
{
  void *ret = NULL;
  cmx_alloc_t *next = cmx_hdl.list;
  while (next != NULL) {
    if (next->rank == proc) {
      ret = next->buf;
      break;
    }
    next = next->next;
  }
  return ret;
}



int cmx_put(
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_put(src, dst, bytes, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_get(
        void *dst, cmxInt src_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    src = (void*)((char*)src+src_offset);
    nb_get(src, dst, bytes, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_acc(
        int datatype, void *scale,
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_acc(datatype, scale, src, dst, bytes, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_puts(
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_puts(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_gets(
        void *dst, cmxInt *dst_stride,
        cmxInt src_offset, cmxInt *src_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    src = (void*)((char*)src+src_offset);
    nb_gets(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_accs(
        int datatype, void *scale,
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_accs(datatype, scale,
            src, src_stride, dst, dst_stride, count, stride_levels, world_proc, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


int cmx_putv(
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    _cmx_giov_t *tiov;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].src[j] = iov[i].loc[j];
        tiov[i].dst[j] = (void*)((char*)dst+iov[i].rem[j]);
      }
    }

    nb_putv(tiov, iov_len, world_proc, &nb);
    nb_wait_for_all(&nb);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}


int cmx_getv(
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;
    _cmx_giov_t *tiov;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].dst[j] = iov[i].loc[j];
        tiov[i].src[j] = (void*)((char*)src+iov[i].rem[j]);
      }
    }

    nb_getv(tiov, iov_len, world_proc, &nb);
    nb_wait_for_all(&nb);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}


int cmx_accv(
        int datatype, void *scale,
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl)
{
    _cmx_request nb;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    _cmx_giov_t *tiov;

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].src[j] = iov[i].loc[j];
        tiov[i].dst[j] = (void*)((char*)dst+iov[i].rem[j]);
      }
    }

    nb_accv(datatype, scale, tiov, iov_len, world_proc, &nb);
    nb_wait_for_all(&nb);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}


/* This operation does not actually do anything with the group argument. This is
 * also true in the comex interface. */
int cmx_fence_all(cmx_igroup_t *group)
{
    int p = 0;
    int count_before = 0;
    int count_after = 0;
    _cmx_request nb;

#if DEBUG
    fprintf(stderr, "[%d] cmx_fence_all(group=%d)\n", g_state.rank, group);
#endif
    /* NOTE: We always fence on the world group */

    /* count how many fence messagse to send */
    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            ++count_before;
        }
    }

#if DEBUG
    fprintf(stderr, "[%d] cmx_fence_all(group=%d) count_before=%d\n",
            g_state.rank, group, count_before);
#endif

    /* check for no outstanding put/get requests */
    if (0 == count_before) {
        return CMX_SUCCESS;
    }

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
    fprintf(stderr, "[%d] cmx_fence_all asm volatile (\"\" : : : \"memory\"); \n",
            g_state.rank, group);
#endif
    asm volatile ("" : : : "memory"); 
#endif

    nb_handle_init(&nb);
    /* optimize by only sending to procs which we have outstanding messages */
    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            int p_master = g_state.master[p];
            header_t *header = NULL;

            /* because we only fence to masters */
            CMX_ASSERT(p_master == p);

            /* prepost recv for acknowledgment */
            nb_recv(NULL, 0, p_master, &nb);

            /* post send of fence request */
            header = malloc(sizeof(header_t));
            CMX_ASSERT(header);
            MAYBE_MEMSET(header, 0, sizeof(header_t));
            header->operation = OP_FENCE;
            header->remote_address = NULL;
            header->local_address = NULL;
            header->length = 0;
            header->rank = 0;
            nb_send_header(header, sizeof(header_t), p_master, &nb);
        }
    }

    nb_wait_for_all(&nb);

    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            fence_array[p] = 0;
            ++count_after;
        }
    }

#if DEBUG
    fprintf(stderr, "[%d] cmx_fence_all(group=%d) count_after=%d\n",
            g_state.rank, group, count_after);
#endif

    CMX_ASSERT(count_before == count_after);

    return CMX_SUCCESS;
}


STATIC int _eager_check(int extra_bytes)
{
    return (((int)sizeof(header_t))+extra_bytes) < eager_threshold;
}


STATIC void _fence_master(int master_rank)
{
#if DEBUG
    printf("[%d] _fence_master(master=%d)\n", g_state.rank, master_rank);
#endif

    if (fence_array[master_rank]) {
        header_t *header = NULL;
        _cmx_request nb;
        nb_handle_init(&nb);

        /* prepost recv for acknowledgment */
        nb_recv(NULL, 0, master_rank, &nb);

        /* post send of fence request */
        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_FENCE;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->length = 0;
        header->rank = 0;
        nb_send_header(header, sizeof(header_t), master_rank, &nb);
        nb_wait_for_all(&nb);
        fence_array[master_rank] = 0;
    }
}


int cmx_fence_proc(int proc, cmx_igroup_t *igroup)
{
    int world_rank = -1;
    int master_rank = -1;

#if DEBUG
    printf("[%d] cmx_fence_proc(proc=%d, group=%d)\n",
            g_state.rank, proc, group);
#endif

    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    _fence_master(master_rank);

    return CMX_SUCCESS;
}


/* cmx_barrier is cmx_fence_all + MPI_Barrier */
int cmx_barrier(cmx_igroup_t *igroup)
{
    int status = 0;
    MPI_Comm comm = MPI_COMM_NULL;

#if DEBUG
    static int count=-1;
    ++count;
    fprintf(stderr, "[%d] cmx_barrier(%d) count=%d\n", g_state.rank, group, count);
#endif

    cmx_fence_all(igroup);
    status = cmx_group_comm(igroup, &comm);
    CMX_ASSERT(CMX_SUCCESS == status);
    status = MPI_Barrier(comm);
    _translate_mpi_error(status, "cmx_barrier:MPI_Barrier");


    return CMX_SUCCESS;
}


STATIC int _packed_size(int *src_stride, int *count, int stride_levels)
{
    int size;
    int i;
    int n1dim;  /* number of 1 dim block */

    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
    CMX_ASSERT(NULL != src_stride);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(count[0] > 0);

#if DEBUG
    fprintf(stderr, "[%d] _packed_size(src_stride=%p, count[0]=%d, stride_levels=%d)\n",
            g_state.rank, src_stride, count[0], stride_levels);
#endif

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* allocate packed buffer now that we know the size */
    size = n1dim * count[0];

    return size;
}


STATIC char* pack(
        char *src, int *src_stride, int *count, int stride_levels, int *size)
{
    int i, j;
    long src_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int packed_index = 0;
    char *packed_buffer = NULL;

    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != src_stride);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(count[0] > 0);
    CMX_ASSERT(NULL != size);

#if DEBUG
    fprintf(stderr, "[%d] pack(src=%p, src_stride=%p, count[0]=%d, stride_levels=%d)\n",
            g_state.rank, src, src_stride, count[0], stride_levels);
#endif

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* allocate packed buffer now that we know the size */
    packed_buffer = malloc(n1dim * count[0]);
    CMX_ASSERT(packed_buffer);

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        src_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += (long) src_bvalue[j] * (long) src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        (void)memcpy(&packed_buffer[packed_index], &src[src_idx], count[0]);
        packed_index += count[0];
    }

    CMX_ASSERT(packed_index == n1dim*count[0]);
    *size = packed_index;

    return packed_buffer;
}


STATIC void unpack(char *packed_buffer,
        char *dst, int *dst_stride, int *count, int stride_levels)
{
    int i, j;
    long dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int dst_bvalue[7], dst_bunit[7];
    int packed_index = 0;

    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
    CMX_ASSERT(NULL != packed_buffer);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != dst_stride);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(count[0] > 0);

#if DEBUG
    fprintf(stderr, "[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
            g_state.rank, dst, dst_stride, count[0], stride_levels);
#endif

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* calculate the destination indices */
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        dst_bvalue[i] = 0;
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            dst_idx += (long) dst_bvalue[j] * (long) dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }

        (void)memcpy(&dst[dst_idx], &packed_buffer[packed_index], count[0]);
        packed_index += count[0];
    }

    CMX_ASSERT(packed_index == n1dim*count[0]);
}


STATIC char* _generate_shm_name(int rank)
{
    int snprintf_retval = 0;
    /* /cmxUUUUUUUUUUPPPPPPPPPPCCCCCCN */
    /* 0000000001111111111222222222233 */
    /* 1234567890123456789012345678901 */
    char *name = NULL;
    static const unsigned int limit = 62;
    static const char letters[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static unsigned int counter[6] = {0};

    CMX_ASSERT(rank >= 0);
    name = malloc(SHM_NAME_SIZE*sizeof(char));
    CMX_ASSERT(name);
    snprintf_retval = snprintf(name, SHM_NAME_SIZE,
            "/cmx%010u%010u%c%c%c%c%c%c", getuid(), getpid(),
            letters[counter[5]],
            letters[counter[4]],
            letters[counter[3]],
            letters[counter[2]],
            letters[counter[1]],
            letters[counter[0]]);
    CMX_ASSERT(snprintf_retval < (int)SHM_NAME_SIZE);
    name[SHM_NAME_SIZE-1] = '\0';
    ++counter[0];
    if (counter[0] >= limit) { ++counter[1]; counter[0] = 0; }
    if (counter[1] >= limit) { ++counter[2]; counter[1] = 0; }
    if (counter[2] >= limit) { ++counter[3]; counter[2] = 0; }
    if (counter[3] >= limit) { ++counter[4]; counter[3] = 0; }
    if (counter[4] >= limit) { ++counter[5]; counter[4] = 0; }
    if (counter[5] >= limit) {
        cmx_error("_generate_shm_name: too many names generated", -1);
    }
#if DEBUG
    fprintf(stderr, "[%d] _generate_shm_name(%d)=%s\n",
            g_state.rank, rank, name);
#endif

    return name;
}


void* cmx_malloc_local(size_t size)
{
    reg_entry_t *reg_entry;
    void *memory = NULL;

    if (size > 0) {
        reg_entry = _cmx_malloc_local(size);
        memory = reg_entry->mapped;
    }
    else {
        memory = NULL;
    }

    return memory;
}


STATIC reg_entry_t* _cmx_malloc_local(size_t size)
{
    char *name = NULL;
    void *memory = NULL;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _cmx_malloc_local(size=%lu)\n",
            g_state.rank, (long unsigned)size);
#endif

    if (0 == size) {
        return NULL;
    }

    /* create my shared memory object */
    name = _generate_shm_name(g_state.rank);
    memory = _shm_create(name, size);
#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] _cmx_malloc_local registering "
            "rank=%d mem=%p size=%lu name=%s mapped=%p\n",
            g_state.rank, g_state.rank, memory,
            (long unsigned)size, name, memory);
#endif

    /* register the memory locally */
#if USE_SICM
#if SICM_OLD 
    reg_entry = reg_cache_insert(
            g_state.rank, memory, size, name, memory, 0, NULL);
#else
    reg_entry = reg_cache_insert(
            g_state.rank, memory, size, name, memory, 0, nill);
#endif
#else
    reg_entry = reg_cache_insert(
            g_state.rank, memory, size, name, memory, 0);
#endif

    if (NULL == reg_entry) {
        cmx_error("_cmx_malloc_local: reg_cache_insert", -1);
    }

    free(name);

    return reg_entry;
}

#if USE_SICM
#if SICM_OLD
STATIC reg_entry_t* _cmx_malloc_local_memdev(size_t size, sicm_device *device)
#else
STATIC reg_entry_t* _cmx_malloc_local_memdev(size_t size, sicm_device_list device)
#endif
{
    char *name = NULL;
    void *memory = NULL;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _cmx_malloc_local(size=%lu)\n",
            g_state.rank, (long unsigned)size);
#endif

    if (0 == size) {
        return NULL;
    }

    /* create my shared memory object */
    name = _generate_shm_name(g_state.rank);
    memory = _shm_create_memdev(name, size, device);
#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] _cmx_malloc_local registering "
            "rank=%d mem=%p size=%lu name=%s mapped=%p\n",
            g_state.rank, g_state.rank, memory,
            (long unsigned)size, name, memory);
#endif

    /* register the memory locally */
    reg_entry = reg_cache_insert(
            g_state.rank, memory, size, name, memory, 1, device);

    if (NULL == reg_entry) {
        cmx_error("_cmx_malloc_local: reg_cache_insert", -1);
    }

    free(name);

    return reg_entry;
}
#endif


int cmx_free_local(void *ptr)
{
    int retval = 0;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    fprintf(stderr, "[%d] cmx_free_local(ptr=%p)\n", g_state.rank, ptr);
#endif

    if (NULL == ptr) {
        return CMX_SUCCESS;
    }

    /* find the registered memory */
    reg_entry = reg_cache_find(g_state.rank, ptr, 0);

    /* unmap the memory */
    retval = munmap(ptr, reg_entry->len);
    if (-1 == retval) {
        perror("cmx_free_local: munmap");
        cmx_error("cmx_free_local: munmap", retval);
    }

    /* remove the shared memory object */
    retval = shm_unlink(reg_entry->name);
    if (-1 == retval) {
        perror("cmx_free_local: shm_unlink");
        cmx_error("cmx_free_local: shm_unlink", retval);
    }

    /* delete the reg_cache entry */
    retval = reg_cache_delete(g_state.rank, ptr);
    CMX_ASSERT(RR_SUCCESS == retval);

    return CMX_SUCCESS;
}

#if USE_SICM
int _cmx_free_local_memdev(void *ptr)
{
    int retval = 0;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _cmx_free_local_memdev(ptr=%p)\n", g_state.rank, ptr);
#endif

    if (NULL == ptr) {
        return CMX_SUCCESS;
    }

    /* find the registered memory */
    reg_entry = reg_cache_find(g_state.rank, ptr, 0);

    /* unmap the memory */
    sicm_free(ptr);
    retval = 0;
    if (-1 == retval) {
        perror("_cmx_free_local_memdev: munmap");
        cmx_error("_cmx_free_local_memdev: munmap", retval);
    }

    /* remove the shared memory object */
    retval = shm_unlink(reg_entry->name);
    if (-1 == retval) {
        perror("_cmx_free_local_memdev: shm_unlink");
        cmx_error("_cmx_free_local_memdev: shm_unlink", retval);
    }

    /* delete the reg_cache entry */
    retval = reg_cache_delete(g_state.rank, ptr);
    CMX_ASSERT(RR_SUCCESS == retval);

    return CMX_SUCCESS;
}
#endif


int cmx_wait_proc(int proc, cmx_igroup_t *igroup)
{
    return cmx_wait_all(igroup);
}


int cmx_wait(_cmx_request* hdl)
{
    int index = 0;

    CMX_ASSERT(NULL != hdl);

#if 0
    /* this condition will likely be tripped if a blocking operation follows a
     * non-blocking operation*/
    if (0 == nb->in_use) {
        fprintf(stderr, "p[%d] cmx_wait Error: invalid handle\n",
                g_state.rank);
    }
#endif

    nb_wait_for_all(hdl);
    nb_unregister_request(hdl);

    return CMX_SUCCESS;
}


/* return 0 if operation is completed, 1 otherwise */
int cmx_test(_cmx_request* hdl, int *status)
{
    int index = 0;

    CMX_ASSERT(NULL != hdl);

#if 0
    /* this condition will likely be tripped if a blocking operation follows a
     * non-blocking operation*/
    if (0 == nb->in_use) {
        fprintf(stderr, "{%d} cmx_test Error: invalid handle\n",
                g_state.rank);
    }
#endif

    if (!nb_test_for_all(hdl)) {
      /* Completed */
        CMX_ASSERT(0 == hdl->send_size);
        CMX_ASSERT(0 == hdl->recv_size);
        *status = 0;
    }
    else {
      /* Not completed */
        *status = 1;
    }

    return CMX_SUCCESS;
}


int cmx_wait_all(cmx_igroup_t *igroup)
{
  int i;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb_list[i] != NULL && nb_list[i]->group == igroup) {
      nb_wait_for_all(nb_list[i]);
      nb_list[i]->in_use = 0;
      nb_list[i] = NULL;
    }
  }

  return CMX_SUCCESS;
}

/* The register and unregister functions are used limit to the number of
 * outstanding non-blocking handles*/
STATIC void nb_register_request(_cmx_request *nb)
{
  int ival = -1;
  int i;
  return;
  for (i=nb_last_request; i<nb_last_request+nb_max_outstanding; i++) {
    int idx = i%nb_max_outstanding;
    if (nb_list[idx] == NULL) {
      ival = idx;
      break;
    }
  }
  if (ival < 0) {
    ival = nb_last_request;
    nb_wait_for_all(nb_list[ival]);
    nb_list[ival] == NULL;
  }
  nb_list[ival] = nb;
  nb_last_request++;
  nb_last_request = nb_last_request%nb_max_outstanding;
}

STATIC void nb_unregister_request(_cmx_request *nb)
{
  int i;
  return;
  for (i=0; i<nb_max_outstanding; i++) {
    if (nb = nb_list[i]) nb_list[i] = NULL;
  }
}

STATIC void nb_handle_init(_cmx_request *nb)
{
  nb->send_size = 0;
  nb->send_head = NULL;
  nb->send_tail = NULL;
  nb->recv_size = 0;
  nb->recv_head = NULL;
  nb->recv_tail = NULL;
  nb->group = NULL;
  nb->in_use = 1;
}

int cmx_nbput(
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_put(src, dst, bytes, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}


int cmx_nbget(
        void *dst, cmxInt src_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    src = (void*)((char*)src+src_offset);
    nb_get(src, dst, bytes, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}


int cmx_nbacc(
        int datatype, void *scale,
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_acc(datatype, scale, src, dst, bytes, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}


int cmx_nbputs(
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels, 
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_puts(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}


int cmx_nbgets(
        void *dst, cmxInt *dst_stride,
        cmxInt src_offset, cmxInt *src_stride,
        cmxInt *count, int stride_levels, 
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl) 
{
    _cmx_request *nb = NULL;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    src = (void*)((char*)src+src_offset);
    nb_gets(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}


int cmx_nbaccs(
        int datatype, void *scale,
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    _cmx_request *nb = NULL;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    dst = (void*)((char*)dst+dst_offset);
    nb_accs(datatype, scale,
            src, src_stride, dst, dst_stride, count, stride_levels, world_proc, hdl);
    nb_register_request(hdl);

    return CMX_SUCCESS;
}

/**
 * non-blocking vector put operation
 * @param iov: descriptor array
 * @param iov_len: length of descriptor array
 * @param proc: remote processor id
 * @param cmx_hdl: handle for data allocation
 * @param req: non-blocking request handle
 */
int cmx_nbputv(
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    _cmx_request *nb = NULL;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    _cmx_giov_t *tiov;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].src[j] = iov[i].loc[j];
        tiov[i].dst[j] = (void*)((char*)dst+iov[i].rem[j]);
      }
    }

    nb_putv(tiov, iov_len, world_proc, hdl);
    nb_register_request(hdl);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}


int cmx_nbgetv(
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    _cmx_request *nb = NULL;
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *src;
    _cmx_giov_t *tiov;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    src = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].dst[j] = iov[i].loc[j];
        tiov[i].src[j] = (void*)((char*)src+iov[i].rem[j]);
      }
    }

    nb_getv(tiov, iov_len, world_proc, hdl);
    nb_register_request(hdl);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}

int cmx_nbaccv(
        int datatype, void *scale,
        cmx_giov_t *iov, cmxInt iov_len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *hdl)
{
    int world_proc = -1;
    cmx_igroup_t *igroup = NULL;
    void *dst;
    _cmx_giov_t *tiov;
    hdl->in_use = 1;

    nb_handle_init(hdl);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_proc = _get_world_rank(igroup, proc);
    dst = find_alloc_ptr(cmx_hdl, world_proc);

    /* copy iov to create a data structure compatible with underlying PR
     * implementation */
    tiov = (_cmx_giov_t*)malloc(iov_len*sizeof(_cmx_giov_t));
    int i,j;
    for (i=0; i<iov_len; i++) {
      cmxInt count = iov[i].count;
      tiov[i].src = malloc(count*sizeof(void*));
      tiov[i].dst = malloc(count*sizeof(void*));
      tiov[i].count = count;
      tiov[i].bytes = iov[i].bytes;
      for (j=0; j<count; j++) {
        tiov[i].src[j] = iov[i].loc[j];
        tiov[i].dst[j] = (void*)((char*)dst+iov[i].rem[j]);
      }
    }

    nb_accv(datatype, scale, tiov, iov_len, world_proc, hdl);
    nb_register_request(hdl);
    for (i=0; i<iov_len; i++) {
      free(tiov[i].src);
      free(tiov[i].dst);
    }
    free(tiov);

    return CMX_SUCCESS;
}


int cmx_rmw(
        int cmx_op, void *ploc, cmxInt rem_offset, int extra,
        int proc, cmx_handle_t cmx_hdl)
{
    header_t *header = NULL;
    char *message = NULL;
    int payload_int = 0;
    long payload_long = 0;
    int length = 0;
    op_t op = OP_NULL;
    long extra_long = (long)extra;
    int world_rank = 0;
    int master_rank = 0;
    cmx_igroup_t *igroup = NULL;
    _cmx_request nb;
    void *prem;

#if DEBUG
    fprintf(stderr, "[%d] cmx_rmw(%d, %p, %p, %d, %d)\n",
            g_state.rank, cmx_op, ploc, prem, extra, proc);
#endif

    nb_handle_init(&nb);
    igroup = cmx_hdl.group;
    CHECK_GROUP(igroup,proc);
    world_rank = _get_world_rank(igroup, proc);
    prem = find_alloc_ptr(cmx_hdl, world_rank);
    master_rank = g_state.master[world_rank];

    switch (cmx_op) {
        case CMX_FETCH_AND_ADD:
            op = OP_FETCH_AND_ADD;
            length = sizeof(int);
            payload_int = extra;
            break;
        case CMX_FETCH_AND_ADD_LONG:
            op = OP_FETCH_AND_ADD;
            length = sizeof(long);
            payload_long = extra_long;
            break;
        case CMX_SWAP:
            op = OP_SWAP;
            length = sizeof(int);
            payload_int = *((int*)ploc);
            break;
        case CMX_SWAP_LONG:
            op = OP_SWAP;
            length = sizeof(long);
            payload_long = *((long*)ploc);
            break;
        default: CMX_ASSERT(0);
    }

    /* create and prepare the header */
    message = malloc(sizeof(header_t) + length);
    CMX_ASSERT(message);
    MAYBE_MEMSET(message, 0, sizeof(header_t) + length);
    header = (header_t*)message;
    header->operation = op;
    header->remote_address = prem;
    header->local_address = ploc;
    header->rank = world_rank;
    header->length = length;
    switch (cmx_op) {
        case CMX_FETCH_AND_ADD:
        case CMX_SWAP:
            (void)memcpy(message+sizeof(header_t), &payload_int, length);
            break;
        case CMX_FETCH_AND_ADD_LONG:
        case CMX_SWAP_LONG:
            (void)memcpy(message+sizeof(header_t), &payload_long, length);
            break;
        default: CMX_ASSERT(0);
    }

    nb_recv(ploc, length, master_rank, &nb); /* prepost recv */
    nb_send_header(message, sizeof(header_t)+length, master_rank, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}


/* Mutex Operations */
int cmx_create_mutexes(int num)
{
    /* always on the world group */
    int my_master = g_state.master[g_state.rank];

    int status = 0;

#if DEBUG
    fprintf(stderr, "[%d] cmx_create_mutexes(num=%d)\n",
            g_state.rank, num);
#endif

    /* preconditions */
    CMX_ASSERT(0 <= num);
    CMX_ASSERT(NULL == num_mutexes);

    num_mutexes = (int*)malloc(group_list->size * sizeof(int));
    /* exchange of mutex counts */
    num_mutexes[group_list->rank] = num;
    status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            num_mutexes, 1, MPI_INT, group_list->comm);
    _translate_mpi_error(status, "cmx_create_mutexes:MPI_Allgather");
    CMX_ASSERT(MPI_SUCCESS == status);

    /* every process sends their own create message to their master */
    {
        _cmx_request nb;
        header_t *header = NULL;

        nb_handle_init(&nb);
        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_CREATE_MUTEXES;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = num;
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(header, sizeof(header_t), my_master, &nb);
        nb_wait_for_all(&nb);
    }

    cmx_barrier(CMX_GROUP_WORLD);

    return CMX_SUCCESS;
}


int cmx_destroy_mutexes()
{
    /* always on the world group */
    int my_master = g_state.master[g_state.rank];

#if DEBUG
    fprintf(stderr, "[%d] cmx_destroy_mutexes()\n", g_state.rank);
#endif

    /* preconditions */
    CMX_ASSERT(num_mutexes);

    /* this call is collective on the world group and this barrier ensures
     * there are no outstanding lock requests */
    cmx_barrier(CMX_GROUP_WORLD);

    /* let masters know they need to participate */
    /* first non-master rank in an SMP node sends the message to master */
    if (_smallest_world_rank_with_same_hostid(group_list) == g_state.rank) {
        _cmx_request nb;
        header_t *header = NULL;

        nb_handle_init(&nb);
        header = malloc(sizeof(header_t));
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_DESTROY_MUTEXES;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = num_mutexes[g_state.rank];
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(header, sizeof(header_t), my_master, &nb);
        nb_wait_for_all(&nb);
    }

    free(num_mutexes);
    num_mutexes = NULL;

    return CMX_SUCCESS;
}


int cmx_lock(int mutex, int proc)
{
    header_t *header = NULL;
    int world_rank = 0;
    int master_rank = 0;
    int ack = 0;
    cmx_igroup_t *igroup = NULL;
    _cmx_request nb;

#if DEBUG
    fprintf(stderr, "[%d] cmx_lock mutex=%d proc=%d\n",
            g_state.rank, mutex, proc);
#endif

    nb_handle_init(&nb);
    CHECK_GROUP(CMX_GROUP_WORLD,proc);
    igroup = CMX_GROUP_WORLD;
    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    header = malloc(sizeof(header_t));
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_LOCK;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = world_rank;
    header->length = mutex;

    nb_recv(&ack, sizeof(int), master_rank, &nb); /* prepost ack */
    nb_send_header(header, sizeof(header_t), master_rank, &nb);
    nb_wait_for_all(&nb);
    CMX_ASSERT(mutex == ack);

    return CMX_SUCCESS;
}


int cmx_unlock(int mutex, int proc)
{
    header_t *header = NULL;
    int world_rank = 0;
    int master_rank = 0;
    cmx_igroup_t *igroup = NULL;
    _cmx_request nb;

#if DEBUG
    fprintf(stderr, "[%d] cmx_unlock mutex=%d proc=%d\n", g_state.rank, mutex, proc);
#endif

    nb_handle_init(&nb);
    CHECK_GROUP(CMX_GROUP_WORLD,proc);
    igroup = CMX_GROUP_WORLD;
    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    fence_array[master_rank] = 1;
    header = malloc(sizeof(header_t));
    CMX_ASSERT(header);
    MAYBE_MEMSET(header, 0, sizeof(header_t));
    header->operation = OP_UNLOCK;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = world_rank;
    header->length = mutex;

    nb_send_header(header, sizeof(header_t), master_rank, &nb);
    nb_wait_for_all(&nb);

    return CMX_SUCCESS;
}

int cmx_malloc(cmx_handle_t *cmx_hdl, cmxInt size, cmx_igroup_t *igroup)
{
#if USE_SICM && TEST_SICM
    char cdevice[32];
#  ifdef TEST_SICM_DEV
    strcpy(cdevice,STR(TEST_SICM_DEV));
#  else
    strcpy(cdevice,"dram");
#  endif
    return cmx_malloc_mem_dev(cmx_hdl, size, group, cdevice);
#else
    reg_entry_t *reg_entries = NULL;
    reg_entry_t my_reg;
    size_t size_entries = 0;
    int my_master = -1;
    int my_world_rank = -1;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    reg_entry_t *reg_entries_local = NULL;
    int status = 0;

    /* preconditions */
    CMX_ASSERT(cmx_hdl);
   
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc(ptrs=%p, size=%lu, group=%d)\n",
            g_state.rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    cmx_barrier(igroup);

    my_world_rank = _get_world_rank(igroup, igroup->rank);
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc my_master=%d\n", g_state.rank, my_master);
#endif

    int smallest_rank_with_same_hostid, largest_rank_with_same_hostid; 
    int num_progress_ranks_per_node, is_node_ranks_packed;
    num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    is_node_ranks_packed = get_progress_rank_distribution_on_node();
    smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == get_my_master_rank_with_same_hostid(g_state.rank,
        g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
    // is_notifier = _smallest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == smallest_rank_with_same_hostid + g_state.node_size*
      ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size);
#else
    // is_notifier = _largest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == (largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size));
#endif
#endif
    if (is_notifier) {
        reg_entries_local = malloc(sizeof(reg_entry_t)*g_state.node_size);
    }

    /* allocate space for registration cache entries */
    size_entries = sizeof(reg_entry_t) * igroup->size;
    reg_entries = malloc(size_entries);
    MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*igroup->size);
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc lr_same_hostid=%d\n", 
      g_state.rank, largest_rank_with_same_hostid);
    fprintf(stderr, "[%d] cmx_malloc igroup size=%d\n", g_state.rank, igroup->size);
    fprintf(stderr, "[%d] cmx_malloc node_size=%d\n", g_state.rank, g_state.node_size);
    fprintf(stderr, "[%d] cmx_malloc is_notifier=%d\n", g_state.rank, is_notifier);
    fprintf(stderr, "[%d] rank, igroup size[5%d]\n",
            g_state.rank, igroup->size);
#endif
#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated reg entries\n",
            g_state.rank);
#endif

    /* allocate and register segment */
    MAYBE_MEMSET(&my_reg, 0, sizeof(reg_entry_t));
    if (0 == size) {
        reg_cache_nullify(&my_reg);
    }
    else {
        my_reg = *_cmx_malloc_local(sizeof(char)*size);
    }

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated and registered local shmem\n",
            g_state.rank);
#endif

    /* exchange buffer address via reg entries */
    reg_entries[igroup->rank] = my_reg;
    status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            reg_entries, sizeof(reg_entry_t), MPI_BYTE, igroup->comm);
    _translate_mpi_error(status, "cmx_malloc:MPI_Allgather");
#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc allgather status [%d]\n",
            g_state.rank, status);
#endif
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allgather reg entries\n",
            g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (NULL == reg_entries[i].buf) {
            /* a proc did not allocate (size==0) */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found NULL buf at %d\n",
                    g_state.rank, i);
#endif
        }
        else if (g_state.rank == reg_entries[i].rank) {
            /* we already registered our own memory, but PR hasn't */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found self at %d\n",
                    g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
        }
        // else if (g_state.hostid[reg_entries[i].rank]
        //         == g_state.hostid[my_world_rank]) 

        else if (g_state.master[reg_entries[i].rank] == 
           g_state.master[get_my_master_rank_with_same_hostid(g_state.rank,
           g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
           num_progress_ranks_per_node, is_node_ranks_packed)] )
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
        else if (g_state.master[reg_entries[i].rank] ==
                g_state.master[(smallest_rank_with_same_hostid + g_state.node_size *
       ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size))])
#else
        else if (g_state.master[reg_entries[i].rank] ==
                g_state.master[(largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size))])
#endif
#endif
            {
            /* same SMP node, need to mmap */
            /* open remote shared memory object */
            void *memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc registering "
                    "rank=%d buf=%p len=%lu name=%s map=%p\n",
                    g_state.rank,
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory);
#endif
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory,0
#if USE_SICM
#if SICM_OLD
                    ,NULL
#else
                    ,nill
#endif
#endif
                  );
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
        }
        else {
#if 0
            /* remote SMP node */
            /* i.e. we know about the mem but don't have local shared access */
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    NULL);
#endif
        }
    }

    /* assign the cmx handle to return to caller */
    cmx_alloc_t *prev = NULL;
    for (i=0; i<igroup->size; ++i) {
      cmx_alloc_t *link = (cmx_alloc_t*)malloc(sizeof(cmx_alloc_t));
      cmx_hdl->list = link;
      link->buf = reg_entries[i].buf;
      link->size = (cmxInt)reg_entries[i].len;
      link->rank = reg_entries[i].rank;
      link->next = prev;
      prev = link;
    }
    cmx_hdl->group = igroup;
    cmx_hdl->rank = igroup->rank;
    cmx_hdl->buf = my_reg.mapped;
    cmx_hdl->bytes = my_reg.len;

    /* send reg entries to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier) {
        _cmx_request nb;
        int reg_entries_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        nb_handle_init(&nb);
        reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
        message_size = sizeof(header_t) + reg_entries_local_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_MALLOC;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, &nb);
        nb_wait_for_all(&nb);
        free(reg_entries_local);
    }

    free(reg_entries);

    cmx_barrier(igroup);

    return CMX_SUCCESS;
#endif
}

int cmx_malloc_mem_dev(cmx_handle_t *cmx_hdl, size_t size, cmx_igroup_t *igroup,
        const char* device)
{
#if (!defined(USE_SICM) || !USE_SICM)
    return cmx_malloc(cmx_hdl,size,igroup);
#else
    cmx_igroup_t *igroup = NULL;
    reg_entry_t *reg_entries = NULL;
    reg_entry_t my_reg;
    size_t size_entries = 0;
    int my_master = -1;
    int my_world_rank = -1;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    reg_entry_t *reg_entries_local = NULL;
    int status = 0;
#if SICM_OLD
    sicm_device *idevice = NULL;
#else
    sicm_device_list idevice;
    idevice.count = 0;
    idevice.devices = NULL;
#endif

    /* preconditions */
    CMX_ASSERT(ptrs);

#if DEBUG
    fprintf(stderr, "[%d] cmx_malloc(ptrs=%p, size=%lu, group=%d)\n",
            g_state.rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    cmx_barrier(igroup);

    my_world_rank = _get_world_rank(igroup, igroup->rank);
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc my_master=%d\n", g_state.rank, my_master);
#endif

#if MASTER_IS_SMALLEST_SMP_RANK
    is_notifier = _smallest_world_rank_with_same_hostid(igroup) == g_state.rank;
#else
    is_notifier = _largest_world_rank_with_same_hostid(igroup) == g_state.rank;
#endif
    if (is_notifier) {
        reg_entries_local = malloc(sizeof(reg_entry_t)*g_state.node_size);
    }

    /* allocate space for registration cache entries */
    size_entries = sizeof(reg_entry_t) * igroup->size;
    reg_entries = malloc(size_entries);
    MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*igroup->size);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated reg entries\n",
            g_state.rank);
#endif

    /* allocate and register segment */
    MAYBE_MEMSET(&my_reg, 0, sizeof(reg_entry_t));
    if (0 == size) {
        reg_cache_nullify(&my_reg);
    }
    else {
      idevice = device_dram;

      if (!strncmp(device,"dram",4)) {
        idevice = device_dram;
      } else if (!strncmp(device,"knl_hbm",7)) {
        idevice = device_knl_hbm;
      } else if (!strncmp(device,"ppc_hbm",7)) {
        idevice = device_ppc_hbm;
      }
      my_reg = *_cmx_malloc_local_memdev(sizeof(char)*size, idevice);
    }

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allocated and registered local shmem\n",
            g_state.rank);
#endif

    /* exchange buffer address via reg entries */
    reg_entries[igroup->rank] = my_reg;
    status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            reg_entries, sizeof(reg_entry_t), MPI_BYTE, igroup->comm);
    _translate_mpi_error(status, "cmx_malloc_mem_dev:MPI_Allgather");
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_malloc allgather reg entries\n",
            g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (NULL == reg_entries[i].buf) {
            /* a proc did not allocate (size==0) */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found NULL buf at %d\n",
                    g_state.rank, i);
#endif
        }
        else if (g_state.rank == reg_entries[i].rank) {
            /* we already registered our own memory, but PR hasn't */
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc found self at %d\n",
                    g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
        }
        else if (g_state.hostid[reg_entries[i].rank]
                == g_state.hostid[my_world_rank]) {
            /* same SMP node, need to mmap */
            /* open remote shared memory object */
            void *memory = _shm_attach_memdev(reg_entries[i].name,
                reg_entries[i].len, idevice);
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_malloc registering "
                    "rank=%d buf=%p len=%lu name=%s map=%p\n",
                    g_state.rank,
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory);
#endif
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory, 1, idevice);
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
        }
        else {
#if 0
            /* remote SMP node */
            /* i.e. we know about the mem but don't have local shared access */
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    NULL);
#endif
        }
    }

    /* assign the cmx handle to return to caller */
    cmx_alloc_t *prev = NULL;
    for (i=0; i<igroup->size; ++i) {
      cmx_alloc_t *link = (cmx_alloc_t*)malloc(sizeof(cmx_alloc_t));
      cmx_hdl->list = link;
      link->buf = reg_entries[i].buf;
      link->size = (cmxInt)reg_entries[i].len;
      link->rank = reg_entries[i].rank;
      link->next = prev;
      prev = link;
    }
    cmx_hdl->group = igroup;
    cmx_hdl->rank = igroup->rank;
    cmx_hdl->buf = my_reg.mapped;
    cmx_hdl->bytes = my_reg.len;

    /* send reg entries to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier) {
        _cmx_request nb;
        int reg_entries_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        nb_handle_init(&nb);
        reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
#if DEBUG
        fprintf(stderr, "[%d] cmx_malloc, reg_entries_local_count[%d]\n",
           g_state.rank, reg_entries_local_count); 
#endif 
        message_size = sizeof(header_t) + reg_entries_local_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_MALLOC;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, &nb);
        nb_wait_for_all(&nb);
        free(reg_entries_local);
    }

    free(reg_entries);

    cmx_barrier(igroup);

    return CMX_SUCCESS;
#endif
}


/* one unnamed semaphore per world process */
void _malloc_semaphore()
{
    char *name = NULL;
    char *names = NULL;
    sem_t *my_sem = NULL;
    int status = 0;
    MPI_Datatype shm_name_type;
    int i = 0;

#if DEBUG
    fprintf(stderr, "[%d] _malloc_semaphore()\n", g_state.rank);
#endif

    status = MPI_Type_contiguous(SHM_NAME_SIZE, MPI_CHAR, &shm_name_type);
    _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_contiguous");
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Type_commit(&shm_name_type);
    _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_commmit");
    CMX_ASSERT(MPI_SUCCESS == status);

    semaphores = (sem_t**)malloc(sizeof(sem_t*) * g_state.size);
    CMX_ASSERT(semaphores);

    name = _generate_shm_name(g_state.rank);
    CMX_ASSERT(name);

#if ENABLE_UNNAMED_SEM
    {
        my_sem = _shm_create(name, sizeof(sem_t));
        /* initialize the memory as an inter-process semaphore */
        if (0 != sem_init(my_sem, 1, 1)) {
            perror("_malloc_semaphore: sem_init");
            cmx_error("_malloc_semaphore: sem_init", -1);
        }
    }
#else
    {
        my_sem = sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 1);
        if (SEM_FAILED == my_sem) {
            if (EEXIST == errno) {
                status = sem_unlink(name);
                if (-1 == status) {
                    perror("_malloc_semaphore: sem_unlink");
                    cmx_error("_malloc_semaphore: sem_unlink", status);
                }
            }
            /* second try */
            my_sem = sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 1);
        }
        if (SEM_FAILED == my_sem) {
            perror("_malloc_semaphore: sem_open");
            cmx_error("_malloc_semaphore: sem_open", -1);
        }
    }
#endif

    /* store my sem in global cache */
    semaphores[g_state.rank] = my_sem;

    names = (char*)malloc(sizeof(char) * SHM_NAME_SIZE * g_state.size);
    CMX_ASSERT(names);

    /* exchange names */
    (void)memcpy(&names[SHM_NAME_SIZE*g_state.rank], name, SHM_NAME_SIZE);
    status = MPI_Allgather(MPI_IN_PLACE, 1, shm_name_type,
            names, 1, shm_name_type, g_state.comm);
    _translate_mpi_error(status, "_malloc_semaphore:MPI_Allgather");
    CMX_ASSERT(MPI_SUCCESS == status);

    /* create/open remote semaphores and store in cache */
    for (i=0; i<g_state.size; ++i) {
        if (g_state.rank == i) {
            continue; /* skip my own rank */
        }
        else if (g_state.hostid[g_state.rank] == g_state.hostid[i]) {
            /* same SMP node */
#if ENABLE_UNNAMED_SEM
            semaphores[i] = _shm_attach(
                    &names[SHM_NAME_SIZE*i], sizeof(sem_t));
            CMX_ASSERT(semaphores[i]);
#else
            semaphores[i] = sem_open(&names[SHM_NAME_SIZE*i], 0);
            if (SEM_FAILED == semaphores[i]) {
                perror("_malloc_semaphore: sem_open");
                cmx_error("_malloc_semaphore: sem_open", -2);
            }
#endif
        }
        else {
            semaphores[i] = NULL;
        }
    }

    sem_name = name;

    free(names);

    status = MPI_Type_free(&shm_name_type);
    _translate_mpi_error(status, "_malloc_semaphore:MPI_Type_free");
    CMX_ASSERT(MPI_SUCCESS == status);
}


void _free_semaphore()
{
    int i;
    int retval;

#if DEBUG
    fprintf(stderr, "[%d] _free_semaphore()\n", g_state.rank);
#endif

    for (i=0; i<g_state.size; ++i) {
        if (g_state.rank == i) {
            /* me */
#if ENABLE_UNNAMED_SEM
            retval = sem_destroy(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_destroy");
                cmx_error("_free_semaphore: sem_destroy", retval);
            }
            retval = munmap(semaphores[i], sizeof(sem_t));
            if (-1 == retval) {
                perror("_free_semaphore: munmap");
                cmx_error("_free_semaphore: munmap", retval);
            }
            retval = shm_unlink(sem_name);
            if (-1 == retval) {
                perror("_free_semaphore: shm_unlink");
                cmx_error("_free_semaphore: shm_unlink", retval);
            }
#else
            retval = sem_close(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_close");
                cmx_error("_free_semaphore: sem_close", retval);
            }
            retval = sem_unlink(sem_name);
            if (-1 == retval) {
                perror("_free_semaphore: sem_unlink");
                cmx_error("_free_semaphore: sem_unlink", retval);
            }
#endif
        }
        else if (g_state.hostid[g_state.rank] == g_state.hostid[i]) {
            /* same SMP node */
#if ENABLE_UNNAMED_SEM
            retval = munmap(semaphores[i], sizeof(sem_t));
            if (-1 == retval) {
                perror("_free_semaphore: munmap");
                cmx_error("_free_semaphore: munmap", retval);
            }
#else
            retval = sem_close(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_close");
                cmx_error("_free_semaphore: sem_close", retval);
            }
#endif
        }
    }

    free(sem_name);
    sem_name = NULL;

    free(semaphores);
    semaphores = NULL;
}

/**
 * Access local buffer from CMX handle
 * @param handle CMX handle for data allocation
 * @param buf pointer to local buffer
 * @return CMX_SUCCESS on success
 */
int cmx_access(cmx_handle_t cmx_hdl, void **buf)
{
  *buf = cmx_hdl.buf;
  return CMX_SUCCESS;
}

int cmx_free(cmx_handle_t cmx_hdl)
{
#if (USE_SICM && TEST_SICM)
    return cmx_free_dev(cmx_hdl);
#else
    cmx_igroup_t *igroup = NULL;
    int my_world_rank = -1;
    int *world_ranks = NULL;
    int my_master = -1;
    void **ptrs = NULL;
    void *ptr;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    rank_ptr_t *rank_ptrs = NULL;
    int status = 0;
    cmx_alloc_t *list, *next;


#if DEBUG
    fprintf(stderr, "[%d] cmx_free(ptr=%p, group=%d)\n", g_state.rank, ptr, group);
#endif

    igroup = cmx_hdl.group;
    cmx_barrier(igroup);

    world_ranks = _get_world_ranks(igroup);
    my_world_rank = world_ranks[igroup->rank];
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free my_master=%d\n", g_state.rank, my_master);
#endif

    int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    int is_node_ranks_packed = get_progress_rank_distribution_on_node();
    int smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    int largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == get_my_master_rank_with_same_hostid(g_state.rank,
        g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
    // is_notifier = _smallest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == smallest_rank_with_same_hostid + g_state.node_size*
      ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size);
#else
    // is_notifier = _largest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == (largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size));
#endif
#endif
    if (is_notifier) {
        rank_ptrs = malloc(sizeof(rank_ptr_t)*g_state.node_size);
    }

    /* allocate receive buffer for exchange of pointers */
    ptrs = (void **)malloc(sizeof(void *) * igroup->size);
    CMX_ASSERT(ptrs);
    ptrs[igroup->rank] = find_alloc_ptr(cmx_hdl, my_world_rank);
    ptr = ptrs[igroup->rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free ptrs allocated and assigned\n",
            g_state.rank);
#endif

    /* exchange of pointers */
    status = MPI_Allgather(MPI_IN_PLACE, sizeof(void *), MPI_BYTE,
            ptrs, sizeof(void *), MPI_BYTE, igroup->comm);
    _translate_mpi_error(status, "cmx_free:MPI_Allgather");
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free ptrs exchanged\n", g_state.rank);
#endif

    /* remove all pointers from registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (i == igroup->rank) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free found self at %d\n", g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy? */
                rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
                rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
                reg_entries_local_count++;
            }
        }
        else if (NULL == ptrs[i]) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free found NULL at %d\n", g_state.rank, i);
#endif
        }
        // else if (g_state.hostid[world_ranks[i]]
        //         == g_state.hostid[g_state.rank]) 
        else if (g_state.master[world_ranks[i]] == 
           g_state.master[get_my_master_rank_with_same_hostid(g_state.rank,
           g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
           num_progress_ranks_per_node, is_node_ranks_packed)] )
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
        else if (g_state.master[world_ranks[i]] ==
                g_state.master[(smallest_rank_with_same_hostid + g_state.node_size *
       ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size))]) 
#else
        else if (g_state.master[world_ranks[i]] ==
                g_state.master[(largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size))]) 
#endif
#endif
        {
            /* same SMP node */
            reg_entry_t *reg_entry = NULL;
            int retval = 0;

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free same hostid at %d\n", g_state.rank, i);
#endif

            /* find the registered memory */
            reg_entry = reg_cache_find(world_ranks[i], ptrs[i], 0);
            CMX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free found reg entry\n", g_state.rank);
#endif

            /* unmap the memory */
            retval = munmap(reg_entry->mapped, reg_entry->len);
            if (-1 == retval) {
                perror("cmx_free: munmap");
                cmx_error("cmx_free: munmap", retval);
            }

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free unmapped mapped memory in reg entry\n",
                    g_state.rank);
#endif

            reg_cache_delete(world_ranks[i], ptrs[i]);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free deleted reg cache entry\n", g_state.rank);
#endif

            if (is_notifier) {
                /* does this need to be a memcpy? */
                rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
                rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
                reg_entries_local_count++;
            }
        }
        else {
#if 0
            reg_cache_delete(world_ranks[i], ptrs[i]);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free deleted reg cache entry\n", g_state.rank);
#endif
#endif
        }
    }

    /* send ptrs to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier) {
        _cmx_request nb;
        int rank_ptrs_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        nb_handle_init(&nb);
        rank_ptrs_local_size = sizeof(rank_ptr_t) * reg_entries_local_count;
        message_size = sizeof(header_t) + rank_ptrs_local_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_FREE;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        (void)memcpy(message+sizeof(header_t), rank_ptrs, rank_ptrs_local_size);
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, &nb);
        nb_wait_for_all(&nb);
        free(rank_ptrs);
    }

    /* free ptrs array */
    free(ptrs);
    free(world_ranks);

    /* remove my ptr from reg cache and free ptr */
    cmx_free_local(ptr);

    /* Is this needed? */
    cmx_barrier(igroup);

    /* clean up the cmx_handle_t struct */
    list = cmx_hdl.list; 
    while (list) {
      next = list;
      list = next->next;
      free(next);
    }

    return CMX_SUCCESS;
#endif
}

int cmx_free_dev(cmx_handle_t cmx_hdl)
{
#if USE_SICM
    cmx_igroup_t *igroup = NULL;
    int my_world_rank = -1;
    int *world_ranks = NULL;
    int my_master = -1;
    void **ptrs = NULL;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    rank_ptr_t *rank_ptrs = NULL;
    int status = 0;


#if DEBUG
    fprintf(stderr, "[%d] cmx_free_dev(ptr=%p, group=%d)\n", g_state.rank, ptr, group);
#endif

    igroup = cmx_hdl.group;
    cmx_barrier(group);
    world_ranks = _get_world_ranks(igroup);
    my_world_rank = world_ranks[igroup->rank];
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free_dev my_master=%d\n", g_state.rank, my_master);
#endif

    int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    int is_node_ranks_packed = get_progress_rank_distribution_on_node();
    int smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    int largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == get_my_master_rank_with_same_hostid(g_state.rank,
        g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
    // is_notifier = _smallest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == smallest_rank_with_same_hostid + g_state.node_size*
      ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size);
#else
    // is_notifier = _largest_world_rank_with_same_hostid(igroup) == g_state.rank;
    int largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(igroup);
    is_notifier = g_state.rank == (largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size));
#endif
#endif
    if (is_notifier) {
        rank_ptrs = malloc(sizeof(rank_ptr_t)*g_state.node_size);
    }

    /* allocate receive buffer for exchange of pointers */
    ptrs = (void **)malloc(sizeof(void *) * igroup->size);
    CMX_ASSERT(ptrs);
    ptrs[igroup->rank] = find_alloc_ptr(handle, my_world_rank);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free_dev ptrs allocated and assigned\n",
            g_state.rank);
#endif

    /* exchange of pointers */
    status = MPI_Allgather(MPI_IN_PLACE, sizeof(void *), MPI_BYTE,
            ptrs, sizeof(void *), MPI_BYTE, igroup->comm);
    _translate_mpi_error(status, "cmx_free_dev:MPI_Allgather");
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] cmx_free_dev ptrs exchanged\n", g_state.rank);
#endif

    /* remove all pointers from registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (i == igroup->rank) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev found self at %d\n", g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy? */
                rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
                rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
                reg_entries_local_count++;
            }
        }
        else if (NULL == ptrs[i]) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev found NULL at %d\n", g_state.rank, i);
#endif
        }
        // else if (g_state.hostid[world_ranks[i]]
        //         == g_state.hostid[g_state.rank]) 
        else if (g_state.master[world_ranks[i]] == 
           g_state.master[get_my_master_rank_with_same_hostid(g_state.rank,
           g_state.node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
           num_progress_ranks_per_node, is_node_ranks_packed)] )
#if 0
#if MASTER_IS_SMALLEST_SMP_RANK
        else if (g_state.master[world_ranks[i]] ==
                g_state.master[(smallest_rank_with_same_hostid + g_state.node_size *
       ((g_state.rank - smallest_rank_with_same_hostid)/g_state.node_size))]) 
#else
        else if (g_state.master[world_ranks[i]] ==
                g_state.master[(largest_rank_with_same_hostid - g_state.node_size *
       ((largest_rank_with_same_hostid - g_state.rank)/g_state.node_size))]) 
#endif
#endif
        {
            /* same SMP node */
            reg_entry_t *reg_entry = NULL;
            int retval = 0;

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev same hostid at %d\n", g_state.rank, i);
#endif

            /* find the registered memory */
            reg_entry = reg_cache_find(world_ranks[i], ptrs[i], 0);
            CMX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev found reg entry\n", g_state.rank);
#endif

            /* free the memory */
            sicm_free(reg_entry->mapped);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev unmapped mapped memory in reg entry\n",
                    g_state.rank);
#endif

            reg_cache_delete(world_ranks[i], ptrs[i]);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free_dev deleted reg cache entry\n", g_state.rank);
#endif

            if (is_notifier) {
                /* does this need to be a memcpy? */
                rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
                rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
                reg_entries_local_count++;
            }
        }
    }

    /* send ptrs to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier) {
        _cmx_request nb;
        int rank_ptrs_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        nb_handle_init(&nb);
        rank_ptrs_local_size = sizeof(rank_ptr_t) * reg_entries_local_count;
        message_size = sizeof(header_t) + rank_ptrs_local_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_FREE;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        (void)memcpy(message+sizeof(header_t), rank_ptrs, rank_ptrs_local_size);
        nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, &nb);
        nb_wait_for_all(&nb);
        free(rank_ptrs);
    }

    /* free ptrs array */
    free(ptrs);
    free(world_ranks);

    /* remove my ptr from reg cache and free ptr */
    _cmx_free_local_memdev(ptr);

    /* Is this needed? */
    cmx_barrier(igroup);

    /* clean up the cmx_handle_t struct */
    list = handle.list; 
    while (list) {
      next = list;
      list = next->next;
      free(next);
    }

    return CMX_SUCCESS;
#else
  return cmx_free(cmx_hdl);
#endif
}


STATIC void _progress_server()
{
    int running = 0;
    char *static_header_buffer = NULL;
    int static_header_buffer_size = 0;
    int extra_size = 0;
    int ierr;

#if DEBUG
    fprintf(stderr, "[%d] _progress_server()\n", g_state.rank);
    fprintf(stderr, "[%d] _progress_server(); node_size[%d]\n", 
       g_state.rank, g_state.node_size);
#endif

    {
        int status = _set_affinity(g_state.node_size);
        if (0 != status) {
            status = _set_affinity(g_state.node_size-1);
            CMX_ASSERT(0 == status);
        }
    }

    /* static header buffer size must be large enough to hold the biggest
     * message that might possibly be sent using a header type message. */
    static_header_buffer_size += sizeof(header_t);
    /* extra header info could be reg entries, one per local rank */
    extra_size = sizeof(reg_entry_t)*g_state.node_size;
    /* or, extra header info could be an acc scale plus stride */
    if ((sizeof(stride_t)+sizeof(DoubleComplex)) > extra_size) {
        extra_size = sizeof(stride_t)+sizeof(DoubleComplex);
    }
    static_header_buffer_size += extra_size;
    /* after all of the above, possibly grow the size based on user request */
    if (static_header_buffer_size < eager_threshold) {
        static_header_buffer_size = eager_threshold;
    }

    /* initialize shared buffers */
    static_header_buffer = (char*)malloc(sizeof(char)*static_header_buffer_size);
    CMX_ASSERT(static_header_buffer);
    static_server_buffer = (char*)malloc(sizeof(char)*static_server_buffer_size);
    CMX_ASSERT(static_server_buffer);

    running = 1;
    while (running) {
        int source = 0;
        int length = 0;
        char *payload = NULL;
        header_t *header = NULL;
        MPI_Status recv_status;

        ierr = MPI_Recv(static_header_buffer, static_header_buffer_size, MPI_CHAR,
                MPI_ANY_SOURCE, CMX_TAG, g_state.comm, &recv_status);
        _translate_mpi_error(ierr, "_progress_server:MPI_Recv");
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        _translate_mpi_error(ierr, "_progress_server:MPI_Get_count");
        source = recv_status.MPI_SOURCE;
#   if DEBUG
        fprintf(stderr, "[%d] progress MPI_Recv source=%d length=%d\n",
                g_state.rank, source, length);
#   endif
        header = (header_t*)static_header_buffer;
        payload = static_header_buffer + sizeof(header_t);
        /* dispatch message handler */
        switch (header->operation) {
            case OP_PUT:
                _put_handler(header, payload, source);
                break;
            case OP_PUT_PACKED:
                _put_packed_handler(header, payload, source);
                break;
            case OP_PUT_DATATYPE:
                _put_datatype_handler(header, payload, source);
                break;
            case OP_PUT_IOV:
                _put_iov_handler(header, source);
                break;
            case OP_GET:
                _get_handler(header, source);
                break;
            case OP_GET_PACKED:
                _get_packed_handler(header, payload, source);
                break;
            case OP_GET_DATATYPE:
                _get_datatype_handler(header, payload, source);
                break;
            case OP_GET_IOV:
                _get_iov_handler(header, source);
                break;
            case OP_ACC_INT:
            case OP_ACC_DBL:
            case OP_ACC_FLT:
            case OP_ACC_CPL:
            case OP_ACC_DCP:
            case OP_ACC_LNG:
                _acc_handler(header, payload, source);
                break;
            case OP_ACC_INT_PACKED:
            case OP_ACC_DBL_PACKED:
            case OP_ACC_FLT_PACKED:
            case OP_ACC_CPL_PACKED:
            case OP_ACC_DCP_PACKED:
            case OP_ACC_LNG_PACKED:
                _acc_packed_handler(header, payload, source);
                break;
            case OP_ACC_INT_IOV:
            case OP_ACC_DBL_IOV:
            case OP_ACC_FLT_IOV:
            case OP_ACC_CPL_IOV:
            case OP_ACC_DCP_IOV:
            case OP_ACC_LNG_IOV:
                _acc_iov_handler(header, payload, source);
                break;
            case OP_FENCE:
                _fence_handler(header, source);
                break;
            case OP_FETCH_AND_ADD:
                _fetch_and_add_handler(header, payload, source);
                break;
            case OP_SWAP:
                _swap_handler(header, payload, source);
                break;
            case OP_CREATE_MUTEXES:
                _mutex_create_handler(header, source);
                break;
            case OP_DESTROY_MUTEXES:
                _mutex_destroy_handler(header, source);
                break;
            case OP_LOCK:
                _lock_handler(header, source);
                break;
            case OP_UNLOCK:
                _unlock_handler(header, source);
                break;
            case OP_QUIT:
                running = 0;
                break;
            case OP_MALLOC:
                _malloc_handler(header, payload, source);
                break;
            case OP_FREE:
                _free_handler(header, payload, source);
                break;
            default:
                fprintf(stderr, "[%d] header operation not recognized: %d\n",
                        g_state.rank, header->operation);
                CMX_ASSERT(0);
        }
    }

    initialized = 0;

    free(static_header_buffer);
    free(static_server_buffer);

    _free_semaphore();

    free(mutexes);
    free(lq_heads);

    ierr = MPI_Barrier(g_state.comm);
    _translate_mpi_error(ierr, "_progress_server:MPI_Barrier");

    /* reg_cache */
    reg_cache_destroy();

    // destroy the communicators
#if DEBUG
    fprintf(stderr, "[%d] before cmx_group_finalize()\n", g_state.rank);
#endif
    cmx_group_finalize();
#if DEBUG
    fprintf(stderr, "[%d] after cmx_group_finalize()\n", g_state.rank);
#endif

#if DEBUG_TO_FILE
    fclose(cmx_trace_file);
#endif

    // assume this is the end of a user's application
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}


STATIC void _put_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int use_eager = _eager_check(header->length);

#if DEBUG
    fprintf(stderr, "[%d] _put_handler rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
            reg_entry, header->remote_address);
    if (use_eager) {
        (void)memcpy(mapped_offset, payload, header->length);
    }
    else {
        char *buf = (char*)mapped_offset;
        int bytes_remaining = header->length;
        do {
            int size = bytes_remaining>max_message_size ?
                max_message_size : bytes_remaining;
            server_recv(buf, size, proc);
            buf += size;
            bytes_remaining -= size;
        } while (bytes_remaining > 0);
    }
}


STATIC void _put_packed_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    char *packed_buffer = NULL;
    stride_t *stride = NULL;
    int use_eager = _eager_check(sizeof(stride_t)+header->length);
#if DEBUG
    int i=0;
#endif

#if DEBUG
    fprintf(stderr, "[%d] _put_packed_handler rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    stride = (stride_t*)payload;
    CMX_ASSERT(stride->stride_levels >= 0);
    CMX_ASSERT(stride->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    fprintf(stderr, "[%d] _put_packed_handler stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride->stride_levels, stride->count[0]);
    for (i=0; i<stride->stride_levels; ++i) {
        fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride->stride[i], i, stride->count[i+1]);
    }
#endif

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, stride->count[0]);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
            reg_entry, header->remote_address);

    if (use_eager) {
        packed_buffer = payload+sizeof(stride_t);
        unpack(packed_buffer, mapped_offset,
                stride->stride, stride->count, stride->stride_levels);
    }
    else {
        if ((unsigned)header->length > static_server_buffer_size) {
            packed_buffer = malloc(header->length);
        }
        else {
            packed_buffer = static_server_buffer;
        }

        {
            /* we receive the buffer backwards */
            char *buf = packed_buffer + header->length;
            int bytes_remaining = header->length;
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                buf -= size;
                server_recv(buf, size, proc);
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }

        unpack(packed_buffer, mapped_offset,
                stride->stride, stride->count, stride->stride_levels);

        if ((unsigned)header->length > static_server_buffer_size) {
            free(packed_buffer);
        }
    }
}


STATIC void _put_datatype_handler(header_t *header, char *payload, int proc)
{
    MPI_Datatype dst_type;
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    stride_t *stride = NULL;
    int ierr;
#if DEBUG
    int i=0;
#endif

#if DEBUG
    fprintf(stderr, "[%d] _put_datatype_handler rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    stride = (stride_t*)payload;
    CMX_ASSERT(stride);
    CMX_ASSERT(stride->stride_levels >= 0);
    CMX_ASSERT(stride->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    fprintf(stderr, "[%d] _put_datatype_handler stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride->stride_levels, stride->count[0]);
    for (i=0; i<stride->stride_levels; ++i) {
        fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride->stride[i], i, stride->count[i+1]);
    }
#endif

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, stride->count[0]);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
            reg_entry, header->remote_address);

    strided_to_subarray_dtype(stride->stride, stride->count,
            stride->stride_levels, MPI_BYTE, &dst_type);
    ierr = MPI_Type_commit(&dst_type);
    _translate_mpi_error(ierr,"_put_datatype_handler:MPI_Type_commit");

    server_recv_datatype(mapped_offset, dst_type, proc);

    ierr = MPI_Type_free(&dst_type);
    _translate_mpi_error(ierr,"_put_datatype_handler:MPI_Type_free");
}


STATIC void _put_iov_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int i = 0;
    char *packed_buffer = NULL;
    int packed_index = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int limit = 0;
    int bytes = 0;
    void **src = NULL;
    void **dst = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _put_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] _put_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_PUT_IOV == header->operation);

    iov_buf = malloc(header->length);
    CMX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    CMX_ASSERT(iov_off == header->length);

#if DEBUG
    fprintf(stderr, "[%d] _put_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        packed_buffer = malloc(bytes*limit);
        CMX_ASSERT(packed_buffer);
    }
    else {
        packed_buffer = static_server_buffer;
    }

    server_recv(packed_buffer, bytes * limit, proc);

    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, dst[i], bytes);
        CMX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(
                reg_entry, dst[i]);

        (void)memcpy(mapped_offset, &packed_buffer[packed_index], bytes);
        packed_index += bytes;
    }
    CMX_ASSERT(packed_index == bytes*limit);

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        free(packed_buffer);
    }

    free(iov_buf);
}


STATIC void _get_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _get_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    CMX_ASSERT(OP_GET == header->operation);
    
    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    {
        char *buf = (char*)mapped_offset;
        int bytes_remaining = header->length;
        do {
            int size = bytes_remaining>max_message_size ?
                max_message_size : bytes_remaining;
            server_send(buf, size, proc);
            buf += size;
            bytes_remaining -= size;
        } while (bytes_remaining > 0);
    }
}


STATIC void _get_packed_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    char *packed_buffer = NULL;
    int packed_index = 0;
    stride_t *stride_src = (stride_t*)payload;

#if DEBUG
    fprintf(stderr, "[%d] _get_packed_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_GET_PACKED == header->operation);

    CMX_ASSERT(stride_src->stride_levels >= 0);
    CMX_ASSERT(stride_src->stride_levels < CMX_MAX_STRIDE_LEVEL);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    packed_buffer = pack(mapped_offset,
            stride_src->stride, stride_src->count, stride_src->stride_levels,
            &packed_index);

    {
        /* we send the buffer backwards */
        char *buf = packed_buffer + packed_index;
        int bytes_remaining = packed_index;
        do {
            int size = bytes_remaining>max_message_size ?
                max_message_size : bytes_remaining;
            buf -= size;
            server_send(buf, size, proc);
            bytes_remaining -= size;
        } while (bytes_remaining > 0);
    }

    free(packed_buffer);
}


STATIC void _get_datatype_handler(header_t *header, char *payload, int proc)
{
    MPI_Datatype src_type;
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    stride_t *stride_src = NULL;
    int ierr;

#if DEBUG
    int i;
    fprintf(stderr, "[%d] _get_datatype_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_GET_DATATYPE == header->operation);

    stride_src = (stride_t*)payload;
    CMX_ASSERT(stride_src);
    CMX_ASSERT(stride_src->stride_levels >= 0);
    CMX_ASSERT(stride_src->stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    for (i=0; i<stride_src->stride_levels; ++i) {
        fprintf(stderr, "\tstride[%d]=%d\n", i, stride_src->stride[i]);
    }
    for (i=0; i<stride_src->stride_levels+1; ++i) {
        fprintf(stderr, "\tcount[%d]=%d\n", i, stride_src->count[i]);
    }
#endif

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    strided_to_subarray_dtype(stride_src->stride, stride_src->count,
            stride_src->stride_levels, MPI_BYTE, &src_type);
    ierr = MPI_Type_commit(&src_type);
    _translate_mpi_error(ierr,"_get_datatype_handler:MPI_Type_commit");

    server_send_datatype(mapped_offset, src_type, proc);

    ierr = MPI_Type_free(&src_type);
    _translate_mpi_error(ierr,"_get_datatype_handler:MPI_Type_free");
}


STATIC void _get_iov_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int i = 0;
    char *packed_buffer = NULL;
    int packed_index = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int limit = 0;
    int bytes = 0;
    void **src = NULL;
    void **dst = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _get_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] _get_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_GET_IOV == header->operation);

    iov_buf = malloc(header->length);
    CMX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    CMX_ASSERT(iov_off == header->length);

#if DEBUG
    fprintf(stderr, "[%d] _get_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        packed_buffer = malloc(bytes*limit);
        CMX_ASSERT(packed_buffer);
    }
    else {
        packed_buffer = static_server_buffer;
    }

    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, src[i], bytes);
        CMX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, src[i]);

        (void)memcpy(&packed_buffer[packed_index], mapped_offset, bytes);
        packed_index += bytes;
    }
    CMX_ASSERT(packed_index == bytes*limit);

    server_send(packed_buffer, packed_index, proc);

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        free(packed_buffer);
    }

    free(iov_buf);
}


STATIC void _acc_handler(header_t *header, char *scale, int proc)
{
    int sizeof_scale = 0;
    int acc_type = 0;
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    char *acc_buffer = NULL;
    int use_eager = 0;

#if DEBUG
    fprintf(stderr, "[%d] _acc_handler\n", g_state.rank);
#endif

    switch (header->operation) {
        case OP_ACC_INT:
            acc_type = CMX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL:
            acc_type = CMX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT:
            acc_type = CMX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG:
            acc_type = CMX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL:
            acc_type = CMX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP:
            acc_type = CMX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: CMX_ASSERT(0);
    }
    use_eager = _eager_check(sizeof_scale+header->length);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    if (use_eager) {
        acc_buffer = scale + sizeof_scale;
    }
    else {
        if ((unsigned)header->length > static_server_buffer_size) {
            acc_buffer = malloc(header->length);
        }
        else {
            acc_buffer = static_server_buffer;
        }

        {
            char *buf = (char*)acc_buffer;
            int bytes_remaining = header->length;

            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                server_recv(buf, size, proc);
                buf += size;
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }

    if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
        sem_wait(semaphores[header->rank]);
        _acc(acc_type, header->length, mapped_offset, acc_buffer, scale);
        sem_post(semaphores[header->rank]);
    }
    else {
        _acc(acc_type, header->length, mapped_offset, acc_buffer, scale);
    }

    if (use_eager) {
    }
    else {
        if ((unsigned)header->length > static_server_buffer_size) {
            free(acc_buffer);
        }
    }
}


STATIC void _acc_packed_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    void *scale = NULL;
    int sizeof_scale = 0;
    int acc_type = 0;
    char *acc_buffer = NULL;
    stride_t *stride = NULL;
    int use_eager = 0;

#if DEBUG
    fprintf(stderr, "[%d] _acc_packed_handler\n", g_state.rank);
#endif

    switch (header->operation) {
        case OP_ACC_INT_PACKED:
            acc_type = CMX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL_PACKED:
            acc_type = CMX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT_PACKED:
            acc_type = CMX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG_PACKED:
            acc_type = CMX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL_PACKED:
            acc_type = CMX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP_PACKED:
            acc_type = CMX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: CMX_ASSERT(0);
    }
    use_eager = _eager_check(sizeof_scale+sizeof(stride_t)+header->length);

    scale = payload;
    stride = (stride_t*)(payload + sizeof_scale);

    if (use_eager) {
        acc_buffer = payload+sizeof_scale+sizeof(stride_t);
    }
    else {
        if ((unsigned)header->length > static_server_buffer_size) {
            acc_buffer = malloc(header->length);
        }
        else {
            acc_buffer = static_server_buffer;
        }

        {
            /* we receive the buffer backwards */
            char *buf = acc_buffer + header->length;
            int bytes_remaining = header->length;
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                buf -= size;
                server_recv(buf, size, proc);
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
        sem_wait(semaphores[header->rank]);
    }
    {
        char *packed_buffer = acc_buffer;
        char *dst = mapped_offset;
        int *dst_stride = stride->stride;
        int *count = stride->count;
        int stride_levels = stride->stride_levels;
        int i, j;
        long dst_idx;  /* index offset of current block position to ptr */
        int n1dim;  /* number of 1 dim block */
        int dst_bvalue[7], dst_bunit[7];
        int packed_index = 0;

        CMX_ASSERT(stride_levels >= 0);
        CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);
        CMX_ASSERT(NULL != packed_buffer);
        CMX_ASSERT(NULL != dst);
        CMX_ASSERT(NULL != dst_stride);
        CMX_ASSERT(NULL != count);
        CMX_ASSERT(count[0] > 0);

#if DEBUG
        fprintf(stderr, "[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
                g_state.rank, dst, dst_stride, count[0], stride_levels);
#endif

        /* number of n-element of the first dimension */
        n1dim = 1;
        for(i=1; i<=stride_levels; i++) {
            n1dim *= count[i];
        }

        /* calculate the destination indices */
        dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

        for(i=2; i<=stride_levels; i++) {
            dst_bvalue[i] = 0;
            dst_bunit[i] = dst_bunit[i-1] * count[i-1];
        }

        for(i=0; i<n1dim; i++) {
            dst_idx = 0;
            for(j=1; j<=stride_levels; j++) {
                dst_idx += (long) dst_bvalue[j] * (long) dst_stride[j-1];
                if((i+1) % dst_bunit[j] == 0) {
                    dst_bvalue[j]++;
                }
                if(dst_bvalue[j] > (count[j]-1)) {
                    dst_bvalue[j] = 0;
                }
            }

            _acc(acc_type, count[0], &dst[dst_idx], &packed_buffer[packed_index], scale);
            packed_index += count[0];
        }

        CMX_ASSERT(packed_index == n1dim*count[0]);
    }
    if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
        sem_post(semaphores[header->rank]);
    }

    if (use_eager) {
    }
    else {
        if ((unsigned)header->length > static_server_buffer_size) {
            free(acc_buffer);
        }
    }
}


STATIC void _acc_iov_handler(header_t *header, char *scale, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int i = 0;
    char *packed_buffer = NULL;
    int packed_index = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int limit = 0;
    int bytes = 0;
    void **src = NULL;
    void **dst = NULL;
    int sizeof_scale = 0;
    int acc_type = 0;

#if DEBUG
    fprintf(stderr, "[%d] _acc_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] _acc_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

#if DEBUG
    fprintf(stderr, "[%d] _acc_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    switch (header->operation) {
        case OP_ACC_INT_IOV:
            acc_type = CMX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL_IOV:
            acc_type = CMX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT_IOV:
            acc_type = CMX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG_IOV:
            acc_type = CMX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL_IOV:
            acc_type = CMX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP_IOV:
            acc_type = CMX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: CMX_ASSERT(0);
    }

    iov_buf = malloc(header->length);
    CMX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    CMX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    CMX_ASSERT(iov_off == header->length);

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        packed_buffer = malloc(bytes*limit);
    }
    else {
        packed_buffer = static_server_buffer;
    }

    server_recv(packed_buffer, bytes*limit, proc);

    if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
        sem_wait(semaphores[header->rank]);
    }
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, dst[i], bytes);
        CMX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, dst[i]);

        _acc(acc_type, bytes, mapped_offset, &packed_buffer[packed_index], scale);
        packed_index += bytes;
    }
    CMX_ASSERT(packed_index == bytes*limit);
    if (CMX_ENABLE_ACC_SELF || CMX_ENABLE_ACC_SMP) {
        sem_post(semaphores[header->rank]);
    }

    if ((unsigned)(bytes*limit) > static_server_buffer_size) {
        free(packed_buffer);
    }

    free(iov_buf);
}


STATIC void _fence_handler(header_t *header, int proc)
{
#if DEBUG
    fprintf(stderr, "[%d] _fence_handler proc=%d\n", g_state.rank, proc);
#endif

    /* preconditions */
    CMX_ASSERT(header);

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
    fprintf(stderr, "[%d] _fence_handler asm volatile (\"\" : : : \"memory\"); \n",
            g_state.rank);
#endif
    asm volatile ("" : : : "memory"); 
#endif

    /* we send the ack back to the originating proc */
    server_send(NULL, 0, proc);
}


STATIC void _fetch_and_add_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int *value_int = NULL;
    long *value_long = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _fetch_and_add_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    fprintf(stderr, "[%d] header rem=%p loc=%p rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    CMX_ASSERT(OP_FETCH_AND_ADD == header->operation);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);
    
    if (sizeof(int) == header->length) {
        value_int = malloc(sizeof(int));
        *value_int = *((int*)mapped_offset); /* "fetch" */
        *((int*)mapped_offset) += *((int*)payload); /* "add" */
        server_send(value_int, sizeof(int), proc);
        free(value_int);
    }
    else if (sizeof(long) == header->length) {
        value_long = malloc(sizeof(long));
        *value_long = *((long*)mapped_offset); /* "fetch" */
        *((long*)mapped_offset) += *((long*)payload); /* "add" */
        server_send(value_long, sizeof(long), proc);
        free(value_long);
    }
    else {
        CMX_ASSERT(0);
    }
}


STATIC void _swap_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int *value_int = NULL;
    long *value_long = NULL;

#if DEBUG
    fprintf(stderr, "[%d] _swap_handler rem=%p loc=%p rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    CMX_ASSERT(OP_SWAP == header->operation);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    CMX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);
    
    if (sizeof(int) == header->length) {
        value_int = malloc(sizeof(int));
        *value_int = *((int*)mapped_offset); /* "fetch" */
        *((int*)mapped_offset) = *((int*)payload); /* "swap" */
        server_send(value_int, sizeof(int), proc);
        free(value_int);
    }
    else if (sizeof(long) == header->length) {
        value_long = malloc(sizeof(long));
        *value_long = *((long*)mapped_offset); /* "fetch" */
        *((long*)mapped_offset) = *((long*)payload); /* "swap" */
        server_send(value_long, sizeof(long), proc);
        free(value_long);
    }
    else {
        CMX_ASSERT(0);
    }
}


STATIC void _mutex_create_handler(header_t *header, int proc)
{
    int i;
    int num = header->length;

#if DEBUG
    fprintf(stderr, "[%d] _mutex_create_handler proc=%d num=%d\n",
            g_state.rank, proc, num);
#endif

    mutexes[proc] = (int*)malloc(sizeof(int) * num);
    lq_heads[proc] = (lock_t**)malloc(sizeof(lock_t*) * num);
    for (i=0; i<num; ++i) {
        mutexes[proc][i] = UNLOCKED;
        lq_heads[proc][i] = NULL;
    }

    server_send(NULL, 0, proc);
}


STATIC void _mutex_destroy_handler(header_t *header, int proc)
{
    int i;
    int num = header->length;

#if DEBUG
    fprintf(stderr, "[%d] _mutex_destroy_handler proc=%d\n", g_state.rank, proc);
#endif

    for (i=0; i<num; ++i) {
        CMX_ASSERT(mutexes[proc][i] == UNLOCKED);
        CMX_ASSERT(lq_heads[proc][i] == NULL);
    }

    free(mutexes[proc]);
    free(lq_heads[proc]);

    server_send(NULL, 0, proc);
}


STATIC void _lock_handler(header_t *header, int proc)
{
    int id = header->length;
    int rank = header->rank;

#if DEBUG
    fprintf(stderr, "[%d] _lock_handler id=%d in rank=%d req by proc=%d\n",
            g_state.rank, id, rank, proc);
#endif

    CMX_ASSERT(0 <= id);
    
    if (UNLOCKED == mutexes[rank][id]) {
        mutexes[rank][id] = proc;
        server_send(&id, sizeof(int), proc);
    }
    else {
        lock_t *lock = NULL;
#if DEBUG
        fprintf(stderr, "[%d] _lq_push rank=%d req_by=%d id=%d\n",
                g_state.rank, rank, proc, id);
#endif
        lock = malloc(sizeof(lock_t));
        lock->next = NULL;
        lock->rank = proc;

        if (lq_heads[rank][id]) {
            /* insert at tail */
            lock_t *lq = lq_heads[rank][id];
            while (lq->next) {
                lq = lq->next;
            }
            lq->next = lock;
        }
        else {
            /* new head */
            lq_heads[rank][id] = lock;
        }
    }
}


STATIC void _unlock_handler(header_t *header, int proc)
{
    int id = header->length;
    int rank = header->rank;

#if DEBUG
    fprintf(stderr, "[%d] _unlock_handler id=%d in rank=%d req by proc=%d\n",
            g_state.rank, id, rank, proc);
#endif

    CMX_ASSERT(0 <= id);
    
    if (lq_heads[rank][id]) {
        /* a lock requester was queued */
        /* find the next lock request and update queue */
        lock_t *lock = lq_heads[rank][id];
        lq_heads[rank][id] = lq_heads[rank][id]->next;
        /* update lock */
        mutexes[rank][id] = lock->rank;
        /* notify next in line */
        server_send(&id, sizeof(int), lock->rank);
        free(lock);
    }
    else {
        /* no enqued request */
        mutexes[rank][id] = UNLOCKED;
    }
}


STATIC void _malloc_handler(
        header_t *header, char *payload, int proc)
{
    int i;
    int n;
    reg_entry_t *reg_entries = (reg_entry_t*)payload;

#if DEBUG
    fprintf(stderr, "[%d] _malloc_handler proc=%d\n", g_state.rank, proc);
#endif

    CMX_ASSERT(header);
    CMX_ASSERT(header->operation == OP_MALLOC);
    n = header->length;

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] _malloc_handler preconditions complete\n", g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<n; ++i) {
        if (NULL == reg_entries[i].buf) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _malloc_handler found NULL at %d\n", g_state.rank, i);
#endif
        }
        else if (g_state.hostid[reg_entries[i].rank]
                == g_state.hostid[g_state.rank]) {
            /* same SMP node, need to mmap */
            /* attach to remote shared memory object */
          void *memory;
#if USE_SICM
          if (reg_entries[i].use_dev) {
            memory = _shm_attach_memdev(reg_entries[i].name, reg_entries[i].len,
                reg_entries[i].device);
          } else {
#endif
            memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if USE_SICM
          }
#endif
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _malloc_handler registering "
                    "rank=%d buf=%p len=%lu name=%s, mapped=%p\n",
                    g_state.rank,
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    (unsigned long)reg_entries[i].len,
                    reg_entries[i].name,
                    memory);
#endif
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    memory
                    ,reg_entries[i].use_dev
#if USE_SICM
                    ,reg_entries[i].device
#endif
);
        }
        else {
#if 0
            /* remote SMP node */
            /* i.e. we know about the mem but don't have local shared access */
            (void)reg_cache_insert(
                    reg_entries[i].rank,
                    reg_entries[i].buf,
                    reg_entries[i].len,
                    reg_entries[i].name,
                    NULL);
#endif
        }
    }

#if DEBUG && DEBUG_VERBOSE
    fprintf(stderr, "[%d] _malloc_handler finished registrations\n", g_state.rank);
#endif

    server_send(NULL, 0, proc); /* ack */
}


STATIC void _free_handler(header_t *header, char *payload, int proc)
{
    int i = 0;
    int n = header->length;
    rank_ptr_t *rank_ptrs = (rank_ptr_t*)payload;

#if DEBUG
    fprintf(stderr, "[%d] _free_handler proc=%d\n", g_state.rank, proc);
#endif

    /* remove all pointers from registration cache */
    for (i=0; i<n; ++i) {
        if (g_state.rank == rank_ptrs[i].rank) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] cmx_free found self at %d\n", g_state.rank, i);
#endif
        }
        else if (NULL == rank_ptrs[i].ptr) {
#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler found NULL at %d\n", g_state.rank, i);
#endif
        }
        else if (g_state.hostid[rank_ptrs[i].rank]
                == g_state.hostid[g_state.rank]) {
            /* same SMP node */
            reg_entry_t *reg_entry = NULL;
            int retval = 0;

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler same hostid at %d\n", g_state.rank, i);
#endif

            /* find the registered memory */
            reg_entry = reg_cache_find(rank_ptrs[i].rank, rank_ptrs[i].ptr, 0);
            CMX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler found reg entry\n", g_state.rank);
#endif

            /* unmap the memory */
#if USE_SICM
            if (reg_entry->use_dev) {
              sicm_free(reg_entry->mapped);
              retval = 0;
            } else {
              retval = munmap(reg_entry->mapped, reg_entry->len);
            }
#else
            retval = munmap(reg_entry->mapped, reg_entry->len);
#endif
            if (-1 == retval) {
                perror("_free_handler: munmap");
                cmx_error("_free_handler: munmap", retval);
            }

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler unmapped mapped memory in reg entry\n",
                    g_state.rank);
#endif

            reg_cache_delete(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler deleted reg cache entry\n",
                    g_state.rank);
#endif

        }
        else {
#if 0
            reg_cache_delete(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
            fprintf(stderr, "[%d] _free_handler deleted reg cache entry\n",
                    g_state.rank);
#endif
#endif
        }
    }

    server_send(NULL, 0, proc); /* ack */
}


STATIC void* _get_offset_memory(reg_entry_t *reg_entry, void *memory)
{
    ptrdiff_t offset = 0;

    CMX_ASSERT(reg_entry);
#if DEBUG_VERBOSE
    fprintf(stderr, "[%d] _get_offset_memory reg_entry->buf=%p memory=%p\n",
            g_state.rank, reg_entry->buf, memory);
#endif
    offset = ((char*)memory) - ((char*)reg_entry->buf);
#if DEBUG_VERBOSE
    fprintf(stderr, "[%d] _get_offset_memory ptrdiff=%lu\n",
            g_state.rank, (unsigned long)offset);
#endif
    return (void*)((char*)(reg_entry->mapped)+offset);
}


STATIC int _is_master(void)
{
    return (g_state.master[g_state.rank] == g_state.rank);
}


STATIC int _get_world_rank(cmx_igroup_t *igroup, int rank)
{
#if 0
    int world_rank;
    int status;

    status = MPI_Group_translate_ranks(igroup->group, 1, &rank,
            g_state.group, &world_rank);
    CHECK_MPI_RETVAL(status);
    CMX_ASSERT(MPI_PROC_NULL != world_rank);

    return world_rank;
#else
    return igroup->world_ranks[rank];
#endif
}


/* gets (in group order) corresponding world ranks for entire group */
STATIC int* _get_world_ranks(cmx_igroup_t *igroup)
{
#if 0
    int i = 0;
    int *group_ranks = (int*)malloc(sizeof(int)*igroup->size);
    int *world_ranks = (int*)malloc(sizeof(int)*igroup->size);
    int status;

    for (i=0; i<igroup->size; ++i) {
        group_ranks[i] = i;
        world_ranks[i] = MPI_PROC_NULL;
    }

    status = MPI_Group_translate_ranks(
            igroup->group, igroup->size, group_ranks,
            g_state.group, world_ranks);
    CMX_ASSERT(MPI_SUCCESS == status);

    for (i=0; i<igroup->size; ++i) {
        CMX_ASSERT(MPI_PROC_NULL != world_ranks[i]);
    }

    free(group_ranks);

    return world_ranks;
#else
#if 0
    MPI_Comm comm = igroup->comm;
    int i = 0;
    int my_world_rank = g_state.rank;
    int *world_ranks = (int*)malloc(sizeof(int)*igroup->size);
    int status;

    for (i=0; i<igroup->size; ++i) {
        world_ranks[i] = MPI_PROC_NULL;
    }

    status = MPI_Allgather(&my_world_rank,1,MPI_INT,world_ranks,
        1,MPI_INT,comm);
    CMX_ASSERT(MPI_SUCCESS == status);

    for (i=0; i<igroup->size; ++i) {
        CMX_ASSERT(MPI_PROC_NULL != world_ranks[i]);
    }

    return world_ranks;
#else
    int size = igroup->size;
    int i = 0;
    int *world_ranks = (int*)malloc(sizeof(int)*size);
    for (i=0; i<size; ++i) {
      world_ranks[i] = igroup->world_ranks[i];
    }
    return world_ranks;
#endif
#endif
}


/* we sometimes need to notify a node master of some event and the rank in
 * charge of doing that is returned by this function */
STATIC int _smallest_world_rank_with_same_hostid(cmx_igroup_t *igroup)
{
    int i = 0;
    int smallest = g_state.rank;
    int *world_ranks = _get_world_ranks(igroup);

    for (i=0; i<igroup->size; ++i) {
        if (g_state.hostid[world_ranks[i]] == g_state.hostid[g_state.rank]) {
            /* found same host as me */
            if (world_ranks[i] < smallest) {
                smallest = world_ranks[i];
            }
        }
    }

    free(world_ranks);

    return smallest;
}


/* we sometimes need to notify a node master of some event and the rank in
 * charge of doing that is returned by this function */
STATIC int _largest_world_rank_with_same_hostid(cmx_igroup_t *igroup)
{
    int i = 0;
    int largest = g_state.rank;
    int *world_ranks = _get_world_ranks(igroup);

    for (i=0; i<igroup->size; ++i) {
        if (g_state.hostid[world_ranks[i]] == g_state.hostid[g_state.rank]) {
            /* found same host as me */
            if (world_ranks[i] > largest) {
                largest = world_ranks[i];
            }
        }
    }

    free(world_ranks);

    return largest;
}


STATIC void* _shm_create(const char *name, size_t size)
{
    void *mapped = NULL;
    int fd = 0;
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] _shm_create(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* create shared memory segment */
    fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd && EEXIST == errno) {
        retval = shm_unlink(name);
        if (-1 == retval) {
            perror("_shm_create: shm_unlink");
            cmx_error("_shm_create: shm_unlink", retval);
        }
    }

    /* try a second time */
    if (-1 == fd) {
        fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    }

    /* finally report error if needed */
    if (-1 == fd) {
        perror("_shm_create: shm_open");
        cmx_error("_shm_create: shm_open", fd);
    }

    /* set the size of my shared memory object */
    retval = ftruncate(fd, size);
    if (-1 == retval) {
        perror("_shm_create: ftruncate");
        cmx_error("_shm_create: ftruncate", retval);
    }

    /* map into local address space */
    mapped = _shm_map(fd, size);

    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_create: close");
        cmx_error("_shm_create: close", -1);
    }

    return mapped;
}

#if USE_SICM
#if SICM_OLD
STATIC void* _shm_create_memdev(const char *name, size_t size, sicm_device* device)
#else
STATIC void* _shm_create_memdev(const char *name, size_t size, sicm_device_list device)
#endif
{
    void *mapped = NULL;
    int fd = 0;
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] _shm_create_memdev(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* create shared memory segment */
    fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd && EEXIST == errno) {
        retval = shm_unlink(name);
        if (-1 == retval) {
            perror("_shm_create_memdev: shm_unlink");
            cmx_error("_shm_create_memdev: shm_unlink", retval);
        }
    }

    /* try a second time */
    if (-1 == fd) {
        fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    }

    /* finally report error if needed */
    if (-1 == fd) {
        perror("_shm_create_memdev: shm_open");
        cmx_error("_shm_create_memdev: shm_open", fd);
    }

    /* the file will be used for arena allocation,
     * so it should not be truncated here */
#if SICM_OLD
    sicm_arena arena = sicm_arena_create_mmapped(0, device, fd, 0, -1, 0);
#else
    sicm_arena arena = sicm_arena_create_mmapped(0, 0, &device, fd, 0, -1, 0);
#endif
    /* map into local address space */
    mapped = _shm_map_arena(fd, size, arena);

    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_create_memdev: close");
        cmx_error("_shm_create_memdev: close", -1);
    }

    return mapped;
}
#endif


STATIC void* _shm_attach(const char *name, size_t size)
{
    void *mapped = NULL;
    int fd = 0;
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] _shm_attach(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* attach to shared memory segment */
    fd = shm_open(name, O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd) {
        perror("_shm_attach: shm_open");
        cmx_error("_shm_attach: shm_open", -1);
    }

    /* map into local address space */
    mapped = _shm_map(fd, size);
    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_attach: close");
        cmx_error("_shm_attach: close", -1);
    }

    return mapped;
}

#if USE_SICM
#if SICM_OLD
STATIC void* _shm_attach_memdev(const char *name, size_t size, sicm_device *device)
#else
STATIC void* _shm_attach_memdev(const char *name, size_t size, sicm_device_list device)
#endif
{
    void *mapped = NULL;
    int fd = 0;
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] _shm_attach_memdev(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* attach to shared memory segment */
    fd = shm_open(name, O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd) {
        perror("_shm_attach_memdev: shm_open");
        cmx_error("_shm_attach_memdev: shm_open", -1);
    }
#if SICM_OLD
    sicm_arena arena = sicm_arena_create_mmapped(0, device, fd, 0, -1, 0);
#else
    sicm_arena arena = sicm_arena_create_mmapped(0, 0, &device, fd, 0, -1, 0);
#endif

    /* map into local address space */
    mapped = _shm_map_arena(fd, size, arena);
    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_attach_memdev: close");
        cmx_error("_shm_attach_memdev: close", -1);
    }

    return mapped;
}
#endif

#if USE_SICM
STATIC void* _shm_map_arena(int fd, size_t size, sicm_arena arena)
{
    void *memory = sicm_arena_alloc(arena, size);
    if (NULL == memory) {
        perror("_shm_map_arena: mmap");
        cmx_error("_shm_map_arena: mmap", -1);
    }

    return memory;
}
#endif
STATIC void* _shm_map(int fd, size_t size)
{
    void *memory  = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == memory) {
        perror("_shm_map: mmap");
        cmx_error("_shm_map: mmap", -1);
    }

    return memory;
}


STATIC int _set_affinity(int cpu)
{
    int status = 0;
#if CMX_SET_AFFINITY
#if HAVE_PTHREAD_SETAFFINITY_NP || HAVE_SCHED_SETAFFINITY
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
#if HAVE_PTHREAD_SETAFFINITY_NP
    status = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (0 != status) {
        perror("pthread_setaffinity_np");
    }
#elif HAVE_SCHED_SETAFFINITY
    status = sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);
    if (0 != status) {
        perror("sched_setaffinity");
    }
#endif
#endif
#endif

    return status;
}


STATIC void check_mpi_retval(int retval, const char *file, int line)
{
    if (MPI_SUCCESS != retval) {
        const char *msg = str_mpi_retval(retval);
        fprintf(stderr, "{%d} MPI Error: %s: line %d: %s\n",
                g_state.rank, file, line, msg);
        MPI_Abort(g_state.comm, retval);
    }
}


STATIC const char *str_mpi_retval(int retval)
{
    const char *msg = NULL;
         if (retval == MPI_SUCCESS      ) { msg = "MPI_SUCCESS";        }
    else if (retval == MPI_ERR_BUFFER   ) { msg = "MPI_ERR_BUFFER";     }
    else if (retval == MPI_ERR_COUNT    ) { msg = "MPI_ERR_COUNT";      }
    else if (retval == MPI_ERR_TYPE     ) { msg = "MPI_ERR_TYPE";       }
    else if (retval == MPI_ERR_TAG      ) { msg = "MPI_ERR_TAG";        }
    else if (retval == MPI_ERR_COMM     ) { msg = "MPI_ERR_COMM";       }
    else if (retval == MPI_ERR_RANK     ) { msg = "MPI_ERR_RANK";       }
    else if (retval == MPI_ERR_ROOT     ) { msg = "MPI_ERR_ROOT";       }
    else if (retval == MPI_ERR_GROUP    ) { msg = "MPI_ERR_GROUP";      }
    else if (retval == MPI_ERR_OP       ) { msg = "MPI_ERR_OP";         }
    else if (retval == MPI_ERR_TOPOLOGY ) { msg = "MPI_ERR_TOPOLOGY";   }
    else if (retval == MPI_ERR_DIMS     ) { msg = "MPI_ERR_DIMS";       }
    else if (retval == MPI_ERR_ARG      ) { msg = "MPI_ERR_ARG";        }
    else if (retval == MPI_ERR_UNKNOWN  ) { msg = "MPI_ERR_UNKNOWN";    }
    else if (retval == MPI_ERR_TRUNCATE ) { msg = "MPI_ERR_TRUNCATE";   }
    else if (retval == MPI_ERR_OTHER    ) { msg = "MPI_ERR_OTHER";      }
    else if (retval == MPI_ERR_INTERN   ) { msg = "MPI_ERR_INTERN";     }
    else if (retval == MPI_ERR_IN_STATUS) { msg = "MPI_ERR_IN_STATUS";  }
    else if (retval == MPI_ERR_PENDING  ) { msg = "MPI_ERR_PENDING";    }
    else if (retval == MPI_ERR_REQUEST  ) { msg = "MPI_ERR_REQUEST";    }
    else if (retval == MPI_ERR_LASTCODE ) { msg = "MPI_ERR_LASTCODE";   }
    else                                  { msg = "DEFAULT";            }
    return msg;
}


STATIC void server_send(void *buf, int count, int dest)
{
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] server_send(buf=%p, count=%d, dest=%d)\n",
            g_state.rank, buf, count, dest);
#endif

    retval = MPI_Send(buf, count, MPI_CHAR, dest,
            CMX_TAG, g_state.comm);
    _translate_mpi_error(retval,"server_send:MPI_Send");

    CHECK_MPI_RETVAL(retval);
}


STATIC void server_send_datatype(void *buf, MPI_Datatype dt, int dest)
{
    int retval = 0;

#if DEBUG
    fprintf(stderr, "[%d] server_send_datatype(buf=%p, ..., dest=%d)\n",
            g_state.rank, buf, dest);
#endif

    retval = MPI_Send(buf, 1, dt, dest, CMX_TAG, g_state.comm);
    _translate_mpi_error(retval,"server_send_datatype:MPI_Send");

    CHECK_MPI_RETVAL(retval);
}


STATIC void server_recv(void *buf, int count, int source)
{
    int retval = 0;
    MPI_Status status;
    int recv_count = 0;

    retval = MPI_Recv(buf, count, MPI_CHAR, source,
            CMX_TAG, g_state.comm, &status);
    _translate_mpi_error(retval,"server_recv:MPI_Recv");

    CHECK_MPI_RETVAL(retval);
    CMX_ASSERT(status.MPI_SOURCE == source);
    CMX_ASSERT(status.MPI_TAG == CMX_TAG);

    retval = MPI_Get_count(&status, MPI_CHAR, &recv_count);
    _translate_mpi_error(retval,"server_recv:MPI_Get_count");
    CHECK_MPI_RETVAL(retval);
    CMX_ASSERT(recv_count == count);
}


STATIC void server_recv_datatype(void *buf, MPI_Datatype dt, int source)
{
    int retval = 0;
    MPI_Status status;

    retval = MPI_Recv(buf, 1, dt, source,
            CMX_TAG, g_state.comm, &status);
    _translate_mpi_error(retval,"server_recv_datatype:MPI_Recv");

    CHECK_MPI_RETVAL(retval);
    CMX_ASSERT(status.MPI_SOURCE == source);
    CMX_ASSERT(status.MPI_TAG == CMX_TAG);
}


STATIC void nb_send_common(void *buf, int count, int dest, _cmx_request *nb, int need_free)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != nb);

    nb->send_size += 1;
    nb_count_event += 1;
    nb_count_send += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = need_free;
    message->stride = NULL;
    message->iov = NULL;
    message->datatype = MPI_DATATYPE_NULL;

    if (NULL == nb->send_head) {
        nb->send_head = message;
    }
    if (NULL != nb->send_tail) {
        nb->send_tail->next = message;
    }
    nb->send_tail = message;

    retval = MPI_Isend(buf, count, MPI_CHAR, dest, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_send_common:MPI_Isend");
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_send_datatype(void *buf, MPI_Datatype dt, int dest, _cmx_request *nb)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != nb);

    nb->send_size += 1;
    nb_count_event += 1;
    nb_count_send += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = 0;
    message->stride = NULL;
    message->iov = NULL;
    message->datatype = dt;

    if (NULL == nb->send_head) {
        nb->send_head = message;
    }
    if (NULL != nb->send_tail) {
        nb->send_tail->next = message;
    }
    nb->send_tail = message;

    retval = MPI_Isend(buf, 1, dt, dest, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_send_datatype:MPI_Isend");
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_send_header(void *buf, int count, int dest, _cmx_request *nb)
{
    nb_send_common(buf, count, dest, nb, 1);
}


STATIC void nb_send_buffer(void *buf, int count, int dest, _cmx_request *nb)
{
    nb_send_common(buf, count, dest, nb, 0);
}


STATIC void nb_recv_packed(void *buf, int count, int source, _cmx_request *nb, stride_t *stride)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != buf);
    CMX_ASSERT(count > 0);
    CMX_ASSERT(NULL != nb);

#if DEBUG
    fprintf(stderr, "[%d] nb_recv_packed(buf=%p, count=%d, source=%d, nb=%p)\n",
            g_state.rank, buf, count, source, nb);
#endif

    nb->recv_size += 1;
    nb_count_event += 1;
    nb_count_recv += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = 1;
    message->stride = stride;
    message->iov = NULL;
    message->datatype = MPI_DATATYPE_NULL;

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_recv_packed:MPI_Irecv");
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_recv_datatype(void *buf, MPI_Datatype dt, int source, _cmx_request *nb)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != buf);
    CMX_ASSERT(NULL != nb);

#if DEBUG
    fprintf(stderr, "[%d] nb_recv_datatype(buf=%p, count=%d, source=%d, nb=%p)\n",
            g_state.rank, buf, count, source, nb);
#endif

    nb->recv_size += 1;
    nb_count_event += 1;
    nb_count_recv += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = 0;
    message->stride = NULL;
    message->iov = NULL;
    message->datatype = dt;

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, 1, dt, source, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_recv_datatype:MPI_Irecv");
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_recv_iov(void *buf, int count, int source, _cmx_request *nb, _cmx_giov_t *iov)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != nb);

#if DEBUG
    fprintf(stderr, "[%d] nb_recv_iov(buf=%p, count=%d, source=%d, nb=%p)\n",
            g_state.rank, buf, count, source, nb);
#endif

    nb->recv_size += 1;
    nb_count_event += 1;
    nb_count_recv += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = 1;
    message->stride = NULL;
    message->iov = iov;
    message->datatype = MPI_DATATYPE_NULL;

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
        CMX_ASSERT(NULL == nb->recv_tail);
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_recv_iov:MPI_Irecv");
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_recv(void *buf, int count, int source, _cmx_request *nb)
{
    int retval = 0;
    message_t *message = NULL;

    CMX_ASSERT(NULL != nb);

#if DEBUG
    fprintf(stderr, "[%d] nb_recv(buf=%p, count=%d, source=%d, nb=%p)\n",
            g_state.rank, buf, count, source, nb);
#endif

    nb->recv_size += 1;
    nb_count_event += 1;
    nb_count_recv += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = NULL;
    message->need_free = 0;
    message->stride = NULL;
    message->iov = NULL;
    message->datatype = MPI_DATATYPE_NULL;

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, CMX_TAG, g_state.comm,
            &(message->request));
    _translate_mpi_error(retval,"nb_recv:MPI_Irecv");
    CHECK_MPI_RETVAL(retval);
}


STATIC int nb_get_handle_index()
{
    int value = 0;

    if (0 == nb_index) {
        value = nb_max_outstanding-1;
    }
    else {
        value = nb_index-1;
    }

    return value;
}


#if 0
STATIC _cmx_request* nb_wait_for_handle()
{
    _cmx_request *nb = NULL;
    int in_use_count = 0;
    int loop_index = nb_index;
    int found = 0;

    /* find first handle that isn't associated with a user-level handle */
    /* make sure the handle we find has processed all events */
    /* the user can accidentally exhaust the available handles */

#if 0
    /* NOTE: it looks like this loop just forces completion of the handle
     * corresponding to nb_index. It should probably test all handles and if it
     * doesn't find  a free one, then use the one at nb_index */
    do {
        ++in_use_count;
        if (in_use_count > nb_max_outstanding) {
            fprintf(stderr,
                    "{%d} nb_wait_for_handle Error: all user-level "
                    "nonblocking handles have been exhausted\n",
                    g_state.rank);
            MPI_Abort(g_state.comm, -1);
        }
        nb = &nb_state[nb_index];
        nb_index++;
        nb_index %= nb_max_outstanding; /* wrap around if needed */
        nb_wait_for_all(nb);
    } while (nb->in_use);
#else
    /* look through list for unused handle */
    do {
        ++in_use_count;
        if (in_use_count > nb_max_outstanding) {
          break;
        }
        nb = &nb_state[loop_index];
        if (!nb->in_use) {
          nb_index = loop_index;
          found = 1;
          break;
        }
        loop_index++;
        loop_index %= nb_max_outstanding; /* wrap around if needed */
    } while (nb->in_use);
    if (!found) {
      nb = &nb_state[nb_index];
      nb_wait_for_all(nb);
    }
    //nb->hdl = nb_index;
    nb_index++;
    nb_index %= nb_max_outstanding; /* wrap around if needed */
    /* make sure in_use flag is set to 1 */
    nb->in_use = 1;
#endif

    return nb;
}
#endif


STATIC void nb_wait_for_send1(_cmx_request *nb)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_wait_for_send1(nb=%p)\n", g_state.rank, nb);
#endif

    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(NULL != nb->send_head);

    {
        MPI_Status status;
        int retval = 0;
        message_t *message_to_free = NULL;

        retval = MPI_Wait(&(nb->send_head->request), &status);
       _translate_mpi_error(retval,"nb_wait_for_send1:MPI_Irecv");
        CHECK_MPI_RETVAL(retval);

        if (nb->send_head->need_free) {
            free(nb->send_head->message);
        }

        if (MPI_DATATYPE_NULL != nb->send_head->datatype) {
            retval = MPI_Type_free(&nb->send_head->datatype);
           _translate_mpi_error(retval,"nb_wait_for_send1:MPI_Type_free");
            CHECK_MPI_RETVAL(retval);
        }

        message_to_free = nb->send_head;
        nb->send_head = nb->send_head->next;
        free(message_to_free);

        CMX_ASSERT(nb->send_size > 0);
        nb->send_size -= 1;
        nb_count_send_processed += 1;
        nb_count_event_processed += 1;

        if (NULL == nb->send_head) {
            nb->send_tail = NULL;
        }
    }
}


/* returns true if operation has completed */
STATIC int nb_test_for_send1(_cmx_request *nb, message_t **save_send_head,
    message_t **prev)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_test_for_send1(nb=%p)\n", g_state.rank, nb);
#endif

    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(NULL != nb->send_head);

    {
        MPI_Status status;
        int retval = 0;
        int flag;
        message_t *message_to_free = NULL;

        retval = MPI_Test(&(nb->send_head->request), &flag, &status);
        _translate_mpi_error(retval,"nb_test_for_send1:MPI_Test");
        CHECK_MPI_RETVAL(retval);

        if (flag) {
          if (nb->send_head->need_free) {
            free(nb->send_head->message);
          }

          if (MPI_DATATYPE_NULL != nb->send_head->datatype) {
            retval = MPI_Type_free(&nb->send_head->datatype);
            _translate_mpi_error(retval,"nb_test_for_send1:MPI_Type_free");
            CHECK_MPI_RETVAL(retval);
          }

          message_to_free = nb->send_head;
          if (*prev) (*prev)->next=nb->send_head->next;
          nb->send_head = nb->send_head->next;
          *save_send_head = NULL;
          free(message_to_free);

          CMX_ASSERT(nb->send_size > 0);
          nb->send_size -= 1;
          nb_count_send_processed += 1;
          nb_count_event_processed += 1;

          if (NULL == nb->send_head) {
            nb->send_tail = NULL;
          }
        } else {
          *prev = nb->send_head;
          *save_send_head = nb->send_head;
          nb->send_head = nb->send_head->next;
        }
        return flag;
    }
}


STATIC void nb_wait_for_recv1(_cmx_request *nb)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_wait_for_recv1(nb=%p)\n", g_state.rank, nb);
#endif

    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(NULL != nb->recv_head);

    {
        MPI_Status status;
        int retval = 0;
        message_t *message_to_free = NULL;

        retval = MPI_Wait(&(nb->recv_head->request), &status);
        _translate_mpi_error(retval,"nb_wait_for_recv1:MPI_Wait");
        CHECK_MPI_RETVAL(retval);

        if (NULL != nb->recv_head->stride) {
            stride_t *stride = nb->recv_head->stride;
            CMX_ASSERT(nb->recv_head->message);
            CMX_ASSERT(stride);
            CMX_ASSERT(stride->ptr);
            CMX_ASSERT(stride->stride);
            CMX_ASSERT(stride->count);
            CMX_ASSERT(stride->stride_levels);
            unpack(nb->recv_head->message, stride->ptr,
                    stride->stride, stride->count, stride->stride_levels);
            free(stride);
        }

        if (NULL != nb->recv_head->iov) {
            int i = 0;
            char *message = nb->recv_head->message;
            int off = 0;
            _cmx_giov_t *iov = nb->recv_head->iov;
            for (i=0; i<iov->count; ++i) {
                (void)memcpy(iov->dst[i], &message[off], iov->bytes);
                off += iov->bytes;
            }
            free(iov->src);
            free(iov->dst);
            free(iov);
        }

        if (nb->recv_head->need_free) {
            free(nb->recv_head->message);
        }

        if (MPI_DATATYPE_NULL != nb->recv_head->datatype) {
            retval = MPI_Type_free(&nb->recv_head->datatype);
            _translate_mpi_error(retval,"nb_wait_for_recv1:MPI_Type_free");
            CHECK_MPI_RETVAL(retval);
        }

        message_to_free = nb->recv_head;
        nb->recv_head = nb->recv_head->next;
        free(message_to_free);

        CMX_ASSERT(nb->recv_size > 0);
        nb->recv_size -= 1;
        nb_count_recv_processed += 1;
        nb_count_event_processed += 1;

        if (NULL == nb->recv_head) {
            nb->recv_tail = NULL;
        }
    }
}


/* returns true if operation has completed */
STATIC int nb_test_for_recv1(_cmx_request *nb, message_t **save_recv_head,
    message_t **prev)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_wait_for_recv1(nb=%p)\n", g_state.rank, nb);
#endif

    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(NULL != nb->recv_head);

    {
        MPI_Status status;
        int retval = 0;
        int flag;
        message_t *message_to_free = NULL;

        retval = MPI_Test(&(nb->recv_head->request), &flag, &status);
        _translate_mpi_error(retval,"nb_test_for_recv1:MPI_Test");
        CHECK_MPI_RETVAL(retval);

        if (flag) {
          if (NULL != nb->recv_head->stride) {
            stride_t *stride = nb->recv_head->stride;
            CMX_ASSERT(nb->recv_head->message);
            CMX_ASSERT(stride);
            CMX_ASSERT(stride->ptr);
            CMX_ASSERT(stride->stride);
            CMX_ASSERT(stride->count);
            CMX_ASSERT(stride->stride_levels);
            unpack(nb->recv_head->message, stride->ptr,
                stride->stride, stride->count, stride->stride_levels);
            free(stride);
          }

          if (NULL != nb->recv_head->iov) {
            int i = 0;
            char *message = nb->recv_head->message;
            int off = 0;
            _cmx_giov_t *iov = nb->recv_head->iov;
            for (i=0; i<iov->count; ++i) {
              (void)memcpy(iov->dst[i], &message[off], iov->bytes);
              off += iov->bytes;
            }
            free(iov->src);
            free(iov->dst);
            free(iov);
          }

          if (nb->recv_head->need_free) {
            free(nb->recv_head->message);
          }

          if (MPI_DATATYPE_NULL != nb->recv_head->datatype) {
            retval = MPI_Type_free(&nb->recv_head->datatype);
           _translate_mpi_error(retval,"nb_test_for_recv1:MPI_Type_free");
            CHECK_MPI_RETVAL(retval);
          }

          message_to_free = nb->recv_head;
          if (*prev) (*prev)->next=nb->recv_head->next;
          nb->recv_head = nb->recv_head->next;
          *save_recv_head = NULL;
          free(message_to_free);

          CMX_ASSERT(nb->recv_size > 0);
          nb->recv_size -= 1;
          nb_count_recv_processed += 1;
          nb_count_event_processed += 1;

          if (NULL == nb->recv_head) {
            nb->recv_tail = NULL;
          }
        } else {
          *prev = nb->recv_head;
          *save_recv_head = nb->recv_head;
           nb->recv_head = nb->recv_head->next;
        }
        return flag;
    }
}


STATIC void nb_wait_for_all(_cmx_request *nb)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_wait_for_all(nb=%p)\n", g_state.rank, nb);
#endif
    int world_proc = g_state.rank;

    if (nb->in_use == 0) return;

    CMX_ASSERT(NULL != nb);

    /* fair processing of requests */
    while (NULL != nb->send_head || NULL != nb->recv_head) {
        if (NULL != nb->send_head) {
            nb_wait_for_send1(nb);
        }
        if (NULL != nb->recv_head) {
            nb_wait_for_recv1(nb);
        }
    }
    nb->in_use = 0;
}

/* Returns 0 if no outstanding requests */

STATIC int nb_test_for_all(_cmx_request *nb)
{
#if DEBUG
    fprintf(stderr, "[%d] nb_test_for_all(nb=%p)\n", g_state.rank, nb);
#endif
    int ret = 0;
    message_t *save_send_head = NULL;
    message_t *save_recv_head = NULL;
    message_t *tmp_send_head;
    message_t *tmp_recv_head;
    message_t *send_prev = NULL;
    message_t *recv_prev = NULL;

    /**
     * TODO: Determine if this condition may be true for a valid series of
     * operations. In particular, if a set of non-blocking operations follow a
     * set of blocking operations.
    CMX_ASSERT(NULL != nb);
    */
    if (nb == NULL) return 0;

    /* check for outstanding requests */
    while (NULL != nb->send_head || NULL != nb->recv_head) {
      if (NULL != nb->send_head) {
        if (!nb_test_for_send1(nb, &tmp_send_head, &send_prev)) {
          ret = 1; 
        }
        if ((NULL == save_send_head) && (ret == 1)) {
          save_send_head = tmp_send_head;
        }
      }
      if (NULL != nb->recv_head) {
        if (!nb_test_for_recv1(nb, &tmp_recv_head, &recv_prev)) {
          ret = 1;
        }
        if ((NULL == save_recv_head) && (ret == 1)) {
          save_recv_head = tmp_recv_head;
        }
      }
    }
    nb->send_head = save_send_head;
    nb->recv_head = save_recv_head;
    if (ret == 0) nb->in_use = 0;
    return ret;
}


STATIC void nb_put(void *src, void *dst, int bytes, int proc, _cmx_request *nb)
{
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(bytes > 0);
    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != nb);

#if DEBUG
    printf("[%d] nb_put(src=%p, dst=%p, bytes=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, dst, bytes, proc, nb);
#endif

    if (CMX_ENABLE_PUT_SELF) {
        /* put to self */
        if (g_state.rank == proc) {
            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }
            (void)memcpy(dst, src, bytes);
            return;
        }
    }

    if (CMX_ENABLE_PUT_SMP) {
        /* put to SMP node */
        // if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) 
        if (g_state.master[proc] == g_state.master[g_state.rank]) 
        {
            reg_entry_t *reg_entry = NULL;
            void *mapped_offset = NULL;

            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }

            reg_entry = reg_cache_find(proc, dst, bytes);
            CMX_ASSERT(reg_entry);
            mapped_offset = _get_offset_memory(reg_entry, dst);
            (void)memcpy(mapped_offset, src, bytes);
            return;
        }
    }

    {
        char *message = NULL;
        int message_size = 0;
        header_t *header = NULL;
        int master_rank = -1;
        int use_eager = _eager_check(bytes);

        master_rank = g_state.master[proc];
        /* only fence on the master */
        fence_array[master_rank] = 1;
        if (use_eager) {
            message_size = sizeof(header_t) + bytes;
        }
        else {
            message_size = sizeof(header_t);
        }
        message = malloc(message_size);
        header = (header_t*)message;
        header->operation = OP_PUT;
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->remote_address = dst;
        header->local_address = src;
        header->rank = proc;
        header->length = bytes;
        if (use_eager) {
            (void)memcpy(message+sizeof(header_t), src, bytes);
            nb_send_header(message, message_size, master_rank, nb);
        }
        else {
            char *buf = (char*)src;
            int bytes_remaining = bytes;
            nb_send_header(header, sizeof(header_t), master_rank, nb);
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                nb_send_buffer(buf, size, master_rank, nb);
                buf += size;
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }
    nb->in_use = 1;
}


STATIC void nb_get(void *src, void *dst, int bytes, int proc, _cmx_request *nb)
{
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(bytes > 0);
    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != nb);

    if (CMX_ENABLE_GET_SELF) {
        /* get from self */
        if (g_state.rank == proc) {
            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }
            (void)memcpy(dst, src, bytes);
            return;
        }
    }

    if (CMX_ENABLE_GET_SMP) {
        /* get from SMP node */
        // if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) 
        if (g_state.master[proc] == g_state.master[g_state.rank]) 
        {
            reg_entry_t *reg_entry = NULL;
            void *mapped_offset = NULL;

            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }

            reg_entry = reg_cache_find(proc, src, bytes);
            CMX_ASSERT(reg_entry);
            mapped_offset = _get_offset_memory(reg_entry, src);
            (void)memcpy(dst, mapped_offset, bytes);
            return;
        }
    }

    {
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];
        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_GET;
        header->remote_address = src;
        header->local_address = dst;
        header->rank = proc;
        header->length = bytes;
        {
            /* prepost all receives */
            char *buf = (char*)dst;
            int bytes_remaining = bytes;
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                nb_recv(buf, size, master_rank, nb);
                buf += size;
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
        nb_send_header(header, sizeof(header_t), master_rank, nb);
    }
    nb->in_use = 1;
}


STATIC void nb_acc(int datatype, void *scale,
        void *src, void *dst, int bytes, int proc, _cmx_request *nb)
{
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(bytes > 0);
    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != nb);

    if (CMX_ENABLE_ACC_SELF) {
        /* acc to self */
        if (g_state.rank == proc) {
            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }
            sem_wait(semaphores[proc]);
            _acc(datatype, bytes, dst, src, scale);
            sem_post(semaphores[proc]);
            return;
        }
    }

    if (CMX_ENABLE_ACC_SMP) {
        /* acc to same SMP node */
        // if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) 
        if (g_state.master[proc] == g_state.master[g_state.rank]) 
        {
            reg_entry_t *reg_entry = NULL;
            void *mapped_offset = NULL;

            if (fence_array[g_state.master[proc]]) {
                _fence_master(g_state.master[proc]);
            }

            reg_entry = reg_cache_find(proc, dst, bytes);
            CMX_ASSERT(reg_entry);
            mapped_offset = _get_offset_memory(reg_entry, dst);
            sem_wait(semaphores[proc]);
            _acc(datatype, bytes, mapped_offset, src, scale);
            sem_post(semaphores[proc]);
            return;
        }
    }

    {
        header_t *header = NULL;
        char *message = NULL;
        int master_rank = -1;
        int message_size = 0;
        int scale_size = 0;
        op_t operation = OP_NULL;
        int use_eager = 0;

        switch (datatype) {
            case CMX_ACC_INT:
                operation = OP_ACC_INT;
                scale_size = sizeof(int);
                break;
            case CMX_ACC_DBL:
                operation = OP_ACC_DBL;
                scale_size = sizeof(double);
                break;
            case CMX_ACC_FLT:
                operation = OP_ACC_FLT;
                scale_size = sizeof(float);
                break;
            case CMX_ACC_CPL:
                operation = OP_ACC_CPL;
                scale_size = sizeof(SingleComplex);
                break;
            case CMX_ACC_DCP:
                operation = OP_ACC_DCP;
                scale_size = sizeof(DoubleComplex);
                break;
            case CMX_ACC_LNG:
                operation = OP_ACC_LNG;
                scale_size = sizeof(long);
                break;
            default: CMX_ASSERT(0);
        }
        use_eager = _eager_check(scale_size+bytes);

        master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        if (use_eager) {
            message_size = sizeof(header_t) + scale_size + bytes;
        }
        else {
            message_size = sizeof(header_t) + scale_size;
        }
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = operation;
        header->remote_address = dst;
        header->local_address = src;
        header->rank = proc;
        header->length = bytes;
        (void)memcpy(message+sizeof(header_t), scale, scale_size);
        if (use_eager) {
            (void)memcpy(message+sizeof(header_t)+scale_size,
                    src, bytes);
            nb_send_header(message, message_size, master_rank, nb);
        }
        else {
            char *buf = (char*)src;
            int bytes_remaining = bytes;
            nb_send_header(message, message_size, master_rank, nb);
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                nb_send_buffer(buf, size, master_rank, nb);
                buf += size;
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }
    nb->in_use = 1;
}


STATIC void nb_puts(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb)
{
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];

#if DEBUG
    fprintf(stderr, "[%d] nb_puts(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    /* if not actually a strided put */
    if (0 == stride_levels) {
        nb_put(src, dst, count[0], proc, nb);
        return;
    }

    /* if not a strided put to self or SMP, use datatype algorithm */
    if (CMX_ENABLE_PUT_DATATYPE
            && (!CMX_ENABLE_PUT_SELF || g_state.rank != proc)
            && (!CMX_ENABLE_PUT_SMP
                || g_state.hostid[proc] != g_state.hostid[g_state.rank])
            && (_packed_size(src_stride, count, stride_levels) > CMX_PUT_DATATYPE_THRESHOLD)) {
        nb_puts_datatype(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }

    /* if not a strided put to self or SMP, use packed algorithm */
    if (CMX_ENABLE_PUT_PACKED
            && (!CMX_ENABLE_PUT_SELF || g_state.rank != proc)
            && (!CMX_ENABLE_PUT_SMP
                || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
        nb_puts_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    /* index mangling */
    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += (long) src_bvalue[j] * (long) src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        for(j=1; j<=stride_levels; j++) {
            dst_idx += (long) dst_bvalue[j] * (long) dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        
        nb_put((char *)src + src_idx, (char *)dst + dst_idx,
                count[0], proc, nb);
    }
}


STATIC void nb_puts_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb)
{
    int i;
    int packed_index = 0;
    char *packed_buffer = NULL;
    stride_t stride;

#if DEBUG
    fprintf(stderr, "[%d] nb_puts_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride.stride_levels = stride_levels;
    stride.count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride.stride[i] = dst_stride[i];
        stride.count[i+1] = count[i+1];
    }
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride.stride[i] = -1;
        stride.count[i+1] = -1;
    }

    CMX_ASSERT(stride.stride_levels >= 0);
    CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    fprintf(stderr, "[%d] nb_puts_packed stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride_levels, count[0]);
    for (i=0; i<stride_levels; ++i) {
        printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride.stride[i], i, stride.count[i+1]);
    }
#endif

    packed_buffer = pack(src, src_stride, count, stride_levels, &packed_index);

    CMX_ASSERT(NULL != packed_buffer);
    CMX_ASSERT(packed_index > 0);

    {
        char *message = NULL;
        int message_size = 0;
        header_t *header = NULL;
        int master_rank = -1;
        int use_eager = _eager_check(sizeof(stride_t)+packed_index);

        master_rank = g_state.master[proc];
        /* only fence on the master */
        fence_array[master_rank] = 1;
        if (use_eager) {
            message_size = sizeof(header_t)+sizeof(stride_t)+packed_index;
        }
        else {
            message_size = sizeof(header_t)+sizeof(stride_t);
        }
        message = malloc(message_size);
        header = (header_t*)message;
        header->operation = OP_PUT_PACKED;
        header->remote_address = dst;
        header->local_address = NULL;
        header->rank = proc;
        header->length = packed_index;
        (void)memcpy(message+sizeof(header_t), &stride, sizeof(stride_t));
        if (use_eager) {
            (void)memcpy(message+sizeof(header_t)+sizeof(stride_t),
                    packed_buffer, packed_index);
            nb_send_header(message, message_size, master_rank, nb);
            free(packed_buffer);
        }
        else {
            /* we send the buffer backwards */
            char *buf = packed_buffer + packed_index;;
            int bytes_remaining = packed_index;
            nb_send_header(message, message_size, master_rank, nb);
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                buf -= size;
                if (size == bytes_remaining) {
                    /* on the last send, mark buffer for deletion */
                    nb_send_header(buf, size, master_rank, nb);
                }
                else {
                    nb_send_buffer(buf, size, master_rank, nb);
                }
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }
    nb->in_use = 1;
}


STATIC void nb_puts_datatype(
        void *src_ptr, int *src_stride_ar,
        void *dst_ptr, int *dst_stride_ar,
        int *count, int stride_levels,
        int proc, _cmx_request *nb)
{
    MPI_Datatype src_type;
    int ierr;
    int i;
    stride_t stride;

#if DEBUG
    fprintf(stderr, "[%d] nb_puts_datatype(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src_ptr, src_stride_ar, dst_ptr, dst_stride_ar,
            count[0], stride_levels, proc, nb);
#endif

    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != src_ptr);
    CMX_ASSERT(NULL != dst_ptr);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    MAYBE_MEMSET(&stride, 0, sizeof(stride_t));
    stride.stride_levels = stride_levels;
    stride.count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride.stride[i] = dst_stride_ar[i];
        stride.count[i+1] = count[i+1];
    }
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride.stride[i] = -1;
        stride.count[i+1] = -1;
    }

    CMX_ASSERT(stride.stride_levels >= 0);
    CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    fprintf(stderr, "[%d] nb_puts_datatype stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride_levels, count[0]);
    for (i=0; i<stride_levels; ++i) {
        fprintf(stderr, "[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride.stride[i], i, stride.count[i+1]);
    }
#endif

    strided_to_subarray_dtype(src_stride_ar, count, stride_levels,
            MPI_BYTE, &src_type);
    ierr = MPI_Type_commit(&src_type);
    _translate_mpi_error(ierr,"nb_puts_datatype:MPI_Type_commit");

    {
        char *message = NULL;
        int message_size = 0;
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];
        /* only fence on the master */
        fence_array[master_rank] = 1;
        message_size = sizeof(header_t) + sizeof(stride_t);
        message = malloc(message_size);
        header = (header_t*)message;
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_PUT_DATATYPE;
        header->remote_address = dst_ptr;
        header->local_address = NULL;
        header->rank = proc;
        header->length = 0;
        (void)memcpy(message+sizeof(header_t), &stride, sizeof(stride_t));
        nb_send_header(message, message_size, master_rank, nb);
        nb_send_datatype(src_ptr, src_type, master_rank, nb);
    }
    nb->in_use = 1;
}


STATIC void nb_gets(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb)
{
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];

    /* if not actually a strided get */
    if (0 == stride_levels) {
        nb_get(src, dst, count[0], proc, nb);
        return;
    }

    /* if not a strided get from self or SMP, use datatype algorithm */
    if (CMX_ENABLE_GET_DATATYPE
            && (!CMX_ENABLE_GET_SELF || g_state.rank != proc)
            && (!CMX_ENABLE_GET_SMP
                || g_state.hostid[proc] != g_state.hostid[g_state.rank])
            && (_packed_size(src_stride, count, stride_levels) > CMX_GET_DATATYPE_THRESHOLD)) {
        nb_gets_datatype(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }

    /* if not a strided get from self or SMP, use packed algorithm */
    if (CMX_ENABLE_GET_PACKED
            && (!CMX_ENABLE_GET_SELF || g_state.rank != proc)
            && (!CMX_ENABLE_GET_SMP
                || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
        nb_gets_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += (long) src_bvalue[j] * (long) src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;
        
        for(j=1; j<=stride_levels; j++) {
            dst_idx += (long) dst_bvalue[j] * (long) dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        
        nb_get((char *)src + src_idx, (char *)dst + dst_idx,
                count[0], proc, nb);
    }
}


STATIC void nb_gets_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb)
{
    int i;
    stride_t stride_src;
    stride_t *stride_dst = NULL;

#if DEBUG
    fprintf(stderr, "[%d] nb_gets_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(count[0] > 0);
    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy src info into structure */
    stride_src.ptr = src;
    stride_src.stride_levels = stride_levels;
    stride_src.count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride_src.stride[i] = src_stride[i];
        stride_src.count[i+1] = count[i+1];
    }
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride_src.stride[i] = -1;
        stride_src.count[i+1] = -1;
    }

    CMX_ASSERT(stride_src.stride_levels >= 0);
    CMX_ASSERT(stride_src.stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride_dst = malloc(sizeof(stride_t));
    CMX_ASSERT(stride_dst);
    stride_dst->ptr = dst;
    stride_dst->stride_levels = stride_levels;
    stride_dst->count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride_dst->stride[i] = dst_stride[i];
        stride_dst->count[i+1] = count[i+1];
    }
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride_dst->stride[i] = -1;
        stride_dst->count[i+1] = -1;
    }

    CMX_ASSERT(stride_dst->stride_levels >= 0);
    CMX_ASSERT(stride_dst->stride_levels < CMX_MAX_STRIDE_LEVEL);

    {
        char *message = NULL;
        int message_size = 0;
        int recv_size = 0;
        char *packed_buffer = NULL;
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];

        message_size = sizeof(header_t) + sizeof(stride_t);
        message = malloc(message_size);
        header = (header_t*)message;
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_GET_PACKED;
        header->remote_address = src;
        header->local_address = dst;
        header->rank = proc;
        header->length = 0;

        recv_size = _packed_size(stride_dst->stride,
                stride_dst->count, stride_dst->stride_levels);
        CMX_ASSERT(recv_size > 0);
        packed_buffer = malloc(recv_size);
        CMX_ASSERT(packed_buffer);
        {
            /* prepost all receives backward */
            char *buf = (char*)packed_buffer + recv_size;
            int bytes_remaining = recv_size;
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                buf -= size;
                if (size == bytes_remaining) {
                    /* on the last recv, indicate a packed recv */
                    nb_recv_packed(buf, size, master_rank, nb, stride_dst);
                }
                else {
                    nb_recv(buf, size, master_rank, nb);
                }
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
        (void)memcpy(message+sizeof(header_t), &stride_src, sizeof(stride_t));
        nb_send_header(message, message_size, master_rank, nb);
    }
    nb->in_use = 1;
}


STATIC void nb_gets_datatype(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, _cmx_request *nb)
{
    MPI_Datatype dst_type;
    int i;
    stride_t stride_src;

#if DEBUG
    fprintf(stderr, "[%d] nb_gets_datatype(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif
#if DEBUG
    for (i=0; i<stride_levels; ++i) {
        fprintf(stderr, "\tsrc_stride[%d]=%d\n", i, src_stride[i]);
    }
    for (i=0; i<stride_levels; ++i) {
        fprintf(stderr, "\tdst_stride[%d]=%d\n", i, dst_stride[i]);
    }
    for (i=0; i<stride_levels+1; ++i) {
        fprintf(stderr, "\tcount[%d]=%d\n", i, count[i]);
    }
#endif

    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(count[0] > 0);
    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy src info into structure */
    MAYBE_MEMSET(&stride_src, 0, sizeof(header_t));
    stride_src.ptr = src;
    stride_src.stride_levels = stride_levels;
    stride_src.count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride_src.stride[i] = src_stride[i];
        stride_src.count[i+1] = count[i+1];
    }
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride_src.stride[i] = -1;
        stride_src.count[i+1] = -1;
    }

    CMX_ASSERT(stride_src.stride_levels >= 0);
    CMX_ASSERT(stride_src.stride_levels < CMX_MAX_STRIDE_LEVEL);

    {
        char *message = NULL;
        int message_size = 0;
        header_t *header = NULL;
        int master_rank = -1;
        int ierr;

        master_rank = g_state.master[proc];

        message_size = sizeof(header_t) + sizeof(stride_t);
        message = malloc(message_size);
        header = (header_t*)message;
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_GET_DATATYPE;
        header->remote_address = src;
        header->local_address = dst;
        header->rank = proc;
        header->length = 0;

        strided_to_subarray_dtype(dst_stride, count, stride_levels, MPI_BYTE, &dst_type);
        ierr = MPI_Type_commit(&dst_type);
        _translate_mpi_error(ierr,"nb_gets_datatype:MPI_Type_commit");

        nb_recv_datatype(dst, dst_type, master_rank, nb);
        (void)memcpy(message+sizeof(header_t), &stride_src, sizeof(stride_t));
        nb_send_header(message, message_size, master_rank, nb);
    }
    nb->in_use = 1;
}


STATIC void nb_accs(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, _cmx_request *nb)
{
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];

    /* if not actually a strided acc */
    if (0 == stride_levels) {
        nb_acc(datatype, scale, src, dst, count[0], proc, nb);
        return;
    }

    /* if not a strided acc to self or SMP, use packed algorithm */
    if (CMX_ENABLE_ACC_PACKED
            && (!CMX_ENABLE_ACC_SELF || g_state.rank != proc)
            && (!CMX_ENABLE_ACC_SMP
                || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
        nb_accs_packed(datatype, scale, src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    /* index mangling */
    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += (long) src_bvalue[j] * (long) src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        for(j=1; j<=stride_levels; j++) {
            dst_idx += (long) dst_bvalue[j] * (long) dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        
        nb_acc(datatype, scale, (char *)src + src_idx, (char *)dst + dst_idx,
                count[0], proc, nb);
    }
}


STATIC void nb_accs_packed(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, _cmx_request *nb)
{
    int i;
    int packed_index = 0;
    char *packed_buffer = NULL;
    stride_t stride;

#if DEBUG
    fprintf(stderr, "[%d] nb_accs_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    CMX_ASSERT(proc >= 0);
    CMX_ASSERT(proc < g_state.size);
    CMX_ASSERT(NULL != scale);
    CMX_ASSERT(NULL != src);
    CMX_ASSERT(NULL != dst);
    CMX_ASSERT(NULL != count);
    CMX_ASSERT(NULL != nb);
    CMX_ASSERT(count[0] > 0);
    CMX_ASSERT(stride_levels >= 0);
    CMX_ASSERT(stride_levels < CMX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride.ptr = dst;
    stride.stride_levels = stride_levels;
    stride.count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride.stride[i] = dst_stride[i];
        stride.count[i+1] = count[i+1];
    }
    /* assign remaining values to invalid */
    for (/*no init*/; i<CMX_MAX_STRIDE_LEVEL; ++i) {
        stride.stride[i] = -1;
        stride.count[i+1] = -1;
    }

    CMX_ASSERT(stride.stride_levels >= 0);
    CMX_ASSERT(stride.stride_levels < CMX_MAX_STRIDE_LEVEL);

#if DEBUG
    fprintf(stderr, "[%d] nb_accs_packed stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride_levels, count[0]);
    for (i=0; i<stride_levels; ++i) {
        printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride.stride[i], i, stride.count[i+1]);
    }
#endif

    packed_buffer = pack(src, src_stride, count, stride_levels, &packed_index);

    CMX_ASSERT(NULL != packed_buffer);
    CMX_ASSERT(packed_index > 0);

    {
        header_t *header = NULL;
        char *message = NULL;
        int message_size = 0;
        int scale_size = 0;
        op_t operation = OP_NULL;
        int master_rank = -1;
        int use_eager = 0;

        switch (datatype) {
            case CMX_ACC_INT:
                operation = OP_ACC_INT_PACKED;
                scale_size = sizeof(int);
                break;
            case CMX_ACC_DBL:
                operation = OP_ACC_DBL_PACKED;
                scale_size = sizeof(double);
                break;
            case CMX_ACC_FLT:
                operation = OP_ACC_FLT_PACKED;
                scale_size = sizeof(float);
                break;
            case CMX_ACC_CPL:
                operation = OP_ACC_CPL_PACKED;
                scale_size = sizeof(SingleComplex);
                break;
            case CMX_ACC_DCP:
                operation = OP_ACC_DCP_PACKED;
                scale_size = sizeof(DoubleComplex);
                break;
            case CMX_ACC_LNG:
                operation = OP_ACC_LNG_PACKED;
                scale_size = sizeof(long);
                break;
            default: CMX_ASSERT(0);
        }
        use_eager = _eager_check(scale_size+sizeof(stride_t)+packed_index);

        master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        if (use_eager) {
            message_size = sizeof(header_t) + scale_size + sizeof(stride_t) + packed_index;
        }
        else {
            message_size = sizeof(header_t) + scale_size + sizeof(stride_t);
        }
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = operation;
        header->remote_address = dst;
        header->local_address = NULL;
        header->rank = proc;
        header->length = packed_index;
        (void)memcpy(message+sizeof(header_t), scale, scale_size);
        (void)memcpy(message+sizeof(header_t)+scale_size, &stride, sizeof(stride_t));
        if (use_eager) {
            (void)memcpy(message+sizeof(header_t)+scale_size+sizeof(stride_t),
                    packed_buffer, packed_index);
            nb_send_header(message, message_size, master_rank, nb);
            free(packed_buffer);
        }
        else {
            /* we send the buffer backwards */
            char *buf = packed_buffer + packed_index;
            int bytes_remaining = packed_index;
            nb_send_header(message, message_size, master_rank, nb);
            do {
                int size = bytes_remaining>max_message_size ?
                    max_message_size : bytes_remaining;
                buf -= size;
                if (size == bytes_remaining) {
                    nb_send_header(buf, size, master_rank, nb);
                }
                else {
                    nb_send_buffer(buf, size, master_rank, nb);
                }
                bytes_remaining -= size;
            } while (bytes_remaining > 0);
        }
    }
    nb->in_use = 1;
}


STATIC void nb_putv(
        _cmx_giov_t *iov, int iov_len,
        int proc, _cmx_request *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector put to self, use packed algorithm */
        if (CMX_ENABLE_PUT_IOV
                && (!CMX_ENABLE_PUT_SELF || g_state.rank != proc)
                && (!CMX_ENABLE_PUT_SMP
                    || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
            nb_putv_packed(&iov[i], proc, nb);
        }
        else {
            int j;
            void **src = iov[i].src;
            void **dst = iov[i].dst;
            int bytes = iov[i].bytes;
            int limit = iov[i].count;
            for (j=0; j<limit; ++j) {
                nb_put(src[j], dst[j], bytes, proc, nb);
            }
        }
    }
}


STATIC void nb_putv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb)
{
    int i = 0;
    void **src = NULL;
    void **dst = NULL;
    int bytes = 0;
    int limit = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int iov_size = 0;
    char *packed_buffer = NULL;
    int packed_size = 0;
    int packed_index = 0;

    src = iov->src;
    dst = iov->dst;
    bytes = iov->bytes;
    limit = iov->count;

#if DEBUG
    fprintf(stderr, "[%d] nb_putv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    CMX_ASSERT(iov_buf);
    iov_off = 0;
    /* copy limit */
    (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int));
    iov_off += sizeof(int);
    /* copy bytes */
    (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int));
    iov_off += sizeof(int);
    /* copy src pointers */
    (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    /* copy dst pointers */
    (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    CMX_ASSERT(iov_off == iov_size);

    /* allocate send buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    CMX_ASSERT(packed_buffer);
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
        packed_index += bytes;
    }
    CMX_ASSERT(packed_index == bytes*limit);

    {
        header_t *header = NULL;
        int master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_PUT_IOV;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = proc;
        header->length = iov_size;
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_header(iov_buf, iov_size, master_rank, nb);
        nb_send_header(packed_buffer, packed_size, master_rank, nb);
    }
}


STATIC void nb_getv(
        _cmx_giov_t *iov, int iov_len,
        int proc, _cmx_request *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector get from self, use packed algorithm */
        if (CMX_ENABLE_GET_IOV
                && (!CMX_ENABLE_GET_SELF || g_state.rank != proc)
                && (!CMX_ENABLE_GET_SMP
                    || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
            nb_getv_packed(&iov[i], proc, nb);
        }
        else {
            int j;
            void **src = iov[i].src;
            void **dst = iov[i].dst;
            int bytes = iov[i].bytes;
            int limit = iov[i].count;
            for (j=0; j<limit; ++j) {
                nb_get(src[j], dst[j], bytes, proc, nb);
            }
        }
    }
}


STATIC void nb_getv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb)
{
    void **src = NULL;
    void **dst = NULL;
    int bytes = 0;
    int limit = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int iov_size = 0;
    _cmx_giov_t *iov_copy = NULL;
    char *packed_buffer = NULL;
    int packed_size = 0;

    src = iov->src;
    dst = iov->dst;
    bytes = iov->bytes;
    limit = iov->count;

#if DEBUG
    fprintf(stderr, "[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    iov_off = 0;
    CMX_ASSERT(iov_buf);
    /* copy limit */
    (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int));
    iov_off += sizeof(int);
    /* copy bytes */
    (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int));
    iov_off += sizeof(int);
    /* copy src pointers */
    (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    /* copy dst pointers */
    (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    CMX_ASSERT(iov_off == iov_size);

    /* copy given iov for later */
    iov_copy = malloc(sizeof(_cmx_giov_t));
    iov_copy->bytes = bytes;
    iov_copy->count = limit;
    iov_copy->src = malloc(sizeof(void*)*iov->count);
    CMX_ASSERT(iov_copy->src);
    (void)memcpy(iov_copy->src, iov->src, sizeof(void*)*iov->count);
    iov_copy->dst = malloc(sizeof(void*)*iov->count);
    CMX_ASSERT(iov_copy->dst);
    (void)memcpy(iov_copy->dst, iov->dst, sizeof(void*)*iov->count);

#if DEBUG
    fprintf(stderr, "[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p copy\n",
            g_state.rank, iov_copy->count, iov_copy->bytes,
            iov_copy->src[0], iov_copy->dst[0]);
#endif

    /* allocate recv buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    CMX_ASSERT(packed_buffer);

    {
        header_t *header = NULL;
        int master_rank = g_state.master[proc];

        header = malloc(sizeof(header_t));
        CMX_ASSERT(header);
        MAYBE_MEMSET(header, 0, sizeof(header_t));
        header->operation = OP_GET_IOV;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = proc;
        header->length = iov_size;
        nb_recv_iov(packed_buffer, packed_size, master_rank, nb, iov_copy);
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_header(iov_buf, iov_size, master_rank, nb);
    }
}


STATIC void nb_accv(
        int datatype, void *scale,
        _cmx_giov_t *iov, int iov_len,
        int proc, _cmx_request *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector acc to self, use packed algorithm */
        if (CMX_ENABLE_ACC_IOV
                && (!CMX_ENABLE_ACC_SELF || g_state.rank != proc)
                && (!CMX_ENABLE_ACC_SMP
                    || g_state.hostid[proc] != g_state.hostid[g_state.rank])) {
            nb_accv_packed(datatype, scale, &iov[i], proc, nb);
        }
        else {
            int j;
            void **src = iov[i].src;
            void **dst = iov[i].dst;
            int bytes = iov[i].bytes;
            int limit = iov[i].count;
            for (j=0; j<limit; ++j) {
                nb_acc(datatype, scale, src[j], dst[j], bytes, proc, nb);
            }
        }
    }
}


STATIC void nb_accv_packed(
        int datatype, void *scale,
        _cmx_giov_t *iov,
        int proc, _cmx_request *nb)
{
    int i = 0;
    void **src = NULL;
    void **dst = NULL;
    int bytes = 0;
    int limit = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int iov_size = 0;
    char *packed_buffer = NULL;
    int packed_size = 0;
    int packed_index = 0;

    src = iov->src;
    dst = iov->dst;
    bytes = iov->bytes;
    limit = iov->count;

#if DEBUG
    fprintf(stderr, "[%d] nb_accv_packed limit=%d bytes=%d loc[0]=%p rem_offset[0]=%d\n",
            g_state.rank, limit, bytes, loc[0], rem[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    iov_off = 0;
    CMX_ASSERT(iov_buf);
    /* copy limit */
    (void)memcpy(&iov_buf[iov_off], &limit, sizeof(int));
    iov_off += sizeof(int);
    /* copy bytes */
    (void)memcpy(&iov_buf[iov_off], &bytes, sizeof(int));
    iov_off += sizeof(int);
    /* copy src pointers */
    (void)memcpy(&iov_buf[iov_off], src, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    /* copy dst pointers */
    (void)memcpy(&iov_buf[iov_off], dst, limit*sizeof(void*));
    iov_off += limit*sizeof(void*);
    CMX_ASSERT(iov_off == iov_size);


    /* allocate send buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    CMX_ASSERT(packed_buffer);
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
        packed_index += bytes;
    }
    CMX_ASSERT(packed_index == bytes*limit);


    {
        header_t *header = NULL;
        char *message = NULL;
        int message_size = 0;
        int scale_size = 0;
        op_t operation = OP_NULL;
        int master_rank = g_state.master[proc];

        switch (datatype) {
            case CMX_ACC_INT:
                operation = OP_ACC_INT_IOV;
                scale_size = sizeof(int);
                break;
            case CMX_ACC_DBL:
                operation = OP_ACC_DBL_IOV;
                scale_size = sizeof(double);
                break;
            case CMX_ACC_FLT:
                operation = OP_ACC_FLT_IOV;
                scale_size = sizeof(float);
                break;
            case CMX_ACC_CPL:
                operation = OP_ACC_CPL_IOV;
                scale_size = sizeof(SingleComplex);
                break;
            case CMX_ACC_DCP:
                operation = OP_ACC_DCP_IOV;
                scale_size = sizeof(DoubleComplex);
                break;
            case CMX_ACC_LNG:
                operation = OP_ACC_LNG_IOV;
                scale_size = sizeof(long);
                break;
            default: CMX_ASSERT(0);
        }

        /* only fence on the master */
        fence_array[master_rank] = 1;

        message_size = sizeof(header_t) + scale_size;
        message = malloc(message_size);
        CMX_ASSERT(message);
        header = (header_t*)message;
        header->operation = operation;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = proc;
        header->length = iov_size;
        (void)memcpy(message+sizeof(header_t), scale, scale_size);
        nb_send_header(message, message_size, master_rank, nb);
        nb_send_header(iov_buf, iov_size, master_rank, nb);
        nb_send_header(packed_buffer, packed_size, master_rank, nb);
    }
}

/**
 * Utility function to catch and translate MPI errors. Returns silently if
 * no error detected.
 * @param ierr: Error code from MPI call
 * @param location: User specified string to indicate location of error
 */
STATIC void _translate_mpi_error(int ierr, const char* location)
{
  if (ierr == MPI_SUCCESS) return;
  char err_string[MPI_MAX_ERROR_STRING];
  int len;
  fprintf(stderr,"p[%d] Error in %s\n",g_state.rank,location);
  MPI_Error_string(ierr,err_string,&len);
  fprintf(stderr,"p[%d] MPI_Error: %s\n",g_state.rank,err_string);
}


/**
 * No checking for data consistency. Assume correctness has already been
 * established elsewhere. Individual elements are assumed to be one byte in size
 * stride_array: physical dimensions of array
 * count: number of elements along each array dimension
 * levels: number of stride levels (should be one less than array dimension)
 * type: MPI_Datatype returned to calling program
 */
STATIC void strided_to_subarray_dtype(int *stride_array, int *count, int levels, MPI_Datatype base_type, MPI_Datatype *type)
{
    int ndims = levels+1;
    int i = 0;
    int ierr = 0;
    int array_of_sizes[7];
    int array_of_starts[7];
    int array_of_subsizes[7];
    int stride = 0;

    ierr = MPI_Type_size(base_type,&stride);
    _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_size");

    /* the pointer to the local buffer points to the first data element
     * in data exchange, not the origin of the local array, so all starts
     * should be zero */
    for (i=0; i<levels; i++) {
        array_of_sizes[i] = stride_array[i]/stride;
        array_of_starts[i] = 0;
        array_of_subsizes[i] = count[i];
        if (array_of_sizes[i] < array_of_subsizes[i]) {
            fprintf(stderr, "p[%d] ERROR [strided_to_subarray_dtype]\n"
                    "stride: %d\n"
                    "stride_array[%d]: %d\n"
                    "array_of_sizes[%d]: %d\n"
                    "array_of_subsizes[%d]: %d\n",
                    g_state.rank,
                    stride,
                    i,stride_array[i],
                    i,array_of_sizes[i],
                    i,array_of_subsizes[i]);
        }
        stride = stride_array[i];
    }
    array_of_sizes[levels] = count[levels];
    array_of_starts[levels] = 0;
    array_of_subsizes[levels] = count[levels];
#if DEBUG
    for (i=0; i<ndims; i++) {
        fprintf(stderr, "p[%d] ndims: %d sizes[%d]: %d subsizes[%d]: %d starts[%d]: %d\n",
                g_state.rank,
                ndims,
                i,array_of_sizes[i],
                i,array_of_subsizes[i],
                i,array_of_starts[i]);
    }
#endif

    ierr = MPI_Type_create_subarray(ndims, array_of_sizes,
            array_of_subsizes, array_of_starts, MPI_ORDER_FORTRAN,
            base_type, type);
    _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_create_subarray");
    if (MPI_SUCCESS != ierr) {
        fprintf(stderr, "p[%d] Error forming MPI_Datatype for one-sided strided operation."
                " Check that stride dimensions are compatible with local block"
                " dimensions\n",g_state.rank);
        for (i=0; i<levels; i++) {
            fprintf(stderr, "p[%d] count[%d]: %d stride[%d]: %d\n",
                    g_state.rank,
                    i,count[i],
                    i,stride_array[i]);
        }
        fprintf(stderr, "p[%d] count[%d]: %d\n",g_state.rank,i,count[i]);
        _translate_mpi_error(ierr,"strided_to_subarray_dtype:MPI_Type_create_subarray");
    }
}
