#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* C and/or system headers */
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <assert.h>
#include <signal.h>

/* 3rd party headers */
#include <mpi.h>

/* our headers */
#include "comex.h"
#include "comex_impl.h"
#include "groups.h"
#include "reg_cache.h"
#include "acc.h"

#define PAUSE_ON_ERROR 1
#define STATIC static inline

/* data structures */

typedef enum {
    OP_PUT = 0,
    OP_PUT_PACKED,
    OP_PUT_IOV,
    OP_GET,
    OP_GET_PACKED,
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
    OP_FREE
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
} rank_ptr_t;


/* static state */
static int *num_mutexes = NULL;     /**< (all) how many mutexes on each process */
static int **mutexes = NULL;        /**< (masters) value is rank of lock holder */
static lock_t ***lq_heads = NULL;   /**< array of lock queues */
static char *sem_name = NULL;       /* local semaphore name */
static sem_t **semaphores = NULL;   /* semaphores for locking within SMP node */
static int initialized = 0;         /* for comex_initialized(), 0=false */
static char *fence_array = NULL;
static pthread_t progress_thread;

static nb_t *nb_state = NULL;       /* keep track of all nonblocking operations */
static int nb_max_outstanding = COMEX_MAX_NB_OUTSTANDING;
static int nb_index = 0;
static int nb_count_event = 0;
static int nb_count_event_processed = 0;
static int nb_count_send = 0;
static int nb_count_send_processed = 0;
static int nb_count_recv = 0;
static int nb_count_recv_processed = 0;

static char *static_acc_buffer = NULL;

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
        comex_error("gethostname failed", errno);
    }
    fprintf(stderr,"%d(%s:%d): Segmentation Violation ... pausing\n",
            g_state.rank, name, getpid());
    pause();

    comex_error("Segmentation Violation error, status=",(int) sig);
}
#endif

/* static function declarations */

/* error checking */
#define CHECK_MPI_RETVAL(retval) check_mpi_retval((retval), __FILE__, __LINE__)
STATIC void check_mpi_retval(int retval, const char *file, int line);
STATIC const char *str_mpi_retval(int retval);

/* server fuctions */
STATIC void server_send(void *buf, int count, int dest);
STATIC void server_recv(void *buf, int count, int source);
STATIC void* _progress_server(void *arg);
STATIC void _put_handler(header_t *header, int proc);
STATIC void _put_packed_handler(header_t *header, int proc);
STATIC void _put_iov_handler(header_t *header, int proc);
STATIC void _get_handler(header_t *header, int proc);
STATIC void _get_packed_handler(header_t *header, int proc);
STATIC void _get_iov_handler(header_t *header, int proc);
STATIC void _acc_handler(header_t *header, char *scale, int proc);
STATIC void _acc_packed_handler(header_t *header, int proc);
STATIC void _acc_iov_handler(header_t *header, int proc);
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
STATIC void nb_send_common(void *buf, int count, int dest, nb_t *nb, int need_free);
STATIC void nb_send_header(void *buf, int count, int dest, nb_t *nb);
STATIC void nb_send_buffer(void *buf, int count, int dest, nb_t *nb);
STATIC void nb_recv_packed(void *buf, int count, int source, nb_t *nb, stride_t *stride);
STATIC void nb_recv_iov(void *buf, int count, int source, nb_t *nb, comex_giov_t *iov);
STATIC void nb_recv(void *buf, int count, int source, nb_t *nb);
STATIC int nb_get_handle_index();
STATIC nb_t* nb_wait_for_handle();
STATIC void nb_wait_for_send(nb_t *nb);
STATIC void nb_wait_for_send1(nb_t *nb);
STATIC void nb_wait_for_recv(nb_t *nb);
STATIC void nb_wait_for_recv1(nb_t *nb);
STATIC void nb_wait_for_all(nb_t *nb);
STATIC void nb_wait_all();
STATIC void nb_put(void *src, void *dst, int bytes, int proc, nb_t *nb);
STATIC void nb_get(void *src, void *dst, int bytes, int proc, nb_t *nb);
STATIC void nb_acc(int datatype, void *scale, void *src, void *dst, int bytes, int proc, nb_t *nb);
STATIC void nb_puts(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_puts_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_gets(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_gets_packed(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_accs(
        int datatype, void *scale,
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_accs_packed(
        int datatype, void *scale,
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb);
STATIC void nb_putv(comex_giov_t *iov, int iov_len, int proc, nb_t *nb);
STATIC void nb_putv_packed(comex_giov_t *iov, int proc, nb_t *nb);
STATIC void nb_getv(comex_giov_t *iov, int iov_len, int proc, nb_t *nb);
STATIC void nb_getv_packed(comex_giov_t *iov, int proc, nb_t *nb);
STATIC void nb_accv(int datatype, void *scale,
        comex_giov_t *iov, int iov_len, int proc, nb_t *nb);
STATIC void nb_accv_packed(int datatype, void *scale,
        comex_giov_t *iov, int proc, nb_t *nb);

/* other functions */
STATIC int packed_size(int *src_stride, int *count, int stride_levels);
STATIC char* pack(char *src, int *src_stride,
                int *count, int stride_levels, int *size);
STATIC void unpack(char *packed_buffer,
                char *dst, int *dst_stride, int *count, int stride_levels);
STATIC char* _generate_shm_name(int rank);
STATIC reg_entry_t* _comex_malloc_local(size_t size);
STATIC void* _get_offset_memory(reg_entry_t *reg_entry, void *memory);
STATIC int _is_master(void);
STATIC int _get_world_rank(comex_igroup_t *igroup, int rank);
STATIC int* _get_world_ranks(comex_igroup_t *igroup);
STATIC int _smallest_world_rank_with_same_hostid(comex_igroup_t *group);
STATIC int _largest_world_rank_with_same_hostid(comex_igroup_t *igroup);
STATIC void _malloc_semaphore(void);
STATIC void _free_semaphore(void);
STATIC void* _shm_create(const char *name, size_t size);
STATIC void* _shm_attach(const char *name, size_t size);
STATIC void* _shm_map(int fd, size_t size);
STATIC int _set_affinity(int cpu);


int comex_init()
{
    int status = 0;
    int init_flag = 0;
    int i = 0;
    int mpi_thread_level_provided;
    
    if (initialized) {
        return 0;
    }
    initialized = 1;

    /* Assert MPI has been initialized */
    status = MPI_Initialized(&init_flag);
    CHECK_MPI_RETVAL(status);
    assert(init_flag);
    
    /* Assert MPI has been initialized with proper threading level */
    status = MPI_Query_thread(&mpi_thread_level_provided);
    CHECK_MPI_RETVAL(status);
    COMEX_ASSERT(MPI_THREAD_MULTIPLE == mpi_thread_level_provided);

    /* env vars */
    {
        char *value = NULL;
        nb_max_outstanding = COMEX_MAX_NB_OUTSTANDING; /* default */
        if ((value = getenv("COMEX_MAX_NB_OUTSTANDING")) != NULL) {
            nb_max_outstanding = atoi(value);
        }
        COMEX_ASSERT(nb_max_outstanding > 0);
    }

    /* groups */
    comex_group_init();

    /* mutexes */
    mutexes = NULL;
    num_mutexes = NULL;
    lq_heads = NULL;

    /* nonblocking message queues */
    nb_state = (nb_t*)malloc(sizeof(nb_t) * nb_max_outstanding);
    COMEX_ASSERT(nb_state);
    for (i = 0; i < nb_max_outstanding; ++i) {
        nb_state[i].in_use = 0;
        nb_state[i].send_size = 0;
        nb_state[i].send_head = NULL;
        nb_state[i].send_tail = NULL;
        nb_state[i].recv_size = 0;
        nb_state[i].recv_head = NULL;
        nb_state[i].recv_tail = NULL;
    }
    nb_index = 0;
    nb_count_event = 0;
    nb_count_event_processed = 0;
    nb_count_send = 0;
    nb_count_send_processed = 0;
    nb_count_recv = 0;
    nb_count_recv_processed = 0;

    /* reg_cache */
    /* note: every process needs a reg cache and it's always based on the
     * world rank and size */
    reg_cache_init(g_state.size);

    _malloc_semaphore();

#if DEBUG
    printf("[%d] comex_init() before progress server\n", g_state.rank);
#endif

#if PAUSE_ON_ERROR
    if ((SigSegvOrig=signal(SIGSEGV, SigSegvHandler)) == SIG_ERR) {
        comex_error("signal(SIGSEGV, ...) error", -1);
    }
#endif

    /* Synch - Sanity Check */
    /* This barrier is on the world worker group */
    MPI_Barrier(group_list->comm);

    status = _set_affinity(g_state.node_rank);
    COMEX_ASSERT(0 == status);

    if (_is_master()) {
        /* TODO: wasteful O(p) storage... */
        mutexes = (int**)malloc(sizeof(int*) * g_state.size);
        COMEX_ASSERT(mutexes);
        /* create one lock queue for each proc for each mutex */
        lq_heads = (lock_t***)malloc(sizeof(lock_t**) * g_state.size);
        COMEX_ASSERT(lq_heads);
        /* start the server */
        pthread_create(&progress_thread, NULL, _progress_server, NULL);
    }

    /* static state */
    fence_array = malloc(sizeof(char) * g_state.size);
    COMEX_ASSERT(fence_array);
    for (i = 0; i < g_state.size; ++i) {
        fence_array[i] = 0;
    }

#if DEBUG
    printf("[%d] comex_init() before barrier\n", g_state.rank);
#endif

    /* Synch - Sanity Check */
    /* This barrier is on the world worker group */
    MPI_Barrier(group_list->comm);

#if DEBUG
    printf("[%d] comex_init() success\n", g_state.rank);
#endif

    return COMEX_SUCCESS;
}


int comex_init_args(int *argc, char ***argv)
{
    int init_flag;
    int status;
    
    status = MPI_Initialized(&init_flag);
    CHECK_MPI_RETVAL(status);
    
    if(!init_flag) {
        int level;
        status = MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &level);
        CHECK_MPI_RETVAL(status);
        COMEX_ASSERT(MPI_THREAD_MULTIPLE == level);
    }
    
    return comex_init();
}


int comex_initialized()
{
#if DEBUG
    if (initialized) {
        printf("[%d] comex_initialized()\n", g_state.rank);
    }
#endif

    return initialized;
}


int comex_finalize()
{
#if DEBUG
    printf("[%d] comex_finalize()\n", g_state.rank);
#endif

    /* it's okay to call multiple times -- extra calls are no-ops */
    if (!initialized) {
        return COMEX_SUCCESS;
    }

    comex_barrier(COMEX_GROUP_WORLD);

    initialized = 0;

    _free_semaphore();

    /* Make sure that all outstanding operations are done */
    comex_wait_all(COMEX_GROUP_WORLD);
    
    comex_barrier(COMEX_GROUP_WORLD);

    /* send quit message to thread */
    if (_is_master()) {
        int my_master = -1;
        header_t *header = NULL;
        nb_t *nb = NULL;

        nb = nb_wait_for_handle();
        my_master = g_state.master[g_state.rank];
        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = OP_QUIT;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = 0;
        nb_send_header(header, sizeof(header_t), my_master, nb);
        nb_wait_for_all(nb);
        pthread_join(progress_thread, NULL);
    }

    free(fence_array);

    free(nb_state);
#if DEBUG
    printf(" %d freed nb_state ptr %p \n", g_state.rank, nb_state);
#endif

    MPI_Barrier(g_state.comm);

    /* reg_cache */
    reg_cache_destroy(g_state.size);

    /* destroy the groups */
#if DEBUG
    printf("[%d] before comex_group_finalize()\n", g_state.rank);
#endif
    comex_group_finalize();
#if DEBUG
    printf("[%d] after comex_group_finalize()\n", g_state.rank);
#endif

#if DEBUG_TO_FILE
    fclose(comex_trace_file);
#endif

    return COMEX_SUCCESS;
}


void comex_error(char *msg, int code)
{
#if DEBUG
    printf("[%d] Received an Error in Communication: (%d) %s\n",
            g_state.rank, code, msg);
#if DEBUG_TO_FILE
    fclose(comex_trace_file);
#endif
#endif
    fprintf(stderr,"[%d] Received an Error in Communication: (%d) %s\n",
            g_state.rank, code, msg);
    
    MPI_Abort(g_state.comm, code);
}


int comex_put(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_put(src, dst, bytes, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_get(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_get(src, dst, bytes, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_acc(
        int datatype, void *scale,
        void *src, void *dst, int bytes,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_acc(datatype, scale, src, dst, bytes, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_puts(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_puts(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_gets(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_gets(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_accs(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_accs(datatype, scale,
            src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_putv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_putv(iov, iov_len, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_getv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_getv(iov, iov_len, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_accv(
        int datatype, void *scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

    nb = nb_wait_for_handle();

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_accv(datatype, scale, iov, iov_len, world_proc, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_fence_all(comex_group_t group)
{
    int p = 0;
    int count_before = 0;
    int count_after = 0;
    nb_t *nb = NULL;

#if DEBUG
    printf("[%d] comex_fence_all(group=%d)\n", g_state.rank, group);
#endif
    /* NOTE: We always fence on the world group */

    /* count how many fence messagse to send */
    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            ++count_before;
        }
    }

#if DEBUG
    printf("[%d] comex_fence_all(group=%d) count_before=%d\n",
            g_state.rank, group, count_before);
#endif

    /* check for no outstanding put/get requests */
    if (0 == count_before) {
        return COMEX_SUCCESS;
    }

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
    printf("[%d] comex_fence_all asm volatile (\"\" : : : \"memory\"); \n",
            g_state.rank, group);
#endif
    asm volatile ("" : : : "memory"); 
#endif

    /* optimize by only sending to procs which we have outstanding messages */
    nb = nb_wait_for_handle();
    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            int p_master = g_state.master[p];
            header_t *header = NULL;

            /* because we only fence to masters */
            COMEX_ASSERT(p_master == p);

            /* prepost recv for acknowledgment */
            nb_recv(NULL, 0, p_master, nb);

            /* post send of fence request */
            header = malloc(sizeof(header_t));
            COMEX_ASSERT(header);
            header->operation = OP_FENCE;
            header->remote_address = NULL;
            header->local_address = NULL;
            header->length = 0;
            header->rank = 0;
            nb_send_header(header, sizeof(header_t), p_master, nb);
        }
    }

    nb_wait_for_all(nb);

    for (p=0; p<g_state.size; ++p) {
        if (fence_array[p]) {
            fence_array[p] = 0;
            ++count_after;
        }
    }

#if DEBUG
    printf("[%d] comex_fence_all(group=%d) count_after=%d\n",
            g_state.rank, group, count_after);
#endif

    COMEX_ASSERT(count_before == count_after);

    return COMEX_SUCCESS;
}


int comex_fence_proc(int proc, comex_group_t group)
{
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;

#if DEBUG
    printf("[%d] comex_fence_proc(proc=%d, group=%d)\n",
            g_state.rank, proc, group);
#endif

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    /* optimize by only sending to procs which we have outstanding messages */
    if (fence_array[world_proc]) {
        int p_master = g_state.master[world_proc];
        header_t *header = NULL;
        nb_t *nb = NULL;

        nb = nb_wait_for_handle();

        /* because we only fence to masters */
        COMEX_ASSERT(p_master == world_proc);

        /* prepost recv for acknowledgment */
        nb_recv(NULL, 0, p_master, nb);

        /* post send of fence request */
        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = OP_FENCE;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->length = 0;
        header->rank = 0;
        nb_send_header(header, sizeof(header_t), p_master, nb);
        nb_wait_for_all(nb);
        fence_array[world_proc] = 0;
    }

    return COMEX_SUCCESS;
}


/* comex_barrier is comex_fence_all + MPI_Barrier */
int comex_barrier(comex_group_t group)
{
    int status = 0;
    MPI_Comm comm = MPI_COMM_NULL;

#if DEBUG
    static int count=-1;
    ++count;
    printf("[%d] comex_barrier(%d) count=%d\n", g_state.rank, group, count);
#endif

    comex_fence_all(group);
    status = comex_group_comm(group, &comm);
    COMEX_ASSERT(COMEX_SUCCESS == status);
    MPI_Barrier(comm);

    return COMEX_SUCCESS;
}


STATIC int packed_size(int *src_stride, int *count, int stride_levels)
{
    int size;
    int i;
    int n1dim;  /* number of 1 dim block */

    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);
    COMEX_ASSERT(NULL != src_stride);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(count[0] > 0);

#if DEBUG
    printf("[%d] packed_size(src_stride=%p, count[0]=%d, stride_levels=%d)\n",
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
    stride_t stride;

    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != src_stride);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(count[0] > 0);
    COMEX_ASSERT(NULL != size);

#if DEBUG
    printf("[%d] pack(src=%p, src_stride=%p, count[0]=%d, stride_levels=%d)\n",
            g_state.rank, src, src_stride, count[0], stride_levels);
#endif

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    /* allocate packed buffer now that we know the size */
    packed_buffer = malloc(n1dim * count[0]);
    COMEX_ASSERT(packed_buffer);

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

    COMEX_ASSERT(packed_index == n1dim*count[0]);
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
    stride_t stride;

    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);
    COMEX_ASSERT(NULL != packed_buffer);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(NULL != dst_stride);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(count[0] > 0);

#if DEBUG
    printf("[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
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

    COMEX_ASSERT(packed_index == n1dim*count[0]);
}


STATIC char* _generate_shm_name(int rank)
{
    int snprintf_retval = 0;
    /* /cmxPPPPPPPPCCCCCCC  */
    /* 00000000011111111112 */
    /* 12345678901234567890 */
    char *name = NULL;
    static const unsigned int limit = 62;
    static const char letters[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static unsigned int counter[7] = {0};
    unsigned int urank = rank;

    COMEX_ASSERT(rank >= 0);
    name = malloc(SHM_NAME_SIZE*sizeof(char));
    COMEX_ASSERT(name);
    snprintf_retval = snprintf(name, SHM_NAME_SIZE,
            "/cmx%08u%c%c%c%c%c%c%c", urank,
            letters[counter[6]],
            letters[counter[5]],
            letters[counter[4]],
            letters[counter[3]],
            letters[counter[2]],
            letters[counter[1]],
            letters[counter[0]]);
    COMEX_ASSERT(snprintf_retval < (int)SHM_NAME_SIZE);
    name[SHM_NAME_SIZE-1] = '\0';
    ++counter[0];
    if (counter[0] >= limit) { ++counter[1]; counter[0] = 0; }
    if (counter[1] >= limit) { ++counter[2]; counter[1] = 0; }
    if (counter[2] >= limit) { ++counter[3]; counter[2] = 0; }
    if (counter[3] >= limit) { ++counter[4]; counter[3] = 0; }
    if (counter[4] >= limit) { ++counter[5]; counter[4] = 0; }
    if (counter[5] >= limit) { ++counter[6]; counter[5] = 0; }
    if (counter[6] >= limit) {
        comex_error("_generate_shm_name: too many names generated", -1);
    }
#if DEBUG
    printf("[%d] _generate_shm_name(%d)=%s\n",
            g_state.rank, rank, name);
#endif

    return name;
}


void* comex_malloc_local(size_t size)
{
    reg_entry_t *reg_entry;
    void *memory = NULL;

    if (size > 0) {
        reg_entry = _comex_malloc_local(size);
        memory = reg_entry->mapped;
    }
    else {
        memory = NULL;
    }

    return memory;
}


STATIC reg_entry_t* _comex_malloc_local(size_t size)
{
    char *name = NULL;
    void *memory = NULL;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    printf("[%d] _comex_malloc_local(size=%lu)\n",
            g_state.rank, (long unsigned)size);
#endif

    if (0 == size) {
        return NULL;
    }

    /* create my shared memory object */
    name = _generate_shm_name(g_state.rank);
    memory = _shm_create(name, size);
#if DEBUG && DEBUG_VERBOSE
    printf("[%d] _comex_malloc_local registering "
            "rank=%d mem=%p size=%lu name=%s mapped=%p\n",
            g_state.rank, g_state.rank, memory,
            (long unsigned)size, name, memory);
#endif

    /* register the memory locally */
    reg_entry = reg_cache_insert(
            g_state.rank, memory, size, name, memory);
    if (NULL == reg_entry) {
        comex_error("_comex_malloc_local: reg_cache_insert", -1);
    }

    free(name);

    return reg_entry;
}


int comex_free_local(void *ptr)
{
    int retval = 0;
    reg_entry_t *reg_entry = NULL;

#if DEBUG
    printf("[%d] comex_free_local(ptr=%p)\n", g_state.rank, ptr);
#endif

    if (NULL == ptr) {
        return COMEX_SUCCESS;
    }

    /* find the registered memory */
    reg_entry = reg_cache_find(g_state.rank, ptr, 0);

    /* unmap the memory */
    retval = munmap(ptr, reg_entry->len);
    if (-1 == retval) {
        perror("comex_free_local: munmap");
        comex_error("comex_free_local: munmap", retval);
    }

    /* remove the shared memory object */
    retval = shm_unlink(reg_entry->name);
    if (-1 == retval) {
        perror("comex_free_local: shm_unlink");
        comex_error("comex_free_local: shm_unlink", retval);
    }

    /* delete the reg_cache entry */
    retval = reg_cache_delete(g_state.rank, ptr);
    COMEX_ASSERT(RR_SUCCESS == retval);

    return COMEX_SUCCESS;
}


int comex_wait_proc(int proc, comex_group_t group)
{
    return comex_wait_all(group);
}


int comex_wait(comex_request_t* hdl)
{
    int index = 0;
    nb_t *nb = NULL;

    COMEX_ASSERT(NULL != hdl);

    index = *(int*)hdl;
    COMEX_ASSERT(index >= 0);
    COMEX_ASSERT(index < nb_max_outstanding);
    nb = &nb_state[index];

    if (0 == nb->in_use) {
        fprintf(stderr, "{%d} comex_wait Error: invalid handle\n",
                g_state.rank);
    }

    nb_wait_for_all(nb);

    nb->in_use = 0;

    return COMEX_SUCCESS;
}


int comex_test(comex_request_t* hdl, int *status)
{
    int index = 0;
    nb_t *nb = NULL;

    COMEX_ASSERT(NULL != hdl);

    index = *(int*)hdl;
    COMEX_ASSERT(index >= 0);
    COMEX_ASSERT(index < nb_max_outstanding);
    nb = &nb_state[index];

    if (0 == nb->in_use) {
        fprintf(stderr, "{%d} comex_test Error: invalid handle\n",
                g_state.rank);
    }

    if (NULL == nb->send_head && NULL == nb->recv_head) {
        COMEX_ASSERT(0 == nb->send_size);
        COMEX_ASSERT(0 == nb->recv_size);
        *status = 0;
        nb->in_use = 0;
    }
    else {
        *status = 1;
    }

    return COMEX_SUCCESS;
}


int comex_wait_all(comex_group_t group)
{
    nb_wait_all();

    return COMEX_SUCCESS;
}


int comex_nbput(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_put(src, dst, bytes, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbget(
        void *src, void *dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_get(src, dst, bytes, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbacc(
        int datatype, void *scale,
        void *src, void *dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_acc(datatype, scale, src, dst, bytes, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbputs(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels, 
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_puts(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbgets(
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels, 
        int proc, comex_group_t group,
        comex_request_t *hdl) 
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_gets(src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbaccs(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_accs(datatype, scale,
            src, src_stride, dst, dst_stride, count, stride_levels, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbputv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_putv(iov, iov_len, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbgetv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_getv(iov, iov_len, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_nbaccv(
        int datatype, void *scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* hdl)
{
    nb_t *nb = NULL;
    int world_proc = -1;
    comex_igroup_t *igroup = NULL;
    comex_request_t _hdl = 0;

    nb = nb_wait_for_handle();
    _hdl = nb_get_handle_index();
    COMEX_ASSERT(&nb_state[_hdl] == nb);
    if (hdl) {
        *hdl = _hdl;
        nb->in_use = 1;
    }

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_proc = _get_world_rank(igroup, proc);

    nb_accv(datatype, scale, iov, iov_len, world_proc, nb);

    return COMEX_SUCCESS;
}


int comex_rmw(
        int comex_op, void *ploc, void *prem, int extra,
        int proc, comex_group_t group)
{
    header_t *header = NULL;
    char *message = NULL;
    int payload_int = 0;
    long payload_long = 0;
    int length = 0;
    int op = 0;
    long extra_long = (long)extra;
    int world_rank = 0;
    int master_rank = 0;
    comex_igroup_t *igroup = NULL;
    nb_t *nb = NULL;

#if DEBUG
    printf("[%d] comex_rmw(%d, %p, %p, %d, %d)\n",
            g_state.rank, comex_op, ploc, prem, extra, proc);
#endif

    CHECK_GROUP(group,proc);
    igroup = comex_get_igroup_from_group(group);
    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    switch (comex_op) {
        case COMEX_FETCH_AND_ADD:
            op = OP_FETCH_AND_ADD;
            length = sizeof(int);
            payload_int = extra;
            break;
        case COMEX_FETCH_AND_ADD_LONG:
            op = OP_FETCH_AND_ADD;
            length = sizeof(long);
            payload_long = extra_long;
            break;
        case COMEX_SWAP:
            op = OP_SWAP;
            length = sizeof(int);
            payload_int = *((int*)ploc);
            break;
        case COMEX_SWAP_LONG:
            op = OP_SWAP;
            length = sizeof(long);
            payload_long = *((long*)ploc);
            break;
        default: COMEX_ASSERT(0);
    }

    /* create and prepare the header */
    message = malloc(sizeof(header_t) + length);
    COMEX_ASSERT(message);
    header = (header_t*)message;
    header->operation = op;
    header->remote_address = prem;
    header->local_address = ploc;
    header->rank = world_rank;
    header->length = length;
    switch (comex_op) {
        case COMEX_FETCH_AND_ADD:
        case COMEX_SWAP:
            (void)memcpy(message+sizeof(header_t), &payload_int, length);
            break;
        case COMEX_FETCH_AND_ADD_LONG:
        case COMEX_SWAP_LONG:
            (void)memcpy(message+sizeof(header_t), &payload_long, length);
            break;
        default: COMEX_ASSERT(0);
    }

    nb = nb_wait_for_handle();
    nb_recv(ploc, length, master_rank, nb); /* prepost recv */
    nb_send_header(message, sizeof(header_t)+length, master_rank, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


/* Mutex Operations */
int comex_create_mutexes(int num)
{
    /* always on the world group */
    int my_master = g_state.master[g_state.rank];

#if DEBUG
    printf("[%d] comex_create_mutexes(num=%d)\n",
            g_state.rank, num);
#endif

    /* preconditions */
    COMEX_ASSERT(0 <= num);
    COMEX_ASSERT(NULL == num_mutexes);

    num_mutexes = (int*)malloc(group_list->size * sizeof(int));
    /* exchange of mutex counts */
    num_mutexes[group_list->rank] = num;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
            num_mutexes, 1, MPI_INT, group_list->comm);

    /* every process sends their own create message to their master */
    {
        nb_t *nb = NULL;
        header_t *header = NULL;

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = OP_CREATE_MUTEXES;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = num;
        nb = nb_wait_for_handle();
        nb_recv(NULL, 0, my_master, nb); /* prepost ack */
        nb_send_header(header, sizeof(header_t), my_master, nb);
        nb_wait_for_all(nb);
    }

    MPI_Barrier(group_list->comm);

    return COMEX_SUCCESS;
}


int comex_destroy_mutexes()
{
    /* always on the world group */
    int my_master = g_state.master[g_state.rank];

#if DEBUG
    printf("[%d] comex_destroy_mutexes()\n", g_state.rank);
#endif

    /* preconditions */
    COMEX_ASSERT(num_mutexes);

    /* this call is collective on the world group and this barrier ensures
     * there are no outstanding lock requests */
    comex_barrier(COMEX_GROUP_WORLD);

    /* let masters know they need to participate */
    /* first non-master rank in an SMP node sends the message to master */
    if (_smallest_world_rank_with_same_hostid(group_list) == g_state.rank) {
        nb_t *nb = NULL;
        header_t *header = NULL;

        header = malloc(sizeof(header_t));
        header->operation = OP_DESTROY_MUTEXES;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = num_mutexes[g_state.rank];
        nb = nb_wait_for_handle();
        nb_recv(NULL, 0, my_master, nb); /* prepost ack */
        nb_send_header(header, sizeof(header_t), my_master, nb);
        nb_wait_for_all(nb);
    }

    free(num_mutexes);
    num_mutexes = NULL;

    return COMEX_SUCCESS;
}


int comex_lock(int mutex, int proc)
{
    header_t *header = NULL;
    int world_rank = 0;
    int master_rank = 0;
    int ack = 0;
    comex_igroup_t *igroup = NULL;
    nb_t *nb = NULL;

#if DEBUG
    printf("[%d] comex_lock mutex=%d proc=%d\n",
            g_state.rank, mutex, proc);
#endif

    CHECK_GROUP(COMEX_GROUP_WORLD,proc);
    igroup = comex_get_igroup_from_group(COMEX_GROUP_WORLD);
    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    header = malloc(sizeof(header_t));
    COMEX_ASSERT(header);
    header->operation = OP_LOCK;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = world_rank;
    header->length = mutex;

    nb = nb_wait_for_handle();
    nb_recv(&ack, sizeof(int), master_rank, nb); /* prepost ack */
    nb_send_header(header, sizeof(header_t), master_rank, nb);
    nb_wait_for_all(nb);
    COMEX_ASSERT(mutex == ack);

    return COMEX_SUCCESS;
}


int comex_unlock(int mutex, int proc)
{
    header_t *header = NULL;
    int world_rank = 0;
    int master_rank = 0;
    comex_igroup_t *igroup = NULL;
    nb_t *nb = NULL;

#if DEBUG
    printf("[%d] comex_unlock mutex=%d proc=%d\n", g_state.rank, mutex, proc);
#endif

    CHECK_GROUP(COMEX_GROUP_WORLD,proc);
    igroup = comex_get_igroup_from_group(COMEX_GROUP_WORLD);
    world_rank = _get_world_rank(igroup, proc);
    master_rank = g_state.master[world_rank];

    fence_array[master_rank] = 1;
    header = malloc(sizeof(header_t));
    COMEX_ASSERT(header);
    header->operation = OP_UNLOCK;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = world_rank;
    header->length = mutex;

    nb = nb_wait_for_handle();
    nb_send_header(header, sizeof(header_t), master_rank, nb);
    nb_wait_for_all(nb);

    return COMEX_SUCCESS;
}


int comex_malloc(void *ptrs[], size_t size, comex_group_t group)
{
    comex_igroup_t *igroup = NULL;
    reg_entry_t *reg_entries = NULL;
    size_t size_entries = 0;
    int my_master = -1;
    int my_world_rank = -1;
    int i;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    reg_entry_t *reg_entries_local = NULL;

    /* preconditions */
    COMEX_ASSERT(ptrs);
   
#if DEBUG
    printf("[%d] comex_malloc(ptrs=%p, size=%lu, group=%d)\n",
            g_state.rank, ptrs, (long unsigned)size, group);
#endif

    /* is this needed? */
    comex_barrier(group);

    igroup = comex_get_igroup_from_group(group);
    my_world_rank = _get_world_rank(igroup, igroup->rank);
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_malloc my_master=%d\n", g_state.rank, my_master);
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

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_malloc allocated reg entries\n",
            g_state.rank);
#endif

    /* allocate and register segment */
    if (0 == size) {
        reg_cache_nullify(&reg_entries[igroup->rank]);
    }
    else {
        reg_entries[igroup->rank] = *_comex_malloc_local(sizeof(char)*size);
    }

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_malloc allocated and registered local shmem\n",
            g_state.rank);
#endif

    /* exchange buffer address via reg entries */
    MPI_Allgather(MPI_IN_PLACE, sizeof(reg_entry_t), MPI_BYTE, reg_entries,
            sizeof(reg_entry_t), MPI_BYTE, igroup->comm);

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_malloc allgather reg entries\n",
            g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (NULL == reg_entries[i].buf) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_malloc found NULL buf at %d\n",
                    g_state.rank, i);
#endif
            continue; /* a proc did not allocate (size==0) */
        }
        if (g_state.rank == reg_entries[i].rank) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_malloc found self at %d\n",
                    g_state.rank, i);
#endif
            if (is_notifier) {
                /* does this need to be a memcpy?? */
                reg_entries_local[reg_entries_local_count++] = reg_entries[i];
            }
            continue; /* we already registered our own memory */
        }
        if (g_state.hostid[reg_entries[i].rank]
                == g_state.hostid[my_world_rank]) {
            /* same SMP node, need to mmap */
            /* open remote shared memory object */
            void *memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_malloc registering "
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
                    memory);
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

    /* assign the ptr array to return to caller */
    for (i=0; i<igroup->size; ++i) {
        ptrs[i] = reg_entries[i].buf;
    }

    /* send reg entries to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier && !_is_master()) {
        nb_t *nb = NULL;
        int reg_entries_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
        message_size = sizeof(header_t) + reg_entries_local_size;
        message = malloc(message_size);
        COMEX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_MALLOC;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
        nb = nb_wait_for_handle();
        nb_recv(NULL, 0, my_master, nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, nb);
        nb_wait_for_all(nb);
        free(reg_entries_local);
    }
    else if (is_notifier) {
        free(reg_entries_local);
    }

    free(reg_entries);

    comex_barrier(group);

    return COMEX_SUCCESS;
}

int comex_malloc_mem_dev(void *ptrs[], size_t size, comex_group_t group,
        const char* device)
{
    return comex_malloc(ptrs,size,group);
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
    printf("[%d] _malloc_semaphore()\n", g_state.rank);
#endif

    status = MPI_Type_contiguous(SHM_NAME_SIZE, MPI_CHAR, &shm_name_type);
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Type_commit(&shm_name_type);
    COMEX_ASSERT(MPI_SUCCESS == status);

    semaphores = (sem_t**)malloc(sizeof(sem_t*) * g_state.size);
    COMEX_ASSERT(semaphores);

    name = _generate_shm_name(g_state.rank);
    COMEX_ASSERT(name);

#if ENABLE_UNNAMED_SEM
    {
        my_sem = _shm_create(name, sizeof(sem_t));
        /* initialize the memory as an inter-process semaphore */
        if (0 != sem_init(my_sem, 1, 1)) {
            perror("_malloc_semaphore: sem_init");
            comex_error("_malloc_semaphore: sem_init", -1);
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
                    comex_error("_malloc_semaphore: sem_unlink", status);
                }
            }
            /* second try */
            my_sem = sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 1);
        }
        if (SEM_FAILED == my_sem) {
            perror("_malloc_semaphore: sem_open");
            comex_error("_malloc_semaphore: sem_open", -1);
        }
    }
#endif

    /* store my sem in global cache */
    semaphores[g_state.rank] = my_sem;

    names = (char*)malloc(sizeof(char) * SHM_NAME_SIZE * g_state.size);
    COMEX_ASSERT(names);

    /* exchange names */
    (void)memcpy(&names[SHM_NAME_SIZE*g_state.rank], name, SHM_NAME_SIZE);
    status = MPI_Allgather(MPI_IN_PLACE, 1, shm_name_type,
            names, 1, shm_name_type, g_state.comm);

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
            COMEX_ASSERT(semaphores[i]);
#else
            semaphores[i] = sem_open(&names[SHM_NAME_SIZE*i], 0);
            if (SEM_FAILED == semaphores[i]) {
                perror("_malloc_semaphore: sem_open");
                comex_error("_malloc_semaphore: sem_open", -2);
            }
#endif
        }
        else {
            semaphores[i] = NULL;
        }
    }

    sem_name = name;

    free(names);
}


void _free_semaphore()
{
    int i;
    int retval;

#if DEBUG
    printf("[%d] _free_semaphore()\n", g_state.rank);
#endif

    for (i=0; i<g_state.size; ++i) {
        if (g_state.rank == i) {
            /* me */
#if ENABLE_UNNAMED_SEM
            retval = sem_destroy(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_destroy");
                comex_error("_free_semaphore: sem_destroy", retval);
            }
            retval = munmap(semaphores[i], sizeof(sem_t));
            if (-1 == retval) {
                perror("_free_semaphore: munmap");
                comex_error("_free_semaphore: munmap", retval);
            }
            retval = shm_unlink(sem_name);
            if (-1 == retval) {
                perror("_free_semaphore: shm_unlink");
                comex_error("_free_semaphore: shm_unlink", retval);
            }
#else
            retval = sem_close(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_close");
                comex_error("_free_semaphore: sem_close", retval);
            }
            retval = sem_unlink(sem_name);
            if (-1 == retval) {
                perror("_free_semaphore: sem_unlink");
                comex_error("_free_semaphore: sem_unlink", retval);
            }
#endif
        }
        else if (g_state.hostid[g_state.rank] == g_state.hostid[i]) {
            /* same SMP node */
#if ENABLE_UNNAMED_SEM
            retval = munmap(semaphores[i], sizeof(sem_t));
            if (-1 == retval) {
                perror("_free_semaphore: munmap");
                comex_error("_free_semaphore: munmap", retval);
            }
#else
            retval = sem_close(semaphores[i]);
            if (-1 == retval) {
                perror("_free_semaphore: sem_close");
                comex_error("_free_semaphore: sem_close", retval);
            }
#endif
        }
    }

    free(sem_name);
    sem_name = NULL;

    free(semaphores);
    semaphores = NULL;
}


int comex_free(void *ptr, comex_group_t group)
{
    comex_igroup_t *igroup = NULL;
    int my_world_rank = -1;
    int *world_ranks = NULL;
    int my_master = -1;
    void **ptrs = NULL;
    int i = 0;
    int is_notifier = 0;
    int reg_entries_local_count = 0;
    rank_ptr_t *rank_ptrs = NULL;

    comex_barrier(group);

#if DEBUG
    printf("[%d] comex_free(ptr=%p, group=%d)\n", g_state.rank, ptr, group);
#endif

    igroup = comex_get_igroup_from_group(group);
    world_ranks = _get_world_ranks(igroup);
    my_world_rank = world_ranks[igroup->rank];
    my_master = g_state.master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_free my_master=%d\n", g_state.rank, my_master);
#endif

#if MASTER_IS_SMALLEST_SMP_RANK
    is_notifier = _smallest_world_rank_with_same_hostid(igroup) == g_state.rank;
#else
    is_notifier = _largest_world_rank_with_same_hostid(igroup) == g_state.rank;
#endif
    if (is_notifier) {
        rank_ptrs = malloc(sizeof(rank_ptr_t)*g_state.node_size);
    }

    /* allocate receive buffer for exchange of pointers */
    ptrs = (void **)malloc(sizeof(void *) * igroup->size);
    COMEX_ASSERT(ptrs);
    ptrs[igroup->rank] = ptr;

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_free ptrs allocated and assigned\n",
            g_state.rank);
#endif

    /* exchange of pointers */
    MPI_Allgather(MPI_IN_PLACE, sizeof(void *), MPI_BYTE,
            ptrs, sizeof(void *), MPI_BYTE, igroup->comm);

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] comex_free ptrs exchanged\n", g_state.rank);
#endif

    /* remove all pointers from registration cache */
    for (i=0; i<igroup->size; ++i) {
        if (i == igroup->rank) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free found self at %d\n", g_state.rank, i);
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
            printf("[%d] comex_free found NULL at %d\n", g_state.rank, i);
#endif
        }
        else if (g_state.hostid[world_ranks[i]]
                == g_state.hostid[g_state.rank]) {
            /* same SMP node */
            reg_entry_t *reg_entry = NULL;
            int retval = 0;

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free same hostid at %d\n", g_state.rank, i);
#endif

            /* find the registered memory */
            reg_entry = reg_cache_find(world_ranks[i], ptrs[i], 0);
            COMEX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free found reg entry\n", g_state.rank);
#endif

            /* unmap the memory */
            retval = munmap(reg_entry->mapped, reg_entry->len);
            if (-1 == retval) {
                perror("comex_free: munmap");
                comex_error("comex_free: munmap", retval);
            }

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free unmapped mapped memory in reg entry\n",
                    g_state.rank);
#endif

            reg_cache_delete(world_ranks[i], ptrs[i]);

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free deleted reg cache entry\n", g_state.rank);
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
            printf("[%d] comex_free deleted reg cache entry\n", g_state.rank);
#endif
#endif
        }
    }

    /* send ptrs to my master */
    /* first non-master rank in an SMP node sends the message to master */
    if (is_notifier && !_is_master()) {
        nb_t *nb = NULL;
        int rank_ptrs_local_size = 0;
        int message_size = 0;
        char *message = NULL;
        header_t *header = NULL;

        rank_ptrs_local_size = sizeof(rank_ptr_t) * reg_entries_local_count;
        message_size = sizeof(header_t) + rank_ptrs_local_size;
        message = malloc(message_size);
        COMEX_ASSERT(message);
        header = (header_t*)message;
        header->operation = OP_FREE;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = 0;
        header->length = reg_entries_local_count;
        memcpy(message+sizeof(header_t), rank_ptrs, rank_ptrs_local_size);
        nb = nb_wait_for_handle();
        nb_recv(NULL, 0, my_master, nb); /* prepost ack */
        nb_send_header(message, message_size, my_master, nb);
        nb_wait_for_all(nb);
        free(rank_ptrs);
    }
    else if (is_notifier) {
        free(rank_ptrs);
    }

    /* free ptrs array */
    free(ptrs);
    free(world_ranks);

    /* remove my ptr from reg cache and free ptr */
    comex_free_local(ptr);

    /* Is this needed? */
    comex_barrier(group);

    return COMEX_SUCCESS;
}

int comex_free_dev(void *ptr, comex_group_t group)
{
    return comex_free(ptr, group);
}

STATIC void* _progress_server(void *arg)
{
    int running = 0;
    char *static_buffer = NULL;

#if DEBUG
    printf("[%d] _progress_server()\n", g_state.rank);
#endif

    {
        int status = _set_affinity(g_state.node_size);
        if (0 != status) {
            status = _set_affinity(g_state.node_size-1);
            COMEX_ASSERT(0 == status);
        }
    }

    /* initialize shared buffers */
    static_buffer = (char*)malloc(sizeof(char)*COMEX_STATIC_BUFFER_SIZE);
    COMEX_ASSERT(static_buffer);
    static_acc_buffer = (char*)malloc(sizeof(char)*COMEX_STATIC_BUFFER_SIZE);
    COMEX_ASSERT(static_acc_buffer);

    running = 1;
    while (running) {
        int source = 0;
        int length = 0;
        char *payload = NULL;
        header_t *header = NULL;
        MPI_Status recv_status;

        MPI_Recv(static_buffer, COMEX_STATIC_BUFFER_SIZE, MPI_CHAR,
                MPI_ANY_SOURCE, COMEX_TAG, g_state.comm, &recv_status);
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        source = recv_status.MPI_SOURCE;
#   if DEBUG
        printf("[%d] progress MPI_Recv source=%d length=%d\n",
                g_state.rank, source, length);
#   endif
        header = (header_t*)static_buffer;
        payload = static_buffer + sizeof(header_t);
        /* dispatch message handler */
        switch (header->operation) {
            case OP_PUT:
                _put_handler(header, source);
                break;
            case OP_PUT_PACKED:
                _put_packed_handler(header, source);
                break;
            case OP_PUT_IOV:
                _put_iov_handler(header, source);
                break;
            case OP_GET:
                _get_handler(header, source);
                break;
            case OP_GET_PACKED:
                _get_packed_handler(header, source);
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
                _acc_packed_handler(header, source);
                break;
            case OP_ACC_INT_IOV:
            case OP_ACC_DBL_IOV:
            case OP_ACC_FLT_IOV:
            case OP_ACC_CPL_IOV:
            case OP_ACC_DCP_IOV:
            case OP_ACC_LNG_IOV:
                _acc_iov_handler(header, source);
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
                printf("[%d] header operation not recognized: %d\n",
                        g_state.rank, header->operation);
                COMEX_ASSERT(0);
        }
    }

    free(static_buffer);
    free(static_acc_buffer);

    return NULL;
}


STATIC void _put_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int retval = 0;
    MPI_Status recv_status;

#if DEBUG
    printf("[%d] _put_handler rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
            reg_entry, header->remote_address);
    retval = MPI_Recv(mapped_offset, header->length, MPI_CHAR,
            proc, COMEX_TAG, g_state.comm, &recv_status);
    CHECK_MPI_RETVAL(retval);
}


STATIC void _put_packed_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int retval = 0;
    MPI_Status recv_status;
    char *packed_buffer = NULL;
    int packed_index = 0;
    stride_t *stride = NULL;
    int i, j;
    long dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int dst_bvalue[7], dst_bunit[7];

#if DEBUG
    printf("[%d] _put_packed_handler rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    stride = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride);
    server_recv(stride, sizeof(stride_t), proc);
    COMEX_ASSERT(stride->stride_levels >= 0);
    COMEX_ASSERT(stride->stride_levels < COMEX_MAX_STRIDE_LEVEL);

#if DEBUG
    printf("[%d] _put_packed_handler stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride->stride_levels, stride->count[0]);
    for (i=0; i<stride->stride_levels; ++i) {
        printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride->stride[i], i, stride->count[i+1]);
    }
#endif

    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        packed_buffer = malloc(header->length);
    }
    else {
        packed_buffer = static_acc_buffer;
    }

    server_recv(packed_buffer, header->length, proc);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, stride->count[0]);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(
            reg_entry, header->remote_address);

    unpack(packed_buffer, mapped_offset,
            stride->stride, stride->count, stride->stride_levels);
    free(stride);
    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        free(packed_buffer);
    }
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
    printf("[%d] _put_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] _put_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_PUT_IOV == header->operation);

    iov_buf = malloc(header->length);
    COMEX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    COMEX_ASSERT(iov_off == header->length);

#if DEBUG
    printf("[%d] _put_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
        packed_buffer = malloc(bytes*limit);
        COMEX_ASSERT(packed_buffer);
    }
    else {
        packed_buffer = static_acc_buffer;
    }

    server_recv(packed_buffer, bytes * limit, proc);

    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, dst[i], bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(
                reg_entry, dst[i]);

        (void)memcpy(mapped_offset, &packed_buffer[packed_index], bytes);
        packed_index += bytes;
    }
    COMEX_ASSERT(packed_index == bytes*limit);

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
        free(packed_buffer);
    }

    free(iov_buf);
}


STATIC void _get_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;

#if DEBUG
    printf("[%d] _get_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    COMEX_ASSERT(OP_GET == header->operation);
    
    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    server_send(mapped_offset, header->length, proc);
}


STATIC void _get_packed_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    char *packed_buffer = NULL;
    int packed_index = 0;
    stride_t *stride_src = NULL;

#if DEBUG
    printf("[%d] _get_packed_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_GET_PACKED == header->operation);

    stride_src = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride_src);
    server_recv(stride_src, sizeof(stride_t), proc);
    COMEX_ASSERT(stride_src->stride_levels >= 0);
    COMEX_ASSERT(stride_src->stride_levels < COMEX_MAX_STRIDE_LEVEL);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    packed_buffer = pack(mapped_offset,
            stride_src->stride, stride_src->count, stride_src->stride_levels,
            &packed_index);

    server_send(packed_buffer, packed_index, proc);

    free(stride_src);
    free(packed_buffer);
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
    printf("[%d] _get_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] _get_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    assert(OP_GET_IOV == header->operation);

    iov_buf = malloc(header->length);
    COMEX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    COMEX_ASSERT(iov_off == header->length);

#if DEBUG
    printf("[%d] _get_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
        packed_buffer = malloc(bytes*limit);
        COMEX_ASSERT(packed_buffer);
    }
    else {
        packed_buffer = static_acc_buffer;
    }

    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, src[i], bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, src[i]);

        (void)memcpy(&packed_buffer[packed_index], mapped_offset, bytes);
        packed_index += bytes;
    }
    COMEX_ASSERT(packed_index == bytes*limit);

    server_send(packed_buffer, packed_index, proc);

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
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
    MPI_Status recv_status;

#if DEBUG
    printf("[%d] _acc_handler\n", g_state.rank);
#endif

    switch (header->operation) {
        case OP_ACC_INT:
            acc_type = COMEX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL:
            acc_type = COMEX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT:
            acc_type = COMEX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG:
            acc_type = COMEX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL:
            acc_type = COMEX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP:
            acc_type = COMEX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: COMEX_ASSERT(0);
    }

    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        acc_buffer = malloc(header->length);
    }
    else {
        acc_buffer = static_acc_buffer;
    }

    MPI_Recv(acc_buffer, header->length, MPI_CHAR,
            proc, COMEX_TAG, g_state.comm, &recv_status);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    sem_wait(semaphores[header->rank]);
    _acc(acc_type, header->length, mapped_offset, acc_buffer, scale);
    sem_post(semaphores[header->rank]);

    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        free(acc_buffer);
    }
}


STATIC void _acc_packed_handler(header_t *header, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    void *scale = NULL;
    int sizeof_scale = 0;
    int acc_type = 0;
    char *acc_buffer = NULL;
    MPI_Status recv_status;
    stride_t *stride = NULL;

#if DEBUG
    printf("[%d] _acc_packed_handler\n", g_state.rank);
#endif

    switch (header->operation) {
        case OP_ACC_INT_PACKED:
            acc_type = COMEX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL_PACKED:
            acc_type = COMEX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT_PACKED:
            acc_type = COMEX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG_PACKED:
            acc_type = COMEX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL_PACKED:
            acc_type = COMEX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP_PACKED:
            acc_type = COMEX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: COMEX_ASSERT(0);
    }

    scale = malloc(sizeof_scale);
    COMEX_ASSERT(scale);
    server_recv(scale, sizeof_scale, proc);

    stride = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride);
    server_recv(stride, sizeof(stride_t), proc);

    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        acc_buffer = malloc(header->length);
    }
    else {
        acc_buffer = static_acc_buffer;
    }

    server_recv(acc_buffer, header->length, proc);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
    mapped_offset = _get_offset_memory(reg_entry, header->remote_address);

    sem_wait(semaphores[header->rank]);
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

        COMEX_ASSERT(stride_levels >= 0);
        COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);
        COMEX_ASSERT(NULL != packed_buffer);
        COMEX_ASSERT(NULL != dst);
        COMEX_ASSERT(NULL != dst_stride);
        COMEX_ASSERT(NULL != count);
        COMEX_ASSERT(count[0] > 0);

#if DEBUG
        printf("[%d] unpack(dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d)\n",
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

        COMEX_ASSERT(packed_index == n1dim*count[0]);
    }
    sem_post(semaphores[header->rank]);

    if ((unsigned)header->length > COMEX_STATIC_BUFFER_SIZE) {
        free(acc_buffer);
    }

    free(stride);
    free(scale);
}


STATIC void _acc_iov_handler(header_t *header, int proc)
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
    void *scale = NULL;
    int sizeof_scale = 0;
    int acc_type = 0;
    MPI_Status recv_status;

#if DEBUG
    printf("[%d] _acc_iov_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] _acc_iov_handler header rem=%p loc=%p rem_rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

#if DEBUG
    printf("[%d] _acc_iov_handler limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    switch (header->operation) {
        case OP_ACC_INT_IOV:
            acc_type = COMEX_ACC_INT;
            sizeof_scale = sizeof(int);
            break;
        case OP_ACC_DBL_IOV:
            acc_type = COMEX_ACC_DBL;
            sizeof_scale = sizeof(double);
            break;
        case OP_ACC_FLT_IOV:
            acc_type = COMEX_ACC_FLT;
            sizeof_scale = sizeof(float);
            break;
        case OP_ACC_LNG_IOV:
            acc_type = COMEX_ACC_LNG;
            sizeof_scale = sizeof(long);
            break;
        case OP_ACC_CPL_IOV:
            acc_type = COMEX_ACC_CPL;
            sizeof_scale = sizeof(SingleComplex);
            break;
        case OP_ACC_DCP_IOV:
            acc_type = COMEX_ACC_DCP;
            sizeof_scale = sizeof(DoubleComplex);
            break;
        default: COMEX_ASSERT(0);
    }

    scale = malloc(sizeof_scale);
    COMEX_ASSERT(scale);
    server_recv(scale, sizeof_scale, proc);

    iov_buf = malloc(header->length);
    COMEX_ASSERT(iov_buf);
    server_recv(iov_buf, header->length, proc);

    limit = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(limit > 0);

    bytes = *((int*)(&iov_buf[iov_off]));
    iov_off += sizeof(int);
    COMEX_ASSERT(bytes > 0);

    src = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    dst = (void**)&iov_buf[iov_off];
    iov_off += sizeof(void*)*limit;

    COMEX_ASSERT(iov_off == header->length);

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
        packed_buffer = malloc(bytes*limit);
    }
    else {
        packed_buffer = static_acc_buffer;
    }

    server_recv(packed_buffer, bytes*limit, proc);

    sem_wait(semaphores[header->rank]);
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        reg_entry = reg_cache_find(
                header->rank, dst[i], bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, dst[i]);

        _acc(acc_type, bytes, mapped_offset, &packed_buffer[packed_index], scale);
        packed_index += bytes;
    }
    COMEX_ASSERT(packed_index == bytes*limit);
    sem_post(semaphores[header->rank]);

    if ((bytes*limit) > COMEX_STATIC_BUFFER_SIZE) {
        free(packed_buffer);
    }

    free(scale);
    free(iov_buf);
}


STATIC void _fence_handler(header_t *header, int proc)
{
#if DEBUG
    printf("[%d] _fence_handler proc=%d\n", g_state.rank, proc);
#endif

    /* preconditions */
    COMEX_ASSERT(header);

#if NEED_ASM_VOLATILE_MEMORY
#if DEBUG
    printf("[%d] _fence_handler asm volatile (\"\" : : : \"memory\"); \n",
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
    printf("[%d] _fetch_and_add_handler proc=%d\n", g_state.rank, proc);
#endif
#if DEBUG
    printf("[%d] header rem=%p loc=%p rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    COMEX_ASSERT(OP_FETCH_AND_ADD == header->operation);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
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
        COMEX_ASSERT(0);
    }
}


STATIC void _swap_handler(header_t *header, char *payload, int proc)
{
    reg_entry_t *reg_entry = NULL;
    void *mapped_offset = NULL;
    int *value_int = NULL;
    long *value_long = NULL;

#if DEBUG
    printf("[%d] _swap_handler rem=%p loc=%p rank=%d len=%d\n",
            g_state.rank,
            header->remote_address,
            header->local_address,
            header->rank,
            header->length);
#endif

    COMEX_ASSERT(OP_SWAP == header->operation);

    reg_entry = reg_cache_find(
            header->rank, header->remote_address, header->length);
    COMEX_ASSERT(reg_entry);
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
        COMEX_ASSERT(0);
    }
}


STATIC void _mutex_create_handler(header_t *header, int proc)
{
    int i;
    int num = header->length;

#if DEBUG
    printf("[%d] _mutex_create_handler proc=%d num=%d\n",
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
    printf("[%d] _mutex_destroy_handler proc=%d\n", g_state.rank, proc);
#endif

    for (i=0; i<num; ++i) {
        COMEX_ASSERT(mutexes[proc][i] == UNLOCKED);
        COMEX_ASSERT(lq_heads[proc][i] == NULL);
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
    printf("[%d] _lock_handler id=%d in rank=%d req by proc=%d\n",
            g_state.rank, id, rank, proc);
#endif

    COMEX_ASSERT(0 <= id);
    
    if (UNLOCKED == mutexes[rank][id]) {
        mutexes[rank][id] = proc;
        server_send(&id, sizeof(int), proc);
    }
    else {
        lock_t *lock = NULL;
#if DEBUG
        printf("[%d] _lq_push rank=%d req_by=%d id=%d\n",
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
    printf("[%d] _unlock_handler id=%d in rank=%d req by proc=%d\n",
            g_state.rank, id, rank, proc);
#endif

    COMEX_ASSERT(0 <= id);
    
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
    printf("[%d] _malloc_handler proc=%d\n", g_state.rank, proc);
#endif

    COMEX_ASSERT(header);
    COMEX_ASSERT(header->operation == OP_MALLOC);
    n = header->length;

#if DEBUG && DEBUG_VERBOSE
    printf("[%d] _malloc_handler preconditions complete\n", g_state.rank);
#endif

    /* insert reg entries into local registration cache */
    for (i=0; i<n; ++i) {
        if (NULL == reg_entries[i].buf) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _malloc_handler found NULL at %d\n", g_state.rank, i);
#endif
        }
        else if (g_state.hostid[reg_entries[i].rank]
                == g_state.hostid[g_state.rank]) {
            /* same SMP node, need to mmap */
            /* attach to remote shared memory object */
            void *memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _malloc_handler registering "
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
                    memory);
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
    printf("[%d] _malloc_handler finished registrations\n", g_state.rank);
#endif

    server_send(NULL, 0, proc); /* ack */
}


STATIC void _free_handler(header_t *header, char *payload, int proc)
{
    int i = 0;
    int n = header->length;
    rank_ptr_t *rank_ptrs = (rank_ptr_t*)payload;

#if DEBUG
    printf("[%d] _free_handler proc=%d\n", g_state.rank, proc);
#endif

    /* remove all pointers from registration cache */
    for (i=0; i<n; ++i) {
        if (g_state.rank == rank_ptrs[i].rank) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] comex_free found self at %d\n", g_state.rank, i);
#endif
        }
        else if (NULL == rank_ptrs[i].ptr) {
#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler found NULL at %d\n", g_state.rank, i);
#endif
        }
        else if (g_state.hostid[rank_ptrs[i].rank]
                == g_state.hostid[g_state.rank]) {
            /* same SMP node */
            reg_entry_t *reg_entry = NULL;
            int retval = 0;

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler same hostid at %d\n", g_state.rank, i);
#endif

            /* find the registered memory */
            reg_entry = reg_cache_find(rank_ptrs[i].rank, rank_ptrs[i].ptr, 0);
            COMEX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler found reg entry\n", g_state.rank);
#endif

            /* unmap the memory */
            retval = munmap(reg_entry->mapped, reg_entry->len);
            if (-1 == retval) {
                perror("_free_handler: munmap");
                comex_error("_free_handler: munmap", retval);
            }

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler unmapped mapped memory in reg entry\n",
                    g_state.rank);
#endif

            reg_cache_delete(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler deleted reg cache entry\n",
                    g_state.rank);
#endif

        }
        else {
#if 0
            reg_cache_delete(rank_ptrs[i].rank, rank_ptrs[i].ptr);

#if DEBUG && DEBUG_VERBOSE
            printf("[%d] _free_handler deleted reg cache entry\n",
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

    COMEX_ASSERT(reg_entry);
#if DEBUG_VERBOSE
    printf("[%d] _get_offset_memory reg_entry->buf=%p memory=%p\n",
            g_state.rank, reg_entry->buf, memory);
#endif
    offset = ((char*)memory) - ((char*)reg_entry->buf);
#if DEBUG_VERBOSE
    printf("[%d] _get_offset_memory ptrdiff=%lu\n",
            g_state.rank, (unsigned long)offset);
#endif
    return (void*)((char*)(reg_entry->mapped)+offset);
}


STATIC int _is_master(void)
{
    return (g_state.master[g_state.rank] == g_state.rank);
}


STATIC int _get_world_rank(comex_igroup_t *igroup, int rank)
{
    int world_rank;
    int status;

    status = MPI_Group_translate_ranks(igroup->group, 1, &rank,
            g_state.group, &world_rank);
    CHECK_MPI_RETVAL(status);
    COMEX_ASSERT(MPI_PROC_NULL != world_rank);

    return world_rank;
}


/* gets (in group order) corresponding world ranks for entire group */
STATIC int* _get_world_ranks(comex_igroup_t *igroup)
{
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
    COMEX_ASSERT(MPI_SUCCESS == status);

    for (i=0; i<igroup->size; ++i) {
        COMEX_ASSERT(MPI_PROC_NULL != world_ranks[i]);
    }

    free(group_ranks);

    return world_ranks;
}


/* we sometimes need to notify a node master of some event and the rank in
 * charge of doing that is returned by this function */
STATIC int _smallest_world_rank_with_same_hostid(comex_igroup_t *igroup)
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
STATIC int _largest_world_rank_with_same_hostid(comex_igroup_t *igroup)
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
    printf("[%d] _shm_create(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* create shared memory segment */
    fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd && EEXIST == errno) {
        retval = shm_unlink(name);
        if (-1 == retval) {
            perror("_shm_create: shm_unlink");
            comex_error("_shm_create: shm_unlink", retval);
        }
    }

    /* try a second time */
    if (-1 == fd) {
        fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    }

    /* finally report error if needed */
    if (-1 == fd) {
        perror("_shm_create: shm_open");
        comex_error("_shm_create: shm_open", fd);
    }

    /* set the size of my shared memory object */
    retval = ftruncate(fd, size);
    if (-1 == retval) {
        perror("_shm_create: ftruncate");
        comex_error("_shm_create: ftruncate", retval);
    }

    /* map into local address space */
    mapped = _shm_map(fd, size);

    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_create: close");
        comex_error("_shm_create: close", -1);
    }

    return mapped;
}


STATIC void* _shm_attach(const char *name, size_t size)
{
    void *mapped = NULL;
    int fd = 0;
    int retval = 0;

#if DEBUG
    printf("[%d] _shm_attach(%s, %lu)\n",
            g_state.rank, name, (unsigned long)size);
#endif

    /* attach to shared memory segment */
    fd = shm_open(name, O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd) {
        perror("_shm_attach: shm_open");
        comex_error("_shm_attach: shm_open", -1);
    }

    /* map into local address space */
    mapped = _shm_map(fd, size);

    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
        perror("_shm_attach: close");
        comex_error("_shm_attach: close", -1);
    }

    return mapped;
}


STATIC void* _shm_map(int fd, size_t size)
{
    void *memory = NULL;

    memory = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == memory) {
        perror("_shm_map: mmap");
        comex_error("_shm_map: mmap", -1);
    }

    return memory;
}


STATIC int _set_affinity(int cpu)
{
    int status = 0;
#if COMEX_SET_AFFINITY
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
    printf("[%d] server_send(buf=%p, count=%d, dest=%d)\n",
            g_state.rank, buf, count, dest);
#endif

    retval = MPI_Send(buf, count, MPI_CHAR, dest,
            COMEX_TAG, group_list->comm);

    CHECK_MPI_RETVAL(retval);
}


STATIC void server_recv(void *buf, int count, int source)
{
    int retval = 0;
    MPI_Status status;
    int recv_count = 0;

    retval = MPI_Recv(buf, count, MPI_CHAR, source,
            COMEX_TAG, g_state.comm, &status);

    CHECK_MPI_RETVAL(retval);
    COMEX_ASSERT(status.MPI_SOURCE == source);
    COMEX_ASSERT(status.MPI_TAG == COMEX_TAG);

    retval = MPI_Get_count(&status, MPI_CHAR, &recv_count);
    CHECK_MPI_RETVAL(retval);
    COMEX_ASSERT(recv_count == count);
}


STATIC void nb_send_common(void *buf, int count, int dest, nb_t *nb, int need_free)
{
    int retval = 0;
    message_t *message = NULL;

    COMEX_ASSERT(NULL != nb);

    nb->send_size += 1;
    nb_count_event += 1;
    nb_count_send += 1;

    message = (message_t*)malloc(sizeof(message_t));
    message->next = NULL;
    message->message = buf;
    message->need_free = need_free;
    message->stride = NULL;
    message->iov = NULL;

    if (NULL == nb->send_head) {
        nb->send_head = message;
    }
    if (NULL != nb->send_tail) {
        nb->send_tail->next = message;
    }
    nb->send_tail = message;

    retval = MPI_Isend(buf, count, MPI_CHAR, dest, COMEX_TAG, g_state.comm,
            &(message->request));
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_send_header(void *buf, int count, int dest, nb_t *nb)
{
    nb_send_common(buf, count, dest, nb, 1);
}


STATIC void nb_send_buffer(void *buf, int count, int dest, nb_t *nb)
{
    nb_send_common(buf, count, dest, nb, 0);
}


STATIC void nb_recv_packed(void *buf, int count, int source, nb_t *nb, stride_t *stride)
{
    int retval = 0;
    message_t *message = NULL;

    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(count > 0);
    COMEX_ASSERT(NULL != nb);

#if DEBUG
    printf("[%d] nb_recv_packed(buf=%p, count=%d, source=%d, nb=%p)\n",
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

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, COMEX_TAG, group_list->comm,
            &(message->request));
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_recv_iov(void *buf, int count, int source, nb_t *nb, comex_giov_t *iov)
{
    int retval = 0;
    message_t *message = NULL;

    COMEX_ASSERT(NULL != nb);

#if DEBUG
    printf("[%d] nb_recv_iov(buf=%p, count=%d, source=%d, nb=%p)\n",
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

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
        COMEX_ASSERT(NULL == nb->recv_tail);
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, COMEX_TAG, group_list->comm,
            &(message->request));
    CHECK_MPI_RETVAL(retval);
}


STATIC void nb_recv(void *buf, int count, int source, nb_t *nb)
{
    int retval = 0;
    message_t *message = NULL;

    COMEX_ASSERT(NULL != nb);

#if DEBUG
    printf("[%d] nb_recv(buf=%p, count=%d, source=%d, nb=%p)\n",
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

    if (NULL == nb->recv_head) {
        nb->recv_head = message;
    }
    if (NULL != nb->recv_tail) {
        nb->recv_tail->next = message;
    }
    nb->recv_tail = message;

    retval = MPI_Irecv(buf, count, MPI_CHAR, source, COMEX_TAG, group_list->comm,
            &(message->request));
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



STATIC nb_t* nb_wait_for_handle()
{
    nb_t *nb = NULL;
    int in_use_count = 0;

    /* find first handle that isn't associated with a user-level handle */
    /* make sure the handle we find has processed all events */
    /* the user can accidentally exhaust the available handles */
    do {
        ++in_use_count;
        if (in_use_count > nb_max_outstanding) {
            fprintf(stderr,
                    "{%d} nb_wait_for_handle Error: all user-level "
                    "nonblocking handles have been exhausted\n",
                    g_state.rank);
            MPI_Abort(g_state.comm, -1);
        }
        nb = &nb_state[nb_index++];
        nb_index %= nb_max_outstanding; /* wrap around if needed */
        nb_wait_for_all(nb);
    } while (nb->in_use);

    return nb;
}


STATIC void nb_wait_for_send(nb_t *nb)
{
#if DEBUG
    printf("[%d] nb_wait_for_send(nb=%p)\n", g_state.rank, nb);
#endif

    COMEX_ASSERT(NULL != nb);

    while (NULL != nb->send_head) {
        nb_wait_for_send1(nb);
    }

    nb->send_tail = NULL;
}


STATIC void nb_wait_for_send1(nb_t *nb)
{
#if DEBUG
    printf("[%d] nb_wait_for_send1(nb=%p)\n", g_state.rank, nb);
#endif

    COMEX_ASSERT(NULL != nb);
    COMEX_ASSERT(NULL != nb->send_head);

    {
        MPI_Status status;
        int retval = 0;
        message_t *message_to_free = NULL;

        retval = MPI_Wait(&(nb->send_head->request), &status);
        CHECK_MPI_RETVAL(retval);

        if (nb->send_head->need_free) {
            free(nb->send_head->message);
        }

        message_to_free = nb->send_head;
        nb->send_head = nb->send_head->next;
        free(message_to_free);

        COMEX_ASSERT(nb->send_size > 0);
        nb->send_size -= 1;
        nb_count_send_processed += 1;
        nb_count_event_processed += 1;

        if (NULL == nb->send_head) {
            nb->send_tail = NULL;
        }
    }
}


STATIC void nb_wait_for_recv(nb_t *nb)
{
#if DEBUG
    printf("[%d] nb_wait_for_recv(nb=%p)\n", g_state.rank, nb);
#endif

    COMEX_ASSERT(NULL != nb);

    while (NULL != nb->recv_head) {
        nb_wait_for_recv1(nb);
    }
}


STATIC void nb_wait_for_recv1(nb_t *nb)
{
#if DEBUG
    printf("[%d] nb_wait_for_recv1(nb=%p)\n", g_state.rank, nb);
#endif

    COMEX_ASSERT(NULL != nb);
    COMEX_ASSERT(NULL != nb->recv_head);

    {
        MPI_Status status;
        int retval = 0;
        message_t *message_to_free = NULL;

        retval = MPI_Wait(&(nb->recv_head->request), &status);
        CHECK_MPI_RETVAL(retval);

        if (NULL != nb->recv_head->stride) {
            stride_t *stride = nb->recv_head->stride;
            COMEX_ASSERT(nb->recv_head->message);
            COMEX_ASSERT(stride);
            COMEX_ASSERT(stride->ptr);
            COMEX_ASSERT(stride->stride);
            COMEX_ASSERT(stride->count);
            COMEX_ASSERT(stride->stride_levels);
            unpack(nb->recv_head->message, stride->ptr,
                    stride->stride, stride->count, stride->stride_levels);
            free(stride);
        }

        if (NULL != nb->recv_head->iov) {
            int i = 0;
            char *message = nb->recv_head->message;
            int off = 0;
            comex_giov_t *iov = nb->recv_head->iov;
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

        message_to_free = nb->recv_head;
        nb->recv_head = nb->recv_head->next;
        free(message_to_free);

        COMEX_ASSERT(nb->recv_size > 0);
        nb->recv_size -= 1;
        nb_count_recv_processed += 1;
        nb_count_event_processed += 1;

        if (NULL == nb->recv_head) {
            nb->recv_tail = NULL;
        }
    }
}


STATIC void nb_wait_for_all(nb_t *nb)
{
#if DEBUG
    printf("[%d] nb_wait_for_all(nb=%p)\n", g_state.rank, nb);
#endif

    COMEX_ASSERT(NULL != nb);

    /* fair processing of requests */
    while (NULL != nb->send_head || NULL != nb->recv_head) {
        if (NULL != nb->send_head) {
            nb_wait_for_send1(nb);
        }
        if (NULL != nb->recv_head) {
            nb_wait_for_recv1(nb);
        }
    }
}


STATIC void nb_wait_all()
{
    int i = 0;

    COMEX_ASSERT(nb_count_event-nb_count_event_processed >= 0);

    for (i=0; i<nb_max_outstanding; ++i) {
        nb_wait_for_all(&nb_state[i]);
        COMEX_ASSERT(nb_state[i].send_size == 0);
        COMEX_ASSERT(nb_state[i].send_head == NULL);
        COMEX_ASSERT(nb_state[i].send_tail == NULL);
        COMEX_ASSERT(nb_state[i].recv_size == 0);
        COMEX_ASSERT(nb_state[i].recv_head == NULL);
        COMEX_ASSERT(nb_state[i].recv_tail == NULL);
    }
    COMEX_ASSERT(nb_count_event == nb_count_event_processed);
    COMEX_ASSERT(nb_count_send == nb_count_send_processed);
    COMEX_ASSERT(nb_count_recv == nb_count_recv_processed);
}


STATIC void nb_put(void *src, void *dst, int bytes, int proc, nb_t *nb)
{
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(bytes > 0);
    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != nb);

#if ENABLE_PUT_SELF
    /* put to self */
    if (g_state.rank == proc) {
        memcpy(dst, src, bytes);
        return;
    }
#endif

#if ENABLE_PUT_SMP
    /* put to SMP node */
    if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) {
        reg_entry_t *reg_entry = NULL;
        void *mapped_offset = NULL;

        reg_entry = reg_cache_find(proc, dst, bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, dst);
        memcpy(mapped_offset, src, bytes);
        return;
    }
#endif

    {
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];
        /* only fence on the master */
        fence_array[master_rank] = 1;
        header = malloc(sizeof(header_t));
        header->operation = OP_PUT;
        header->remote_address = dst;
        header->local_address = src;
        header->rank = proc;
        header->length = bytes;
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_buffer(src, bytes, master_rank, nb);
    }
}


STATIC void nb_get(void *src, void *dst, int bytes, int proc, nb_t *nb)
{
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(bytes > 0);
    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != nb);

#if ENABLE_GET_SELF
    /* get from self */
    if (g_state.rank == proc) {
        memcpy(dst, src, bytes);
        return;
    }
#endif

#if ENABLE_GET_SMP
    /* get from SMP node */
    if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) {
        reg_entry_t *reg_entry = NULL;
        void *mapped_offset = NULL;

        reg_entry = reg_cache_find(proc, src, bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, src);
        memcpy(dst, mapped_offset, bytes);
        return;
    }
#endif

    {
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];
        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = OP_GET;
        header->remote_address = src;
        header->local_address = dst;
        header->rank = proc;
        header->length = bytes;
        nb_recv(dst, bytes, master_rank, nb); /* prepost receive */
        nb_send_header(header, sizeof(header_t), master_rank, nb);
    }
}


STATIC void nb_acc(int datatype, void *scale,
        void *src, void *dst, int bytes, int proc, nb_t *nb)
{
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(bytes > 0);
    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != nb);

#if ENABLE_ACC_SELF
    /* acc to self */
    if (g_state.rank == proc) {
        sem_wait(semaphores[proc]);
        _acc(datatype, bytes, dst, src, scale);
        sem_post(semaphores[proc]);
        return;
    }
#endif

#if ENABLE_ACC_SMP
    /* acc to same SMP node */
    if (g_state.hostid[proc] == g_state.hostid[g_state.rank]) {
        reg_entry_t *reg_entry = NULL;
        void *mapped_offset = NULL;

        reg_entry = reg_cache_find(proc, dst, bytes);
        COMEX_ASSERT(reg_entry);
        mapped_offset = _get_offset_memory(reg_entry, dst);
        sem_wait(semaphores[proc]);
        _acc(datatype, bytes, mapped_offset, src, scale);
        sem_post(semaphores[proc]);
        return;
    }
#endif

    {
        header_t *header = NULL;
        char *message = NULL;
        int master_rank = -1;
        int message_size = 0;
        int scale_size = 0;
        op_t operation = 0;

        switch (datatype) {
            case COMEX_ACC_INT:
                operation = OP_ACC_INT;
                scale_size = sizeof(int);
                break;
            case COMEX_ACC_DBL:
                operation = OP_ACC_DBL;
                scale_size = sizeof(double);
                break;
            case COMEX_ACC_FLT:
                operation = OP_ACC_FLT;
                scale_size = sizeof(float);
                break;
            case COMEX_ACC_CPL:
                operation = OP_ACC_CPL;
                scale_size = sizeof(SingleComplex);
                break;
            case COMEX_ACC_DCP:
                operation = OP_ACC_DCP;
                scale_size = sizeof(DoubleComplex);
                break;
            case COMEX_ACC_LNG:
                operation = OP_ACC_LNG;
                scale_size = sizeof(long);
                break;
            default: COMEX_ASSERT(0);
        }

        master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        message_size = sizeof(header_t) + scale_size;
        message = malloc(message_size);
        COMEX_ASSERT(message);
        header = (header_t*)message;
        header->operation = operation;
        header->remote_address = dst;
        header->local_address = src;
        header->rank = proc;
        header->length = bytes;
        memcpy(message+sizeof(header_t), scale, scale_size);
        nb_send_header(message, message_size, master_rank, nb);
        nb_send_buffer(src, bytes, master_rank, nb);
    }
}


STATIC void nb_puts(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb)
{
    int i, j;
    long src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int dst_bvalue[7], dst_bunit[7];

#if DEBUG
    printf("[%d] nb_puts(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    /* if not actually a strided put */
    if (0 == stride_levels) {
        nb_put(src, dst, count[0], proc, nb);
        return;
    }

#if ENABLE_PUT_PACKED
#if ENABLE_PUT_SELF
    /* if not a strided put to self, use packed algorithm */
    if (g_state.rank != proc)
#endif
    {
        nb_puts_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }
#endif

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
        int *count, int stride_levels, int proc, nb_t *nb)
{
    int i, j;
    long src_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int src_bvalue[7], src_bunit[7];
    int packed_index = 0;
    char *packed_buffer = NULL;
    stride_t *stride = NULL;

#if DEBUG
    printf("[%d] nb_puts_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(NULL != nb);
    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride);
    stride->stride_levels = stride_levels;
    stride->count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride->stride[i] = dst_stride[i];
        stride->count[i+1] = count[i+1];
    }
    for (/*no init*/; i<COMEX_MAX_STRIDE_LEVEL; ++i) {
        stride->stride[i] = -1;
        stride->count[i+1] = -1;
    }

    COMEX_ASSERT(stride->stride_levels >= 0);
    COMEX_ASSERT(stride->stride_levels < COMEX_MAX_STRIDE_LEVEL);

#if DEBUG
    printf("[%d] nb_puts_packed stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride_levels, count[0]);
    for (i=0; i<stride_levels; ++i) {
        printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride->stride[i], i, stride->count[i+1]);
    }
#endif

    packed_buffer = pack(src, src_stride, count, stride_levels, &packed_index);

    COMEX_ASSERT(NULL != packed_buffer);
    COMEX_ASSERT(packed_index > 0);

    {
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];
        /* only fence on the master */
        fence_array[master_rank] = 1;
        header = malloc(sizeof(header_t));
        header->operation = OP_PUT_PACKED;
        header->remote_address = dst;
        header->local_address = NULL;
        header->rank = proc;
        header->length = packed_index;
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_header(stride, sizeof(stride_t), master_rank, nb);
        nb_send_header(packed_buffer, packed_index, master_rank, nb);
    }
}


STATIC void nb_gets(
        void *src, int *src_stride, void *dst, int *dst_stride,
        int *count, int stride_levels, int proc, nb_t *nb)
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

#if ENABLE_GET_PACKED
#if ENABLE_GET_SELF
    /* if not a strided get from self, use packed algorithm */
    if (g_state.rank != proc)
#endif
    {
        nb_gets_packed(src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }
#endif

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
        int *count, int stride_levels, int proc, nb_t *nb)
{
    int i;
    stride_t *stride_src = NULL;
    stride_t *stride_dst = NULL;

#if DEBUG
    printf("[%d] nb_gets_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(NULL != nb);
    COMEX_ASSERT(count[0] > 0);
    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);

    /* copy src info into structure */
    stride_src = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride_src);
    stride_src->ptr = src;
    stride_src->stride_levels = stride_levels;
    stride_src->count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride_src->stride[i] = src_stride[i];
        stride_src->count[i+1] = count[i+1];
    }
    for (/*no init*/; i<COMEX_MAX_STRIDE_LEVEL; ++i) {
        stride_src->stride[i] = -1;
        stride_src->count[i+1] = -1;
    }

    COMEX_ASSERT(stride_src->stride_levels >= 0);
    COMEX_ASSERT(stride_src->stride_levels < COMEX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride_dst = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride_dst);
    stride_dst->ptr = dst;
    stride_dst->stride_levels = stride_levels;
    stride_dst->count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride_dst->stride[i] = dst_stride[i];
        stride_dst->count[i+1] = count[i+1];
    }
    for (/*no init*/; i<COMEX_MAX_STRIDE_LEVEL; ++i) {
        stride_dst->stride[i] = -1;
        stride_dst->count[i+1] = -1;
    }

    COMEX_ASSERT(stride_dst->stride_levels >= 0);
    COMEX_ASSERT(stride_dst->stride_levels < COMEX_MAX_STRIDE_LEVEL);

    {
        int recv_size = 0;
        char *packed_buffer = NULL;
        header_t *header = NULL;
        int master_rank = -1;

        master_rank = g_state.master[proc];

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = OP_GET_PACKED;
        header->remote_address = src;
        header->local_address = dst;
        header->rank = proc;
        header->length = 0;

        recv_size = packed_size(stride_dst->stride,
                stride_dst->count, stride_dst->stride_levels);
        COMEX_ASSERT(recv_size > 0);
        packed_buffer = malloc(recv_size);
        COMEX_ASSERT(packed_buffer);
        nb_recv_packed(packed_buffer, recv_size, master_rank, nb, stride_dst);
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_header(stride_src, sizeof(stride_t), master_rank, nb);
    }
}


STATIC void nb_accs(
        int datatype, void *scale,
        void *src, int *src_stride,
        void *dst, int *dst_stride,
        int *count, int stride_levels,
        int proc, nb_t *nb)
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

#if ENABLE_ACC_PACKED
#if ENABLE_ACC_SELF
    /* if not a strided acc to self, use packed algorithm */
    if (g_state.rank != proc)
#endif
    {
        nb_accs_packed(datatype, scale, src, src_stride, dst, dst_stride, count, stride_levels, proc, nb);
        return;
    }
#endif

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
        int proc, nb_t *nb)
{
    int i;
    int packed_index = 0;
    char *packed_buffer = NULL;
    stride_t *stride = NULL;

#if DEBUG
    printf("[%d] nb_accs_packed(src=%p, src_stride=%p, dst=%p, dst_stride=%p, count[0]=%d, stride_levels=%d, proc=%d, nb=%p)\n",
            g_state.rank, src, src_stride, dst, dst_stride,
            count[0], stride_levels, proc, nb);
#endif

    COMEX_ASSERT(proc >= 0);
    COMEX_ASSERT(proc < g_state.size);
    COMEX_ASSERT(NULL != scale);
    COMEX_ASSERT(NULL != src);
    COMEX_ASSERT(NULL != dst);
    COMEX_ASSERT(NULL != count);
    COMEX_ASSERT(NULL != nb);
    COMEX_ASSERT(count[0] > 0);
    COMEX_ASSERT(stride_levels >= 0);
    COMEX_ASSERT(stride_levels < COMEX_MAX_STRIDE_LEVEL);

    /* copy dst info into structure */
    stride = malloc(sizeof(stride_t));
    COMEX_ASSERT(stride);
    stride->ptr = dst;
    stride->stride_levels = stride_levels;
    stride->count[0] = count[0];
    for (i=0; i<stride_levels; ++i) {
        stride->stride[i] = dst_stride[i];
        stride->count[i+1] = count[i+1];
    }
    /* assign remaining values to invalid */
    for (/*no init*/; i<COMEX_MAX_STRIDE_LEVEL; ++i) {
        stride->stride[i] = -1;
        stride->count[i+1] = -1;
    }

    COMEX_ASSERT(stride->stride_levels >= 0);
    COMEX_ASSERT(stride->stride_levels < COMEX_MAX_STRIDE_LEVEL);

#if DEBUG
    printf("[%d] nb_accs_packed stride_levels=%d, count[0]=%d\n",
            g_state.rank, stride_levels, count[0]);
    for (i=0; i<stride_levels; ++i) {
        printf("[%d] stride[%d]=%d count[%d+1]=%d\n",
                g_state.rank, i, stride->stride[i], i, stride->count[i+1]);
    }
#endif

    packed_buffer = pack(src, src_stride, count, stride_levels, &packed_index);

    COMEX_ASSERT(NULL != packed_buffer);
    COMEX_ASSERT(packed_index > 0);

    {
        header_t *header = NULL;
        int scale_size = 0;
        op_t operation = 0;
        int master_rank = -1;

        switch (datatype) {
            case COMEX_ACC_INT:
                operation = OP_ACC_INT_PACKED;
                scale_size = sizeof(int);
                break;
            case COMEX_ACC_DBL:
                operation = OP_ACC_DBL_PACKED;
                scale_size = sizeof(double);
                break;
            case COMEX_ACC_FLT:
                operation = OP_ACC_FLT_PACKED;
                scale_size = sizeof(float);
                break;
            case COMEX_ACC_CPL:
                operation = OP_ACC_CPL_PACKED;
                scale_size = sizeof(SingleComplex);
                break;
            case COMEX_ACC_DCP:
                operation = OP_ACC_DCP_PACKED;
                scale_size = sizeof(DoubleComplex);
                break;
            case COMEX_ACC_LNG:
                operation = OP_ACC_LNG_PACKED;
                scale_size = sizeof(long);
                break;
            default: COMEX_ASSERT(0);
        }

        master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = operation;
        header->remote_address = dst;
        header->local_address = NULL;
        header->rank = proc;
        header->length = packed_index;
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_buffer(scale, scale_size, master_rank, nb);
        nb_send_header(stride, sizeof(stride_t), master_rank, nb);
        nb_send_header(packed_buffer, packed_index, master_rank, nb);
    }
}


STATIC void nb_putv(
        comex_giov_t *iov, int iov_len,
        int proc, nb_t *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector put to self, use packed algorithm */
        if (ENABLE_PUT_IOV && (!ENABLE_PUT_SELF || g_state.rank != proc)) {
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


STATIC void nb_putv_packed(comex_giov_t *iov, int proc, nb_t *nb)
{
    int i = 0;
    int j = 0;;
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
    printf("[%d] nb_putv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    COMEX_ASSERT(iov_buf);
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
    COMEX_ASSERT(iov_off == iov_size);

    /* allocate send buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    COMEX_ASSERT(packed_buffer);
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
        packed_index += bytes;
    }
    COMEX_ASSERT(packed_index == bytes*limit);

    {
        header_t *header = NULL;
        int master_rank = g_state.master[proc];

        /* only fence on the master */
        fence_array[master_rank] = 1;

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
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
        comex_giov_t *iov, int iov_len,
        int proc, nb_t *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector get from self, use packed algorithm */
        if (ENABLE_GET_IOV && (!ENABLE_GET_SELF || g_state.rank != proc)) {
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


STATIC void nb_getv_packed(comex_giov_t *iov, int proc, nb_t *nb)
{
    int i = 0;
    int j = 0;;
    void **src = NULL;
    void **dst = NULL;
    int bytes = 0;
    int limit = 0;
    char *iov_buf = NULL;
    int iov_off = 0;
    int iov_size = 0;
    comex_giov_t *iov_copy = NULL;
    char *packed_buffer = NULL;
    int packed_size = 0;

    src = iov->src;
    dst = iov->dst;
    bytes = iov->bytes;
    limit = iov->count;

#if DEBUG
    printf("[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    iov_off = 0;
    COMEX_ASSERT(iov_buf);
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
    COMEX_ASSERT(iov_off == iov_size);

    /* copy given iov for later */
    iov_copy = malloc(sizeof(comex_giov_t));
    iov_copy->bytes = bytes;
    iov_copy->count = limit;
    iov_copy->src = malloc(sizeof(void*)*iov->count);
    COMEX_ASSERT(iov_copy->src);
    (void)memcpy(iov_copy->src, iov->src, sizeof(void*)*iov->count);
    iov_copy->dst = malloc(sizeof(void*)*iov->count);
    COMEX_ASSERT(iov_copy->dst);
    (void)memcpy(iov_copy->dst, iov->dst, sizeof(void*)*iov->count);

#if DEBUG
    printf("[%d] nb_getv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p copy\n",
            g_state.rank, iov_copy->count, iov_copy->bytes,
            iov_copy->src[0], iov_copy->dst[0]);
#endif

    /* allocate recv buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    COMEX_ASSERT(packed_buffer);

    {
        header_t *header = NULL;
        int master_rank = g_state.master[proc];

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
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
        comex_giov_t *iov, int iov_len,
        int proc, nb_t *nb)
{
    int i = 0;

    for (i=0; i<iov_len; ++i) {
        /* if not a vector acc to self, use packed algorithm */
        if (ENABLE_ACC_IOV && (!ENABLE_ACC_SELF || g_state.rank != proc)) {
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
        comex_giov_t *iov,
        int proc, nb_t *nb)
{
    int i = 0;
    int j = 0;;
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
    printf("[%d] nb_accv_packed limit=%d bytes=%d src[0]=%p dst[0]=%p\n",
            g_state.rank, limit, bytes, src[0], dst[0]);
#endif

    /* allocate compressed iov */
    iov_size = 2*limit*sizeof(void*) + 2*sizeof(int);
    iov_buf = malloc(iov_size);
    iov_off = 0;
    COMEX_ASSERT(iov_buf);
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
    COMEX_ASSERT(iov_off == iov_size);

    /* allocate send buffer */
    packed_size = bytes * limit;
    packed_buffer = malloc(packed_size);
    COMEX_ASSERT(packed_buffer);
    packed_index = 0;
    for (i=0; i<limit; ++i) {
        (void)memcpy(&packed_buffer[packed_index], src[i], bytes);
        packed_index += bytes;
    }
    COMEX_ASSERT(packed_index == bytes*limit);

    {
        header_t *header = NULL;
        int scale_size = 0;
        op_t operation = 0;
        int master_rank = g_state.master[proc];

        switch (datatype) {
            case COMEX_ACC_INT:
                operation = OP_ACC_INT_IOV;
                scale_size = sizeof(int);
                break;
            case COMEX_ACC_DBL:
                operation = OP_ACC_DBL_IOV;
                scale_size = sizeof(double);
                break;
            case COMEX_ACC_FLT:
                operation = OP_ACC_FLT_IOV;
                scale_size = sizeof(float);
                break;
            case COMEX_ACC_CPL:
                operation = OP_ACC_CPL_IOV;
                scale_size = sizeof(SingleComplex);
                break;
            case COMEX_ACC_DCP:
                operation = OP_ACC_DCP_IOV;
                scale_size = sizeof(DoubleComplex);
                break;
            case COMEX_ACC_LNG:
                operation = OP_ACC_LNG_IOV;
                scale_size = sizeof(long);
                break;
            default: COMEX_ASSERT(0);
        }

        /* only fence on the master */
        fence_array[master_rank] = 1;

        header = malloc(sizeof(header_t));
        COMEX_ASSERT(header);
        header->operation = operation;
        header->remote_address = NULL;
        header->local_address = NULL;
        header->rank = proc;
        header->length = iov_size;
        nb_send_header(header, sizeof(header_t), master_rank, nb);
        nb_send_buffer(scale, scale_size, master_rank, nb);
        nb_send_header(iov_buf, iov_size, master_rank, nb);
        nb_send_header(packed_buffer, packed_size, master_rank, nb);
    }
}

