/* cmx header file */
#ifndef _P_ENVIRONMENT_H
#define _P_ENVIRONMENT_H

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
#include <string>

/* 3rd party headers */
#include <mpi.h>

#include "acc.hpp"
#include "defines.hpp"
#include "p_structs.hpp"
#include "p_group.hpp"
#include "reg_cache.hpp"

#define CMX_MAX_NB_OUTSTANDING 256
#define CMX_MAX_STRIDE_LEVEL 8
#define CMX_TAG 27624
#define CMX_STATIC_BUFFER_SIZE (2u*1048576u)
#define SHM_NAME_SIZE 31
#define UNLOCKED -1 /* performance or correctness related settings */
#if defined(__bgq__) || defined(__bgp__)
#define ENABLE_UNNAMED_SEM 1
#else
#define ENABLE_UNNAMED_SEM 0
#endif
#define NEED_ASM_VOLATILE_MEMORY 0
#define MASTER_IS_SMALLEST_SMP_RANK 0
#define CMX_SET_AFFINITY 0
#define ENABLE_PUT_SELF 1
#define ENABLE_GET_SELF 1
#define ENABLE_ACC_SELF 1
#define ENABLE_PUT_SMP 1
#define ENABLE_GET_SMP 1
#define ENABLE_ACC_SMP 1
#define ENABLE_PUT_PACKED 1
#define ENABLE_GET_PACKED 1
#define ENABLE_ACC_PACKED 1
#define ENABLE_PUT_DATATYPE 1
#define ENABLE_GET_DATATYPE 1
#define ENABLE_ACC_DATATYPE 0
#define ENABLE_PUT_IOV 1
#define ENABLE_GET_IOV 1
#define ENABLE_ACC_IOV 1

namespace CMX {
/* structure to keep track of global layout */

typedef struct {
  MPI_Comm comm;  /* whole comm; all ranks */
  MPI_Group group;/* whole group; all ranks */
  int size;       /* comm size */
  int rank;       /* comm rank */
  int *master;    /* master[size] rank of a given rank's master */
  long *hostid;   /* hostid[size] hostid of SMP node for a given rank */
  MPI_Comm node_comm;  /* node comm; SMP ranks */
  int node_size;       /* node comm size */
  int node_rank;       /* node comm rank */
} cmx_group_world_t;

extern int _initialized;         /* for cmx_initialized(), 0=false */

class p_Environment {
public:

#if 0
/**
 * Return an instance of the p_Environment singleton
 * @return pointer to p_Environment singleton
 */
static p_Environment *instance(); 

/**
 * Return an instance of the p_Environment singleton. Initialize instance
 * with argc and argv if it does not already exist
 * @param[in] argc number of arguments
 * @param[in] argv list of arguments
 * @return pointer to p_Environment singleton
 */
static p_Environment *instance(int *argc, char ***argv); 
#endif

/**
 * Initialize CMX environment.
 */
p_Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
~p_Environment();

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void wait(_cmx_request *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(p_Group *group);

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool test(_cmx_request *hdl);

/**
 * Return a pointer to struct containing global state. This is used by
 * allocations
 * @return global state
 */
cmx_group_world_t* getGlobalState();

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void p_error(const std::string msg, int code);

/**
 * Get group corresponding to world group
 * @return pointer to world group
 */
p_Group* getWorldGroup();

public:
/* server functions */
void server_send(void *buf, int count, int dest);
void server_send_datatype(void *buf, MPI_Datatype dt, int dest);
void server_recv(void *buf, int count, int source);
void server_recv_datatype(void *buf, MPI_Datatype dt, int source);
void _progress_server();
void _put_handler(header_t *header, char *payload, int proc);
void _put_packed_handler(header_t *header, char *payload, int proc);
void _put_datatype_handler(header_t *header, char *payload, int proc);
void _put_iov_handler(header_t *header, int proc);
void _get_handler(header_t *header, int proc);
void _get_packed_handler(header_t *header, char *payload, int proc);
void _get_datatype_handler(header_t *header, char *payload, int proc);
void _get_iov_handler(header_t *header, int proc);
void _acc_handler(header_t *header, char *scale, int proc);
void _acc_packed_handler(header_t *header, char *payload, int proc);
void _acc_iov_handler(header_t *header, char *scale, int proc);
void _fence_handler(header_t *header, int proc);
void _fetch_and_add_handler(header_t *header, char *payload, int proc);
void _swap_handler(header_t *header, char *payload, int proc);
void _mutex_create_handler(header_t *header, int proc);
void _mutex_destroy_handler(header_t *header, int proc);
void _lock_handler(header_t *header, int proc);
void _unlock_handler(header_t *header, int proc);
void _malloc_handler(header_t *header, char *payload, int proc);
void _free_handler(header_t *header, char *payload, int proc);

/* worker functions */
void nb_send_common(void *buf, int count, int dest, _cmx_request *nb, int need_free);
void nb_send_datatype(void *buf, MPI_Datatype dt, int dest, _cmx_request *nb);
void nb_send_header(void *buf, int count, int dest, _cmx_request *nb);
void nb_send_buffer(void *buf, int count, int dest, _cmx_request *nb);
void nb_recv_packed(void *buf, int count, int source, _cmx_request *nb, stride_t *stride);
void nb_recv_datatype(void *buf, MPI_Datatype dt, int source, _cmx_request *nb);
void nb_recv_iov(void *buf, int count, int source, _cmx_request *nb, _cmx_giov_t *iov);
void nb_recv(void *buf, int count, int source, _cmx_request *nb);
void nb_wait_for_send1(_cmx_request *nb);
void nb_wait_for_recv1(_cmx_request *nb);
void nb_wait_for_all(_cmx_request *nb);
int nb_test_for_all(_cmx_request *nb);
void nb_register_request(_cmx_request *nb);
void nb_unregister_request(_cmx_request *nb);
void nb_handle_init(_cmx_request *nb);
void nb_put(void *src, void *dst, int bytes, int proc, _cmx_request *nb);
void nb_get(void *src, void *dst, int bytes, int proc, _cmx_request *nb);
void nb_acc(int datatype, void *scale, void *src, void *dst, int bytes, int proc, _cmx_request *nb);
void nb_puts(
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_puts_packed(
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_puts_datatype(
    void *src_ptr, int *src_stride_ar,
    void *dst_ptr, int *dst_stride_ar,
    int *count, int stride_levels,
    int proc, _cmx_request *nb);
void nb_gets(
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_gets_packed(
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_gets_datatype(
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_accs(
    int datatype, void *scale,
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_accs_packed(
    int datatype, void *scale,
    void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, _cmx_request *nb);
void nb_putv(_cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
void nb_putv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
void nb_getv(_cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
void nb_getv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
void nb_accv(int datatype, void *scale,
    _cmx_giov_t *iov, int iov_len, int proc, _cmx_request *nb);
void nb_accv_packed(int datatype, void *scale,
    _cmx_giov_t *iov, int proc, _cmx_request *nb);
void _fence_master(int master_rank);
int _eager_check(int extra_bytes);
int nb_test_for_send1(_cmx_request *nb, message_t **save_send_head,
        message_t **prev);
int nb_test_for_recv1(_cmx_request *nb, message_t **save_recv_head,
        message_t **prev);

/* group functions */
void _group_init(void);

/* other functions */
int _packed_size(int *src_stride, int *count, int stride_levels);
char* pack(char *src, int *src_stride,
    int *count, int stride_levels, int *size);
void unpack(char *packed_buffer,
    char *dst, int *dst_stride, int *count, int stride_levels);
char* _generate_shm_name(int rank);
void* malloc_local(size_t size);
reg_entry_t* _malloc_local(size_t size);
int free_local(void *ptr);
void* _get_offset_memory(reg_entry_t *reg_entry, void *memory);
int _is_master(void);
int _get_world_rank(p_Group *group, int rank);
int* _get_world_ranks(p_Group *group);
int _smallest_world_rank_with_same_hostid(p_Group *group);
int _largest_world_rank_with_same_hostid(p_Group *group);
void _malloc_semaphore(void);
void _free_semaphore(void);
void* _shm_create(const char *name, size_t size);
void* _shm_attach(const char *name, size_t size);
void* _shm_map(int fd, size_t size);
int _set_affinity(int cpu);
void _translate_mpi_error(int ierr, const char* location);
void strided_to_subarray_dtype(int *stride_array, int *count, int levels,
    MPI_Datatype base_type, MPI_Datatype *type);
int get_num_progress_ranks_per_node();
int get_progress_rank_distribution_on_node();
int get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed);
int get_my_rank_to_free(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed);
void check_mpi_retval(int retval, const char *file, int line);
const char *str_mpi_retval(int retval);
long xgethostid();
static int cmplong(const void *p1, const void *p2);



private:

p_Environment *p_instance;

p_Group *p_CMX_GROUP_WORLD;

/* useful for debugging */
int _cmx_me;

/* static state */
int *num_mutexes;     /**< (all) how many mutexes on each process */
int **mutexes;        /**< (masters) value is rank of lock holder */
lock_t ***lq_heads;   /**< array of lock queues */
char *sem_name;       /* local semaphore name */
sem_t **semaphores;   /* semaphores for locking within SMP node */
char *fence_array;

int nb_max_outstanding;
int nb_last_request;
int nb_index;
int nb_count_event;
int nb_count_event_processed;
int nb_count_send;
int nb_count_send_processed;
int nb_count_recv;
int nb_count_recv_processed;

char *static_server_buffer;
int static_server_buffer_size;
int eager_threshold;
int max_message_size;

int CMX_ENABLE_PUT_SELF;
int CMX_ENABLE_GET_SELF;
int CMX_ENABLE_ACC_SELF;
int CMX_ENABLE_PUT_SMP;
int CMX_ENABLE_GET_SMP;
int CMX_ENABLE_ACC_SMP;
int CMX_ENABLE_PUT_PACKED;
int CMX_ENABLE_GET_PACKED;
int CMX_ENABLE_ACC_PACKED;
int CMX_ENABLE_PUT_DATATYPE;
int CMX_ENABLE_GET_DATATYPE;
int CMX_PUT_DATATYPE_THRESHOLD;
int CMX_GET_DATATYPE_THRESHOLD;
int CMX_ENABLE_PUT_IOV;
int CMX_ENABLE_GET_IOV;
int CMX_ENABLE_ACC_IOV;

/* Non-blocking handle list */
_cmx_request **nb_list;

};
}

#endif /* _P_ENVIRONMENT_H */
