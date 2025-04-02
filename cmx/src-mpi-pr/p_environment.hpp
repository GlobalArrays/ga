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
#include <memory>

/* 3rd party headers */
#include <mpi.h>

#include "acc.hpp"
#include "defines.hpp"
#include "p_structs.hpp"
#include "p_group.hpp"
#include "node_config.hpp"
#include "shmem.hpp"
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

/**
 * Return an instance of the p_Environment singleton
 * @return pointer to p_Environment singleton
 */
static p_Environment *instance(); 

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void wait(_cmx_request *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(Group *group);

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool test(_cmx_request *hdl);

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void p_error(const std::string msg, int code);

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void fence(Group *group);

/**
 * Fence specific process in a group
 * @param proc process to fence
 * @param group group containing target process
 */
void fenceProc(int proc, Group *group);

/**
 * Get group corresponding to world group
 * @return pointer to world group
 */
Group* getWorldGroup();

/**
 * Translates the ranks of processes in one group to those in another group.  The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] group_from the group to translate ranks from 
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
int translateRanks(int n, Group *group_from, int *ranks_from,
    Group *group_to, int *ranks_to);

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group the group to translate ranks from 
 * @param[in] group_ranks the ranks to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
int translateWorld(int n, Group *group, int *group_ranks, int *world_ranks);

/**
 * Close down most environment functions. Should be called right before
 * MPI_Finalize
 */
void finalize();

protected:

/* worker functions. Invoked by global arrays to send messages to progress rank */
void nb_wait_for_all(_cmx_request *nb);
bool nb_test_for_all(_cmx_request *nb);
void nb_put(void *src, void *dst, int64_t bytes, int proc, _cmx_request *nb);
void nb_get(void *src, void *dst, int64_t bytes, int proc, _cmx_request *nb);
void nb_acc(int datatype, void *scale, void *src, void *dst, int64_t bytes, int proc, _cmx_request *nb);
void nb_puts(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_gets(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_accs(
    int datatype, void *scale,
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_putv(_cmx_giov_t *iov, int64_t iov_len, int proc, _cmx_request *nb);
void nb_getv(_cmx_giov_t *iov, int64_t iov_len, int proc, _cmx_request *nb);
void nb_accv(int datatype, void *scale,
    _cmx_giov_t *iov, int64_t iov_len, int proc, _cmx_request *nb);
void _fence_master(int master_rank);

/* allocate/free functions */
int dist_malloc(void **ptrs, int64_t bytes, Group *group);
int dist_free(void *ptr, Group *group);

/* non-blocking handle initialization */
void nb_register_request(_cmx_request *nb);

/* read-modify-write */
int rmw(int op, void *ploc, void *prem, int extra, int proc,  Group *group);

private:

/**
 * Initialize CMX environment.
 */
p_Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
~p_Environment();

/* non-blocking implementation functions */
void nb_send_common(void *buf, int count, int dest, _cmx_request *nb, int need_free);
void nb_send_datatype(void *buf, MPI_Datatype dt, int dest, _cmx_request *nb);
void nb_send_header(void *buf, int count, int dest, _cmx_request *nb);
void nb_send_buffer(void *buf, int count, int dest, _cmx_request *nb);
void nb_recv_packed(void *buf, int count, int source, _cmx_request *nb,
    stride_t *stride);
void nb_recv_datatype(void *buf, MPI_Datatype dt, int source, _cmx_request *nb);
void nb_recv_iov(void *buf, int count, int source, _cmx_request *nb, _cmx_giov_t *iov);
void nb_recv(void *buf, int count, int source, _cmx_request *nb);
void nb_wait_for_send1(_cmx_request *nb);
void nb_wait_for_recv1(_cmx_request *nb);
void nb_puts_packed(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_puts_datatype(
    void *src_ptr, int64_t *src_stride_ar,
    void *dst_ptr, int64_t *dst_stride_ar,
    int64_t *count, int stride_levels,
    int proc, _cmx_request *nb);
void nb_gets_packed(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_gets_datatype(
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_accs_packed(
    int datatype, void *scale,
    void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *nb);
void nb_putv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
void nb_getv_packed(_cmx_giov_t *iov, int proc, _cmx_request *nb);
void nb_accv_packed(int datatype, void *scale,
    _cmx_giov_t *iov, int proc, _cmx_request *nb);
int _eager_check(int extra_bytes);

/* non-blocking handle implementations */
void nb_unregister_request(_cmx_request *nb);
void nb_request_init(_cmx_request *nb);
int nb_test_for_send1(_cmx_request *nb, message_t **save_send_head,
        message_t **prev);
int nb_test_for_recv1(_cmx_request *nb, message_t **save_recv_head,
        message_t **prev);
void init_message(message_t *message);

/* server functions */
void _progress_server();
void server_send(void *buf, int64_t count, int dest);
void server_send_datatype(void *buf, MPI_Datatype dt, int dest);
void server_recv(void *buf, int64_t count, int source);
void server_recv_datatype(void *buf, MPI_Datatype dt, int source);
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

/* group functions */
void _group_init(void);

/* information on network */
int _get_world_rank(Group *group, int rank);
int* _get_world_ranks(Group *group);
int _smallest_world_rank_with_same_hostid(Group *group);
int _largest_world_rank_with_same_hostid(Group *group);
long xgethostid();
int get_num_progress_ranks_per_node();
int get_progress_rank_distribution_on_node();
int get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed);
int get_my_rank_to_free(int rank, int split_group_size,
    int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
    int num_progress_ranks_per_node, int is_node_ranks_packed);

/* pack/unpack data */
int64_t _packed_size(int64_t *src_stride, int64_t *count, int stride_levels);
char* pack(char *src, int64_t *src_stride,
    int64_t *count, int stride_levels, int64_t *size);
void unpack(char *packed_buffer,
    char *dst, int64_t *dst_stride, int64_t *count, int stride_levels);
void strided_to_subarray_dtype(int64_t *stride_array, int64_t *count, int levels,
    MPI_Datatype base_type, MPI_Datatype *type);

/* other functions */
char* _generate_shm_name(int rank);
void* _get_offset_memory(reg_entry_t *reg_entry, void *memory);
void _malloc_semaphore(void);
void _free_semaphore(void);
void* _shm_create(const char *name, size_t size);
void* _shm_attach(const char *name, size_t size);
void* _shm_map(int fd, size_t size);
int _set_affinity(int cpu);
static int cmplong(const void *p1, const void *p2);
reg_entry_t* _malloc_local(size_t size);

/* MPI utility functions */
void _translate_mpi_error(int ierr, const char* location);
void check_mpi_retval(int retval, const char *file, int line);
const char *str_mpi_retval(int retval);

private:

friend class p_Allocation; // protected functions are accessible from

static p_Environment *p_instance;

Group* p_CMX_GROUP_WORLD;

/* useful for debugging */
int _cmx_me;

/* static state */
int *num_mutexes;     /**< (all) how many mutexes on each process */
int **mutexes;        /**< (masters) value is rank of lock holder */
std::vector<std::vector<lock_t*> > lq_heads;   /**< array of lock queues */
std::string sem_name;       /* local semaphore name */
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

/* Some helper classes */
p_NodeConfig p_config;
p_Shmem p_shmem;
p_Register p_register;

};
}

#endif /* _P_ENVIRONMENT_H */
