#ifndef CMX_IMPL_H_
#define CMX_IMPL_H_

#include <semaphore.h>

#include <mpi.h>
#include <assert.h>

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

#define DEBUG 0
#define DEBUG_VERBOSE 0
#define DEBUG_TO_FILE 0
#if DEBUG_TO_FILE
    FILE *cmx_trace_file;
#   define printf(...) fprintf(cmx_trace_file, __VA_ARGS__); fflush(cmx_trace_file)
#else
#   define printf(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#endif

#define CMX_STRINGIFY(x) #x
//#ifdef NDEBUG
//#define CMX_ASSERT(WHAT) ((void) (0))
//#else
#define CMX_ASSERT(WHAT) \
  ((WHAT) \
   ? (void) (0) \
   : cmx_assert_fail (CMX_STRINGIFY(WHAT), __FILE__, __LINE__, __func__))
//#endif

typedef int cmxInt;

typedef struct group_link {
  struct group_link *next;/**< next group in linked list */
  MPI_Comm comm;          /**< whole comm; all ranks */
  MPI_Group group;        /**< whole group; all ranks */
  int size;               /**< comm size */
  int rank;               /**< comm rank */
  int *world_ranks;       /**< list of world ranks corresponding to local ranks */
} cmx_igroup_t;

typedef cmx_igroup_t* cmx_group_t;

typedef struct alloc_link {
  struct alloc_link *next;
  int rank;
  void *buf;
  cmxInt size;
} cmx_alloc_t;

typedef struct {
  cmx_igroup_t *group;
  cmxInt bytes;
  cmx_alloc_t *list;
  int rank;
  void *buf;
} _cmx_handle;

/* structure to describe strided data transfers */
typedef struct {
  void *ptr;
  int stride_levels;
  cmxInt stride[CMX_MAX_STRIDE_LEVEL];
  cmxInt count[CMX_MAX_STRIDE_LEVEL+1];
} stride_t;

/* Internal struct for vector communication */
typedef struct {
  void **src; /**< array of source starting addresses */
  void **dst; /**< array of destination starting addresses */
  int count; /**< size of address arrays (src[count],dst[count]) */
  int bytes; /**< length in bytes for each src[i]/dst[i] pair */
} _cmx_giov_t;

typedef struct message_link {
  struct message_link *next;
  void *message;
  MPI_Request request;
  MPI_Datatype datatype;
  int need_free;
  stride_t *stride;
  _cmx_giov_t *iov;
} message_t;

typedef struct {
  int in_use;
  cmxInt send_size;
  message_t *send_head;
  message_t *send_tail;
  cmxInt recv_size;
  message_t *recv_head;
  message_t *recv_tail;
  cmx_igroup_t *group;
} _cmx_request;

typedef struct {
  int rank;
  void *ptr;
} rank_ptr_t;

extern cmx_igroup_t *group_list;

extern cmx_igroup_t *CMX_GROUP_WORLD;

extern int _cmx_me;

/* TODO: Problem with this function since cmx_error is defined in cmx.h
 *  * On the other hand, this function is currently not used */
#if 1
static inline void cmx_assert_fail(
        const char *assertion,
        const char *file,
        unsigned int line,
        const char *function)
{
#if 0
    fprintf(stderr, "[%d] %s:%u: %s: Assertion `%s' failed",
            g_state.rank, file, line, function, assertion);
#endif
    fprintf(stderr, "p[%d] ASSERT %s:%u: %s: Assertion `%s' failed\n",
            _cmx_me, file, line, function, assertion);
    fflush(stderr);
#if DEBUG
    printf("[%d] %s:%u: %s: Assertion `%s' failed",
            _cmx_me, file, line, function, assertion);
#endif
    assert(-1);
/*    cmx_error("cmx_assert_fail", -1); */
}
#endif
#endif /* CMX_IMPL_H_ */
