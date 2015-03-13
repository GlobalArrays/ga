#ifndef COMEX_IMPL_H_
#define COMEX_IMPL_H_

#include <semaphore.h>

#include <mpi.h>

#include "groups.h"

#define COMEX_MAX_NB_OUTSTANDING 8
#define COMEX_MAX_STRIDE_LEVEL 8
#define COMEX_TAG 27624
#define COMEX_STATIC_BUFFER_SIZE (2u*1048576u)
#define SHM_NAME_SIZE 20
#define UNLOCKED -1

/* performance or correctness related settings */
#if defined(__bgq__) || defined(__bgp__)
#define ENABLE_UNNAMED_SEM 1
#else
#define ENABLE_UNNAMED_SEM 0
#endif
#define NEED_ASM_VOLATILE_MEMORY 0
#define MASTER_IS_SMALLEST_SMP_RANK 0
#define COMEX_SET_AFFINITY 0
#define ENABLE_PUT_SELF 1
#define ENABLE_GET_SELF 1
#define ENABLE_ACC_SELF 1
#define ENABLE_PUT_SMP 1
#define ENABLE_GET_SMP 1
#define ENABLE_ACC_SMP 1
#define ENABLE_PUT_PACKED 1
#define ENABLE_GET_PACKED 1
#define ENABLE_ACC_PACKED 1
#define ENABLE_PUT_IOV 1
#define ENABLE_GET_IOV 1
#define ENABLE_ACC_IOV 1

#define DEBUG 0
#define DEBUG_VERBOSE 0
#define DEBUG_TO_FILE 0
#if DEBUG_TO_FILE
    FILE *comex_trace_file;
#   define printf(...) fprintf(comex_trace_file, __VA_ARGS__); fflush(comex_trace_file)
#else
#   define printf(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)
#endif

#define COMEX_STRINGIFY(x) #x
#ifdef NDEBUG
#define COMEX_ASSERT(WHAT) ((void) (0))
#else
#define COMEX_ASSERT(WHAT) \
    ((WHAT) \
     ? (void) (0) \
     : comex_assert_fail (COMEX_STRINGIFY(WHAT), __FILE__, __LINE__, __func__))
#endif

static inline void comex_assert_fail(
        const char *assertion,
        const char *file,
        unsigned int line,
        const char *function)
{
    fprintf(stderr, "[%d] %s:%u: %s: Assertion `%s' failed",
            g_state.rank, file, line, function, assertion);
    fflush(stderr);
#if DEBUG
    printf("[%d] %s:%u: %s: Assertion `%s' failed",
            g_state.rank, file, line, function, assertion);
#endif
    comex_error("comex_assert_fail", -1);
}

#endif /* COMEX_IMPL_H_ */
