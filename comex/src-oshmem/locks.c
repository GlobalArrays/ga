#include "locks.h"
#include "comex.h"
#include "comex_impl.h"
#include <stdlib.h>
#include <shmem.h>


/* Internal lock storage (regular memory, not symmetric) */
static long *g_locks = NULL;
static int g_locks_npes = 0;
static int *g_locks_map = NULL;

int comex_create_mutexes(int num) {
    int npes = l_state.n_pes;
    int pe = l_state.pe;
    g_locks_npes = npes;
    if (g_locks) {
        shmem_free(g_locks);
        g_locks = NULL;
    }
    if (g_locks_map) {
        free(g_locks_map);
        g_locks_map = NULL;
    }
    // Step 1: create and sum num_mutexes
    int *num_mutexes = (int*)shmem_malloc(sizeof(int) * npes);
    if (!num_mutexes) return COMEX_FAILURE;
    for (int i = 0; i < npes; ++i) num_mutexes[i] = 0;
    num_mutexes[pe] = num;
    int *sum_mutexes = (int*)shmem_malloc(sizeof(int) * npes);
    if (!sum_mutexes) { shmem_free(num_mutexes); return COMEX_FAILURE; }
    shmem_int_sum_to_all(sum_mutexes, num_mutexes, npes, 0, 0, npes, NULL, NULL);
    // Step 2: allocate g_locks_map and fill as described
    g_locks_map = (int*)malloc(sizeof(int) * (npes + 1));
    if (!g_locks_map) { shmem_free(num_mutexes); shmem_free(sum_mutexes); return COMEX_FAILURE; }
    g_locks_map[0] = npes;
    for (int i = 1; i <= npes; ++i) {
        g_locks_map[i] = g_locks_map[i-1] + sum_mutexes[i-1];
    }
    shmem_free(num_mutexes);
    shmem_free(sum_mutexes);
    if (g_locks) {
        shmem_free(g_locks);
        g_locks = NULL;
    }
    size_t locks_len = g_locks_map[npes];
    g_locks = (long*)shmem_malloc(sizeof(long) * locks_len);
    if (!g_locks) { free(g_locks_map); g_locks_map = NULL; return COMEX_FAILURE; }
    for (size_t i = 0; i < locks_len; ++i) g_locks[i] = (long)i;
    return COMEX_SUCCESS;
}

int comex_destroy_mutexes(void) {
    if (g_locks) {
        shmem_free(g_locks);
        g_locks = NULL;
    }
    if (g_locks_map) {
        free(g_locks_map);
        g_locks_map = NULL;
    }
    g_locks_npes = 0;
    return COMEX_SUCCESS;
}

int comex_lock(int mutex, int proc) {
    if (proc < 0 || proc >= l_state.n_pes) return COMEX_FAILURE;
    if (mutex < 0 || mutex >= (g_locks_map[proc+1] - g_locks_map[proc])) return COMEX_FAILURE;
    long imutex = g_locks_map[proc] + mutex;
    shmem_set_lock(&g_locks[imutex]);
    return COMEX_SUCCESS;
}

int comex_unlock(int mutex, int proc) {
    if (proc < 0 || proc >= l_state.n_pes) return COMEX_FAILURE;
    if (mutex < 0 || mutex >= (g_locks_map[proc+1] - g_locks_map[proc])) return COMEX_FAILURE;
    long imutex = g_locks_map[proc] + mutex;
    shmem_clear_lock(&g_locks[imutex]);
    return COMEX_SUCCESS;
}

void locks_set_internal(int pe) {
    if (!g_locks) return;
    if (pe < 0 || pe >= l_state.n_pes) return;
    shmem_set_lock(&g_locks[pe]);
}

void locks_clear_internal(int pe) {
    if (!g_locks) return;
    if (pe < 0 || pe >= l_state.n_pes) return;
    shmem_clear_lock(&g_locks[pe]);
}
