#include "locks.h"
#include "comex.h"
#include "comex_impl.h"
#include <stdlib.h>
#include <shmem.h>

/* Internal lock storage (symmetric) */
static long *g_locks = NULL;
static int g_num_mutexes = 0;
static int g_total_per_pe = 0;

int comex_create_mutexes(int num) {
    if (num < 0) return COMEX_FAILURE;
    g_num_mutexes = num;
    g_total_per_pe = l_state.n_pes + g_num_mutexes;
    size_t total = (size_t)l_state.n_pes * (size_t)g_total_per_pe;
    g_locks = (long*)shmem_malloc(sizeof(long) * total);
    if (!g_locks) return COMEX_FAILURE;
    for (size_t i = 0; i < total; ++i) g_locks[i] = (long)i;
    shmem_barrier_all();
    return COMEX_SUCCESS;
}

int comex_destroy_mutexes(void) {
    if (g_locks) {
        shmem_free(g_locks);
        g_locks = NULL;
    }
    g_num_mutexes = 0;
    shmem_barrier_all();
    return COMEX_SUCCESS;
}

int comex_lock(int mutex, int proc) {
    if (g_total_per_pe <= 0) return COMEX_FAILURE;
    if (proc < 0 || proc >= l_state.n_pes) return COMEX_FAILURE;
    if (mutex < 0 || mutex >= g_num_mutexes) return COMEX_FAILURE;
    long imutex = (long)(proc + 1) * (long)l_state.n_pes + (long)mutex;
    if ((size_t)imutex >= (size_t)g_total_per_pe) return COMEX_FAILURE;
    shmem_set_lock(&g_locks[imutex]);
    return COMEX_SUCCESS;
}

int comex_unlock(int mutex, int proc) {
    if (g_total_per_pe <= 0) return COMEX_FAILURE;
    if (proc < 0 || proc >= l_state.n_pes) return COMEX_FAILURE;
    if (mutex < 0 || mutex >= g_num_mutexes) return COMEX_FAILURE;
    long imutex = (long)(proc + 1) * (long)l_state.n_pes + (long)mutex;
    if ((size_t)imutex >= (size_t)g_total_per_pe) return COMEX_FAILURE;
    shmem_clear_lock(&g_locks[imutex]);
    return COMEX_SUCCESS;
}

void locks_set_internal(int pe) {
    if (!g_locks) return;
    if (pe < 0 || pe >= l_state.n_pes) return;
    /* reserved internal slot is the first slot for each PE */
    size_t idx = (size_t)pe * (size_t)g_total_per_pe + 0;
    shmem_set_lock(&g_locks[idx]);
}

void locks_clear_internal(int pe) {
    if (!g_locks) return;
    if (pe < 0 || pe >= l_state.n_pes) return;
    size_t idx = (size_t)pe * (size_t)g_total_per_pe + 0;
    shmem_clear_lock(&g_locks[idx]);
}
