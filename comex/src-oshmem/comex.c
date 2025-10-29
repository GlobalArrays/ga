#if HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "comex.h"
#include "comex_impl.h"
#include "groups.h"
#include "nb.h"
#include <mpi.h>

local_state l_state;

static int initialized = 0;

/* symmetric array used for pointer exchange during comex_malloc (world-only) */
static void **g_all_ptrs = NULL; /* symmetric heap allocation for pointers */
static long *g_all_sizes = NULL; /* symmetric heap allocation for sizes */

/* local allocation registry to store allocation info for future use */
typedef struct {
    void *ptr;
    size_t size;
} alloc_entry_t;

static alloc_entry_t *local_allocs = NULL;
static int local_alloc_count = 0;

int comex_init() {
    if (initialized) return COMEX_SUCCESS;

    /* Initialize SHMEM */
#if defined(_OPENSHMEM)
    shmem_init();
#else
    /* Many implementations define shmem_init in shmem.h; call it anyway */
    shmem_init();
#endif
    l_state.pe = shmem_my_pe();
    l_state.n_pes = shmem_n_pes();

    /* allocate symmetric array for pointer exchange (one pointer per PE)
     * allocate on symmetric heap so address is same on all PEs
     */
    g_all_ptrs = (void**)shmem_malloc(sizeof(void*) * l_state.n_pes);
    if (!g_all_ptrs) {
        fprintf(stderr,"[%d] comex: shmem_malloc failed for g_all_ptrs\n", l_state.pe);
        return COMEX_FAILURE;
    }
    /* initialize local copy to NULL */
    for (int i=0;i<l_state.n_pes;i++) g_all_ptrs[i] = NULL;

    /* allocate symmetric array for size exchange (one long per PE) */
    g_all_sizes = (long*)shmem_malloc(sizeof(long) * l_state.n_pes);
    if (!g_all_sizes) {
        fprintf(stderr,"[%d] comex: shmem_malloc failed for g_all_sizes\n", l_state.pe);
        return COMEX_FAILURE;
    }
    for (int i=0;i<l_state.n_pes;i++) g_all_sizes[i] = 0;

    comex_group_init();

    /* barrier for sanity */
    shmem_barrier_all();

    initialized = 1;
    return COMEX_SUCCESS;
}

int comex_init_args(int *argc, char ***argv) {
    (void)argc; (void)argv;
    return comex_init();
}

int comex_init_comm(MPI_Comm comm) {
    int cmp = MPI_UNEQUAL;
    int ierr = MPI_Comm_compare(comm, MPI_COMM_WORLD, &cmp);
    if (ierr != MPI_SUCCESS) {
        comex_error("comex_init_comm: MPI_Comm_compare failed", ierr);
        return COMEX_FAILURE;
    }
    /* Accept only MPI_COMM_WORLD or communicators congruent with it */
    if (!(cmp == MPI_IDENT || cmp == MPI_CONGRUENT)) {
        comex_error("comex_init_comm: only MPI_COMM_WORLD is supported by SHMEM backend", -1);
        return COMEX_FAILURE;
    }
    return comex_init();
}

int comex_initialized() {
    return initialized;
}

void comex_error(const char *msg, int code) {
    fprintf(stderr,"[%d] COMEX ERROR: %s (code=%d)\n", l_state.pe, msg, code);
    shmem_global_exit(code);
}

int comex_finalize() {
    if (!initialized) return COMEX_SUCCESS;

    /* wait for outstanding NB ops */
    shmem_quiet();
    shmem_barrier_all();

    /* free symmetric array */
    if (g_all_ptrs) shmem_free(g_all_ptrs);
    g_all_ptrs = NULL;

    comex_group_finalize();

    shmem_finalize();

    initialized = 0;
    return COMEX_SUCCESS;
}

/* helper: translate group rank (group is world-only) to world PE */
static int translate_group_rank_to_pe(int proc, comex_group_t group, int *pe_out) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (proc < 0 || proc >= l_state.n_pes) return COMEX_FAILURE;
    *pe_out = proc;
    return COMEX_SUCCESS;
}

int comex_put(void *src, void *dst, int bytes, int proc, comex_group_t group) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    /* SHMEM semantics: shmem_putmem(target, source, nelems, pe) */
    shmem_putmem(dst, src, bytes, pe);
    /* ensure local completion before returning (match COMEX semantics) */
    shmem_quiet();
    return COMEX_SUCCESS;
}

int comex_get(void *src, void *dst, int bytes, int proc, comex_group_t group) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    shmem_getmem(dst, src, bytes, pe);
    shmem_quiet();
    return COMEX_SUCCESS;
}

int comex_nbput(void *src, void *dst, int bytes, int proc, comex_group_t group, comex_request_t *hdl) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;
    int idx = comex_nb_reserve();
    if (idx < 0) return COMEX_FAILURE;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    e->op = 0; e->src = src; e->dst = dst; e->bytes = bytes; e->target_pe = pe;

    shmem_put_nbi(dst, src, bytes, pe);
    /* do NOT call shmem_quiet() here; leave it to wait/test */

    if (hdl) *hdl = idx;
    return COMEX_SUCCESS;
}

int comex_nbget(void *src, void *dst, int bytes, int proc, comex_group_t group, comex_request_t *hdl) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;
    int idx = comex_nb_reserve();
    if (idx < 0) return COMEX_FAILURE;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    e->op = 1; e->src = src; e->dst = dst; e->bytes = bytes; e->target_pe = pe;

    shmem_get_nbi(dst, src, bytes, pe);

    if (hdl) *hdl = idx;
    return COMEX_SUCCESS;
}

int comex_wait(comex_request_t *nb_handle) {
    if (!nb_handle) return COMEX_FAILURE;
    int idx = *nb_handle;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    if (!e) return COMEX_FAILURE;

    /* Ensure completion of outstanding non-blocking operations */
    shmem_quiet();
    comex_nb_release(idx);
    return COMEX_SUCCESS;
}

int comex_test(comex_request_t *nb_handle, int *status) {
    if (!nb_handle || !status) return COMEX_FAILURE;
    int idx = *nb_handle;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    if (!e) return COMEX_FAILURE;

    /* Best-effort: call quiet and consider it complete */
    shmem_quiet();
    *status = 0; /* completed */
    comex_nb_release(idx);
    return COMEX_SUCCESS;
}

int comex_wait_all(comex_group_t group) {
    (void)group; /* group ignored; world-only */
    shmem_quiet();
    return COMEX_SUCCESS;
}

int comex_malloc(void **ptr_arr, size_t bytes, comex_group_t group) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!ptr_arr) return COMEX_FAILURE;

    /* Step 1: publish local requested size into g_all_sizes on all PEs */
    long local_bytes = (long)bytes;
    for (int target=0; target<l_state.n_pes; ++target) {
        shmem_long_put(&g_all_sizes[l_state.pe], &local_bytes, 1, target);
    }
    /* ensure the size writes have arrived everywhere */
    shmem_barrier_all();

    /* Step 2: locally compute the maximum size across all PEs */
    long max_bytes = 0;
    for (int i=0; i<l_state.n_pes; ++i) {
        if (g_all_sizes[i] > max_bytes) max_bytes = g_all_sizes[i];
    }

    if (max_bytes <= 0) return COMEX_FAILURE;

    /* Step 3: allocate symmetric buffer of size max_bytes */
    void *local_ptr = shmem_malloc((size_t)max_bytes);
    if (!local_ptr) return COMEX_FAILURE;

    /* Step 4: publish local_ptr into g_all_ptrs on all PEs */
    for (int target=0; target<l_state.n_pes; ++target) {
        /* use 64-bit put to transfer pointer value */
        shmem_put64(&g_all_ptrs[l_state.pe], &local_ptr, 1, target);
    }
    shmem_barrier_all();

    /* copy symmetric array into ptr_arr */
    for (int i=0;i<l_state.n_pes;i++) ptr_arr[i] = g_all_ptrs[i];

    /* Step 5: record local allocation for future use */
    alloc_entry_t *tmp = (alloc_entry_t*)realloc(local_allocs, sizeof(alloc_entry_t)*(local_alloc_count+1));
    if (!tmp) {
        /* allocation record failure is non-fatal for RMA but warn */
        fprintf(stderr, "[%d] comex_malloc: warning: failed to record allocation info\n", l_state.pe);
    } else {
        local_allocs = tmp;
        local_allocs[local_alloc_count].ptr = local_ptr;
        /* record the originally requested size (bytes) rather than the
         * actual symmetric allocation size (max_bytes) so callers can
         * retrieve the originally requested allocation size later. */
        local_allocs[local_alloc_count].size = bytes;
        local_alloc_count++;
    }

    return COMEX_SUCCESS;
}

int comex_free(void *ptr, comex_group_t group) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!ptr) return COMEX_FAILURE;
    /* free the symmetric memory */
    shmem_free(ptr);

    /* remove entry from local_allocs if present */
    if (local_allocs && local_alloc_count > 0) {
        int found = -1;
        for (int i = 0; i < local_alloc_count; ++i) {
            if (local_allocs[i].ptr == ptr) { found = i; break; }
        }
        if (found >= 0) {
            /* shift remaining entries down */
            for (int j = found; j < local_alloc_count-1; ++j) {
                local_allocs[j] = local_allocs[j+1];
            }
            local_alloc_count--;
            if (local_alloc_count == 0) {
                free(local_allocs);
                local_allocs = NULL;
            } else {
                alloc_entry_t *tmp = (alloc_entry_t*)realloc(local_allocs, sizeof(alloc_entry_t)*local_alloc_count);
                if (tmp) local_allocs = tmp;
            }
        }
    }

    shmem_barrier_all();
    return COMEX_SUCCESS;
}

/* Simple fence semantics */
int comex_fence_all(comex_group_t group) {
    (void)group;
    shmem_quiet();
    return COMEX_SUCCESS;
}

int comex_fence_proc(int proc, comex_group_t group) {
    (void)group; (void)proc;
    shmem_quiet();
    return COMEX_SUCCESS;
}

/* Minimal group functions: wrapper to groups module */
int comex_group_create(int n, int *pid_list, comex_group_t group, comex_group_t *new_group) {
    return comex_group_create(n,pid_list,group,new_group);
}

int comex_group_free(comex_group_t group) {
    return comex_group_free(group);
}

int comex_group_rank(comex_group_t group, int *rank) {
    return comex_group_rank(group, rank);
}

int comex_group_size(comex_group_t group, int *size) {
    return comex_group_size(group, size);
}

int comex_group_comm(comex_group_t group, MPI_Comm *comm) {
    return comex_group_comm(group, comm);
}
