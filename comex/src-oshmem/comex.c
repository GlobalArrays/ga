#if HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "comex.h"
#include "comex_impl.h"
#include "groups.h"
#include "nb.h"
#include "acc.h"
#include <mpi.h>
#include <execinfo.h>

local_state l_state;

static int initialized = 0;

/* Simple debug printing macro (safe if l_state.pe is uninitialized) */
#define COMEX_DBG(fmt, ...) \
    do { \
        int _pe = l_state.pe; \
        /* Debug print removed */ \
    } while(0)

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

/* auxiliary struct used for non-blocking accumulate bookkeeping */
typedef struct {
    void *tmp;        /* temporary local buffer holding fetched remote dst */
    void *scale;      /* local copy of scale */
    int scale_bytes;  /* size of scale */
    void *src_local;  /* local source pointer */
    int needs_finalize; /* if 1: wait should perform acc->put; if 0: aux already did acc and issued put_nbi */
} nbacc_aux_t;

/* Lock storage and helpers live in locks.c; include locks.h for API. */

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
    if (l_state.pe==0) {
      printf("\nUsing Open SHMEM runtime\n\n");
    }

    /* allocate symmetric array for pointer exchange (one pointer per PE)
     * allocate on symmetric heap so address is same on all PEs
     */
    g_all_ptrs = (void**)shmem_malloc(sizeof(void*) * l_state.n_pes);
    if (!g_all_ptrs) {
        /* Debug print removed */
        return COMEX_FAILURE;
    }
    /* initialize local copy to NULL */
    for (int i=0;i<l_state.n_pes;i++) g_all_ptrs[i] = NULL;

    /* allocate symmetric array for size exchange (one long per PE) */
    g_all_sizes = (long*)shmem_malloc(sizeof(long) * l_state.n_pes);
    if (!g_all_sizes) {
        /* Debug print removed */
        return COMEX_FAILURE;
    }
    for (int i=0;i<l_state.n_pes;i++) g_all_sizes[i] = 0;

    comex_group_init();

    /* barrier for sanity */
    shmem_barrier_all();

    /* ensure lock array is allocated (reserve zero user mutexes by default)
     * so internal code can use the reserved per-PE internal locks without
     * requiring callers to create mutexes explicitly. We call the locks
     * creation API which allocates symmetric lock storage. */
    if (comex_create_mutexes(0) != COMEX_SUCCESS) {
        /* Debug print removed */
        return COMEX_FAILURE;
    }

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
    COMEX_DBG("comex_init_comm: comm OK (cmp=%d)", cmp);
    return comex_init();
}

int comex_initialized() {
    return initialized;
}

void comex_error(const char *msg, int code) {
    /* Debug print removed */
    abort();
}

int comex_finalize() {
    if (!initialized) return COMEX_SUCCESS;

    /* wait for outstanding NB ops */
    shmem_quiet();
    shmem_barrier_all();

    /* free symmetric arrays */
    if (g_all_ptrs) shmem_free(g_all_ptrs);
    g_all_ptrs = NULL;
    if (g_all_sizes) shmem_free(g_all_sizes);
    g_all_sizes = NULL;
    /* destroy symmetric locks if allocated (locks.c handles free/barrier) */
    comex_destroy_mutexes();

    comex_group_finalize();

    shmem_finalize();

    /* free local allocation registry */
    if (local_allocs) {
        free(local_allocs);
        local_allocs = NULL;
        local_alloc_count = 0;
    }

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

    COMEX_DBG("comex_put: src=%p dst=%p bytes=%d proc=%d -> pe=%d", src, dst, bytes, proc, pe);
    /* SHMEM semantics: shmem_putmem(target, source, nelems, pe) */
    shmem_putmem(dst, src, bytes, pe);
    /* ensure local completion before returning (match COMEX semantics) */
    shmem_quiet();
    shmem_fence();
    COMEX_DBG("comex_put: completed");
    return COMEX_SUCCESS;
}

int comex_get(void *src, void *dst, int bytes, int proc, comex_group_t group) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;
    COMEX_DBG("comex_get: src=%p dst=%p bytes=%d proc=%d -> pe=%d", src, dst, bytes, proc, pe);
    /* Debug print removed */
    shmem_getmem(dst, src, bytes, pe);
    shmem_quiet();
    COMEX_DBG("comex_get: completed");
    return COMEX_SUCCESS;
}

int comex_nbput(void *src, void *dst, int bytes, int proc, comex_group_t group, comex_request_t *hdl) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;
    int idx = comex_nb_reserve();
    if (idx < 0) return COMEX_FAILURE;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    e->op = 0; e->src = src; e->dst = dst; e->bytes = bytes; e->target_pe = pe;

    COMEX_DBG("comex_nbput: src=%p dst=%p bytes=%d proc=%d idx=%d", src, dst, bytes, proc, idx);
    /* Use the byte-wise non-blocking put variant so void* is accepted */
    shmem_putmem_nbi(dst, src, bytes, pe);
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

    COMEX_DBG("comex_nbget: src=%p dst=%p bytes=%d proc=%d idx=%d", src, dst, bytes, proc, idx);
    /* Debug print removed */
    /* Use the byte-wise non-blocking get variant so void* is accepted */
    shmem_getmem_nbi(dst, src, bytes, pe);

    if (hdl) *hdl = idx;
    return COMEX_SUCCESS;
}

int comex_wait(comex_request_t *nb_handle) {
    if (!nb_handle) return COMEX_FAILURE;
    int idx = *nb_handle;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    if (!e) return COMEX_FAILURE;
    /* Only call shmem_quiet if this NB entry is still active. If we do
     * call shmem_quiet(), process all active NB entries that require post-
     * processing (e.g., nb-acc entries) and then release them. */
    if (e->active) {
        shmem_quiet();
        /* process all active NB entries */
        for (int i = 0; i < COMEX_MAX_NB_OUTSTANDING; ++i) {
            comex_nb_entry_t *ent = comex_nb_get_entry(i);
            if (!ent || !ent->active) continue;
            /* handle different op types; 0=put,1=get,2=acc (we use 2 for nbacc)
             * For acc entries we expect ent->aux to point to an aux struct
             * containing temporary buffer and scale copy. */
            if (ent->op >= COMEX_ACC_OFF) {
                /* nb-acc completion: perform local accumulate on the fetched
                 * temporary buffer and write it back to the remote destination */
                nbacc_aux_t *aux = (nbacc_aux_t*)ent->aux;
                if (aux) {
                    if (aux->needs_finalize) {
                        /* perform accumulation: tmp = tmp + scale*src_local */
                        _acc(ent->op, ent->bytes, aux->tmp, aux->src_local, aux->scale);
                        /* write back the updated buffer */
                        shmem_putmem(ent->dst, aux->tmp, ent->bytes, ent->target_pe);
                        shmem_quiet();
                    } else {
                        /* aux already performed local acc and issued a nonblocking put;
                         * here we only need to free temporaries after quiet. */
                        /* nothing to do */
                    }
                    /* free temporaries */
                    free(aux->tmp);
                    free(aux->scale);
                    free(aux);
                }
            }
            /* release the entry (for puts/gets we just release after quiet) */
            comex_nb_release(i);
        }
    } else {
        /* entry already inactive; simply release this handle if needed */
        comex_nb_release(idx);
    }
    shmem_fence();
    return COMEX_SUCCESS;
}

int comex_test(comex_request_t *nb_handle, int *status) {
    if (!nb_handle || !status) return COMEX_FAILURE;
    int idx = *nb_handle;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    if (!e) return COMEX_FAILURE;

    /* Best-effort: call quiet and then process completion for this entry */
    shmem_quiet();

    if (e->active) {
        if (e->op >= COMEX_ACC_OFF) {
            nbacc_aux_t *aux = (nbacc_aux_t*)e->aux;
            if (aux) {
                if (aux->needs_finalize) {
                    _acc(e->op, e->bytes, aux->tmp, aux->src_local, aux->scale);
                    shmem_putmem(e->dst, aux->tmp, e->bytes, e->target_pe);
                    shmem_quiet();
                } else {
                    /* already performed acc and issued put_nbi; nothing to apply */
                }
                free(aux->tmp);
                free(aux->scale);
                free(aux);
            }
        }
        comex_nb_release(idx);
    }
    *status = 0; /* completed */
    return COMEX_SUCCESS;
}

int comex_wait_all(comex_group_t group) {
    (void)group; /* group ignored; world-only */
    /* Ensure local completion of outstanding NBI operations, then scan
     * the NB table and finalize any entries (e.g., nb-acc entries need
     * post-processing: local accumulate on fetched tmp buffer and remote
     * put back). */
    shmem_quiet();

    for (int i = 0; i < COMEX_MAX_NB_OUTSTANDING; ++i) {
        comex_nb_entry_t *ent = comex_nb_get_entry(i);
        if (!ent || !ent->active) continue;

        /* For accumulate entries (op values reserved at COMEX_ACC_OFF and
         * above) perform the aux-based finalize; otherwise, simple puts/gets
         * are already completed by shmem_quiet and we can release the entry. */
        if (ent->op >= COMEX_ACC_OFF) {
            nbacc_aux_t *aux = (nbacc_aux_t*)ent->aux;
            if (aux) {
                if (aux->needs_finalize) {
                    /* apply local accumulate into tmp using stored scale/src */
                    _acc(ent->op, ent->bytes, aux->tmp, aux->src_local, aux->scale);
                    /* write back updated buffer to remote destination */
                    shmem_putmem(ent->dst, aux->tmp, ent->bytes, ent->target_pe);
                    shmem_quiet();
                } else {
                    /* already performed acc and issued put_nbi; nothing further */
                }
                free(aux->tmp);
                free(aux->scale);
                free(aux);
            }
        }
        /* release the NB entry */
        comex_nb_release(i);
    }
    shmem_fence();

    return COMEX_SUCCESS;
}

int comex_malloc(void **ptr_arr, size_t bytes, comex_group_t group) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!ptr_arr) return COMEX_FAILURE;

    /* Step 1: publish local requested size into g_all_sizes at our own
     * index. Other PEs will fetch values from this symmetric array.
     * Avoid looping over targets when writing; instead write locally and
     * use SHMEM get operations when reading values from other PEs below. */
    long local_bytes = (long)bytes;
    g_all_sizes[l_state.pe] = local_bytes;
    /* ensure the size write is visible to other PEs */
    shmem_barrier_all();

    /* Step 2: compute the maximum size across all PEs by fetching each
     * PE's published size from its symmetric array entry. We use
     * shmem_long_g to read the remote value rather than relying on remote
     * puts during publication. */
    long max_bytes = 0;
    for (int i = 0; i < l_state.n_pes; ++i) {
        long v = shmem_long_g(&g_all_sizes[i], i);
        if (v > max_bytes) max_bytes = v;
    }

    if (max_bytes <= 0) return COMEX_FAILURE;

    /* Step 3: allocate symmetric buffer of size max_bytes */
    void *local_ptr = shmem_malloc((size_t)max_bytes);
    if (!local_ptr) return COMEX_FAILURE;

    /* Step 4: publish local_ptr into g_all_ptrs on all PEs */
    /* publish our pointer in the symmetric array only at our own index */
    g_all_ptrs[l_state.pe] = local_ptr;
    /* ensure other PEs can read our pointer */
    shmem_barrier_all();

    /* fetch each PE's pointer from its symmetric array entry */
    for (int i = 0; i < l_state.n_pes; ++i) {
        /* read pointer value from PE i into our local ptr_arr[i] */
        shmem_getmem(&ptr_arr[i], &g_all_ptrs[i], sizeof(void*), i);
    }

    /* Step 5: record local allocation for future use */
    alloc_entry_t *tmp = (alloc_entry_t*)realloc(local_allocs, sizeof(alloc_entry_t)*(local_alloc_count+1));
    if (!tmp) {
        /* Debug print removed */
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
    shmem_fence();
    return COMEX_SUCCESS;
}

int comex_fence_proc(int proc, comex_group_t group) {
    (void)group; (void)proc;
    shmem_quiet();
    shmem_fence();
    return COMEX_SUCCESS;
}

/* Minimal group functions: wrapper to groups module */
/* The group management functions are provided by the common groups module
 * (compiled elsewhere). Do not redefine them here — remove wrapper
 * definitions to avoid multiple-definition linker errors. */

/* Simple implementations / stubs for other COMEX functions expected by the
 * rest of the codebase. These are minimal (functional but not optimized)
 * placeholders so the SHMEM backend links cleanly. They should be
 * strengthened later to provide full semantics (mutexes, proper atomic
 * accumulates, device-aware frees, etc.). */

int comex_barrier(comex_group_t group) {
    /* Ensure local completion then perform an MPI barrier on the group's
     * communicator (world-only for this backend). */
    MPI_Comm comm = MPI_COMM_WORLD;
    shmem_quiet();
    shmem_barrier_all();
    MPI_Barrier(comm);
    return COMEX_SUCCESS;
}

int comex_free_dev(void *ptr, comex_group_t group) {
    /* Device-aware free not implemented for SHMEM backend; fall back to
     * regular comex_free semantics. */
    return comex_free(ptr, group);
}

void* comex_malloc_local(size_t bytes) {
    /* Local allocation for temporary buffers — use heap allocation. */
    return malloc(bytes);
}

int comex_free_local(void *ptr) {
    if (!ptr) return COMEX_FAILURE;
    free(ptr);
    return COMEX_SUCCESS;
}

/* Locking helpers moved to locks.c/locks.h */
#include "locks.h"

/* Minimal (placeholder) accumulate implementations. These simply return
 * success without performing real atomic accumulation. They must be
 * implemented properly (get-modify-put under remote locks) as a next
 * development step. */
int comex_acc(int op, void *scale, void *src, void *dst, int bytes,
              int proc, comex_group_t group) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    /* if target is local, perform local accumulate directly */
    if (pe == l_state.pe) {
        locks_set_internal(pe);
        _acc(op, bytes, dst, src, scale);
        locks_clear_internal(pe);
        return COMEX_SUCCESS;
    }

    /* acquire remote lock on target PE using reserved internal lock slot */
    /* Use locks module to acquire the reserved per-PE internal lock. */
    locks_set_internal(pe);

    /* temporary buffer to hold remote destination */
    void *tmp = malloc((size_t)bytes);
    if (!tmp) { locks_clear_internal(pe); return COMEX_FAILURE; }

    /* remote-get the destination into tmp */
    shmem_getmem(tmp, dst, bytes, pe);
    //shmem_quiet();

    /* perform local accumulate: tmp += scale * src */
    _acc(op, bytes, tmp, src, scale);

    /* write the result back to remote destination */
    shmem_putmem(dst, tmp, bytes, pe);
    shmem_quiet();

    free(tmp);
    /* release lock */
    locks_clear_internal(pe);

    return COMEX_SUCCESS;
}

int comex_accs(int op, void *scale, void *src, int *src_stride,
               void *dst, int *dst_stride, int *count, int stride_levels,
               int proc, comex_group_t group) {
    int i, j;
    long src_idx, dst_idx;
    int n1dim;
    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1], src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1], dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    void *get_buf;
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for(i=2; i<=stride_levels; i++) {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    /* lock for atomicity if needed */
    locks_set_internal(pe);

    get_buf = malloc(count[0]);
    if (!get_buf) return COMEX_FAILURE;

    for(i=0; i<n1dim; i++) {
        src_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }

        /* get remote data */
        if (comex_get((char*)dst + dst_idx, get_buf, count[0], pe, group) != COMEX_SUCCESS) {
          free(get_buf);
          locks_clear_internal(pe);
          return COMEX_FAILURE;
        }
        /* local accumulate */
        _acc(op, count[0], get_buf, (char*)src + src_idx, scale);
        /* put back to remote */
        if (comex_put(get_buf, (char*)dst + dst_idx, count[0], pe, group) != COMEX_SUCCESS) {
            free(get_buf);
            locks_clear_internal(pe);
            return COMEX_FAILURE;
        }
    }

    free(get_buf);
    locks_clear_internal(pe);
    return COMEX_SUCCESS;
}

int comex_accv(int op, void *scale, comex_giov_t *darr, int len,
               int proc, comex_group_t group) {
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            comex_acc(op, scale, src[j], dst[j], bytes, proc, group);
        }
    }
    return COMEX_SUCCESS;
}

/* Additional stubs to satisfy callers in ARMCI and other layers. These are
 * intentionally minimal and must be replaced with full implementations
 * (strided/vector operations, nonblocking accumulates, rmw, etc.) in a
 * follow-up iteration. For now they enable linking and basic testing. */

int comex_puts(void *src, int *src_stride, void *dst, int *dst_stride,
               int *count, int stride_levels, int proc, comex_group_t group) {
    /* Implement strided put by iterating over contiguous blocks.
     * count[0] is the size in bytes of each contiguous block.
     * count[1..stride_levels] specify repetition counts for higher dims. */
    int i, j;
    long src_idx, dst_idx;
    int n1dim;
    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1], src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1], dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

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
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }
        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        int rc = comex_put((char*)src + src_idx, (char*)dst + dst_idx, count[0], pe, group);
        if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
    }
    return COMEX_SUCCESS;
}

int comex_putv(comex_giov_t *darr, int len, int proc, comex_group_t group) {
    if (!darr || len <= 0) return COMEX_FAILURE;
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            int rc = comex_put(src[j], dst[j], bytes, proc, group);
            if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
        }
    }
    return COMEX_SUCCESS;
}

int comex_gets(void *src, int *src_stride, void *dst, int *dst_stride,
               int *count, int stride_levels, int proc, comex_group_t group) {
    if (translate_group_rank_to_pe(proc, group, &proc) != COMEX_SUCCESS) return COMEX_FAILURE;
    int n1dim = 1;
    for (int i = 1; i <= stride_levels; ++i) n1dim *= count[i];

    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1];
    int src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];

    src_bvalue[0] = dst_bvalue[0] = 0;
    src_bunit[0] = dst_bunit[0] = 1;
    for (int i = 1; i <= stride_levels; ++i) {
        src_bvalue[i] = 0; dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i];
        dst_bunit[i] = dst_bunit[i-1] * count[i];
    }

    for (int i = 0; i < n1dim; ++i) {
        long src_idx = 0, dst_idx = 0;
        for (int j = 1; j <= stride_levels; ++j) {
            src_idx += (long)src_bvalue[j] * (long)src_stride[j-1];
            if ((i+1) % src_bunit[j] == 0) src_bvalue[j]++;
            if (src_bvalue[j] > (count[j]-1)) src_bvalue[j] = 0;
        }
        for (int j = 1; j <= stride_levels; ++j) {
            dst_idx += (long)dst_bvalue[j] * (long)dst_stride[j-1];
            if ((i+1) % dst_bunit[j] == 0) dst_bvalue[j]++;
            if (dst_bvalue[j] > (count[j]-1)) dst_bvalue[j] = 0;
        }
        char *src_block = (char*)src + src_idx;
        char *dst_block = (char*)dst + dst_idx;
        int rc = comex_get(src_block, dst_block, count[0], proc, group);
        if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
    }
    return COMEX_SUCCESS;
}

int comex_getv(comex_giov_t *darr, int len, int proc, comex_group_t group) {
    if (!darr || len <= 0) return COMEX_FAILURE;
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            int rc = comex_get(src[j], dst[j], bytes, proc, group);
            if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
        }
    }
    return COMEX_SUCCESS;
}

int comex_nbputs(void *src, int *src_stride, void *dst, int *dst_stride,
                 int *count, int stride_levels, int proc, comex_group_t group,
                 comex_request_t* nb_handle) {
    /* Non-blocking strided put: issue one NB put per contiguous block.
     * If nb_handle is provided, return the last reserved handle. If not,
     * we still issue NB puts but do not return a handle. */
    int i, j;
    long src_idx, dst_idx;
    int n1dim;
    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1], src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1], dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    comex_request_t last = -1;
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

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
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }
        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        comex_request_t h = -1;
        int rc = comex_nbput((char*)src + src_idx, (char*)dst + dst_idx, count[0], pe, group, &h);
        if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
        last = h;
    }
    if (nb_handle) *nb_handle = last;
    return COMEX_SUCCESS;
}

int comex_nbputv(comex_giov_t *darr, int len, int proc, comex_group_t group,
                 comex_request_t* nb_handle) {
    if (!darr || len <= 0) return COMEX_FAILURE;
    comex_request_t last = -1;
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            comex_request_t h = -1;
            int rc = comex_nbput(src[j], dst[j], bytes, proc, group, &h);
            if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
            last = h;
        }
    }
    if (nb_handle) *nb_handle = last;
    return COMEX_SUCCESS;
}

int comex_nbgets(void *src, int *src_stride, void *dst, int *dst_stride,
                 int *count, int stride_levels, int proc, comex_group_t group,
                 comex_request_t *nb_handle) {
    /* Non-blocking strided get: issue one NB get per contiguous block.
     * If nb_handle is provided, return the last reserved handle. */
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    int i, j;
    long src_idx, dst_idx;
    int n1dim;
    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1], src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1], dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    comex_request_t last = -1;

    n1dim = 1;
    for (i = 1; i <= stride_levels; i++) {
        n1dim *= count[i];
    }

    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for (i = 2; i <= stride_levels; i++) {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i-1] * count[i-1];
        dst_bunit[i] = dst_bunit[i-1] * count[i-1];
    }

    for (i = 0; i < n1dim; i++) {
        src_idx = 0;
        for (j = 1; j <= stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride[j-1];
            if ((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if (src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;
        for (j = 1; j <= stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride[j-1];
            if ((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if (dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }

        char *src_block = (char*)src + src_idx;
        char *dst_block = (char*)dst + dst_idx;
        comex_request_t h = -1;
        int rc = comex_nbget(src_block, dst_block, count[0], pe, group, &h);
        if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
        last = h;
    }
    if (nb_handle) *nb_handle = last;
    return COMEX_SUCCESS;
}

int comex_nbgetv(comex_giov_t *darr, int len, int proc, comex_group_t group,
                 comex_request_t* nb_handle) {
    if (!darr || len <= 0) return COMEX_FAILURE;
    comex_request_t last = -1;
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            comex_request_t h = -1;
            int rc = comex_nbget(src[j], dst[j], bytes, proc, group, &h);
            if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
            last = h;
        }
    }
    if (nb_handle) *nb_handle = last;
    return COMEX_SUCCESS;
}

int comex_nbacc(int op, void *scale, void *src, void *dst, int bytes,
                int proc, comex_group_t group, comex_request_t *nb_handle) {
    int pe;
    if (translate_group_rank_to_pe(proc, group, &pe) != COMEX_SUCCESS) return COMEX_FAILURE;

    int idx = comex_nb_reserve();
    if (idx < 0) return COMEX_FAILURE;
    comex_nb_entry_t *e = comex_nb_get_entry(idx);
    if (!e) { comex_nb_release(idx); return COMEX_FAILURE; }

    /* store metadata: use ent->op to carry the comex accumulate op code */
    e->op = op; /* COMEX_ACC_* value */
    e->src = src;    /* local source pointer */
    e->dst = dst;    /* remote destination pointer */
    e->bytes = bytes;
    e->target_pe = pe;

    /* prepare auxiliary info */
    nbacc_aux_t *aux = (nbacc_aux_t*)malloc(sizeof(nbacc_aux_t));
    if (!aux) { comex_nb_release(idx); return COMEX_FAILURE; }
    aux->scale = NULL; aux->tmp = NULL; aux->scale_bytes = 0; aux->src_local = src;

    /* determine scale size based on op */
    int scale_bytes = 0;
    if (op == COMEX_ACC_DBL) scale_bytes = sizeof(double);
    else if (op == COMEX_ACC_FLT) scale_bytes = sizeof(float);
    else if (op == COMEX_ACC_INT) scale_bytes = sizeof(int);
    else if (op == COMEX_ACC_LNG) scale_bytes = sizeof(long);
    else if (op == COMEX_ACC_CPL) scale_bytes = sizeof(SingleComplex);
    else if (op == COMEX_ACC_DCP) scale_bytes = sizeof(DoubleComplex);
    else scale_bytes = sizeof(double);

    aux->scale = malloc((size_t)scale_bytes);
    if (aux->scale && scale) memcpy(aux->scale, scale, (size_t)scale_bytes);
    aux->scale_bytes = scale_bytes;
    /* default: aux requires finalization (apply acc and put) unless we
     * perform the acc now and issue a nonblocking put immediately. */
    aux->needs_finalize = 1;

    if (pe == l_state.pe) {
        /* local target: perform synchronous accumulate now and leave aux NULL
         * so comex_wait will simply release the NB entry. */
        locks_set_internal(pe);
        _acc(op, bytes, dst, src, aux->scale);
        locks_clear_internal(pe);
        free(aux->scale);
        free(aux);
        e->aux = NULL;
    } else {
        /* remote target: perform blocking get of remote destination into tmp,
         * perform local accumulate immediately, then issue a non-blocking put
         * of the updated buffer back to the remote destination. The NB entry
         * remains active until the non-blocking put completes (wait/test
         * will call shmem_quiet and then free the aux). */
        aux->tmp = malloc((size_t)bytes);
        if (!aux->tmp) { free(aux->scale); free(aux); comex_nb_release(idx); return COMEX_FAILURE; }
        e->aux = aux;
        /* blocking get the remote destination into tmp (shmem_getmem blocks
         * until the data is available locally; an immediate shmem_quiet() is
         * therefore unnecessary). */
        shmem_getmem(aux->tmp, dst, bytes, pe);
        /* perform local accumulate into tmp */
        _acc(op, bytes, aux->tmp, src, aux->scale);
        /* issue non-blocking put of updated tmp back to remote destination */
        shmem_putmem_nbi(dst, aux->tmp, bytes, pe);
        /* mark that finalization is NOT needed in wait/test (we already did acc & issued put) */
        aux->needs_finalize = 0;
    }

    if (nb_handle) *nb_handle = idx;
    return COMEX_SUCCESS;
}

int comex_nbaccs(int op, void *scale, void *src, int *src_stride,
                 void *dst, int *dst_stride, int *count, int stride_levels,
                 int proc, comex_group_t group, comex_request_t* nb_handle) {
    int i, j;
    long src_idx, dst_idx;
    int n1dim;
    int src_bvalue[COMEX_MAX_STRIDE_LEVEL+1], src_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    int dst_bvalue[COMEX_MAX_STRIDE_LEVEL+1], dst_bunit[COMEX_MAX_STRIDE_LEVEL+1];
    comex_request_t last_handle = -1;

    n1dim = 1;
    for(i=1; i<=stride_levels; i++) {
        n1dim *= count[i];
    }

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
        dst_idx = 0;
        for(j=1; j<=stride_levels; j++) {
            src_idx += src_bvalue[j] * src_stride[j-1];
            if((i+1) % src_bunit[j] == 0) {
                src_bvalue[j]++;
            }
            if(src_bvalue[j] > (count[j]-1)) {
                src_bvalue[j] = 0;
            }
        }
        for(j=1; j<=stride_levels; j++) {
            dst_idx += dst_bvalue[j] * dst_stride[j-1];
            if((i+1) % dst_bunit[j] == 0) {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j]-1)) {
                dst_bvalue[j] = 0;
            }
        }
        comex_request_t h = -1;
        int rc = comex_nbacc(op, scale, (char*)src + src_idx, (char*)dst + dst_idx, count[0], proc, group, &h);
        if (rc != COMEX_SUCCESS) {
            return COMEX_FAILURE;
        }
        last_handle = h;
    }
    if (nb_handle) *nb_handle = last_handle;
    return COMEX_SUCCESS;
}

int comex_nbaccv(int op, void *scale, comex_giov_t *darr, int len,
                 int proc, comex_group_t group, comex_request_t* nb_handle) {
    comex_request_t last = -1;
    for (int i = 0; i < len; ++i) {
        void **src = darr[i].src;
        void **dst = darr[i].dst;
        int bytes = darr[i].bytes;
        int limit = darr[i].count;
        for (int j = 0; j < limit; ++j) {
            comex_request_t h = -1;
            int rc = comex_nbacc(op, scale, src[j], dst[j], bytes, proc, group, &h);
            if (rc != COMEX_SUCCESS) return COMEX_FAILURE;
            last = h;
        }
    }
    if (nb_handle) *nb_handle = last;
    return COMEX_SUCCESS;
}

int comex_malloc_mem_dev(void **ptr_arr, size_t bytes, comex_group_t group, const char *device) {
    (void)device;
    return comex_malloc(ptr_arr, bytes, group);
}

int comex_rmw(int op, void *ploc, void *prem, int extra, int proc, comex_group_t group) {
    int world_proc;
    if (comex_group_translate_world(group, proc, &world_proc) != COMEX_SUCCESS) {
        comex_error("comex_rmw: group translation failed", -1);
        return COMEX_FAILURE;
    }

    switch (op) {
        case COMEX_FETCH_AND_ADD:
            *(int*)ploc = shmem_int_fadd((int*)prem, extra, world_proc);
            break;
        case COMEX_FETCH_AND_ADD_LONG:
            *(long*)ploc = shmem_long_fadd((long*)prem, (long)extra, world_proc);
            break;
        case COMEX_SWAP:
            *(int*)ploc = shmem_int_swap((int*)prem, *(int*)ploc, world_proc);
            break;
        case COMEX_SWAP_LONG:
            *(long*)ploc = shmem_long_swap((long*)prem, *(long*)ploc, world_proc);
            break;
        default:
            comex_error("comex_rmw: unknown op", op);
            return COMEX_FAILURE;
    }
    return COMEX_SUCCESS;
}

int comex_wait_proc(int proc, comex_group_t group) {
    (void)proc; (void)group;
    shmem_quiet();
    return COMEX_SUCCESS;
}

int comex_group_translate_world(comex_group_t group, int group_rank, int *world_rank) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!world_rank) return COMEX_FAILURE;
    *world_rank = group_rank;
    return COMEX_SUCCESS;
}
