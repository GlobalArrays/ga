#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comex.h"
#include "comex_impl.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    comex_init();
    int me = 0;
    int npes = 0;
    comex_group_rank(COMEX_GROUP_WORLD, &me);
    comex_group_size(COMEX_GROUP_WORLD, &npes);

    if (npes < 2) {
        if (me == 0) fprintf(stderr, "Test requires at least 2 PEs\n");
        comex_finalize();
        return 1;
    }

    /* mutexes handled internally by comex_acc; no test-level mutex creation needed */

    /* allocate symmetric region: one double per PE (max size = 8 bytes) */
    void **ptrs = malloc(sizeof(void*) * npes);
    if (!ptrs) { fprintf(stderr, "[%d] malloc failed\n", me); comex_finalize(); return 1; }

    /* allocate (me+1) doubles on each PE so sizes vary by rank */
    size_t my_elems = (size_t)(me + 1);
    if (comex_malloc(ptrs, my_elems * sizeof(double), COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
        fprintf(stderr, "[%d] comex_malloc failed\n", me);
        comex_finalize(); return 1;
    }

    /* initialize local symmetric buffer to 0 for all elements */
    double *mybuf = (double*)ptrs[me];
    for (size_t i = 0; i < my_elems; ++i) mybuf[i] = 0.0;
    double scale = 1.0;

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 0) {
        double val = 3.5;
        int target = 1;
        size_t target_elems = (size_t)(target + 1);
        size_t bytes = target_elems * sizeof(double);
        /* prepare a src buffer with target_elems copies of val */
        double *src_buf = (double*)malloc(bytes);
        for (size_t i = 0; i < target_elems; ++i) src_buf[i] = val;
        if (comex_acc(COMEX_ACC_DBL, &scale, src_buf, ptrs[target], (int)bytes, 1, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_acc failed\n");
        } else {
            fprintf(stderr, "[0] issued blocking acc to PE %d (bytes=%zu)\n", target, bytes);
        }
        free(src_buf);
    }

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 1) {
        printf("[1] after blocking acc, buffer =");
        for (size_t i = 0; i < my_elems; ++i) printf(" %f", mybuf[i]);
        printf(" (expected all 3.5)\n");
    }

    comex_barrier(COMEX_GROUP_WORLD);

    /* Now test nonblocking accumulate from PE 0 to PE 1 */
    if (me == 0) {
        double val = 2.25;
        int target = 1;
        size_t target_elems = (size_t)(target + 1);
        size_t bytes = target_elems * sizeof(double);
        double *src_buf = (double*)malloc(bytes);
        for (size_t i = 0; i < target_elems; ++i) src_buf[i] = val;
        comex_request_t h = -1;
        if (comex_nbacc(COMEX_ACC_DBL, &scale, src_buf, ptrs[target], (int)bytes, 1, COMEX_GROUP_WORLD, &h) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_nbacc failed\n");
        } else {
            fprintf(stderr, "[0] issued nbacc handle=%d\n", h);
            comex_wait(&h);
            fprintf(stderr, "[0] wait returned for nbacc\n");
        }
        free(src_buf);
    }

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 1) {
        printf("[1] after nbacc, buffer =");
        for (size_t i = 0; i < my_elems; ++i) printf(" %f", mybuf[i]);
        printf(" (expected all 5.75)\n");
    }

    comex_barrier(COMEX_GROUP_WORLD);
    comex_free(ptrs[me], COMEX_GROUP_WORLD);
    free(ptrs);

    comex_finalize();
    return 0;
}
