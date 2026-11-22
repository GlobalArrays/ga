#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comex.h"
#include "comex_impl.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    comex_init();

    int me = -1, npes = 0;
    comex_group_rank(COMEX_GROUP_WORLD, &me);
    comex_group_size(COMEX_GROUP_WORLD, &npes);

    if (npes < 2) {
        if (me == 0) fprintf(stderr, "put_get_test requires at least 2 PEs\n");
        comex_finalize();
        return 1;
    }

    void **ptrs = malloc(sizeof(void*) * npes);
    if (!ptrs) { if (me==0) fprintf(stderr, "malloc failed\n"); comex_finalize(); return 1; }

    /* allocate (me+1) doubles on each PE so sizes vary by rank */
    size_t my_elems = (size_t)(me + 1);
    if (comex_malloc(ptrs, my_elems * sizeof(double), COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
        if (me==0) fprintf(stderr, "comex_malloc failed\n"); comex_finalize(); return 1;
    }

    double *mybuf = (double*)ptrs[me];
    for (size_t i = 0; i < my_elems; ++i) mybuf[i] = 0.0;

    /* Test comex_put: PE 0 writes to PE 1's buffer */
    comex_barrier(COMEX_GROUP_WORLD);
    if (me == 0) {
        int target = 1;
        size_t target_elems = (size_t)(target + 1);
        size_t bytes = target_elems * sizeof(double);
        double *src_buf = (double*)malloc(bytes);
        for (size_t i = 0; i < target_elems; ++i) src_buf[i] = 42.5;
        if (comex_put(src_buf, ptrs[target], (int)bytes, 1, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_put failed\n");
        } else {
            fprintf(stderr, "[0] issued comex_put to PE 1 (bytes=%zu)\n", bytes);
        }
        free(src_buf);
    }
    comex_barrier(COMEX_GROUP_WORLD);
    if (me == 1) {
        printf("[1] after comex_put, buffer =");
        for (size_t i = 0; i < my_elems; ++i) printf(" %f", mybuf[i]);
        printf(" (expected all 42.5)\n");
    }

    /* Test comex_get: PE 0 fetches from PE 1 into a local variable */
    if (me == 1) {
        /* set remote values across the buffer */
        for (size_t i = 0; i < my_elems; ++i) mybuf[i] = 7.25 + (double)i;
    }
    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 0) {
        int target = 1;
        size_t target_elems = (size_t)(target + 1);
        size_t bytes = target_elems * sizeof(double);
        double *r = (double*)malloc(bytes);
        if (comex_get(ptrs[target], r, (int)bytes, 1, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_get failed\n");
        } else {
            printf("[0] after comex_get, fetched =");
            for (size_t i = 0; i < target_elems; ++i) printf(" %f", r[i]);
            printf(" (expected 7.25 .. 7.25+N)\n");
        }
        free(r);
    }

    comex_barrier(COMEX_GROUP_WORLD);

    comex_free(ptrs[me], COMEX_GROUP_WORLD);
    free(ptrs);
    comex_finalize();
    return 0;
}
