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

    /* create some mutexes */
    comex_create_mutexes(64);

    /* allocate symmetric region: one double per PE (max size = 8 bytes) */
    void **ptrs = malloc(sizeof(void*) * npes);
    if (!ptrs) { fprintf(stderr, "[%d] malloc failed\n", me); comex_finalize(); return 1; }

    if (comex_malloc(ptrs, sizeof(double), COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
        fprintf(stderr, "[%d] comex_malloc failed\n", me);
        comex_finalize(); return 1;
    }

    /* initialize local symmetric buffer to 0 */
    double *mybuf = (double*)ptrs[me];
    *mybuf = 0.0;
    double scale = 1.0;

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 0) {
        double val = 3.5;
        if (comex_acc(COMEX_ACC_DBL, &scale, &val, ptrs[1], sizeof(double), 1, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_acc failed\n");
        } else {
            fprintf(stderr, "[0] issued blocking acc to PE 1\n");
        }
    }

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 1) {
        double res = *mybuf;
        printf("[1] after blocking acc, value = %f (expected 3.5)\n", res);
    }

    comex_barrier(COMEX_GROUP_WORLD);

    /* Now test nonblocking accumulate from PE 0 to PE 1 */
    if (me == 0) {
        double val = 2.25;
        comex_request_t h = -1;
        if (comex_nbacc(COMEX_ACC_DBL, &scale, &val, ptrs[1], sizeof(double), 1, COMEX_GROUP_WORLD, &h) != COMEX_SUCCESS) {
            fprintf(stderr, "[0] comex_nbacc failed\n");
        } else {
            fprintf(stderr, "[0] issued nbacc handle=%d\n", h);
            comex_wait(&h);
            fprintf(stderr, "[0] wait returned for nbacc\n");
        }
    }

    comex_barrier(COMEX_GROUP_WORLD);

    if (me == 1) {
        double res = *mybuf;
        printf("[1] after nbacc, value = %f (expected 3.5 + 2.25 = 5.75)\n", res);
    }

    comex_barrier(COMEX_GROUP_WORLD);
    comex_free(ptrs[me], COMEX_GROUP_WORLD);
    free(ptrs);

    comex_finalize();
    return 0;
}
