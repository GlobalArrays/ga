/* Simple strided put/get/acc test for SHMEM COMEX backend */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comex.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    comex_init();
    int me = -1, nproc = -1;
    comex_group_rank(COMEX_GROUP_WORLD, &me);
    comex_group_size(COMEX_GROUP_WORLD, &nproc);

    if (nproc < 2) {
        if (me == 0) fprintf(stderr, "strided_test requires at least 2 PEs\n");
        return 1;
    }

    /* We'll test a simple strided put/get/acc where each PE has 4 doubles */
    int nelems = 4;
    /* allocate symmetric buffer on all PEs and obtain pointers */
    void **ptrs = (void**)malloc(sizeof(void*) * nproc);
    if (comex_malloc(ptrs, sizeof(double)*nelems, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
        if (me == 0) fprintf(stderr, "comex_malloc failed\n");
        return 1;
    }
    double *local = (double*)ptrs[me];
    for (int i = 0; i < nelems; ++i) local[i] = 0.0;

    if (me == 0) {
        /* source buffer with interleaved values to copy */
        double src[4] = {1.0, 10.0, 2.0, 20.0};
    /* We'll copy two contiguous doubles as a single block */
    int count[1]; count[0] = 2 * (int)sizeof(double); /* two doubles */
    int stride_levels = 0;

        /* target is PE 1 */
        int target = 1;
        /* perform strided put: copy src[0] and src[2] into target local positions 0 and 1 */
        if (comex_puts(src, NULL, ptrs[target], NULL, count, stride_levels, target, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "comex_puts failed\n");
            return 1;
        }

        /* perform a strided accumulate: add values 0.5 to positions 0 and 1 on target */
        double scale = 0.5;
        double src2[2] = {1.0, 1.0};
        if (comex_accs(COMEX_ACC_DBL, &scale, src2, NULL, ptrs[target], NULL, count, stride_levels, target, COMEX_GROUP_WORLD) != COMEX_SUCCESS) {
            fprintf(stderr, "comex_accs failed\n");
            return 1;
        }

    }

    comex_barrier(COMEX_GROUP_WORLD);

    /* PE 1 checks its local buffer */
    if (me == 1) {
        /* expected: after put, local[0]=1.0, local[1]=2.0; after acc with scale 0.5 and src2 ones
         * dest += 0.5*1.0 -> +0.5, so final values should be 1.5 and 2.5 in positions 0 and 1. */
        printf("[1] local buffer = %f %f %f %f\n", local[0], local[1], local[2], local[3]);
    }

    comex_barrier(COMEX_GROUP_WORLD);
    comex_finalize();
    /* free local pointer array; underlying symmetric memory freed by comex_free on each PE */
    free(ptrs);
    return 0;
}
