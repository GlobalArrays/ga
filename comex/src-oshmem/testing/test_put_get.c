#include <stdio.h>
#include <string.h>
#include "comex.h"

int main(int argc, char **argv) {
    comex_init_args(&argc, &argv);

    int me;
    comex_group_rank(COMEX_GROUP_WORLD, &me);

    const int bytes = 16;
    char *local = (char*)comex_malloc_local(bytes);
    if (!local) {
        fprintf(stderr,"[%d] malloc_local failed\n", me);
        return 1;
    }

    for (int i=0;i<bytes;i++) local[i] = (char)(me + 1);

    if (me == 0) {
        /* put to peer 1 if exists */
        if (comex_group_size(COMEX_GROUP_WORLD, &bytes) == COMEX_SUCCESS) {
            /* best-effort: assume at least 2 PEs in run */
        }
    }

    /* simple barrier */
    comex_barrier(COMEX_GROUP_WORLD);

    comex_free_local(local);
    comex_finalize();
    return 0;
}
