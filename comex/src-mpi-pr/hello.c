#include <assert.h>
#include <stdio.h>

#include <mpi.h>

#include "comex.h"

int main(int argc, char **argv)
{
    int rank = 0;
    int size = 0;
    void **memory = NULL;

    MPI_Init(&argc, &argv);
    comex_init();

    comex_group_size(COMEX_GROUP_WORLD, &size);
    comex_group_rank(COMEX_GROUP_WORLD, &rank);

    memory = malloc(sizeof(void*) * size);
    assert(memory);
    comex_malloc(memory, 1024*(rank+1), COMEX_GROUP_WORLD);

    comex_free(memory[rank], COMEX_GROUP_WORLD);

    comex_finalize();
    MPI_Finalize();

    return 0;
}
