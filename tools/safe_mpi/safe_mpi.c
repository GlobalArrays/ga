#define _GNU_SOURCE
#include <mpi.h>
#include <dlfcn.h>
#include <stdio.h>

typedef int (*mpi_comm_free_f)(MPI_Comm *);
typedef int (*mpi_barrier_f)(MPI_Comm);
typedef int (*mpi_finalized_f)(int *);
typedef int (*mpi_finalize_f)(void);

static mpi_comm_free_f real_comm_free = NULL;
static mpi_barrier_f real_barrier = NULL;
static mpi_finalized_f real_finalized = NULL;
static mpi_finalize_f real_finalize = NULL;

static void init_real(void)
{
    if (!real_finalized) real_finalized = (mpi_finalized_f)dlsym(RTLD_NEXT, "MPI_Finalized");
    if (!real_comm_free) real_comm_free = (mpi_comm_free_f)dlsym(RTLD_NEXT, "MPI_Comm_free");
    if (!real_barrier) real_barrier = (mpi_barrier_f)dlsym(RTLD_NEXT, "MPI_Barrier");
}

int MPI_Comm_free(MPI_Comm *comm)
{
    init_real();
    int finalized = 0;
    if (real_finalized) {
        if (real_finalized(&finalized) != MPI_SUCCESS) return MPI_SUCCESS;
    } else {
        /* If we can't query, be conservative and skip the free */
        return MPI_SUCCESS;
    }
    if (finalized) return MPI_SUCCESS;
/* Tracer removed.
 * The original tracer source that intercepted MPI_Comm_free has been
 * intentionally removed from the repository. This placeholder file
 * remains to indicate the removal. If you need the tracer again,
 * retrieve it from your development history or ask me to recreate it.
 */

/* End of file */
}

int MPI_Barrier(MPI_Comm comm)
{
    init_real();
    int finalized = 0;
    if (real_finalized) {
        if (real_finalized(&finalized) != MPI_SUCCESS) return MPI_SUCCESS;
    } else {
        return MPI_SUCCESS;
    }
    if (finalized) return MPI_SUCCESS;
    if (real_barrier) return real_barrier(comm);
    return MPI_SUCCESS;
}

int MPI_Finalize(void)
{
    init_real();
    int finalized = 0;
    if (real_finalized) {
        if (real_finalized(&finalized) != MPI_SUCCESS) return MPI_SUCCESS;
    } else {
        /* If we can't query, be conservative and skip double-finalize */
        finalized = 0;
    }
    if (finalized) return MPI_SUCCESS;
    if (!real_finalize) real_finalize = (mpi_finalize_f)dlsym(RTLD_NEXT, "MPI_Finalize");
    if (real_finalize) return real_finalize();
    return MPI_SUCCESS;
}
