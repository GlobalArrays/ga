#include <mpi.h>

/**
 * Wrappers for some MPI functions to avoid compilation issue on some platform
 * that cannot find mpi.h using GPU compiler wrappers
 */

/* Return the rank of this processor on MPI_COMM_WORLD */
int MPI_Wrapper_world_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

/* Abort job with error code
 * err: error code
 */
void MPI_Wrapper_abort(int err)
{
  MPI_Abort(MPI_COMM_WORLD,err);
}
