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

/**
 * Return the rank of this processor on communicator comm
 * @param comm communicator for which rank is desired
 * @return rank of process on communicator comm
 */
int MPI_Wrapper_comm_rank(MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

/* Abort job with error code
 * err: error code
 */
void MPI_Wrapper_abort(int err)
{
  MPI_Abort(MPI_COMM_WORLD,err);
}
