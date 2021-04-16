/* cmx header file */
#ifndef _CMX_P_GROUP_H
#define _CMX_P_GROUP_H

#include <mpi.h>

namespace CMX {

class p_Group {

public:

/**
 * Constructor for new group derived from an MPI communicator. This
 * constructor assumes that all processes ranks refer to processes in
 * the MPI communicator. It is designed to set up the world group.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] comm MPI commuicator that defines ranks in pid_list
 */
p_Group(int n, int *pid_list, MPI_Comm comm);

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
p_Group(int n, int *pid_list, p_Group *group);

/**
 * Destructor
 */
~p_Group();

/**
 * Return the rank of process in group
 * @return rank of calling process in group
 */
int rank();

/**
 * Return size of group
 * @return size
 */
int size();

/**
 * Perform a barrier over all communication in the group
 * @return CMX_SUCCESS on success
 */
int barrier();

/**
 * Return internal MPI_Comm, if there is one
 * @return MPI communicator
 */
MPI_Comm MPIComm();

/**
 * Set world ranks for processors in group. Assume that MPI communicator in
 * argument list corresponds to the communicator that contains all working
 * processes
 */
void setWorldRanks(const MPI_Comm &world);

/**
 * Get the world rank of a processor in the group
 * @param rank rank of process in group
 * @return rank of process in world group
 */
int getWorldRank(int rank);

private:

/**
 * Function to avoid duplication in constructors
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] mpi_comm MPI commuicator that defines ranks in pid_list
 * @param[out] comm communicator for new group
 * @param[out] rank calling process rank for new group
 */
void setup(int n, int *pid_list, MPI_Comm mpi_comm, MPI_Comm *comm, int *rank);

MPI_Comm p_comm;
int p_rank;
int p_size;
int *p_world_ranks;

static p_Group *p_world_group;
};

} // namespace CMX

#endif /* _CMX_P_GROUP_H */
