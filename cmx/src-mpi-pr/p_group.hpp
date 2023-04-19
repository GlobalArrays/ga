/* cmx header file */
#ifndef _CMX_P_GROUP_H
#define _CMX_P_GROUP_H

#include <vector>
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
 * Get world group. This function is used for converting an MPI communicator,
 * provided by the environmental initialization, into a group
 * @param[in] comm MPI communicator defining the world group
 * @return pointer to world group
 */
static p_Group* getWorldGroup(MPI_Comm comm = MPI_COMM_NULL);

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

/**
 * Get a complete list of world ranks for processes in this group
 * @return list of world ranks
 */
std::vector<int> getWorldRanks();

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

/**
 * Simple constructor used to create the world group
 */
p_Group();

MPI_Comm p_comm;
int p_rank;
int p_size;
int *p_world_ranks;

static p_Group *p_world_group;
};

} // namespace CMX

#endif /* _CMX_P_GROUP_H */
