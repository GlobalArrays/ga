/* cmx header file */
#ifndef _CMX_COMMON_GROUP_H
#define _CMX_COMMON_GROUP_H

#include <mpi.h>

#include "cmx_impl.hpp"

namespace CMX {

class Group {

public:

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
Group(int n, int *pid_list, Group *group);

/**
 * Destructor
 */
~Group();

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
 * Construct a group directly from existing p_Group. This is needed for the
 * world group
 * @param[in] group pointer to an implementation instance
 */
Group(p_Group *group);

protected:

/**
 * Constructor for new group derived from an MPI communicator. This
 * constructor assumes that all processes ranks refer to processes in
 * the MPI communicator. It is designed to set up the world group.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] comm MPI commuicator that defines ranks in pid_list
 */
Group(int n, int *pid_list, MPI_Comm comm);

private:

p_Group *p_group;
};

} // namespace CMX

#endif /* _CMX_GROUP_H */
