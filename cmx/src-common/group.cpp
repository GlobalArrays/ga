#include "group.hpp"
#include "cmx_impl.hpp"

namespace CMX {

/**
 * Constructor for new group derived from an MPI communicator. This
 * constructor assumes that all processes ranks refer to processes in
 * the MPI communicator. It is designed to set up the world group.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] comm MPI commuicator that defines ranks in pid_list
 */
Group::Group(int n, int *pid_list, MPI_Comm comm)
{
  p_group = new p_Group(n, pid_list, comm);
}

/**
 * Construct a group directly from existing p_Group. This is needed for the
 * world group
 * @param[in] group pointer to an implementation instance
 */
Group::Group(p_Group* group)
{
  p_group = group;
}

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
Group::Group(int n, int *pid_list, Group *group)
{
  p_group = new p_Group(n, pid_list, group->p_group);
}

/**
 * Destructor
 */
Group::~Group()
{
}

/**
 * Return the rank of process in group
 * @return rank of calling process in group
 */
int Group::rank()
{
  return p_group->rank();
}

/**
 * Return size of group
 * @return size
 */
int Group::size()
{
  return p_group->size();
}

/**
 * Perform a barrier over all communication in the group
 * @return CMX_SUCCESS on success
 */
int Group::barrier()
{
  p_group->barrier();
}

/**
 * Return internal MPI_Comm, if there is one
 * @return MPI communicator
 */
MPI_Comm Group::MPIComm()
{
  return p_group->MPIComm();
}

} // CMX namespace
