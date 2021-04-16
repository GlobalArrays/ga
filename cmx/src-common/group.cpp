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
CMX_Group::CMX_Group(int n, int *pid_list, MPI_Comm comm)
{
  p_group = new p_Group(n, pid_list, comm);
}

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
CMX_Group::CMX_Group(int n, int *pid_list, CMX_Group *group)
{
  p_group = new p_Group(n, pid_list, group->p_group);
}

/**
 * Destructor
 */
CMX_Group::~CMX_Group()
{
  delete p_group;
}

/**
 * Return the rank of process in group
 * @return rank of calling process in group
 */
int CMX_Group::rank()
{
  return p_group->rank();
}

/**
 * Return size of group
 * @return size
 */
int CMX_Group::size()
{
  return p_group->size();
}

/**
 * Perform a barrier over all communication in the group
 * @return CMX_SUCCESS on success
 */
int CMX_Group::barrier()
{
  p_group->barrier();
}

/**
 * Return internal MPI_Comm, if there is one
 * @return MPI communicator
 */
MPI_Comm CMX_Group::MPIComm()
{
  return p_group->MPIComm();
}

} // CMX namespace
