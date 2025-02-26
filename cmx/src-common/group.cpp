#include "group.hpp"

namespace CMX {

Group* Group::p_world_group = NULL;

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
  p_new_group = true;
  p_group = new p_Group(n, pid_list, comm);
}

/**
 * Construct a group directly from existing p_Group. This is needed for the
 * world group
 * @param[in] group pointer to an implementation instance
 */
Group::Group(p_Group* group)
{
  p_new_group = false;
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
  p_new_group = true;
  p_group = new p_Group(n, pid_list, group->p_group);
}

/**
 * Destructor
 */
Group::~Group()
{
  if (p_new_group) delete p_group;
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
  return p_group->barrier();
}

/**
 * Return internal MPI_Comm, if there is one
 * @return MPI communicator
 */
MPI_Comm Group::MPIComm()
{
  return p_group->MPIComm();
}

/**
  * Get world group
  * @return pointer to world group
  */
Group* Group::getWorldGroup(MPI_Comm comm = MPI_COMM_NULL)
{
  p_world_group = new Group(p_Group::getWorldGroup(comm));
  return p_world_group;
}

/**
 * Get world group.
 * @return pointer to world group
 */
Group* Group::getWorldGroup()
{
  return p_world_group;
}

/**
 * Get the world rank of a processor in the group
 * @param rank rank of process in group
 * @return rank of process in world group
 */
int Group::getWorldRank(int rank)
{
  return p_group->getWorldRank(rank);
}

/**
 *  * Get  world ranks of all processors in the group
 *   * @return list of ranks in the world group
 *    */
std::vector<int> Group::getWorldRanks()
{
  return p_group->getWorldRanks();
}

} // CMX namespace
