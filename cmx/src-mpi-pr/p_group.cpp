#include <vector>

#include "p_group.hpp"
#include <cstdio>

#include "defines.hpp"

#define CMX_ASSERT(WHAT) ((void)(0))

namespace CMX {

p_Group* p_Group::p_world_group = NULL;

/**
 * Function to avoid duplication in constructors
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] mpi_comm MPI communicator that defines ranks in pid_list
 * @param[out] o_comm communicator for new group
 * @param[out] o_rank calling process rank for new group
 */
void p_Group::setup(int n, int *pid_list, MPI_Comm mpi_comm, MPI_Comm *o_comm, int *o_rank)
{
  MPI_Group wgrp;
  MPI_Group lgrp;
  int size;
  /* Get world group from comm_world */
  MPI_Comm_group(mpi_comm, &wgrp);
  int me;
  MPI_Comm_rank(mpi_comm,&me);
  MPI_Comm_size(mpi_comm,&size);
  MPI_Group_size(wgrp,&size);
  /* Create subgroup from world group */
  int status = MPI_Group_incl(wgrp, n, pid_list, &lgrp);
  {
    int grp_me;
    MPI_Comm comm, comm1, comm2;
    int lvl=1, local_ldr_pos;
    status = MPI_Group_rank(lgrp, &grp_me);
    CMX_ASSERT(MPI_SUCCESS == status);
    if (grp_me == MPI_UNDEFINED) {
      /* FIXME: keeping the group around for now */
      return;
    }
    /* SK: sanity check for the following bitwise operations */
    CMX_ASSERT(grp_me>=0);
    /* FIXME: can be optimized away */
    status = MPI_Comm_dup(MPI_COMM_SELF, &comm);
    CMX_ASSERT(MPI_SUCCESS == status);
    local_ldr_pos = grp_me;
    while(n>lvl) {
      int tag=0;
      /* ^ is bitwise XOR operation. Bit is 1 if bits are different, 0 otherwise
       * It looks like local_ldr_pos^lvl is a way of generating another
       * processor rank that can be used as a remote process. It may result in
       * remote_ldr_pos = local_ldr_pos (e.g. local_ldr_pos = 1 and lvl = 2) */
      int remote_ldr_pos = local_ldr_pos^lvl;
      /* If n is a power of 2, it looks like this condition is always satisfied.
       * May not be if n is not a power of 2 */
      if (remote_ldr_pos < n) {
        int remote_leader = pid_list[remote_ldr_pos];
        MPI_Comm peer_comm = mpi_comm;
        int high = (local_ldr_pos<remote_ldr_pos)?0:1;
        status = MPI_Intercomm_create(
            comm, 0, peer_comm, remote_leader, tag, &comm1);
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_free(&comm);
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Intercomm_merge(comm1, high, &comm2);
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_free(&comm1);
        CMX_ASSERT(MPI_SUCCESS == status);
        comm = comm2;
      }
      /* & is bitwise AND operation. Bit is 1 if both bits are 1
       * ~ operator is bitwise NOT operator. Inverts all bits,
       * so ~0 is a string of 1 bits. lvl is a power of 2, so its bitwise
       * representation is a 1 at the position represented by the power of 2 and
       * zeros everywhere else. (~0)^lvl is the inverse of this with a string of
       * 1s everywhere except at the position represented by lvl. The final
       * bitwise AND operation with local_ldr_pos */
      local_ldr_pos &= ((~0)^lvl);
      /* left shift the bits 1 space. Same as multiply by 2 */
      lvl<<=1;
    }
    *o_comm = comm;
    /* cleanup temporary group (from MPI_Group_incl above) */
    status = MPI_Group_free(&lgrp);
    CMX_ASSERT(MPI_SUCCESS == status);
    /* rank of new comm */
    int size;
    status = MPI_Comm_size(*o_comm, &size);
    CMX_ASSERT(MPI_SUCCESS == status);
    CMX_ASSERT(n == size);
    status = MPI_Comm_rank(*o_comm, &p_rank);
    CMX_ASSERT(MPI_SUCCESS == status);
  }
}

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in mpi_comm
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
p_Group::p_Group(int n, int *pid_list, MPI_Comm mpi_comm)
{
  setup(n, pid_list, mpi_comm, &p_comm, &p_rank);
  /* Assume that the first and only time this constructor is called is to create
   * the world group */
  p_size = n;
  p_world_ranks = new int[n];
  if (p_world_group != NULL) {
    int i;
    for (i=0; i<n; i++) p_world_ranks[i] = i;
  } else {
    CMX_ASSERT(0);
  }
}

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the parent group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 * @param[in] group parent group that is spawning off this group
 */
p_Group::p_Group(int n, int *pid_list, p_Group *group)
{
  MPI_Comm mpi_comm = group->MPIComm();
  setup(n, pid_list, mpi_comm, &p_comm, &p_rank);
  p_size = n;
  p_world_ranks = new int[n];
  /* Set world ranks */
  int i, w_me;
  MPI_Comm world = p_world_group->MPIComm();
  int ierr = MPI_Comm_rank(world,&w_me);
  printf("p[%d] world_rank: %d\n",p_rank,w_me);
  ierr = MPI_Allgather(&w_me,1,MPI_INT,p_world_ranks,
            1,MPI_INT,p_comm);
  printf("p[%d] ranks: %d %d %d %d\n",p_rank,p_world_ranks[0],
      p_world_ranks[1],p_world_ranks[2],p_world_ranks[3]);
}

/**
   * Simple constructor used to create the world group
    */
p_Group::p_Group()
{
}

/**
 * Destructor
 */
p_Group::~p_Group()
{
  MPI_Comm_free(&p_comm);
  if (p_world_ranks) delete [] p_world_ranks;
}

/**
 * Get world group
 * @return pointer to world group
 */
p_Group* p_Group::getWorldGroup(MPI_Comm comm)
{
  if (comm != MPI_COMM_NULL) {
    if (p_world_group == NULL) {
      p_world_group = new p_Group;
      int rank, size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      p_world_group->p_comm = comm;
      p_world_group->p_rank = rank;
      p_world_group->p_size = size;
      p_world_group->p_world_ranks = new int[size];
      int i;
      for (i=0; i<size; i++) p_world_group->p_world_ranks[i] = i;
    } else {
      CMX_ASSERT(0);
    }
  }
  return p_world_group;
}

/**
 * Return the rank of process in group
 * @return rank of calling process in group
 */
int p_Group::rank()
{
  return p_rank;
}

/**
 * Return size of group
 * @return size
 */
int p_Group::size()
{
  return p_size;
}

/**
 * Perform a barrier over all communication in the group
 * @return CMX_SUCCESS on success
 */
int p_Group::barrier()
{
  int status = MPI_Barrier(p_comm);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (status == MPI_SUCCESS) {
    return CMX_SUCCESS;
  }
  return CMX_FAILURE;
}

/**
 * Return internal MPI_Comm, if there is one
 * @return MPI communicator
 */
MPI_Comm p_Group::MPIComm()
{
  return p_comm;
}

/**
 * Set world ranks for processors in group. Assume that MPI communicator in
 * argument list corresponds to the communicator that contains all working
 * processes
 */
void p_Group::setWorldRanks(const MPI_Comm &world)
{

}

/**
 * Get the world rank of a processor in the group
 * @param rank rank of process in group
 * @return rank of process in world group
 */
int p_Group::getWorldRank(int rank)
{
  CMX_ASSERT(rank >= 0);
  CMX_ASSERT(rank < p_size);
  if (p_world_ranks) {
    return p_world_ranks[rank];
  } 
  return -1;
}

/**
 * Get a complete list of world ranks for processes in this group
 * @return list of world ranks
 */
std::vector<int> p_Group::getWorldRanks()
{
  std::vector<int> ret;
  if (p_world_ranks) {
    int i;
    for (i=0; i<p_size; i++) {
       ret.push_back(p_world_ranks[i]);
    }
  }
  return ret;
}

} // CMX namespace
