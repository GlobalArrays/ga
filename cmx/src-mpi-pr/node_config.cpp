#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include "node_config.hpp"

/* This needs to be filled in */
#define CMX_ASSERT(WHAT)

/* cmx header file */
namespace CMX {

/**
 * Simple constructor
 */
p_NodeConfig::p_NodeConfig()
{
}

/* Utility function used by qsort */
static int cmplong(const void *p1, const void *p2)
{
  return *((long*)p1) - *((long*)p2);
}

/**
 * Initialize all functionality in p_NodeConfig object
 * @param comm MPI communicator that acts as world for application
 */
void p_NodeConfig::init(MPI_Comm comm)
{
  int status = 0;
  int i = 0;
  int smallest_rank_with_same_hostid = 0;
  int largest_rank_with_same_hostid = 0;
  int size_node = 0;
  long *sorted = NULL;
  int count = 0;
  Group *world;

  int size, rank;
  MPI_Comm_size(comm,&rank);
  MPI_Comm_size(comm,&size);
  /* dup comm and get group, rank, and size */
  status = MPI_Comm_dup(comm, &(g_state.comm));
  CMX_ASSERT(MPI_SUCCESS == status);
  status = MPI_Comm_group(g_state.comm, &(g_state.group));
  CMX_ASSERT(MPI_SUCCESS == status);
  status = MPI_Comm_rank(g_state.comm, &(g_state.rank));
  CMX_ASSERT(MPI_SUCCESS == status);
  status = MPI_Comm_size(g_state.comm, &(g_state.size));
  CMX_ASSERT(MPI_SUCCESS == status);
  world = Group::getWorldGroup(g_state.comm);
  
  /* need to figure out which proc is master on each node */
  g_state.hostid = new long[g_state.size];
  g_state.hostid[g_state.rank] = xgethostid();
  status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
      g_state.hostid, 1, MPI_LONG, g_state.comm);
  CMX_ASSERT(MPI_SUCCESS == status);
  /* First create a temporary node communicator and then
   *       * split further into number of gruoups within the node */
  MPI_Comm temp_node_comm;
  int temp_node_size;
  /* create node comm */
  /* MPI_Comm_split requires a non-negative color, so sort and sanitize */
  sorted = new long[g_state.size];
  (void)memcpy(sorted, g_state.hostid, sizeof(long)*g_state.size);
  qsort(sorted, g_state.size, sizeof(long), cmplong);
  for (i=0; i<g_state.size-1; ++i) {
    if (sorted[i] == g_state.hostid[g_state.rank])
    {
      break;
    }
    if (sorted[i] != sorted[i+1]) {
      count += 1;
    }
  }
  delete [] sorted;
  /* split up into node communicators */
  status = MPI_Comm_split(comm, count,
      g_state.rank, &temp_node_comm);
  MPI_Comm_size(comm,&size);
  int node_group_size, node_group_rank;
  MPI_Comm_size(temp_node_comm, &node_group_size);
  MPI_Comm_rank(temp_node_comm, &node_group_rank);
  g_state.node_size = node_group_size;
  g_state.node_rank = node_group_rank;
  int node_rank0;
  node_rank0 = (node_group_rank == 0) ? 1 : 0;
  MPI_Allreduce(&node_rank0, &p_num_nodes, 1, MPI_INT, MPI_SUM,
      g_state.comm);
  smallest_rank_with_same_hostid = g_state.rank;
  largest_rank_with_same_hostid = g_state.rank;
  for (i=0; i<g_state.size; ++i) {
    if (g_state.hostid[i] == g_state.hostid[g_state.rank]) {
      ++size_node;
      if (i < smallest_rank_with_same_hostid) {
        smallest_rank_with_same_hostid = i;
      }
      if (i > largest_rank_with_same_hostid) {
        largest_rank_with_same_hostid = i;
      }
    }
  }

  /* Get number of Progress-Ranks per node from environment variable
   *      * equal to 1 by default */
  int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
  /* Perform check on the number of Progress-Ranks */
  if (size_node < 2 * num_progress_ranks_per_node) {
//    cmx_error("ranks per node, must be at least",
//        2 * num_progress_ranks_per_node);
  }
  if (size_node % num_progress_ranks_per_node > 0) {
//    cmx_error("number of ranks per node must be multiple"
//        " of number of process groups per node", -1);
  }
  int is_node_ranks_packed = get_progress_rank_distribution_on_node();
  int split_group_size;
  split_group_size = node_group_size / num_progress_ranks_per_node;
//  MPI_Comm_free(&temp_node_comm);
  g_state.node_comm = temp_node_comm;
  g_state.master = new int[g_state.size];
  g_state.master[g_state.rank] 
    = this->get_my_master_rank_with_same_hostid(g_state.rank, 
      split_group_size, world);
  status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
      g_state.master, 1, MPI_INT, g_state.comm);
  
  CMX_ASSERT(MPI_SUCCESS == status);
}

/**
 * Simple destructor
 */
p_NodeConfig::~p_NodeConfig()
{
  delete [] g_state.hostid;
  delete [] g_state.master;
}

/**
 * Get rank on world group for rank in group
 * @param[in] group group on which rank is defined
 * @param[in] rank rank of process in group
 * @return rank of process in world group
 */
int p_NodeConfig::get_world_rank(Group *group, int rank)
{
  MPI_Group grp;
  MPI_Group wgrp;
  int ret;
  MPI_Comm_group(group->MPIComm(),&grp);
  MPI_Comm_group(g_state.comm,&wgrp);
  MPI_Group_translate_ranks(grp,1,&rank,wgrp,&ret);
  return ret;
}

/**
 * Return world ranks for all ranks in group
 * @param[in] group group for which ranks are requested
 * @return list of world group ranks for all ranks in group
 */
std::vector<int> p_NodeConfig::get_world_ranks(Group *group)
{
  int size = group->size();
  int i = 0;
  std::vector<int> world_ranks(size);
  for (i=0; i<size; ++i) {
    world_ranks[i] = get_world_rank(group,i);
  }
  return world_ranks;
}

/**
 * Return smallest rank in group with same host ID as
 * calling process
 * @param[in] group group for which smallest rank is requested
 * @return smallest rank with same host ID
 */
int p_NodeConfig::smallest_world_rank_with_same_hostid(Group *group)
{
#if 0
  int i = 0;
  int smallest = world_rank;

  long my_hostid = xgethostid();
  for (i=0; i<g_state.size; ++i) {
    if (g_state.hostid[i] == my_hostid) {
      /* found same host as me */
      int wrank = get_world_rank(group, i);
      if (wrank < smallest) {
        smallest = wrank;
      }
    }
  }
  return smallest;
#else
  MPI_Comm comm = group->MPIComm();
  int my_world_rank = group->getWorldRank(group->rank());
  std::vector<int> world_ranks = get_world_ranks(group);
  return smallest_world_rank_with_same_hostid(comm, my_world_rank, world_ranks);
#endif

}

/**
 * Return smallest rank in group with same host ID as
 * calling process
 * @param[in] comm communicator for which smallest rank is requested
 * @param[in] world_rank world rank of calling process
 * @param[in] world_ranks list of world ranks of all processes in comm
 * @return smallest rank with same host ID
 */
int p_NodeConfig::smallest_world_rank_with_same_hostid(MPI_Comm comm, int world_rank,
    std::vector<int> &world_ranks)
{
  int i = 0;

  long my_hostid = g_state.hostid[world_rank];
  int smallest = my_hostid;
  for (i=0; i<world_ranks.size(); ++i) {
    if (g_state.hostid[world_ranks[i]] == my_hostid) {
      /* found same host as me */
      if (world_ranks[i] < smallest) {
        smallest = world_ranks[i];
      }
    }
  }

  return smallest;
}

/**
 * Return largest rank in group with same host ID as
 * calling process
 * @param[in] group group for which largest rank is requested
 * @return largest rank with same host ID
 */
int p_NodeConfig::largest_world_rank_with_same_hostid(Group *group)
{
  MPI_Comm comm = group->MPIComm();
  int my_world_rank = group->getWorldRank(group->rank());
  std::vector<int> world_ranks = get_world_ranks(group);
  return largest_world_rank_with_same_hostid(comm, my_world_rank, world_ranks);
}

/**
 * Return largest rank in group with same host ID as
 * calling process
 * @param[in] group group for which largest rank is requested
 * @param[in] world_rank world rank of calling process
 * @param[in] world_ranks list of world ranks of all processes in comm
 * @return largest rank with same host ID
 */
int p_NodeConfig::largest_world_rank_with_same_hostid(MPI_Comm comm, int world_rank,
    std::vector<int> &world_ranks)
{
  int i = 0;

  long my_hostid = g_state.hostid[world_rank];
  int largest = g_state.rank;
  for (i=0; i<world_ranks.size(); ++i) {
    if (g_state.hostid[world_ranks[i]] == my_hostid) {
      /* found same host as me */
      if (world_ranks[i] > largest) {
        largest = world_ranks[i];
      }
    }
  }

  return largest;
}

/**
 * Wrapper for function that returns a unique host ID for all processes
 * that share a node
 * @return unique host ID for node
 */
long p_NodeConfig::xgethostid()
{
  long nodeid = gethostid();
  return nodeid;
}

/**
 * Return the number of progress ranks for a node
 * @return number of progress ranks
 */
int p_NodeConfig::get_num_progress_ranks_per_node()
{
  int num_progress_ranks_per_node;
  const char* num_progress_ranks_env_var
    = getenv("GA_NUM_PROGRESS_RANKS_PER_NODE");
  if (num_progress_ranks_env_var != NULL &&
      num_progress_ranks_env_var[0] != '\0') {
    int env_number = atoi(getenv("GA_NUM_PROGRESS_RANKS_PER_NODE"));
    if ( env_number > 0 && env_number < 16)
      num_progress_ranks_per_node = env_number;
#if DEBUG
    printf("num_progress_ranks_per_node: %d\n", num_progress_ranks_per_node);
#endif
  }
  else {
    num_progress_ranks_per_node = 1;
  }
  return num_progress_ranks_per_node;
}

/**
 * Return distribution of progress ranks on node
 * @return this function returns the values
 *   PACKED: ranks are assigned consecutively to each progress rank
 *   CYCLIC: ranks are assigned in round-robin to each progress rank
 * @return true if distribution is packed
 */
int p_NodeConfig::get_progress_rank_distribution_on_node()
{
  const char* progress_ranks_packed_env_var
    = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_PACKED");
  const char* progress_ranks_cyclic_env_var
    = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC");
  int is_node_ranks_packed;
  int rank_packed =0;
  int rank_cyclic =0;

  if (progress_ranks_packed_env_var != NULL &&
      progress_ranks_packed_env_var[0] != '\0') {
    if (strchr(progress_ranks_packed_env_var, 'y') != NULL ||
        strchr(progress_ranks_packed_env_var, 'Y') != NULL ||
        strchr(progress_ranks_packed_env_var, '1') != NULL ) {
      rank_packed = 1;
    }
  }
  if (progress_ranks_cyclic_env_var != NULL &&
      progress_ranks_cyclic_env_var[0] != '\0') {
    if (strchr(progress_ranks_cyclic_env_var, 'y') != NULL ||
        strchr(progress_ranks_cyclic_env_var, 'Y') != NULL ||
        strchr(progress_ranks_cyclic_env_var, '1') != NULL ) {
      rank_cyclic = 1;
    }
  }
  if (rank_packed == 1 || rank_cyclic == 0) is_node_ranks_packed = 1;
  if (rank_packed == 0 && rank_cyclic == 1) is_node_ranks_packed = 0;
  return is_node_ranks_packed;
}


/**
 * Find master rank for all processes with same host ID
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] group pointer to group containing calling process
 * @return world group rank of master process
 */
int p_NodeConfig::get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    Group *group)
{
  MPI_Comm comm = group->MPIComm();
  int my_world_rank = group->getWorldRank(rank);
  std::vector<int> world_ranks = get_world_ranks(group);
  return get_my_master_rank_with_same_hostid(rank, split_group_size, comm,
      my_world_rank, world_ranks);
}

/**
 * Find master rank for all processes with same host ID on the world group.
 * This function returns the global rank of the progress rank for the processor
 * corresponding to rank
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] comm MPI communicator over which CMX is initialized
 * @param[in] world_rank world rank of calling process
 * @param[in] world_ranks list of world ranks of all processes in comm
 * @return world group rank of master process
 */
int p_NodeConfig::get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    MPI_Comm comm, int world_rank, std::vector<int> &world_ranks)
{
  int my_master;
  int smallest_rank_with_same_hostid = this->smallest_world_rank_with_same_hostid(comm,
      world_rank, world_ranks);
  int largest_rank_with_same_hostid = this->largest_world_rank_with_same_hostid(comm,
      world_rank, world_ranks);
  int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
  int is_node_ranks_packed = get_progress_rank_distribution_on_node();

#if MASTER_IS_SMALLEST_SMP_RANK
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_master = smallest_rank_with_same_hostid
      + split_group_size *
      ((rank - smallest_rank_with_same_hostid)/split_group_size);
  } else {
    if(num_progress_ranks_per_node == 1) {
      my_master = 2 * (split_group_size *
          ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a
       * node between two sockets
       *          * with even and odd
       *          numbering  */
      if(rank % 2 == 0) {
        my_master = 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        my_master = 1 + 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      }
    }
  }
#else
  /* By default creates largest SMP rank as Master */
  if(is_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_master = largest_rank_with_same_hostid
      - split_group_size *
      ((largest_rank_with_same_hostid - rank)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) {
      my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
          ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node
       * between two sockets
       *          * with even and odd
       *          numbering  */
      if(rank % 2 == 0) {
        my_master = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
        my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      }
    }
  }
#endif
  return my_master;
}

/**
 * Find notifier rank that notifies progress rank when shutting down
 * environment
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] group pointer to group containing calling process
 * @return notifier rank
 */
int p_NodeConfig::get_my_rank_to_free(int rank, int split_group_size, Group *group)
{
  int my_rank_to_free;
  int largest_rank_with_same_hostid = this->largest_world_rank_with_same_hostid(group);
  int smallest_rank_with_same_hostid = this->smallest_world_rank_with_same_hostid(group);
  int are_node_ranks_packed = get_progress_rank_distribution_on_node();

#if MASTER_IS_SMALLEST_SMP_RANK
  /* By default creates largest SMP rank as Master */
  if(are_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_rank_to_free = largest_rank_with_same_hostid
      - split_group_size *
      ((largest_rank_with_same_hostid - rank)/split_group_size);
  }
  else {
    if(num_progress_ranks_per_node == 1) { 
      my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
          ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_rank_to_free = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
        my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
            ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      }
    }
  }
#else
  if(are_node_ranks_packed) {
    /* Contiguous packing of ranks on a node */
    my_rank_to_free = smallest_rank_with_same_hostid
      + split_group_size *
      ((rank - smallest_rank_with_same_hostid)/split_group_size);
  }
  else {
    if(get_num_progress_ranks_per_node() == 1) { 
      my_rank_to_free = 2 * (split_group_size *
          ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
    } else {
      /* Cyclic packing of ranks on a node between two sockets
       * with even and odd numbering  */
      if(rank % 2 == 0) {
        my_rank_to_free = 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        my_rank_to_free = 1 + 2 * (split_group_size *
            ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      }
    }
  }
#endif
  return my_rank_to_free;
}

/**
 * Return rank of calling process in world group
 * @return rank of calling process
 */
int p_NodeConfig::rank()
{
  return g_state.rank;
}

/**
 * Return total number of ranks in world (including progress ranks)
 * @return size of world group
 */
int p_NodeConfig::world_size()
{
  return g_state.size;
}

/**
 * Number of ranks on the node. Actually, the number of ranks served by
 * one progress rank (plus the progress rank itself).
 * @return number of ranks on the node
 */
int p_NodeConfig::node_size()
{
  return g_state.node_size;
}

/**
 * return master proc for proc iproc
 * @param iproc process
 * @return master process for iproc
 */
int p_NodeConfig::master(int iproc)
{
  return g_state.master[iproc];
}

/**
  Calling process is a progress rank
 * @return true if progress rank
 */
bool p_NodeConfig::is_master()
{
  return (g_state.rank == g_state.master[g_state.rank]);
}

/**
   * Number of nodes
    * @return total number of nodes in system
     */
int p_NodeConfig::num_nodes()
{
  return p_num_nodes;
}

/**
 * Return host id for node hosting process
 * @param proc world rank of process
 * @return node host id
 */
long p_NodeConfig::hostid(int proc)
{
  return g_state.hostid[proc];
}

/**
   * Return rank of calling process on node communicator
    * @return node rank
     */
int p_NodeConfig::node_rank()
{
  return g_state.node_rank;
}

/**
 * Return the global communicator for all processes in the system
 * (including progress ranks)
 * @return global communicator
 */
MPI_Comm p_NodeConfig::global_comm()
{
  return g_state.comm;
}

/**
 * Return a communicator for all processes within a single SMP node
 * @return node communicator
 */
MPI_Comm p_NodeConfig::node_comm()
{
  return g_state.node_comm;
}

} // namespace CMX
