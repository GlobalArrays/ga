/* cmx header file */
#ifndef _CMX_P_NODE_CONFIG_H
#define _CMX_P_NODE_CONFIG_H

#include <mpi.h>
#include <vector>
#include "cmx_group.hpp"

namespace CMX {

enum config{PACKED, CYCLIC};

typedef struct {
  MPI_Comm comm;       /* world communicator, all ranks */
  MPI_Group group;     /* world group, all ranks */
  int size;            /* world comm size */
  int rank;            /* global rank */
  int *master;         /* master rank for each rank */
  long *hostid;        /* hostid hostid of SMP node for each rank */
  MPI_Comm node_comm;  /* node communicator */
  int node_size;       /* node communicator size */
  int node_rank;       /* node communicator rank */
} cmx_global_config_t;

class p_NodeConfig {

public:

/**
 * Simple constructor
 */
p_NodeConfig();

/**
 * Simple destructor
 */
~p_NodeConfig();

/**
 * Initialize all functionality in p_NodeConfig object
 * @param comm MPI communicator that acts as world for application
 */
void init(MPI_Comm comm);

/**
 * Get rank on world group for rank in group
 * @param[in] group group on which rank is defined
 * @param[in] rank rank of process in group
 * @return rank of process in world group
 */
int get_world_rank(Group *group, int rank);

/**
 * Return world ranks for all ranks in group
 * @param[in] group group for which ranks are requested
 * @return list of world group ranks for all ranks in group
 */
std::vector<int> get_world_ranks(Group *group);

/**
 * Return smallest rank in group with same host ID as
 * calling process
 * @param[in] group group for which smallest rank is requested
 * @return smallest rank with same host ID
 */
int smallest_world_rank_with_same_hostid(Group *group);

/**
 * Return smallest rank in group with same host ID as
 * calling process
 * @param[in] comm MPI communicator for which smallest rank is requested
 * @param[in] world_rank world rank of calling process
 * @param[in] world_ranks list of world ranks of all processes in comm
 * @return smallest rank with same host ID
 */
int smallest_world_rank_with_same_hostid(MPI_Comm comm, int world_rank,
    std::vector<int> &world_ranks);

/**
 * Return largest rank in group with same host ID as
 * calling process
 * @param[in] group group for which largest rank is requested
 * @return largest rank with same host ID
 */
int largest_world_rank_with_same_hostid(Group *group);

/**
 * Return largest rank in group with same host ID as
 * calling process
 * @param[in] comm MPI communicator for which largest rank is requested
 * @param[in] world_rank world rank of calling process
 * @param[in] world_ranks list of world ranks of all processes in comm
 * @return largest rank with same host ID
 */
int largest_world_rank_with_same_hostid(MPI_Comm comm, int world_rank,
    std::vector<int> &work_ranks);

/**
 * Wrapper for function that returns a unique host ID for all processes
 * that share a node
 * @return unique host ID for node
 */
long xgethostid();

/**
 * Return the number of progress ranks for a node
 * @return number of progress ranks
 */
int get_num_progress_ranks_per_node();

/**
 * Return distribution of progress ranks on node
 * @return this function returns the values
 *   PACKED: ranks are assigned consecutively to each progress rank
 *   CYCLIC: ranks are assigned in round-robin to each progress rank
 * @return true if distribution is packed
 */
int get_progress_rank_distribution_on_node();


/**
 * Find master rank for all processes with same host ID
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] group pointer to group containing calling process
 * @return world group rank of master process
 */
int get_my_master_rank_with_same_hostid(int rank, int split_group_size, Group *group);

/**
 * Find master rank for all processes with same host ID
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] comm MPI communicator containing containing calling process
 * @param[in] world_rank world rank of calling process
 * @return world group rank of master process
 */
int get_my_master_rank_with_same_hostid(int rank, int split_group_size,
    MPI_Comm comm, int world_rank, std::vector<int> &world_ranks);

/**
 * Find notifier rank that notifies progress rank when shutting down
 * environment
 * @param[in] rank rank of calling process
 * @param[in] split_group_size number of processes of calling group
 * @param[in] group pointer to group containing calling process
 * @return notifier rank
 */
int get_my_rank_to_free(int rank, int split_group_size, Group *group);

/**
 * Return rank of calling process in world group
 * @return rank of calling process
 */
int rank();

/**
 * return master proc for proc iproc
 * @param iproc some process
 * @return master process for iproc
 */
int master(int iproc);

/**
 * Calling process is a progress rank
 * @return true if progress rank
 */
bool is_master();

/**
 * Number of ranks on the node. Actually, the number of ranks served by
 * one progress rank (plus the progress rank itself).
 * @return number of ranks on the node
 */
int node_size();

/**
 * Number of nodes
 * @return total number of nodes in system
 */
int num_nodes();

/**
 * Return host id for node hosting process
 * @param proc world rank of process
 * @return node host id
 */
long hostid(int proc);

/**
 * Return rank of calling process on node communicator
 * @return node rank
 */
int node_rank();

/**
 * Return total number of ranks in world (including progress ranks)
 * @return size of world group
 */
int world_size();

/**
 * Return the global communicator for all processes in the system
 * (including progress ranks)
 * @return global communicator
 */
MPI_Comm global_comm();

/**
 * Return a communicator for all processes within a single SMP node
 * @return node communicator
 */
MPI_Comm node_comm();

private:

cmx_global_config_t g_state;

int p_num_nodes;

};

} // namespace CMX

#endif /* _CMX_P_NODE_CONFIG_H */
