#ifndef _XGA_GROUP_H
#define _XGA_GROUP_H

#include "cmx_group.hpp"
#include "xga_environment.hpp"

#define MAX_HOST_NAME_LEN 256

namespace XGA {

class Group {

public:

/**
 * Constructor for new group. This constructor assumes that all processes ranks
 * refer to processes in the group 'group'.
 * @param[in] n number of processes in the new group
 * @param[in] pid_list list of process ranks in the new group
 */
Group(int n, int *pid_list, Group *group);

/** Construct Group from p_Group
 * @param[in] group pointer to p_Group object
 * @return pointer to Group object
 */

/**
 * Destructor
 */
~Group();

/**
 * Get world group. Use this to initialize world group from a communicator.
 * @return pointer to world group
 */
static Group* getWorldGroup(MPI_Comm comm);

/**
 * Get world group.
 * @return pointer to world group
 */
static Group* getWorldGroup();


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
 * Get the world rank of a processor in the group
 * @param rank rank of process in group
 * @return rank of process in world group
 */
int getWorldRank(int rank);

/**
 * Get the local rank of process on process group
 * given the world rank
 * @param rank world rank
 * @return local rank
 */
int getLocalRank(int rank);

/**
 * Get  world ranks of all processors in the group
 * @return list of ranks in the world group
 */
std::vector<int> getWorldRanks();

friend class p_GA;

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

/**
 * Extract undelying runtime group
 * @return CMX runtime group
 */
CMX::Group* getCMXGroup();

/**
 * Construct a group directly from existing CMX::Group. This is needed for the
 * world group
 * @param[in] group pointer to an implementation instance
 */
Group(CMX::Group* group);

private:

friend class Environment;

CMX::Group* p_group;
 
bool p_new_group; // Check to see if p_group needs to be deleted in destructor

static Group *p_world_group;

};
}
#endif //_XGA_GROUP_H
