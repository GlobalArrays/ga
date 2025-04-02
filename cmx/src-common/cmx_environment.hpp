/* cmx header file */
#ifndef _CMX_COMMON_ENVIRONMENT_H
#define _CMX_COMMON_ENVIRONMENT_H

#include <mpi.h>

#include <stdlib.h>

#include "defines.hpp"
#include "cmx_group.hpp"
#include "p_environment.hpp"

namespace CMX {

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
static void CMXError(char *msg, int code)
{
  printf("Environment Abort: %s code: %d\n",msg,code);
  MPI_Abort(MPI_COMM_WORLD,code);
}

class p_Environment;
class Group;

class Environment {

public:

/**
 * Return an instance of the p_Environment singleton
 * @return pointer to p_Environment singleton
 */
static Environment *instance(); 

/**
 * Return an instance of the p_Environment singleton. Initialize instance
 * with argc and argv if it does not already exist
 * @param[in] argc number of arguments
 * @param[in] argv list of arguments
 * @return pointer to p_Environment singleton
 */
static Environment *instance(int *argc, char ***argv); 

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void wait(cmx_request *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(CMX::Group *group);

/**
 * clean up environment and shut down libraries
 */
void finalize();

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool test(cmx_request *hdl);

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void fence(Group *group);

/**
 * Translates the ranks of processes in one group to those in another group. The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] group_from the group to translate ranks from 
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
int translateRanks(int n, Group *group_from, int *ranks_from,
    Group *group_to, int *ranks_to);

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group the group to translate ranks from 
 * @param[in] group_ranks the ranks to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
int translateWorld(int n, Group *group, int *group_ranks, int *world_ranks);

/**
 * Get world group
 * @return pointer to world group
 */
Group* getWorldGroup();

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void error(char *msg, int code);

protected:

/**
 * Initialize CMX environment.
 */
Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
virtual ~Environment();

private:

p_Environment *p_Impl;

static Environment *p_instance;

Group *p_cmx_world_group;

//friend class p_Environment;

};
}

#endif /* _CMX_H */
