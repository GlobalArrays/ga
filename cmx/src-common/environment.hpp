/* cmx header file */
#ifndef _CMX_COMMON_ENVIRONMENT_H
#define _CMX_COMMON_ENVIRONMENT_H

#include <mpi.h>

#include <stdlib.h>

#include "group.hpp"
#include "cmx_impl.hpp"

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
void wait(cmx_request_t *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(p_Group *group);

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool test(cmx_request_t *hdl);

#if 0
/**
 * Translates the ranks of processes in one group to those in another group. The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
virtual int translateRanks(int n, int *ranks_from, CMX_Group *group_to, int *ranks_to);

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group_ranks the rank to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
virtual int translateWorld(int n, int *group_ranks, int *world_ranks);

/**
 * Get world group
 * @return pointer to world group
 */
virtual CMX_Group* getWorldGroup();

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void error(char *msg, int code);
#endif

protected:

/**
 * Initialize CMX environment.
 */
Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
~Environment();

private:

p_Environment *p_Impl;
static Environment *p_instance;

CMX_Group *p_WORLD_GROUP;

};
}

#endif /* _CMX_H */
