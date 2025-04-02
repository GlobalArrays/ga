#include <stdlib.h>
#include <mpi.h>

#include "cmx_environment.hpp"



namespace CMX {

Environment *Environment::p_instance = NULL;

/**
 * Initialize the environment
 */
Environment* Environment::instance()
{
  if (p_instance == NULL) {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      int argc;
      char **argv;
      MPI_Init(&argc, &argv);
    }
    p_instance = new Environment();
  }
  return p_instance;
}

/**
 * Initialize the environment with arguments
 * @param[in] argc number of arguments
 * @param[in] argv list of arguments
 */
Environment *Environment::instance(int *argc, char ***argv)
{
  if (p_instance == NULL) {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      MPI_Init(argc, argv);
    }
    p_instance = new Environment();
  }
  return p_instance;
}

/**
 * clean up environment and shut down libraries
 */
void Environment::finalize()
{
  p_Impl->finalize();
  delete p_cmx_world_group;
  p_cmx_world_group = NULL;
}

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void Environment::wait(cmx_request *hdl)
{
  p_Impl->wait(hdl);
}

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void Environment::waitAll(Group *group)
{
  p_Impl->waitAll(group);
}

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool Environment::test(cmx_request *hdl)
{
  return p_Impl->test(hdl);
}

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void Environment::fence(Group *group)
{
  p_Impl->fence(group);
}

/**
 * Get world group
 * @return pointer to world group
 */
Group* Environment::getWorldGroup()
{
  return p_cmx_world_group;
}

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void Environment::error(char *msg, int code)
{
}

/**
 * Translates the ranks of processes in one group to those in another group.  The
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
int Environment::translateRanks(int n, Group *group_from,
    int *ranks_from, Group *group_to, int *ranks_to)
{
  return p_Impl->translateRanks(n,group_from,ranks_from,group_to,ranks_to);
}

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
int Environment::translateWorld(int n, Group *group, int *group_ranks,
    int *world_ranks)
{
  return p_Impl->translateWorld(n,group,group_ranks,world_ranks);
}

/**
 * Initialize CMX environment.
 */
Environment::Environment()
{
  p_Impl = p_Environment::instance();
  p_cmx_world_group = p_Impl->getWorldGroup();
}

/**
 * Terminate CMX environment and clean up resources.
 */
Environment::~Environment()
{
  if (p_instance) delete p_instance;
}

};
