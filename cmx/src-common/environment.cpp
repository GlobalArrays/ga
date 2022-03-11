#include <mpi.h>

#include <stdlib.h>

#include "environment.hpp"

CMX::Environment *CMX::Environment::p_instance = NULL;

namespace CMX {

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
    p_instance = Environment::instance();
  }
  p_Impl = NULL;
  p_Impl = p_Environment::instance();
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
    p_instance = Environment::instance();
  }
  p_Impl = p_Environment::instance();
  return p_instance;
}

/**
 * clean up environment and shut down libraries
 */
void Environment::finalize()
{
//  delete p_CMX_GROUP_WORLD;
//  printf("Cleaned up group\n");
//  delete p_Impl;
}

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void Environment::wait(cmx_request_t *hdl)
{
  p_Impl->wait(hdl);
}

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void Environment::waitAll(Group *group)
{
  p_Impl->waitAll(group->p_group);
}

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool Environment::test(cmx_request_t *hdl)
{
  return p_Impl->test(hdl);
}

/**
 * Get world group
 * @return pointer to world group
 */
Group* Environment::getWorldGroup()
{
  return p_CMX_GROUP_WORLD;
}

/**
 * Initialize CMX environment.
 */
Environment::Environment()
{
  p_CMX_GROUP_WORLD = new Group(p_Impl->getWorldGroup());
}

/**
 * Terminate CMX environment and clean up resources.
 */
Environment::~Environment()
{
  delete p_CMX_GROUP_WORLD;
}

};
