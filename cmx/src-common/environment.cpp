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
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void Environment::wait(cmx_request_t *hdl)
{
}

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void Environment::waitAll(p_Group *group)
{
}

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool Environment::test(cmx_request_t *hdl)
{
  return false;
}

/**
 * Initialize CMX environment.
 */
Environment::Environment()
{
  p_Impl = new p_Environment();
}

/**
 * Terminate CMX environment and clean up resources.
 */
Environment::~Environment()
{
  delete p_Impl;
}

};
