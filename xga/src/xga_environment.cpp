#include <stdlib.h>
#include <mpi.h>

#include "xga_environment.hpp"



namespace XGA {

Environment *Environment::p_instance = NULL;

/**
 * Initialize the environment
 */
Environment* Environment::instance()
{
  if (p_instance == NULL) {
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
  delete p_world_group;
}

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void Environment::wait(xga_request *hdl)
{
  cmxhdl_t *cmx_hdl = hdl->ahandle;
  while (cmx_hdl != NULL) {
    p_Impl->wait(&cmx_hdl->req);
    cmx_hdl->active = false;
    cmx_hdl->previous = NULL;
    cmx_hdl->index = -1;
    cmxhdl_t *tmp = cmx_hdl;
    cmx_hdl = cmx_hdl->next;
    tmp->next = NULL;
  }
}

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void Environment::waitAll(Group *group)
{
  p_Impl->waitAll(group->getCMXGroup());
}

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool Environment::test(xga_request *hdl)
{
  cmxhdl_t *cmx_hdl = hdl->ahandle;
  bool ret = true;
  /* if operation is complete, clean up list of CMX handles */
  while (cmx_hdl != NULL) {
    bool ttest = p_Impl->test(&cmx_hdl->req);
    if (ttest) {
      cmx_hdl->active = false;
      cmx_hdl->previous = NULL;
      cmx_hdl->index = -1;
      cmxhdl_t *tmp = cmx_hdl;
      cmx_hdl = cmx_hdl->next;
      tmp->next = NULL;
    } else {
      cmx_hdl = cmx_hdl->next;
    }
    ret = ret && ttest;
  }
  return ret;
}

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void Environment::fence(Group *group)
{
  p_Impl->fence(group->getCMXGroup());
}

/**
 * Get world group
 * @return pointer to world group
 */
Group* Environment::getWorldGroup()
{
  return p_world_group;
}

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void Environment::error(const char *msg, int code)
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
  return p_Impl->translateRanks(n,group_from->getCMXGroup(),ranks_from,
      group_to->getCMXGroup(),ranks_to);
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
  return p_Impl->translateWorld(n,group->getCMXGroup(),
      group_ranks,world_ranks);
}

/**
 * Initialize CMX environment.
 */
Environment::Environment()
{
  lastXGAhandle = -1;
  lastCMXhandle = -1;
  xga_nb_tag = -1;
  p_Impl = CMX::Environment::instance();
  p_world_group = new Group(p_Impl->getWorldGroup());
  /* initialize non-blocking handles */
  NBInit();
}

/**
 * Terminate CMX environment and clean up resources.
 */
Environment::~Environment()
{
  if (p_instance) delete p_instance;
}

/**
 * Initialize some parameters for non-blocking calls
 */
void Environment::NBInit()
{
  int i;
  for (i=0; i<MAX_NUM_NB_HDLS; i++) {
    xga_ihdl_array[i].ahandle = NULL;
    xga_ihdl_array[i].group = NULL;
    xga_ihdl_array[i].index = i;
    xga_ihdl_array[i].active = false;

    cmx_ihdl_array[i].next = NULL;
    cmx_ihdl_array[i].previous = NULL;
    cmx_ihdl_array[i].index = -1;
    cmx_ihdl_array[i].active = false;
  }
}

/**
 * Get a non-blocking handle for an XGA non-blocking call
 * @param[out] req XGA non-blocking request handle
 */
void Environment::getXGARequest(xga_request **req)
{
  int i, idx;
  lastXGAhandle++;
  lastXGAhandle = lastXGAhandle%MAX_NUM_NB_HDLS;
  idx = -1;
  /* look for inactive handle */
  for (i=lastXGAhandle; i<lastXGAhandle+MAX_NUM_NB_HDLS; i++) {
    int itmp = i%MAX_NUM_NB_HDLS;
    if (!xga_ihdl_array[itmp].active) {
      idx = itmp;
      break;
    }
  }
  if (idx == -1) {
    wait(&xga_ihdl_array[lastXGAhandle]);
    idx = lastXGAhandle;
  }
  xga_ihdl_array[idx].ahandle = NULL;
  xga_ihdl_array[idx].group = NULL;
  xga_ihdl_array[idx].active = true;
  *req = &xga_ihdl_array[idx];
}

/**
 * Get a non-blocking handle for a CMX non-blocking call
 * This handle is automatically added to the list of CMX
 * non-blocking calls associated with the xga_request
 * @param[in] req XGA non-blocking request handle
 * @return CMX non-blocking request handle
 */
CMX::cmx_request* Environment::getCMXRequest(xga_request *req)
{
  /* loop through list of CMX handles and find one that can
   * be used */
  lastCMXhandle++;
  lastCMXhandle = lastCMXhandle%MAX_NUM_NB_HDLS;
  int i;
  int idx = lastCMXhandle;
  for (i=lastCMXhandle; i<lastCMXhandle+MAX_NUM_NB_HDLS; i++) {
    int itmp = i%MAX_NUM_NB_HDLS;
    if (!cmx_ihdl_array[itmp].active) {
      idx = itmp;
      break;
    }
  }
  /* if selected handle is still active, complete it */
  if (cmx_ihdl_array[idx].active) {
    p_Impl->wait(&cmx_ihdl_array[idx].req);
    if (cmx_ihdl_array[idx].previous != NULL) {
      /*link is not first in linked list */
      cmx_ihdl_array[idx].previous->next = cmx_ihdl_array[idx].next;
      if (cmx_ihdl_array[idx].next != NULL) {
        cmx_ihdl_array[idx].next->previous = cmx_ihdl_array[idx].previous;
      }
    } else {
      /* link is first in linked list */
      req->ahandle = cmx_ihdl_array[idx].next;
      if (cmx_ihdl_array[idx].next != NULL) {
        cmx_ihdl_array[idx].next->previous = NULL;
      }
    }
  }
  cmx_ihdl_array[idx].active = true;
  cmx_ihdl_array[idx].previous = NULL;
  printf("p[%d] INDEX: %d\n",p_world_group->rank(),idx);
  if (req->ahandle != NULL) {
    req->ahandle->previous = &cmx_ihdl_array[idx];
  }
  cmx_ihdl_array[idx].next = req->ahandle;
  req->ahandle = &cmx_ihdl_array[idx];
  cmx_ihdl_array[idx].index = idx;
  lastCMXhandle = idx;
  return &cmx_ihdl_array[idx].req;
}

};
