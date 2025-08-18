#ifndef _XGA_ENVIRONMENT_H
#define _XGA_ENVIRONMENT_H

#include "cmx_environment.hpp"
#include "xga_group.hpp"

namespace XGA {

class Group;

/**
 *                      NOTES
 * The non-blocking xga_request is and element in a list of structs that
 * point to a linked list of non-blocking CMX calls. When a new XGA
 * non-blocking call is created, the code looks at the list of
 * XGA handles and tries to find one that is not currently being used. If
 * it can't find one, it calls wait on an existing call and recycles that
 * handle for the new call.
 */

/* We create a linked list of individual CMX non-blocking calls. This
 * list represents the collection of CMX non-blocking calls used to
 * create a single XGA non-blocking call. Each element in the linked 
 * list is of type cmxhdl_t.
 * handle: cmx_request handle for CMX non-blocking call
 * next: pointer to next cmxhdl_t element in list
 * previous: pointer to previous cmxhdl_t element in list
 * index: index that points back to xga_nbhdl_array list. This can be
 *        used to remove this link from XGA linked list if this
 *        non-blocking call must be cleared to make room for a new
 *        request.
 * active: indicates that this represents an outstanding CMX
 *         request
 */
typedef struct struct_cmxhdl_t {
  CMX::cmx_request req;
  struct_cmxhdl_t *next;
  struct_cmxhdl_t *previous;
  int index;
  bool active;
} cmxhdl_t;

/* We create an array of type xga_request. Each of the elements in this
 * array is the head of the CMX handle linked list that is associated with
 * each non-blocking XGA call.
 * ahandle: head node in a linked list of CMX handles
 * group: group associated with this request
 * index: location of this element in the xga_request list
 * active: flag indicating that this handle is in use
 * If count is 0 or ahandle is null, there are no outstanding CMX calls
 * associated with this XGA handle
 */
typedef struct{
  cmxhdl_t *ahandle;
  Group *group;
  int index;
  bool active;
} xga_request;

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
void wait(xga_request *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(Group *group);

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
bool test(xga_request *hdl);

/**
 * Fence on all processes in group
 * @param group fence all processes in group
 */
void fence(Group *group);

/**
 * Sync system across all processors in a group
 * @param group sync all processes in group
 */
void sync(Group *group);

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
void error(const char *msg, int code);

friend class p_GA;

protected:

/**
 * Initialize CMX environment.
 */
Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
virtual ~Environment();

/**
 * Initialize some parameters for non-blocking calls
 */
void NBInit();

/**
 * Get a non-blocking handle for an XGA non-blocking call
 * @param[out] req pointer to XGA non-blocking request handle
 */
void getXGARequest(xga_request **req);

/**
 * Get a non-blocking handle for a CMX non-blocking call
 * This handle is automatically added to the list of CMX
 * non-blocking calls associated with the xga_request
 * @param[in] req XGA non-blocking request handle
 * @return CMX non-blocking request handle
 */
CMX::cmx_request* getCMXRequest(xga_request *req);

private:

static const int nb_max_outstanding = 256;

CMX::Environment *p_Impl;

static Environment *p_instance;

Group *p_world_group;

std::vector<bool> p_fence_array;

#define MAX_NUM_NB_HDLS 256

/**
 * Array of headers for non-blocking XGA calls. Then cmxhdl_t elements in
 * cmx_ihdl_array index back into the xga_ihdl_array
 */
xga_request xga_ihdl_array[MAX_NUM_NB_HDLS];
cmxhdl_t cmx_ihdl_array[MAX_NUM_NB_HDLS];

/**
 * Global parameters for managing non-blocking handles
 */
int lastXGAhandle; /* last assigned XGA handle */
int lastCMXhandle; /* last assigned CMX handle */
unsigned int xga_nb_tag; /* counter for unique tags on
                                     * non-blocking CMX calls*/
};
}
#endif // XGA_ENVIRONMENT_H
