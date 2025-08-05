#ifndef _NB_UTIL_H
#define _NB_UTIL_H
#include "cmx_environment.hpp"

namespace XGA {

  /**
   *                      NOTES
   * The non-blocking xga_request is and element in a list of structs that
   * point to a linked list of non-blocking CMX calls. When a new XGA
   * non-blocking call is created, the code looks at the list of
   * XGA handles and tries to find one that is not currently being used.  If
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

}
#endif
