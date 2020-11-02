#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "globalp.h"
#include "base.h"
#include "ga-papi.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#define DEBUG 0

/* WARNING: The maximum value MAX_NUM_NB_HDLS can assume is 256. If it is any larger,
 * the 8-bit field defined in gai_hbhdl_t will exceed its upper limit of 255 in
 * some parts of the nbutil.c code */
#define MAX_NUM_NB_HDLS 256
static int nb_max_outstanding = MAX_NUM_NB_HDLS;

/**
 *                      NOTES
 * The non-blocking GA handle indexes into a list of structs that point to a
 * linked list of non-blocking ARMCI calls. The first link in the list is
 * contained in the GA struct. Conversely, each link in the non-blocking list
 * points to the GA handle that contains the head of the list. When a new GA
 * non-blocking call is created, the code looks at the list of GA handles and
 * tries to find one that is not currently being used. If it can't find one, it
 * calls wait on an existing call and recycles that handle for the new call.
 *
 * Similarly, each GA call consists of multiple ARMCI non-blocking calls. The
 * handles for each of these calls are assembled into a list. If no handle is
 * available, the ARMCI_Wait function is called on a handle, freeing it for use.
 * The handle is also removed from the linked list pointed to by the original GA
 * struct. It is possible in this scheme that a GA struct has a linked list that
 * contains no links. When wait of test is called in this case, the struct is
 * marked as inactive and then returns without performing any ARMCI operations.
 */

/* The structure of gai_nbhdl_t (this is our internal handle). It maps directly
 * to a 32-bit integer*/
typedef struct {
    unsigned int ihdl_index:8;
    unsigned int ga_nbtag:24;
} gai_nbhdl_t;


/* We create an array of type struct_armci_hdl_t. This list represents the
 * number of available ARMCI non-blocking calls that are available to create
 * non-blocking GA calls. Each element in the armci handle linked list is of
 * type ga_armcihdl_t.
 * handle: int handle or gai_nbhdl_t struct that represents ARMCI handle for
 *         non-blocking call
 * next: pointer to next element in list
 * previous: pointer to previous element in list
 * ga_hdlarr_index: index that points back to ga_nbhdl_array list.
 *                  This can be used to remove this link from GA linked list if
 *                  this armci request must be cleared to make room for a new
 *                  request.
 * active: indicates that this represents an outstanding ARMCI non-blocking
 * request
 */
typedef struct struct_armcihdl_t {
    armci_hdl_t handle;
    struct struct_armcihdl_t *next;
    struct struct_armcihdl_t *previous;
    int ga_hdlarr_index;
    int active;
} ga_armcihdl_t;


/* We create an array of type ga_nbhdl_array_t. Each of the elements in this
 * array is the head of the armci handle linked list that is associated with
 * each GA call.
 * ahandle: head node in a linked list of ARMCI handles
 * count: total number of ARMCI handles in linked list
 * ga_nbtag: unique tag that matches tag in handle (gai_nbhdl_t)
 * If count is 0 or ahandle is null, there are no outstanding armci calls
 * associated with this GA handle
 */
typedef struct{
    ga_armcihdl_t *ahandle;
    int count;
    int ga_nbtag;
    int active;
} ga_nbhdl_array_t;

/**
 * Array of headers for non-blocking GA calls. The ihdl_index element of the
 * non-blocking handle indexes into this array. The maximum number of
 * outstanding non-blocking GA calls is nb_max_outstanding.
 */
static ga_nbhdl_array_t ga_ihdl_array[MAX_NUM_NB_HDLS];

/**
 * Array of armci handles. This is used to construct linked lists of ARMCI
 * non-blocking calls. The maximum number of outstanding ARMCI non-blocking
 * calls is nb_max_outstanding.
 */
static ga_armcihdl_t armci_ihdl_array[MAX_NUM_NB_HDLS];

static int lastGAhandle = -1; /* last assigned ga handle */
static int lastARMCIhandle = -1; /* last assigned armci handle */

/**
 * get a unique tag for each individual ARMCI call. These tags currently repeat
 * after 16777216=2^24 non-blocking calls
 */
static unsigned int ga_nb_tag = -1;
unsigned int get_next_tag(){
  ga_nb_tag++;
  ga_nb_tag = ga_nb_tag%16777216;
  return ga_nb_tag;
  /* return(++ga_nb_tag); */
}

/**
 * Initialize some data structures used in the non-blocking function calls
 */
void gai_nb_init()
{
  int i;
  char *value;
  /* This is a hideous kluge, but some users want to be able to set this
   * externally. The fact that only integer handles are exchanged between GA and
   * the underlying runtime make it very difficult to handle in a more elegant
   * manner. */
  nb_max_outstanding = MAX_NUM_NB_HDLS; /* default */
  value = getenv("COMEX_MAX_NB_OUTSTANDING");
  if (NULL != value) {
    nb_max_outstanding = atoi(value);
  }
  if (nb_max_outstanding <1 || nb_max_outstanding > MAX_NUM_NB_HDLS) {
    pnga_error("Illegal number of outstanding Non-block requests specified",
        nb_max_outstanding);
  }
  for (i=0; i<nb_max_outstanding; i++) {
    ga_ihdl_array[i].ahandle = NULL;
    ga_ihdl_array[i].count = 0;
    ga_ihdl_array[i].active = 0;
    ga_ihdl_array[i].ga_nbtag = -1;
    armci_ihdl_array[i].next = NULL;
    armci_ihdl_array[i].previous = NULL;
    armci_ihdl_array[i].active = 0;
    ARMCI_INIT_HANDLE(&armci_ihdl_array[i].handle);
  }
}

/**
 * Called from ga_put/get before every call to a non-blocking armci request.
 * Find an available armic non-blocking handle. If none is available,
 * complete an existing outstanding armci request and return the
 * corresponding handle.
 */
armci_hdl_t* get_armci_nbhandle(Integer *nbhandle)
{
  int i, top, idx, iloc;
  gai_nbhdl_t *inbhandle = (gai_nbhdl_t *)nbhandle;
  int index = inbhandle->ihdl_index;
  ga_armcihdl_t* next = ga_ihdl_array[index].ahandle;

  lastARMCIhandle++;
  lastARMCIhandle = lastARMCIhandle%nb_max_outstanding;
  top = lastARMCIhandle+nb_max_outstanding;
  /* default index if no handles are available */
  iloc = lastARMCIhandle;
  for (i=lastARMCIhandle; i<top; i++) {
    idx = i%nb_max_outstanding;
    if (armci_ihdl_array[idx].active == 0) {
      iloc = idx;
      break;
    }
  }
  /* if selected handle represents an outstanding request, complete it */
  if (armci_ihdl_array[iloc].active == 1) {
    int iga_hdl = armci_ihdl_array[iloc].ga_hdlarr_index;
    ARMCI_Wait(&armci_ihdl_array[iloc].handle);
    /* clean up linked list that this handle used to be a link in */
    if (armci_ihdl_array[iloc].previous != NULL) {
      /* link is not first in linked list */
      armci_ihdl_array[iloc].previous->next = armci_ihdl_array[iloc].next;
      if (armci_ihdl_array[iloc].next != NULL) {
        armci_ihdl_array[iloc].next->previous = armci_ihdl_array[iloc].previous;
      }
    } else {
      /* link is first in linked list. Need to update header */
      ga_ihdl_array[iga_hdl].ahandle = armci_ihdl_array[iloc].next;
      if (armci_ihdl_array[iloc].next != NULL) {
        armci_ihdl_array[iloc].next->previous = NULL;
      }
    }
    ga_ihdl_array[iga_hdl].count--;
  }
  /* Initialize armci handle and add this operation to the linked list
   * corresponding to nbhandle */
  ARMCI_INIT_HANDLE(&armci_ihdl_array[iloc].handle);
  armci_ihdl_array[iloc].active = 1;
  armci_ihdl_array[iloc].previous = NULL;
  if (ga_ihdl_array[index].ahandle) {
    ga_ihdl_array[index].ahandle->previous = &armci_ihdl_array[iloc];
  }
  armci_ihdl_array[iloc].next = ga_ihdl_array[index].ahandle;
  ga_ihdl_array[index].ahandle =  &armci_ihdl_array[iloc];
  armci_ihdl_array[iloc].ga_hdlarr_index = index;
  ga_ihdl_array[index].count++;

  /* reset lastARMCIhandle to iloc */
  lastARMCIhandle = iloc;

  return &armci_ihdl_array[iloc].handle;
}

/**
 * the wait routine which is called inside pnga_nbwait. This always returns
 * zero. The return value is not checked in the code.
 */ 
int nga_wait_internal(Integer *nbhandle){
  gai_nbhdl_t *inbhandle = (gai_nbhdl_t *)nbhandle;
  int index = inbhandle->ihdl_index;
  int retval = 0;
  int tag = inbhandle->ga_nbtag;
  /* check if tags match. If they don't then this request was already completed
   * so the handle can be used for another GA non-blocking call. Just return in
   * this case */
  if (tag == ga_ihdl_array[index].ga_nbtag) {
    if (ga_ihdl_array[index].active == 0) {
      printf("p[%d] nga_wait_internal: GA NB handle inactive\n",GAme);
    }
    ga_armcihdl_t* next = ga_ihdl_array[index].ahandle;
    /* Loop over linked list and complete all remaining armci non-blocking calls */
    while(next) {
      ga_armcihdl_t* tmp = next->next;
      /* Complete the call */
      ARMCI_Wait(&next->handle);
      /* reinitialize armci_hlt_t data structure */
      next->next = NULL;
      next->previous = NULL;
      next->active = 0;
      ARMCI_INIT_HANDLE(&next->handle);
      next = tmp;
    }
    ga_ihdl_array[index].ahandle = NULL;
    ga_ihdl_array[index].count = 0;
    ga_ihdl_array[index].active = 0;
  }

  return(retval);
}


/**
 * the test routine which is called inside nga_nbtest. Return 0 if operation is
 * completed
 */ 
int nga_test_internal(Integer *nbhandle)
{
  gai_nbhdl_t *inbhandle = (gai_nbhdl_t *)nbhandle;
  int index = inbhandle->ihdl_index;
  int retval = 0;
  int tag = inbhandle->ga_nbtag;

  /* check if tags match. If they don't then this request was already completed
   * so the handle can be used for another GA non-blocking call. Just return in
   * this case */
  if (tag == ga_ihdl_array[index].ga_nbtag) {
    ga_armcihdl_t* next = ga_ihdl_array[index].ahandle;
    /* Loop over linked list and test all remaining armci non-blocking calls */
    while(next) {
      int ret = ARMCI_Test(&next->handle);
      ga_armcihdl_t *tmp = next->next;
      if (ret == 0) {
        /* operation is completed so remove it from linked list */
        if (next->previous != NULL) {
          /* operation is not first element in list */
          next->previous->next = next->next;
          if (next->next != NULL) {
            next->next->previous = next->previous;
          }
        } else {
          /* operation is first element in list */
          ga_ihdl_array[index].ahandle = next->next;
          if (next->next != NULL) {
            next->next->previous = NULL;
          }
        }
        next->previous = NULL;
        next->next = NULL;
        next->active = 0;
        ga_ihdl_array[index].count--;
      }
      next = tmp;
    }
    if (ga_ihdl_array[index].count == 0) {
      ga_ihdl_array[index].ahandle = NULL;
      ga_ihdl_array[index].active = 0;
    }
    if (ga_ihdl_array[index].count > 0) retval = 1;
  }

  return(retval);
}

/**
 * Find a free GA non-blocking handle.
 */
void ga_init_nbhandle(Integer *nbhandle)
{
  int i, top, idx, iloc;
  gai_nbhdl_t *inbhandle = (gai_nbhdl_t *)nbhandle;
  lastGAhandle++;
  lastGAhandle = lastGAhandle%nb_max_outstanding;
  top = lastGAhandle+nb_max_outstanding;
  /* default index if no handles are available */
  idx = lastGAhandle;
  for (i=lastGAhandle; i<top; i++) {
    iloc = i%nb_max_outstanding;
    if (ga_ihdl_array[iloc].ahandle == NULL) {
      idx = iloc;
      break;
    }
  }
  /* If no free handle is found, clear the oldest handle */
  if (ga_ihdl_array[idx].ahandle != NULL) {
    Integer itmp;
    /* find value of itmp corresponding to oldest handle */
    gai_nbhdl_t *oldhdl = (gai_nbhdl_t*)&itmp;
    oldhdl->ihdl_index = idx;
    oldhdl->ga_nbtag = ga_ihdl_array[idx].ga_nbtag;
    nga_wait_internal(&itmp);
  }
  inbhandle->ihdl_index = idx;
  inbhandle->ga_nbtag = get_next_tag();
  ga_ihdl_array[idx].ahandle = NULL;
  ga_ihdl_array[idx].count = 0;
  ga_ihdl_array[idx].active = 1;
  ga_ihdl_array[idx].ga_nbtag = inbhandle->ga_nbtag;

  /* reset lastGAhandle to idx */
  lastGAhandle = idx;
  return;
}
