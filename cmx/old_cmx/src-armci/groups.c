/*
#if HAVE_CONFIG_H
#   include "config.h"
#endif
*/

#include <stdlib.h>
#include <assert.h>

#include "armci.h"
#include "message.h"
#include "cmx.h"
#include "groups.h"

/* the HEAD of the group linked list */
armci_igroup_t *_iarm_group_list = NULL;

/* ARMCI has the notion of a default group and a world group. */
ARMCI_Group ARMCI_Default_Proc_Group = 0;

/**
 * Return the igroup instance given the group id.
 *
 * The group linked list is searched sequentially until the given group
 * is found. It is an error if this function is called before
 * armci_group_init(). An error occurs if the given group is not found.
 *
 * @param id group identifier
 */
armci_igroup_t* armci_get_igroup_from_group(ARMCI_Group id)
{
  armci_igroup_t *current_group_list_item = _iarm_group_list;

  assert(group_list != NULL);
  while (current_group_list_item != NULL) {
    if (current_group_list_item->id == id) {
      return current_group_list_item;
    }
    current_group_list_item = current_group_list_item->next;
  }
  cmx_error("cmx group lookup failed", -1);

  return NULL;
}

/**
 * Return the cmx_group_corresponding to an ARMCI_Group
 * @param id ARMCI Group identifier
 * @return CMX group
 */
cmx_group_t armci_get_cmx_group(ARMCI_Group id)
{
  return (armci_get_igroup_from_group(id)->group);
}

/**
 * Creates and associates an ARMCI_Group with an armci_igroup_t.
 * This only places the new group in the internal group list and
 * assigns it an external ARMCI_Group id.
 */
void iarm_create_group_and_igroup(
    ARMCI_Group *id, armci_igroup_t **igroup)
{
  int maxID = 0;
  armci_igroup_t *new_group_list_item = NULL;
  armci_igroup_t *last_group_list_item = NULL;

  /* find the last group in the group linked list */
  last_group_list_item = _iarm_group_list;
  while (last_group_list_item->next != NULL) {
    if (last_group_list_item->id > maxID)
      maxID = last_group_list_item->id;
    last_group_list_item = last_group_list_item->next;
  }
  if (last_group_list_item->id > maxID)
    maxID = last_group_list_item->id;

  /* create, init, and insert the new node for the linked list */
  new_group_list_item = malloc(sizeof(armci_igroup_t));
  new_group_list_item->id = maxID + 1;
  new_group_list_item->next = NULL;
  new_group_list_item->handle_list = NULL;
  last_group_list_item->next = new_group_list_item;

  /* return the group id and cmx igroup */
  *igroup = new_group_list_item;
  *id = new_group_list_item->id;
}

/**
 * Destroy the given armci_igroup_t
 */
void iarm_igroup_delete(armci_igroup_t *igroup)
{
  int status;
  armci_handle_link_t *curr_hdl;
  armci_handle_link_t *next_hdl;
  armci_igroup_t *previous_group_list_item = NULL;
  armci_igroup_t *current_group_list_item = NULL;

  assert(*igroup);

  /* find the group in the group linked list */
  current_group_list_item = _iarm_group_list;
  while (current_group_list_item != igroup) {
    previous_group_list_item = current_group_list_item;
    current_group_list_item = previous_group_list_item->next;
    if (current_group_list_item == NULL) {
      printf("Error deleting group: group not found\n");
    }
  }

  /* Remove all handles associated with this group */
  curr_hdl = igroup->handle_list;
  while (curr_hdl != NULL) {
    next_hdl = curr_hdl->next;
    cmx_free(*(curr_hdl->handle));
    curr_hdl = next_hdl;
  }
  /* Fix up link list of groups before deleting group */
  previous_group_list_item->next = current_group_list_item->next;
  free(igroup);
}

/**
 * Initialize group linked list. Prepopulate with world group.
 */
void armci_group_init()
{
  /* create the head of the group linked list */
  assert(group_list == NULL);
  _iarm_group_list = malloc(sizeof(armci_igroup_t));
  /* First item in list contains the world group */
  _iarm_group_list->id = -1;
  _iarm_group_list->next = NULL;
  _iarm_group_list->handle_list = NULL;
  _iarm_group_list->group = CMX_GROUP_WORLD;
  /* Second item in list contains the mirror group */
  armci_igroup_t *mirror = malloc(sizeof(armci_igroup_t));
  mirror->id = 0;
  mirror->next = NULL;
  mirror->handle_list = NULL;
  mirror->group = CMX_GROUP_WORLD;
  /* Create link from world group */
  _iarm_group_list->next = mirror;
}

/**
 * Clean up group link list
 */
void armci_group_finalize()
{
  armci_igroup_t *current_group_list_item = _iarm_group_list;
  armci_igroup_t *previous_group_list_item = NULL;

  current_group_list_item = current_group_list_item->next;

  while (current_group_list_item != NULL) {
    previous_group_list_item = current_group_list_item;
    current_group_list_item = current_group_list_item->next;
    if (current_group_list_item)
    iarm_igroup_delete(previous_group_list_item);
  }
  _iarm_group_list = NULL;
}

/**
 *  Add a CMX handle to the list in the group
 */
void iarm_igroup_add_handle(ARMCI_Group id, cmx_handle_t *cmx_hdl)
{
  armci_igroup_t *current_group_list_item = _iarm_group_list;
  armci_handle_link_t *curr_hdl = NULL;
  armci_handle_link_t *new_hdl = NULL;

  /* find the group*/
  while (current_group_list_item != NULL) {
    if (current_group_list_item->id == id) {
      break;
    }
    current_group_list_item = current_group_list_item->next;
  }

  /* add handle to group. Start by finding last member of
   * handle list */
  curr_hdl = current_group_list_item->handle_list;
  while (curr_hdl != NULL && curr_hdl->next != NULL) {
    curr_hdl = curr_hdl->next;
  }
  new_hdl = malloc(sizeof(armci_handle_link_t));
  if (curr_hdl == NULL) {
    new_hdl->next = NULL;
    new_hdl->prev = NULL;
    new_hdl->handle = cmx_hdl;
    current_group_list_item->handle_list = new_hdl;
  } else {
    curr_hdl->next = new_hdl;
    new_hdl->next = NULL;
    new_hdl->prev = curr_hdl;
    new_hdl->handle = cmx_hdl;
  }
}

int ARMCI_Group_rank(ARMCI_Group *id, int *rank)
{
  armci_igroup_t *grp;
  grp = armci_get_igroup_from_group(*id);
  return cmx_group_rank(grp->group, rank);
}


void ARMCI_Group_size(ARMCI_Group *id, int *size)
{
  armci_igroup_t *grp;
  grp = armci_get_igroup_from_group(*id);
  cmx_group_size(grp->group, size);
}


int ARMCI_Absolute_id(ARMCI_Group *id, int group_rank)
{
  armci_igroup_t *grp;
  int ierr;
  int world_rank;
  grp = armci_get_igroup_from_group(*id);
  ierr = cmx_group_translate_world(grp->group, group_rank, &world_rank);
  assert(CMX_SUCCESS == ierr);
  return world_rank;
}


void ARMCI_Group_set_default(ARMCI_Group *id) 
{
    ARMCI_Default_Proc_Group = *id;
}


void ARMCI_Group_get_default(ARMCI_Group *group_out)
{
    *group_out = ARMCI_Default_Proc_Group;
}


void ARMCI_Group_get_world(ARMCI_Group *group_out)
{
    *group_out = 0;
}


void ARMCI_Group_free(ARMCI_Group *id)
{
  armci_igroup_t *grp;
  cmx_group_t cmx_grp;
  grp = armci_get_igroup_from_group(*id);
  cmx_grp = grp->group;
  cmx_group_free(cmx_grp);
  iarm_igroup_delete(grp);
}

void ARMCI_Group_create_child(
        int n, int *pid_list, ARMCI_Group *id_child, ARMCI_Group *id_parent)
{
  armci_igroup_t *old_grp;
  armci_igroup_t *new_grp;
  cmx_group_t child;
  cmx_group_t parent;
  old_grp = armci_get_igroup_from_group(*id_parent);
  parent = old_grp->group;
  cmx_group_create(n, pid_list, parent, &child);
  iarm_create_group_and_igroup(id_child, &new_grp);
  new_grp->group = child;
  new_grp->id = *id_child;
  new_grp->handle_list = NULL;
  new_grp->next = NULL;
}


void ARMCI_Group_create(int n, int *pid_list, ARMCI_Group *group_out)
{
  armci_igroup_t *old_grp;
  armci_igroup_t *new_grp;
  cmx_group_t child;
  cmx_group_t parent;
  old_grp = armci_get_igroup_from_group(ARMCI_Default_Proc_Group);
  parent = old_grp->group;
  cmx_group_create(n, pid_list, parent, &child);
  iarm_create_group_and_igroup(group_out, &new_grp);
  new_grp->group = child;
  new_grp->id = *group_out;
  new_grp->handle_list = NULL;
  new_grp->next = NULL;
}
