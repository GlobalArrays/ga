#include <stdlib.h>
#include <assert.h>

#include "groups.h"
#include "cmx.h"

/*
#define USE_MPI_ERRORS_RETURN
*/

/* the HEAD of the group linked list */
cmx_igroup_t *group_list = NULL;

/* static functions implemented in this file */
static void cmx_create_group(cmx_igroup_t **group);
static void cmx_igroup_finalize(cmx_igroup_t *group);

cmx_igroup_t *CMX_GROUP_WORLD;

/* Add a variable to keep track of world rank. Useful for debugging */
int _world_me;

/**
 * Creates a cmx group object and puts it in the linked list of groups. Does not
 * assign properties to the group
 */
void cmx_create_group(cmx_igroup_t **group)
{
  cmx_igroup_t *new_group_list_item = NULL;
  cmx_igroup_t *last_group_list_item = NULL;

  /* find the last group in the group linked list */
  last_group_list_item = group_list;
  while (last_group_list_item->next != NULL) {
    last_group_list_item = last_group_list_item->next;
  }

  /* create, init, and insert the new node for the linked list */
  new_group_list_item = (cmx_igroup_t*)malloc(sizeof(cmx_igroup_t));
  new_group_list_item->comm = MPI_COMM_NULL;
  new_group_list_item->group = MPI_GROUP_NULL;
  new_group_list_item->next = NULL;
  new_group_list_item->win_list = NULL;
  last_group_list_item->next = new_group_list_item;

  /* return the group id */
  *group = new_group_list_item;
}


int cmx_group_rank(cmx_igroup_t *group, int *rank)
{
  int status;
  status = MPI_Group_rank(group->group, rank);
  if (status != MPI_SUCCESS) {
    cmx_error("MPI_Group_rank: Failed ", status);
  }

  return CMX_SUCCESS;
}


int cmx_group_size(cmx_igroup_t *group, int *size)
{
  int status;

  status = MPI_Group_size(group->group, size);
  if (status != MPI_SUCCESS) {
    cmx_error("MPI_Group_size: Failed ", status);
  }

  return CMX_SUCCESS;
}


/** 
 * Returns the MPI_Comm object backing the given group. 
 * 
 * @param group: group handle 
 * @param comm the communicator handle 
 */ 
int cmx_group_comm(cmx_igroup_t *group, MPI_Comm *comm)
{
  *comm = group->comm;
  return CMX_SUCCESS;
}


int cmx_group_translate_world(cmx_igroup_t *group, int group_rank, int *world_rank)
{
  if (CMX_GROUP_WORLD == group) {
    *world_rank = group_rank;
  }
  else {
    int status = MPI_Group_translate_ranks(
        group->group, 1, &group_rank, CMX_GROUP_WORLD->group, world_rank);
    if (status != MPI_SUCCESS) {
      cmx_error("MPI_Group_translate_ranks: Failed ", status);
    }
  }

  return CMX_SUCCESS;
}


/**
 * Destroys the given cmx group.
 */
void cmx_igroup_finalize(cmx_igroup_t *group)
{
  int status;
  win_link_t *curr_win;
  win_link_t *next_win;

  assert(group);

  if (group->group != MPI_GROUP_NULL) {
    status = MPI_Group_free(&group->group);
    if (status != MPI_SUCCESS) {
      cmx_error("MPI_Group_free: Failed ", status);
    }
  }

  if (group->comm != MPI_COMM_NULL) {
    status = MPI_Comm_free(&group->comm);
    if (status != MPI_SUCCESS) {
      cmx_error("MPI_Comm_free: Failed ", status);
    }
  }

  /* Remove all windows associated with this group */
  curr_win = group->win_list;
  while (curr_win != NULL) {
    next_win = curr_win->next;
    MPI_Win_free(&curr_win->win);
    free(curr_win);
    curr_win = next_win;
  }
}


int cmx_group_free(cmx_igroup_t *group)
{
  cmx_igroup_t *current_group_list_item = group_list;
  cmx_igroup_t *previous_group_list_item = NULL;

  /* find the group to free */
  while (current_group_list_item != NULL) {
    if (current_group_list_item == group) {
      break;
    }
    previous_group_list_item = current_group_list_item;
    current_group_list_item = current_group_list_item->next;
  }
  /* make sure we found a group */
  assert(current_group_list_item != NULL);
  /* remove the group from the linked list */
  if (previous_group_list_item != NULL) {
    previous_group_list_item->next = current_group_list_item->next;
  }
  /* free the group */
  cmx_igroup_finalize(current_group_list_item);
  free(current_group_list_item);

  return CMX_SUCCESS;
}


int cmx_group_create(
    int n, int *pid_list, cmx_igroup_t *group_parent, cmx_igroup_t **group_child)
{
  int status;
  int grp_me;
  cmx_igroup_t *igroup_child = NULL;
  MPI_Group      *mpi_group_child = NULL;
  MPI_Comm       *mpi_comm_child = NULL;
  cmx_igroup_t *igroup_parent = NULL;
  MPI_Group      *mpi_group_parent = NULL;
  MPI_Comm       *mpi_comm_parent = NULL;

  /* create the node in the linked list of groups and */
  /* get the child's MPI_Group and MPI_Comm, to be populated shortly */
  cmx_create_group(&igroup_child);
  mpi_group_child = &(igroup_child->group);
  mpi_comm_child  = &(igroup_child->comm);

  /* get the parent's MPI_Group and MPI_Comm */
  igroup_parent = group_parent;
  mpi_group_parent = &(igroup_parent->group);
  mpi_comm_parent  = &(igroup_parent->comm);

  status = MPI_Group_incl(*mpi_group_parent, n, pid_list, mpi_group_child);
  if (status != MPI_SUCCESS) {
    cmx_error("MPI_Group_incl: Failed ", status);
  }

  {
    MPI_Comm comm, comm1, comm2;
    int lvl=1, local_ldr_pos;
    MPI_Group_rank(*mpi_group_child, &grp_me);
    if (grp_me == MPI_UNDEFINED) {
      /* FIXME: keeping the group around for now */
      return CMX_SUCCESS;
    }
    /* SK: sanity check for the following bitwise operations */
    assert(grp_me>=0);
    MPI_Comm_dup(MPI_COMM_SELF, &comm); /* FIXME: can be optimized away */
    local_ldr_pos = grp_me;
    while(n>lvl) {
      int tag=0;
      int remote_ldr_pos = local_ldr_pos^lvl;
      if (remote_ldr_pos < n) {
        int remote_leader = pid_list[remote_ldr_pos];
        MPI_Comm peer_comm = *mpi_comm_parent;
        int high = (local_ldr_pos<remote_ldr_pos)?0:1;
        MPI_Intercomm_create(
            comm, 0, peer_comm, remote_leader, tag, &comm1);
        MPI_Comm_free(&comm);
        MPI_Intercomm_merge(comm1, high, &comm2);
        MPI_Comm_free(&comm1);
        comm = comm2;
      }
      local_ldr_pos &= ((~0)^lvl);
      lvl<<=1;
    }
    *mpi_comm_child = comm;
#ifdef USE_MPI_ERRORS_RETURN
    MPI_Comm_set_errhandler(*mpi_comm_child,MPI_ERRORS_RETURN);
#endif
    /* cleanup temporary group (from MPI_Group_incl above) */
    MPI_Group_free(mpi_group_child);
    /* get the actual group associated with comm */
    MPI_Comm_group(*mpi_comm_child, mpi_group_child);
  }
  *group_child = igroup_child;
    MPI_Group_rank(*mpi_group_parent, &grp_me);
  return CMX_SUCCESS;
}


/**
 * Initialize group linked list. Prepopulate with world group.
 */
void cmx_group_init() 
{
  MPI_Comm_rank(MPI_COMM_WORLD,&_world_me);
  /* create the head of the group linked list on the world group
     and set it equal to CMX_GROUP_WORLD */
  assert(group_list == NULL);
  group_list = malloc(sizeof(cmx_igroup_t));
  group_list->next = NULL;
  group_list->win_list = NULL;
#ifdef USE_MPI_ERRORS_RETURN
  MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
#endif

  /* save MPI world group and communicatior in CMX_GROUP_WORLD */
  group_list->comm = l_state.world_comm;
  MPI_Comm_group(group_list->comm, &(group_list->group));
  CMX_GROUP_WORLD = group_list;
}


void cmx_group_finalize()
{
  cmx_igroup_t *current_group_list_item = group_list;
  cmx_igroup_t *previous_group_list_item = NULL;

  /* don't free the world group (the list head) */
  current_group_list_item = current_group_list_item->next;

  while (current_group_list_item != NULL) {
    previous_group_list_item = current_group_list_item;
    current_group_list_item = current_group_list_item->next;
    cmx_igroup_finalize(previous_group_list_item);
    free(previous_group_list_item);
  }

  /* ok, now free the world group, but not the world comm */
  MPI_Group_free(&(group_list->group));
  free(group_list);
  group_list = NULL;
}

/**
 *  Add an MPI window to the window list in the group
 */
void cmx_igroup_add_win(cmx_igroup_t *group, MPI_Win win)
{
  cmx_igroup_t *igroup = group;
  win_link_t *curr_win = NULL;
  win_link_t *new_win = NULL;

  /* add window to group. Start by finding last member of window list */
  curr_win = igroup->win_list;
  while (curr_win != NULL && curr_win->next != NULL) {
    curr_win = curr_win->next;
  }
  new_win = (win_link_t*)malloc(sizeof(win_link_t));
  int rank;
  MPI_Comm_rank(group->comm,&rank);
  if (curr_win == NULL) {
    new_win->next = NULL;
    new_win->prev = NULL;
    new_win->win = win;
    igroup->win_list = new_win;
  } else {
    curr_win->next = new_win;
    new_win->next = NULL;
    new_win->prev = curr_win;
    new_win->win = win;
  }
}

/**
 *  Remove an MPI window from the window list in the group. This only removes
 *  the window from the group list, it does not get rid of the window itself.
 */
void cmx_igroup_delete_win(cmx_igroup_t *group, MPI_Win win)
{
  cmx_igroup_t *current_group_list_item = group;
  win_link_t *curr_win;
  win_link_t *prev_win;

  /* find the window in list */
  /* NOTE: Might be able to simplify this if win_link_t only has next and not
   * prev data members. Need to see if prev is used anywhere */
  curr_win = current_group_list_item->win_list;
  if (curr_win != NULL) {
    while (curr_win->win != win) {
      curr_win = curr_win->next;
    }
    if (curr_win->prev != NULL && curr_win->next != NULL) {
      /* window is not at start or end of list */
      prev_win = curr_win->prev;
      prev_win->next = curr_win->next;
      (curr_win->next)->prev = prev_win;
    } else if (curr_win->prev == NULL && curr_win->next != NULL) {
      /* window is at the start of the list */
      (curr_win->next)->prev = NULL;
      current_group_list_item->win_list = curr_win->next;
    } else if (curr_win->next == NULL && curr_win->prev != NULL) {
      /* window is at the end of the list */
      (curr_win->prev)->next = NULL;
    } else {
      /* only one window in list */
      current_group_list_item->win_list = NULL;
    }
    free(curr_win);
  }
}
