#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include <mpi.h>

#if defined(__CRAYXT) || defined(__CRAYXE)
#  include <pmi.h> 
#endif

#include "cmx.h"
#include "cmx_impl.h"
#include "groups.h"


/* world group state */
cmx_group_world_t g_state = {
    MPI_COMM_NULL,
    MPI_GROUP_NULL,
    -1,
    -1,
    NULL,
    NULL,
    MPI_COMM_NULL,
    -1,
    -1
};

/* the HEAD of the group linked list */
cmx_igroup_t *group_list = NULL;

#define RANK_OR_PID (g_state.rank >= 0 ? g_state.rank : getpid())

/* static functions implemented in this file */
static void _create_igroup(cmx_igroup_t **igroup);
static void _igroup_free(cmx_igroup_t *igroup);
static long xgethostid();


/**
 * Creates and associates a cmx group with a cmx igroup.
 *
 * This does *not* initialize the members of the cmx igroup.
 */
static void _create_igroup(cmx_igroup_t **igroup)
{
    cmx_igroup_t *new_group_list_item = NULL;
    cmx_igroup_t *last_group_list_item = NULL;

#if DEBUG
    printf("[%d] _create_group(...)\n", RANK_OR_PID);
#endif

    /* create, init, and insert the new node for the linked list */
    new_group_list_item = malloc(sizeof(cmx_igroup_t));
    new_group_list_item->next = NULL;
    new_group_list_item->comm = MPI_COMM_NULL;
    new_group_list_item->group = MPI_GROUP_NULL;
    new_group_list_item->size = -1;
    new_group_list_item->rank = -1;
    new_group_list_item->world_ranks = NULL;

    /* find the last group in the group linked list and insert */
    if (group_list) {
        last_group_list_item = group_list;
        while (last_group_list_item->next != NULL) {
            last_group_list_item = last_group_list_item->next;
        }
        last_group_list_item->next = new_group_list_item;
    }
    else {
        group_list = new_group_list_item;
    }

    /* return the group id and cmx igroup */
    *igroup = new_group_list_item;
}


int cmx_group_rank(cmx_igroup_t *igroup, int *rank)
{
    *rank = igroup->rank;

#if DEBUG
    printf("[%d] cmx_group_rank(group=%d, *rank=%d)\n",
            RANK_OR_PID, group, *rank);
#endif

    return CMX_SUCCESS;
}


int cmx_group_size(cmx_igroup_t *igroup, int *size)
{
    *size = igroup->size;

#if DEBUG
    printf("[%d] cmx_group_size(group=%d, *size=%d)\n",
            RANK_OR_PID, group, *size);
#endif

    return CMX_SUCCESS;
}


int cmx_group_comm(cmx_igroup_t *igroup, MPI_Comm *comm)
{
    *comm = igroup->comm;

#if DEBUG
    printf("[%d] cmx_group_comm(group=%d, comm)\n",
            RANK_OR_PID, group);
#endif

    return CMX_SUCCESS;
}

int cmx_group_translate_ranks(int n, cmx_group_t group_from,
    int *ranks_from, cmx_group_t group_to, int *ranks_to)
{
  int i;
  if (group_from == group_to) {
    for (i=0; i<n; i++) {
      ranks_to[i] = ranks_from[i];
    }
  } else {
    int status;
    status = MPI_Group_translate_ranks(group_from->group, n, ranks_from,
        group_to->group, ranks_to);
    if (status != MPI_SUCCESS) {
      cmx_error("MPI_Group_translate_ranks: Failed ", status);
    }
  }
  return CMX_SUCCESS;
}

int cmx_group_translate_world(
        cmx_igroup_t *igroup, int group_rank, int *world_rank)
{
#if DEBUG
    printf("[%d] cmx_group_translate_world("
            "group=%d, group_rank=%d, world_rank)\n",
            RANK_OR_PID, group, group_rank);
#endif

    if (CMX_GROUP_WORLD == igroup) {
        *world_rank = group_rank;
    }
    else {
        int status;

        CMX_ASSERT(group_list); /* first group is world worker group */
        status = MPI_Group_translate_ranks(igroup->group, 1, &group_rank,
                group_list->group, world_rank);
    }

    return CMX_SUCCESS;
}


/**
 * Destroys the given cmx igroup.
 */
static void _igroup_free(cmx_igroup_t *igroup)
{
    int status;

#if DEBUG
    printf("[%d] _igroup_free\n",
            RANK_OR_PID);
#endif

    CMX_ASSERT(igroup);

    if (igroup->group != MPI_GROUP_NULL) {
        status = MPI_Group_free(&igroup->group);
        if (status != MPI_SUCCESS) {
            cmx_error("MPI_Group_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd group\n", RANK_OR_PID);
#endif

    if (igroup->comm != MPI_COMM_NULL) {
        status = MPI_Comm_free(&igroup->comm);
        if (status != MPI_SUCCESS) {
            cmx_error("MPI_Comm_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd comm\n", RANK_OR_PID);
#endif
    if (igroup->world_ranks != NULL) {
      free(igroup->world_ranks);
    }

    free(igroup);
}


int cmx_group_free(cmx_igroup_t *igroup)
{
    cmx_igroup_t *previous_group_list_item = NULL;
    cmx_igroup_t *current_group_list_item = group_list;

#if DEBUG
    printf("[%d] cmx_group_free(id=%d)\n", RANK_OR_PID, id);
#endif

    /* find the group to free */
    while (current_group_list_item != NULL) {
      if (current_group_list_item == igroup) {
        break;
      }
      previous_group_list_item = current_group_list_item;
      current_group_list_item = current_group_list_item->next;
    }

    /* make sure we found a group */
    CMX_ASSERT(current_group_list_item != NULL);
    /* remove the group from the linked list */
    if (previous_group_list_item != NULL) {
        previous_group_list_item->next = current_group_list_item->next;
    }
    /* free the igroup */
    _igroup_free(current_group_list_item);

    return CMX_SUCCESS;
}

void _igroup_set_world_ranks(cmx_igroup_t *igroup)
{
  int i = 0;
  int my_world_rank = g_state.rank;
  igroup->world_ranks = (int*)malloc(sizeof(int)*igroup->size);
  int status;

  for (i=0; i<igroup->size; ++i) {
    igroup->world_ranks[i] = MPI_PROC_NULL;
  }

  status = MPI_Allgather(&my_world_rank,1,MPI_INT,igroup->world_ranks,
      1,MPI_INT,igroup->comm);
  CMX_ASSERT(MPI_SUCCESS == status);

  for (i=0; i<igroup->size; ++i) {
    CMX_ASSERT(MPI_PROC_NULL != igroup->world_ranks[i]);
  }
}

int cmx_group_create(
        int n, int *pid_list, cmx_igroup_t *igroup_parent, cmx_igroup_t **igroup_child)
{
    int status = 0;
    int grp_me = 0;
    MPI_Group      *group_child = NULL;
    MPI_Comm       *comm_child = NULL;
    MPI_Group      *group_parent = NULL;
    MPI_Comm       *comm_parent = NULL;

#if DEBUG
    printf("[%d] cmx_group_create("
            "n=%d, pid_list=%p, id_parent=%d, id_child)\n",
            RANK_OR_PID, n, pid_list, id_parent);
    {
        int p;
        printf("[%d] pid_list={%d", RANK_OR_PID, pid_list[0]);
        for (p=1; p<n; ++p) {
            printf(",%d", pid_list[p]);
        }
        printf("}\n");
    }
#endif

    /* create the node in the linked list of groups and */
    /* get the child's MPI_Group and MPI_Comm, to be populated shortly */
    _create_igroup(igroup_child);
    group_child = &((*igroup_child)->group);
    comm_child  = &((*igroup_child)->comm);

    /* get the parent's MPI_Group and MPI_Comm */
    group_parent = &(igroup_parent->group);
    comm_parent  = &(igroup_parent->comm);

    status = MPI_Group_incl(*group_parent, n, pid_list, group_child);
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    printf("[%d] cmx_group_create before crazy logic\n", RANK_OR_PID);
#endif
    {
        MPI_Comm comm, comm1, comm2;
        int lvl=1, local_ldr_pos;
        status = MPI_Group_rank(*group_child, &grp_me);
        CMX_ASSERT(MPI_SUCCESS == status);
        if (grp_me == MPI_UNDEFINED) {
            /* FIXME: keeping the group around for now */
#if DEBUG
    printf("[%d] cmx_group_create aborting -- not in group\n", RANK_OR_PID);
#endif
            return CMX_SUCCESS;
        }
        /* SK: sanity check for the following bitwise operations */
        CMX_ASSERT(grp_me>=0);
        /* FIXME: can be optimized away */
        status = MPI_Comm_dup(MPI_COMM_SELF, &comm);
        CMX_ASSERT(MPI_SUCCESS == status);
        local_ldr_pos = grp_me;
        while(n>lvl) {
            int tag=0;
            int remote_ldr_pos = local_ldr_pos^lvl;
            if (remote_ldr_pos < n) {
                int remote_leader = pid_list[remote_ldr_pos];
                MPI_Comm peer_comm = *comm_parent;
                int high = (local_ldr_pos<remote_ldr_pos)?0:1;
                status = MPI_Intercomm_create(
                        comm, 0, peer_comm, remote_leader, tag, &comm1);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Intercomm_merge(comm1, high, &comm2);
                CMX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm1);
                CMX_ASSERT(MPI_SUCCESS == status);
                comm = comm2;
            }
            local_ldr_pos &= ((~0)^lvl);
            lvl<<=1;
        }
        *comm_child = comm;
        /* cleanup temporary group (from MPI_Group_incl above) */
        status = MPI_Group_free(group_child);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* get the actual group associated with comm */
        status = MPI_Comm_group(*comm_child, group_child);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* rank and size of new comm */
        status = MPI_Comm_size((*igroup_child)->comm, &((*igroup_child)->size));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_rank((*igroup_child)->comm, &((*igroup_child)->rank));
        CMX_ASSERT(MPI_SUCCESS == status);
    }
#if DEBUG
    printf("[%d] cmx_group_create after crazy logic\n", RANK_OR_PID);
#endif
    _igroup_set_world_ranks(*igroup_child);

    return CMX_SUCCESS;
}


static int cmplong(const void *p1, const void *p2)
{
    return *((long*)p1) - *((long*)p2);
}

/**
 * Initialize group linked list. Prepopulate with world group.
 */
void cmx_group_init() 
{
    int status = 0;
    int i = 0;
    int smallest_rank_with_same_hostid = 0;
    int largest_rank_with_same_hostid = 0;
    int size_node = 0;
    cmx_igroup_t *igroup = NULL;
    long *sorted = NULL;
    int count = 0;
    
    /* populate g_state */

    /* dup MPI_COMM_WORLD and get group, rank, and size */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(g_state.comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_group(g_state.comm, &(g_state.group));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_rank(g_state.comm, &(g_state.rank));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_size(g_state.comm, &(g_state.size));
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG_TO_FILE
    {
        char pathname[80];
        sprintf(pathname, "trace.%d.log", g_state.rank);
        cmx_trace_file = fopen(pathname, "w");
        CMX_ASSERT(NULL != cmx_trace_file);

        printf("[%d] cmx_group_init()\n", RANK_OR_PID);
    }
#endif

    /* need to figure out which proc is master on each node */
    g_state.hostid = (long*)malloc(sizeof(long)*g_state.size);
    g_state.hostid[g_state.rank] = xgethostid();
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
            g_state.hostid, 1, MPI_LONG, g_state.comm);
    CMX_ASSERT(MPI_SUCCESS == status);
     /* First create a temporary node communicator and then
      * split further into number of gruoups within the node */
     MPI_Comm temp_node_comm;
     int temp_node_size;
    /* create node comm */
    /* MPI_Comm_split requires a non-negative color,
     * so sort and sanitize */
    sorted = (long*)malloc(sizeof(long) * g_state.size);
    (void)memcpy(sorted, g_state.hostid, sizeof(long)*g_state.size);
    qsort(sorted, g_state.size, sizeof(long), cmplong);
    for (i=0; i<g_state.size-1; ++i) {
        if (sorted[i] == g_state.hostid[g_state.rank]) 
        {
            break;
        }
        if (sorted[i] != sorted[i+1]) {
            count += 1;
        }
    }
    free(sorted);
#if DEBUG
    printf("count: %d\n", count);
#endif
    status = MPI_Comm_split(MPI_COMM_WORLD, count,
            g_state.rank, &temp_node_comm);
    int node_group_size, node_group_rank;
    MPI_Comm_size(temp_node_comm, &node_group_size);
    MPI_Comm_rank(temp_node_comm, &node_group_rank);
    int node_rank0, num_nodes;
    node_rank0 = (node_group_rank == 0) ? 1 : 0;
    MPI_Allreduce(&node_rank0, &num_nodes, 1, MPI_INT, MPI_SUM,
        g_state.comm);
    smallest_rank_with_same_hostid = g_state.rank;
    largest_rank_with_same_hostid = g_state.rank;
    for (i=0; i<g_state.size; ++i) {
        if (g_state.hostid[i] == g_state.hostid[g_state.rank]) {
            ++size_node;
            if (i < smallest_rank_with_same_hostid) {
                smallest_rank_with_same_hostid = i;
            }
            if (i > largest_rank_with_same_hostid) {
                largest_rank_with_same_hostid = i;
            }
        }
    }
    /* Get number of Progress-Ranks per node from environment variable
     * equal to 1 by default */
    int num_progress_ranks_per_node = get_num_progress_ranks_per_node();
    /* Perform check on the number of Progress-Ranks */
    if (size_node < 2 * num_progress_ranks_per_node) {  
        cmx_error("ranks per node, must be at least", 
            2 * num_progress_ranks_per_node);
    }
    if (size_node % num_progress_ranks_per_node > 0) {  
        cmx_error("number of ranks per node must be multiple of number of process groups per node", -1);
    }
    int is_node_ranks_packed = get_progress_rank_distribution_on_node();
    int split_group_size;
    split_group_size = node_group_size / num_progress_ranks_per_node;
     MPI_Comm_free(&temp_node_comm);
    g_state.master = (int*)malloc(sizeof(int)*g_state.size);
    g_state.master[g_state.rank] = get_my_master_rank_with_same_hostid(g_state.rank, 
        split_group_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
        num_progress_ranks_per_node, is_node_ranks_packed);
#if DEBUG
    printf("[%d] rank; split_group_size: %d\n", g_state.rank, split_group_size);
    printf("[%d] rank; largest_rank_with_same_hostid[%d]; my master is:[%d]\n",
        g_state.rank, largest_rank_with_same_hostid, g_state.master[g_state.rank]);
#endif
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
            g_state.master, 1, MPI_INT, g_state.comm);
    CMX_ASSERT(MPI_SUCCESS == status);

    CMX_ASSERT(group_list == NULL);

    // put split group stamps
    int proc_split_group_stamp;
    int num_split_groups;
    num_split_groups = num_nodes * num_progress_ranks_per_node;
    int* split_group_list = (int*)malloc(sizeof(int)*num_split_groups);
    int split_group_index = 0;
    int j;
    for (i=0; i<g_state.size; i++) {
      for (j=0; j<i; j++) {
        if (g_state.master[i] == g_state.master[j])
            break; 
      }
      if(i == j) {
        split_group_list[split_group_index] = g_state.master[i];
        split_group_index++;
      }
    }
    // label each process
    for (j=0; j<num_split_groups; j++) {
      if (split_group_list[j] == g_state.master[g_state.rank]) {
        proc_split_group_stamp = j;
      }
    }
#if DEBUG
    printf("proc_split_group_stamp[%ld]: %ld\n", 
       g_state.rank, proc_split_group_stamp);
#endif
    free(split_group_list);
    /* create a comm of only the workers */
    if (g_state.master[g_state.rank] == g_state.rank) {
        /* I'm a master */
        MPI_Comm delete_me;
        status = MPI_Comm_split(g_state.comm, 0, g_state.rank, &delete_me);
        CMX_ASSERT(MPI_SUCCESS == status);
        /* masters don't need their own comm */
        if (MPI_COMM_NULL != delete_me) {
            MPI_Comm_free(&delete_me);
        }
        CMX_GROUP_WORLD = NULL;
#if DEBUG
        printf("Creating comm: I AM MASTER[%ld]\n", g_state.rank);
#endif
    } else {
        /* I'm a worker */
        /* create the head of the group linked list */
        igroup = (cmx_igroup_t*)malloc(sizeof(cmx_igroup_t));
        status = MPI_Comm_split(
                g_state.comm, 1, g_state.rank, &(igroup->comm));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_group(igroup->comm, &(igroup->group));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_rank(igroup->comm, &(igroup->rank));
        CMX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_size(igroup->comm, &(igroup->size));
        igroup->next = NULL;
        CMX_ASSERT(MPI_SUCCESS == status);
        _igroup_set_world_ranks(igroup);
        CMX_ASSERT(igroup->world_ranks != NULL);
#if DEBUG
        printf("Creating comm: I AM WORKER[%ld]\n", g_state.rank);
#endif
        CMX_GROUP_WORLD = igroup;
        group_list = igroup;
    }
    status = MPI_Comm_split(MPI_COMM_WORLD, proc_split_group_stamp,
            g_state.rank, &(g_state.node_comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    /* node rank */
    status = MPI_Comm_rank(g_state.node_comm, &(g_state.node_rank));
    CMX_ASSERT(MPI_SUCCESS == status);
    /* node size */
    status = MPI_Comm_size(g_state.node_comm, &(g_state.node_size));
    CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    printf("node_rank[%d]/ size[%d]\n", g_state.node_rank, g_state.node_size);
    if (g_state.master[g_state.rank] == g_state.rank) {
        printf("[%d] world %d/%d\tI'm a master\n",
            RANK_OR_PID, g_state.rank, g_state.size);
    }
    else {
        printf("[%d] world %d/%d\tI'm a worker\n",
            RANK_OR_PID, g_state.rank, g_state.size);
    }
#endif
}


void cmx_group_finalize()
{
    int status;
    cmx_igroup_t *current_group_list_item = group_list;
    cmx_igroup_t *previous_group_list_item = NULL;

#if DEBUG
    printf("[%d] cmx_group_finalize()\n", RANK_OR_PID);
#endif

    /* This loop will also clean up CMX_GROUP_WORLD */
    while (current_group_list_item != NULL) {
        previous_group_list_item = current_group_list_item;
        current_group_list_item = current_group_list_item->next;
        _igroup_free(previous_group_list_item);
    }

    free(g_state.master);
    free(g_state.hostid);
    status = MPI_Comm_free(&(g_state.node_comm));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Group_free(&(g_state.group));
    CMX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_free(&(g_state.comm));
    CMX_ASSERT(MPI_SUCCESS == status);
}


static long xgethostid()
{
#if defined(__CRAYXT) || defined(__CRAYXE)
#warning CRAY
    int nodeid;
#  if defined(__CRAYXT)
    PMI_Portals_get_nid(g_state.rank, &nodeid);
#  elif defined(__CRAYXE)
    PMI_Get_nid(g_state.rank, &nodeid);
#  endif
#else
    long nodeid = gethostid();
#endif

    return nodeid;
}
