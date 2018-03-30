#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include <mpi.h>

#if defined(__bgp__)
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#elif defined(__bgq__)
#  include <mpix.h>
#elif defined(__CRAYXT) || defined(__CRAYXE)
#  include <pmi.h> 
#endif

#include "comex.h"
#include "comex_impl.h"
#include "groups.h"


/* world group state */
comex_group_world_t g_state = {
    MPI_COMM_NULL,
    MPI_GROUP_NULL,
    -1,
    -1,
    NULL,
    NULL
};
/* the HEAD of the group linked list */
comex_igroup_t *group_list = NULL;

#define RANK_OR_PID (g_state.rank >= 0 ? g_state.rank : getpid())

/* static functions implemented in this file */
static void _create_group_and_igroup(comex_group_t *id, comex_igroup_t **igroup);
static void _igroup_free(comex_igroup_t *igroup);
static long xgethostid();


/**
 * Return the comex igroup instance given the group id.
 *
 * The group linked list is searched sequentially until the given group
 * is found. It is an error if this function is called before
 * comex_group_init(). An error occurs if the given group is not found.
 */
comex_igroup_t* comex_get_igroup_from_group(comex_group_t id)
{
    comex_igroup_t *current_group_list_item = group_list;

#if DEBUG
    printf("[%d] comex_get_igroup_from_group(%d)\n", RANK_OR_PID, id);
#endif

    COMEX_ASSERT(group_list != NULL);
    while (current_group_list_item != NULL) {
        if (current_group_list_item->id == id) {
            return current_group_list_item;
        }
        current_group_list_item = current_group_list_item->next;
    }
    comex_error("comex group lookup failed", -1);

    return NULL;
}


/**
 * Creates and associates a comex group with a comex igroup.
 *
 * This does *not* initialize the members of the comex igroup.
 */
static void _create_group_and_igroup(
        comex_group_t *id, comex_igroup_t **igroup)
{
    comex_igroup_t *new_group_list_item = NULL;
    comex_igroup_t *last_group_list_item = NULL;

#if DEBUG
    printf("[%d] _create_group_and_igroup(...)\n", RANK_OR_PID);
#endif

    /* create, init, and insert the new node for the linked list */
    new_group_list_item = malloc(sizeof(comex_igroup_t));
    new_group_list_item->next = NULL;
    new_group_list_item->id = -1;
    new_group_list_item->comm = MPI_COMM_NULL;
    new_group_list_item->group = MPI_GROUP_NULL;
    new_group_list_item->size = -1;
    new_group_list_item->rank = -1;

    /* find the last group in the group linked list and insert */
    if (group_list) {
        last_group_list_item = group_list;
        while (last_group_list_item->next != NULL) {
            last_group_list_item = last_group_list_item->next;
        }
        last_group_list_item->next = new_group_list_item;
        new_group_list_item->id = last_group_list_item->id + 1;
    }
    else {
        group_list = new_group_list_item;
        new_group_list_item->id = COMEX_GROUP_WORLD;
    }

    /* return the group id and comex igroup */
    *igroup = new_group_list_item;
    *id = new_group_list_item->id;
}


int comex_group_rank(comex_group_t group, int *rank)
{
    comex_igroup_t *igroup = comex_get_igroup_from_group(group);
    *rank = igroup->rank;

#if DEBUG
    printf("[%d] comex_group_rank(group=%d, *rank=%d)\n",
            RANK_OR_PID, group, *rank);
#endif

    return COMEX_SUCCESS;
}


int comex_group_size(comex_group_t group, int *size)
{
    comex_igroup_t *igroup = comex_get_igroup_from_group(group);
    *size = igroup->size;

#if DEBUG
    printf("[%d] comex_group_size(group=%d, *size=%d)\n",
            RANK_OR_PID, group, *size);
#endif

    return COMEX_SUCCESS;
}


int comex_group_comm(comex_group_t group, MPI_Comm *comm)
{
    comex_igroup_t *igroup = comex_get_igroup_from_group(group);
    *comm = igroup->comm;

#if DEBUG
    printf("[%d] comex_group_comm(group=%d, comm)\n",
            RANK_OR_PID, group);
#endif

    return COMEX_SUCCESS;
}


int comex_group_translate_world(
        comex_group_t group, int group_rank, int *world_rank)
{
#if DEBUG
    printf("[%d] comex_group_translate_world("
            "group=%d, group_rank=%d, world_rank)\n",
            RANK_OR_PID, group, group_rank);
#endif

    if (COMEX_GROUP_WORLD == group) {
        *world_rank = group_rank;
    }
    else {
        int status;
        comex_igroup_t *igroup = comex_get_igroup_from_group(group);

        COMEX_ASSERT(group_list); /* first group is world worker group */
        status = MPI_Group_translate_ranks(igroup->group, 1, &group_rank,
                group_list->group, world_rank);
    }

    return COMEX_SUCCESS;
}


/**
 * Destroys the given comex igroup.
 */
static void _igroup_free(comex_igroup_t *igroup)
{
    int status;

#if DEBUG
    printf("[%d] _igroup_free\n",
            RANK_OR_PID);
#endif

    COMEX_ASSERT(igroup);

    if (igroup->group != MPI_GROUP_NULL) {
        status = MPI_Group_free(&igroup->group);
        if (status != MPI_SUCCESS) {
            comex_error("MPI_Group_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd group\n", RANK_OR_PID);
#endif

    if (igroup->comm != MPI_COMM_NULL) {
        status = MPI_Comm_free(&igroup->comm);
        if (status != MPI_SUCCESS) {
            comex_error("MPI_Comm_free: Failed ", status);
        }
    }
#if DEBUG
    printf("[%d] free'd comm\n", RANK_OR_PID);
#endif

    free(igroup);
}


int comex_group_free(comex_group_t id)
{
    comex_igroup_t *current_group_list_item = group_list;
    comex_igroup_t *previous_group_list_item = NULL;

#if DEBUG
    printf("[%d] comex_group_free(id=%d)\n", RANK_OR_PID, id);
#endif

    /* find the group to free */
    while (current_group_list_item != NULL) {
        if (current_group_list_item->id == id) {
            break;
        }
        previous_group_list_item = current_group_list_item;
        current_group_list_item = current_group_list_item->next;
    }
    /* make sure we found a group */
    COMEX_ASSERT(current_group_list_item != NULL);
    /* remove the group from the linked list */
    if (previous_group_list_item != NULL) {
        previous_group_list_item->next = current_group_list_item->next;
    }
    /* free the igroup */
    _igroup_free(current_group_list_item);

    return COMEX_SUCCESS;
}


int comex_group_create(
        int n, int *pid_list, comex_group_t id_parent, comex_group_t *id_child)
{
    int status = 0;
    int grp_me = 0;
    comex_igroup_t *igroup_child = NULL;
    MPI_Group      *group_child = NULL;
    MPI_Comm       *comm_child = NULL;
    comex_igroup_t *igroup_parent = NULL;
    MPI_Group      *group_parent = NULL;
    MPI_Comm       *comm_parent = NULL;

#if DEBUG
    printf("[%d] comex_group_create("
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
    _create_group_and_igroup(id_child, &igroup_child);
    group_child = &(igroup_child->group);
    comm_child  = &(igroup_child->comm);

    /* get the parent's MPI_Group and MPI_Comm */
    igroup_parent = comex_get_igroup_from_group(id_parent);
    group_parent = &(igroup_parent->group);
    comm_parent  = &(igroup_parent->comm);

    status = MPI_Group_incl(*group_parent, n, pid_list, group_child);
    COMEX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    printf("[%d] comex_group_create before crazy logic\n", RANK_OR_PID);
#endif
    {
        MPI_Comm comm, comm1, comm2;
        int lvl=1, local_ldr_pos;
        status = MPI_Group_rank(*group_child, &grp_me);
        COMEX_ASSERT(MPI_SUCCESS == status);
        if (grp_me == MPI_UNDEFINED) {
            /* FIXME: keeping the group around for now */
#if DEBUG
    printf("[%d] comex_group_create aborting -- not in group\n", RANK_OR_PID);
#endif
            return COMEX_SUCCESS;
        }
        /* SK: sanity check for the following bitwise operations */
        COMEX_ASSERT(grp_me>=0);
        /* FIXME: can be optimized away */
        status = MPI_Comm_dup(MPI_COMM_SELF, &comm);
        COMEX_ASSERT(MPI_SUCCESS == status);
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
                COMEX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm);
                COMEX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Intercomm_merge(comm1, high, &comm2);
                COMEX_ASSERT(MPI_SUCCESS == status);
                status = MPI_Comm_free(&comm1);
                COMEX_ASSERT(MPI_SUCCESS == status);
                comm = comm2;
            }
            local_ldr_pos &= ((~0)^lvl);
            lvl<<=1;
        }
        *comm_child = comm;
        /* cleanup temporary group (from MPI_Group_incl above) */
        status = MPI_Group_free(group_child);
        COMEX_ASSERT(MPI_SUCCESS == status);
        /* get the actual group associated with comm */
        status = MPI_Comm_group(*comm_child, group_child);
        COMEX_ASSERT(MPI_SUCCESS == status);
        /* rank and size of new comm */
        status = MPI_Comm_size(igroup_child->comm, &(igroup_child->size));
        COMEX_ASSERT(MPI_SUCCESS == status);
        status = MPI_Comm_rank(igroup_child->comm, &(igroup_child->rank));
        COMEX_ASSERT(MPI_SUCCESS == status);
    }
#if DEBUG
    printf("[%d] comex_group_create after crazy logic\n", RANK_OR_PID);
#endif

    return COMEX_SUCCESS;
}


static int cmplong(const void *p1, const void *p2)
{
    return *((long*)p1) - *((long*)p2);
}


/**
 * Initialize group linked list. Prepopulate with world group.
 */
void comex_group_init() 
{
    int status = 0;
    int i = 0;
    int smallest_rank_with_same_hostid = 0;
    int largest_rank_with_same_hostid = 0;
    comex_group_t group = 0;
    comex_igroup_t *igroup = NULL;
    long *sorted = NULL;
    int count = 0;
    
    /* populate g_state */

    /* dup MPI_COMM_WORLD and get group, rank, and size */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(g_state.comm));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_group(g_state.comm, &(g_state.group));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_rank(g_state.comm, &(g_state.rank));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_size(g_state.comm, &(g_state.size));
    COMEX_ASSERT(MPI_SUCCESS == status);

#if DEBUG_TO_FILE
    {
        char pathname[80];
        sprintf(pathname, "trace.%d.log", g_state.rank);
        comex_trace_file = fopen(pathname, "w");
        COMEX_ASSERT(NULL != comex_trace_file);

        printf("[%d] comex_group_init()\n", RANK_OR_PID);
    }
#endif

    /* need to figure out which proc is master on each node */
    g_state.hostid = (long*)malloc(sizeof(long)*g_state.size);
    g_state.hostid[g_state.rank] = xgethostid();
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_LONG,
            g_state.hostid, 1, MPI_LONG, g_state.comm);
    COMEX_ASSERT(MPI_SUCCESS == status);

    smallest_rank_with_same_hostid = g_state.rank;
    largest_rank_with_same_hostid = g_state.rank;
    for (i=0; i<g_state.size; ++i) {
        if (g_state.hostid[i] == g_state.hostid[g_state.rank]) {
            if (i < smallest_rank_with_same_hostid) {
                smallest_rank_with_same_hostid = i;
            }
            if (i > largest_rank_with_same_hostid) {
                largest_rank_with_same_hostid = i;
            }
        }
    }

    g_state.master = (int*)malloc(sizeof(int)*g_state.size);
#if MASTER_IS_SMALLEST_SMP_RANK
    g_state.master[g_state.rank] = smallest_rank_with_same_hostid;
#else
    g_state.master[g_state.rank] = largest_rank_with_same_hostid;
#endif
    status = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
            g_state.master, 1, MPI_INT, g_state.comm);
    COMEX_ASSERT(MPI_SUCCESS == status);

    COMEX_ASSERT(group_list == NULL);

    /* create the head of the group linked list */
    _create_group_and_igroup(&group, &igroup);
    /* create a comm of only the workers (every rank is a worker) */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(igroup->comm));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_group(igroup->comm, &(igroup->group));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_rank(igroup->comm, &(igroup->rank));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_size(igroup->comm, &(igroup->size));
    COMEX_ASSERT(MPI_SUCCESS == status);

    /* create node comm */
    /* MPI_Comm_split requires a non-negative color,
     * so sort and sanitize */
    sorted = (long*)malloc(sizeof(long) * g_state.size);
    (void)memcpy(sorted, g_state.hostid, sizeof(long)*g_state.size);
    qsort(sorted, g_state.size, sizeof(long), cmplong);
    for (i=0; i<g_state.size-1; ++i) {
        if (sorted[i] == g_state.hostid[g_state.rank]) {
            break;
        }
        if (sorted[i] != sorted[i+1]) {
            count += 1;
        }
    }
    free(sorted);
    status = MPI_Comm_split(MPI_COMM_WORLD, count,
            g_state.rank, &(g_state.node_comm));
    COMEX_ASSERT(MPI_SUCCESS == status);
    /* node rank */
    status = MPI_Comm_rank(g_state.node_comm, &(g_state.node_rank));
    COMEX_ASSERT(MPI_SUCCESS == status);
    /* node size */
    status = MPI_Comm_size(g_state.node_comm, &(g_state.node_size));
    COMEX_ASSERT(MPI_SUCCESS == status);

#if DEBUG
    if (g_state.master[g_state.rank] == g_state.rank) {
        printf("[%d] world %d/%d\tI'm a worker with a master thread\n",
            RANK_OR_PID, g_state.rank, g_state.size);
    }
    else {
        printf("[%d] world %d/%d\tI'm a worker\n",
            RANK_OR_PID, g_state.rank, g_state.size);
    }
#endif
}


void comex_group_finalize()
{
    int status;
    comex_igroup_t *current_group_list_item = group_list;
    comex_igroup_t *previous_group_list_item = NULL;

#if DEBUG
    printf("[%d] comex_group_finalize()\n", RANK_OR_PID);
#endif

    while (current_group_list_item != NULL) {
        previous_group_list_item = current_group_list_item;
        current_group_list_item = current_group_list_item->next;
        _igroup_free(previous_group_list_item);
    }

    free(g_state.master);
    free(g_state.hostid);
    status = MPI_Comm_free(&(g_state.node_comm));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Group_free(&(g_state.group));
    COMEX_ASSERT(MPI_SUCCESS == status);
    status = MPI_Comm_free(&(g_state.comm));
    COMEX_ASSERT(MPI_SUCCESS == status);
}


static long xgethostid()
{
#if defined(__bgp__)
#warning BGP
    long nodeid;
    int matched,midplane,nodecard,computecard;
    char rack_row,rack_col;
    char location[128];
    char location_clean[128];
    (void) memset(location, '\0', 128);
    (void) memset(location_clean, '\0', 128);
    _BGP_Personality_t personality;
    Kernel_GetPersonality(&personality, sizeof(personality));
    BGP_Personality_getLocationString(&personality, location);
    matched = sscanf(location, "R%c%c-M%1d-N%2d-J%2d",
            &rack_row, &rack_col, &midplane, &nodecard, &computecard);
    COMEX_ASSERT(matched == 5);
    sprintf(location_clean, "%2d%02d%1d%02d%02d",
            (int)rack_row, (int)rack_col, midplane, nodecard, computecard);
    nodeid = atol(location_clean);
#elif defined(__bgq__)
#warning BGQ
    int nodeid;
    MPIX_Hardware_t hw;
    MPIX_Hardware(&hw);

    nodeid = hw.Coords[0] * hw.Size[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[2] * hw.Size[3] * hw.Size[4]
        + hw.Coords[3] * hw.Size[4]
        + hw.Coords[4];
#elif defined(__CRAYXT) || defined(__CRAYXE)
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
