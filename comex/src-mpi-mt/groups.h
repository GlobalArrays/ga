/**
 * Private header file for comex groups backed by MPI_comm.
 *
 * The rest of the comex group functions are defined in the public comex.h.
 *
 * @author Jeff Daily
 */
#ifndef _COMEX_GROUPS_H_
#define _COMEX_GROUPS_H_

#include <mpi.h>

#include "comex.h"

typedef struct {
    MPI_Comm comm;  /**< whole comm; all ranks */
    MPI_Group group;/**< whole group; all ranks */
    int size;       /**< comm size */
    int rank;       /**< comm rank */
    long *hostid;   /**< hostid[size] hostid of SMP node for a given rank */
    MPI_Comm node_comm;  /**< node comm; SMP ranks */
    int node_size;       /**< node comm size */
    int node_rank;       /**< node comm rank */
} comex_group_world_t;

extern comex_group_world_t g_state;

typedef struct group_link {
    struct group_link *next;/**< next group in linked list */
    comex_group_t id;       /**< user-space id for this group */
    MPI_Comm comm;          /**< whole comm; all ranks */
    MPI_Group group;        /**< whole group; all ranks */
    int size;               /**< comm size */
    int rank;               /**< comm rank */
} comex_igroup_t;

/** list of worker groups */
extern comex_igroup_t *group_list;

extern void comex_group_init();
extern void comex_group_finalize();
extern comex_igroup_t* comex_get_igroup_from_group(comex_group_t group);

/* verify that proc is part of group */
#define CHECK_GROUP(GROUP,PROC) do {                                \
    int size;                                                       \
    COMEX_ASSERT(GROUP >= 0);                                       \
    COMEX_ASSERT(COMEX_SUCCESS == comex_group_size(GROUP,&size));   \
    COMEX_ASSERT(PROC >= 0);                                        \
    COMEX_ASSERT(PROC < size);                                      \
} while(0)

#endif /* _COMEX_GROUPS_H_ */
