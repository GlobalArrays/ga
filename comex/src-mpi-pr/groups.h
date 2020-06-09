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
    int *master;    /**< master[size] rank of a given rank's master */
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
    int *world_ranks;       /**< list of ranks in MPI_COMM_WORLD */
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

static int get_num_progress_ranks_per_node()
{
    int num_progress_ranks_per_node;
    const char* num_progress_ranks_env_var = getenv("GA_NUM_PROGRESS_RANKS_PER_NODE");
    if (num_progress_ranks_env_var != NULL && num_progress_ranks_env_var[0] != '\0') {
       int env_number = atoi(getenv("GA_NUM_PROGRESS_RANKS_PER_NODE"));
       if ( env_number > 0 && env_number < 16)
           num_progress_ranks_per_node = env_number;
#if DEBUG
       printf("num_progress_ranks_per_node: %d\n", num_progress_ranks_per_node);
#endif
    }
    else {
        num_progress_ranks_per_node = 1;
    }
    return num_progress_ranks_per_node;
}

static int get_progress_rank_distribution_on_node() {
   const char* progress_ranks_packed_env_var = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_PACKED");
   const char* progress_ranks_cyclic_env_var = getenv("GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC");
   int is_node_ranks_packed;
   int rank_packed =0;
   int rank_cyclic =0;

   if (progress_ranks_packed_env_var != NULL && progress_ranks_packed_env_var[0] != '\0') {
     if (strchr(progress_ranks_packed_env_var, 'y') != NULL ||
            strchr(progress_ranks_packed_env_var, 'Y') != NULL ||
            strchr(progress_ranks_packed_env_var, '1') != NULL ) {
       rank_packed = 1;
     }
   }
   if (progress_ranks_cyclic_env_var != NULL && progress_ranks_cyclic_env_var[0] != '\0') {
     if (strchr(progress_ranks_cyclic_env_var, 'y') != NULL ||
            strchr(progress_ranks_cyclic_env_var, 'Y') != NULL ||
            strchr(progress_ranks_cyclic_env_var, '1') != NULL ) {
       rank_cyclic = 1;
     }
   }
   if (rank_packed == 1 || rank_cyclic == 0) is_node_ranks_packed = 1;
   if (rank_packed == 0 && rank_cyclic == 1) is_node_ranks_packed = 0;
   return is_node_ranks_packed;
}

static int get_my_master_rank_with_same_hostid(int rank, int split_group_size,
        int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
        int num_progress_ranks_per_node, int is_node_ranks_packed)
{
   int my_master;

#if MASTER_IS_SMALLEST_SMP_RANK
    if(is_node_ranks_packed) {
      /* Contiguous packing of ranks on a node */
      my_master = smallest_rank_with_same_hostid
           + split_group_size *
         ((rank - smallest_rank_with_same_hostid)/split_group_size);
    }
    else {
      if(num_progress_ranks_per_node == 1) { 
        my_master = 2 * (split_group_size *
           ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        /* Cyclic packing of ranks on a node between two sockets
         * with even and odd numbering  */
        if(rank % 2 == 0) {
          my_master = 2 * (split_group_size *
             ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
        } else {
          my_master = 1 + 2 * (split_group_size *
             ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
        }
      }
    }
#else
    /* By default creates largest SMP rank as Master */
    if(is_node_ranks_packed) {
      /* Contiguous packing of ranks on a node */
      my_master = largest_rank_with_same_hostid
           - split_group_size *
         ((largest_rank_with_same_hostid - rank)/split_group_size);
    }
    else {
      if(num_progress_ranks_per_node == 1) { 
        my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
           ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
        /* Cyclic packing of ranks on a node between two sockets
         * with even and odd numbering  */
        if(rank % 2 == 0) {
          my_master = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
             ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
        } else {
          my_master = largest_rank_with_same_hostid - 2 * (split_group_size *
             ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
        }
      }
    }
#endif
   return my_master;
}

static int get_my_rank_to_free(int rank, int split_group_size,
        int smallest_rank_with_same_hostid, int largest_rank_with_same_hostid,
        int num_progress_ranks_per_node, int is_node_ranks_packed)
{
   int my_rank_to_free;

#if MASTER_IS_SMALLEST_SMP_RANK
    /* By default creates largest SMP rank as Master */
    if(is_node_ranks_packed) {
      /* Contiguous packing of ranks on a node */
      my_rank_to_free = largest_rank_with_same_hostid
           - split_group_size *
         ((largest_rank_with_same_hostid - rank)/split_group_size);
    }
    else {
      if(num_progress_ranks_per_node == 1) { 
          my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
             ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
      } else {
          /* Cyclic packing of ranks on a node between two sockets
           * with even and odd numbering  */
          if(rank % 2 == 0) {
            my_rank_to_free = largest_rank_with_same_hostid - 1 - 2 * (split_group_size *
               ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
          } else {
            my_rank_to_free = largest_rank_with_same_hostid - 2 * (split_group_size *
               ( ((largest_rank_with_same_hostid - rank)/2) / split_group_size));
          }
      }
    }
#else
    if(is_node_ranks_packed) {
      /* Contiguous packing of ranks on a node */
      my_rank_to_free = smallest_rank_with_same_hostid
           + split_group_size *
         ((rank - smallest_rank_with_same_hostid)/split_group_size);
    }
    else {
      if(num_progress_ranks_per_node == 1) { 
          my_rank_to_free = 2 * (split_group_size *
             ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
      } else {
        /* Cyclic packing of ranks on a node between two sockets
         * with even and odd numbering  */
        if(rank % 2 == 0) {
          my_rank_to_free = 2 * (split_group_size *
             ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
        } else {
          my_rank_to_free = 1 + 2 * (split_group_size *
             ( ((rank - smallest_rank_with_same_hostid)/2) / split_group_size));
        }
      }
    }
#endif
   return my_rank_to_free;
}

#endif /* _COMEX_GROUPS_H_ */
