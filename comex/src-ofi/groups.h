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
#include "comex_impl.h" 

typedef struct group_link {
    struct group_link *next;
    comex_group_t id;
    MPI_Comm comm;
    MPI_Group group;
} comex_igroup_t;

extern void comex_group_init();
extern void comex_group_finalize();
extern comex_igroup_t* comex_get_igroup_from_group(comex_group_t group);

#define VALIDATE_GROUP_AND_PROC(group, proc)                                              \
  do {                                                                                    \
      int group_size;                                                                     \
      EXPR_CHKANDJUMP((group >= 0), "invalid group");                                     \
      COMEX_CHKANDJUMP(comex_group_size(group, &group_size), "failed to get group size"); \
      EXPR_CHKANDJUMP((proc >= 0), "invalid proc");                                       \
      EXPR_CHKANDJUMP((proc < group_size), "invalid proc");                               \
  } while(0)

#endif /* _COMEX_GROUPS_H_ */
