/**
 * Private header file for cmx groups backed by MPI_comm.
 *
 * The rest of the cmx group functions are defined in the public cmx.h.
 *
 * @author Bruce Palmer
 */
#ifndef _CMX_GROUPS_H_
#define _CMX_GROUPS_H_

#include <mpi.h>

#include "cmx_impl.h"
#include "cmx.h"

extern void cmx_group_init();
extern void cmx_group_finalize();
extern void cmx_igroup_add_win(cmx_igroup_t *group, MPI_Win win);
extern void cmx_igroup_delete_win(cmx_igroup_t *group, MPI_Win win);

#endif /* _CMX_GROUPS_H_ */
