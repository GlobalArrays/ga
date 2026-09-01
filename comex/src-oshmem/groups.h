/**
 * @file groups.h
 * @brief Group management for the OpenSHMEM COMEX backend.
 *
 * This backend implements a minimal group interface. Only the world group
 * (`COMEX_GROUP_WORLD`) is supported. Attempts to create or use subgroups
 * will fail with `COMEX_FAILURE`.
 */

#ifndef COMEX_GROUPS_H
#define COMEX_GROUPS_H

#include "comex.h"

/** Initialize the group subsystem. For the SHMEM backend this sets up any
 *  internal state required for supporting the world group. */
void comex_group_init();

/** Finalize the group subsystem and release any resources. */
void comex_group_finalize();

/**
 * Determine the calling process's rank in the given group.
 *
 * For this backend only `COMEX_GROUP_WORLD` is supported.
 *
 * @param[in]  group group handle (must be COMEX_GROUP_WORLD)
 * @param[out] rank  pointer to store the rank
 * @return COMEX_SUCCESS on success, COMEX_FAILURE if group is unsupported
 */
int comex_group_rank(comex_group_t group, int *rank);

/**
 * Return the size (number of processes) in the given group.
 *
 * For this backend only `COMEX_GROUP_WORLD` is supported.
 *
 * @param[in]  group group handle
 * @param[out] size  pointer to store the number of processes
 * @return COMEX_SUCCESS on success, COMEX_FAILURE if group is unsupported
 */
int comex_group_size(comex_group_t group, int *size);

/**
 * Group creation is not supported in this backend. This function always
 * returns `COMEX_FAILURE`.
 */
int comex_group_create(int n, int *pid_list, comex_group_t group, comex_group_t *new_group);

/** Destroy a group. No-op for the world-only backend. */
int comex_group_free(comex_group_t group);

/**
 * Return an MPI communicator corresponding to the given COMEX group.
 *
 * When `group == COMEX_GROUP_WORLD`, this function returns `MPI_COMM_WORLD`.
 * For any other group it returns `COMEX_FAILURE`.
 *
 * @param[in]  group comex group handle
 * @param[out] comm  pointer to MPI_Comm to be filled on success
 * @return COMEX_SUCCESS or COMEX_FAILURE
 */
int comex_group_comm(comex_group_t group, MPI_Comm *comm);

#endif /* COMEX_GROUPS_H */
