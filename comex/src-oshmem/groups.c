#include "comex_impl.h"
#include "groups.h"
#include "comex.h"

#include <assert.h>

void comex_group_init() {
    /* world group is implicit for SHMEM backend */
}

void comex_group_finalize() {
    /* nothing to free for world-only implementation */
}

int comex_group_rank(comex_group_t group, int *rank) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!rank) return COMEX_FAILURE;
    *rank = l_state.pe;
    return COMEX_SUCCESS;
}

int comex_group_size(comex_group_t group, int *size) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (!size) return COMEX_FAILURE;
    *size = l_state.n_pes;
    return COMEX_SUCCESS;
}

int comex_group_create(int n, int *pid_list, comex_group_t group, comex_group_t *new_group) {
    /* Subgroups are not supported in this backend */
    (void)n; (void)pid_list; (void)group; (void)new_group;
    return COMEX_FAILURE;
}

int comex_group_free(comex_group_t group) {
    (void)group; /* nothing to free */
    return COMEX_SUCCESS;
}

/**
 * Return an MPI communicator that corresponds to the given COMEX group.
 *
 * For the OpenSHMEM backend we only support the world group. When
 * `group == COMEX_GROUP_WORLD` this function returns `MPI_COMM_WORLD` in
 * `*comm`. For any other group, the backend does not support subgroup
 * communicators and the call will fail returning `COMEX_FAILURE`.
 *
 * Note: returning `MPI_COMM_WORLD` provides a convenient communicator for
 * code paths that expect an MPI communicator for the global group. The
 * returned communicator must not be freed by the caller.
 *
 * @param[in]  group comex group handle
 * @param[out] comm  pointer to MPI_Comm to be filled (on success)
 * @return COMEX_SUCCESS when `group == COMEX_GROUP_WORLD`, otherwise
 *         COMEX_FAILURE
 */
int comex_group_comm(comex_group_t group, MPI_Comm *comm) {
    if (group != COMEX_GROUP_WORLD) return COMEX_FAILURE;
    if (comm) *comm = MPI_COMM_WORLD;
    return COMEX_SUCCESS;
}
