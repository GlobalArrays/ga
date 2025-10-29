/**
 * @file comex_impl.h
 * @brief Local backend state and SHMEM includes for the OpenSHMEM COMEX
 *        backend.
 *
 * This header defines the backend-local `local_state` structure which
 * stores the current PE id and the number of PEs. It also includes the
 * appropriate OpenSHMEM header available on the system.
 */

#ifndef COMEX_IMPL_H_
#define COMEX_IMPL_H_

#include <stddef.h>

/* Try both common SHMEM headers */
#if defined(__has_include)
# if __has_include(<shmem.h>)
#  include <shmem.h>
# elif __has_include(<openshmem.h>)
#  include <openshmem.h>
# else
#  include <shmem.h>
# endif
#else
/* If the compiler doesn't support __has_include try the common name */
# include <shmem.h>
#endif

/**
 * Local state for the SHMEM COMEX backend.
 *
 * - `pe` is the calling process's PE id (0..n_pes-1).
 * - `n_pes` is the total number of PEs.
 */
typedef struct {
    int pe;      /**< my PE id */
    int n_pes;   /**< number of PEs */
} local_state;

/** Backend-local state instance (defined in comex.c). */
extern local_state l_state;

#endif /* COMEX_IMPL_H_ */
