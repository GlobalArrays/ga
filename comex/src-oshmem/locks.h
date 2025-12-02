#ifndef COMEX_LOCKS_H
#define COMEX_LOCKS_H

#include <stddef.h>

/* Create num user mutexes (per-PE). Returns COMEX_SUCCESS or COMEX_FAILURE. */
int comex_create_mutexes(int num);
int comex_destroy_mutexes(void);

/* Lock/unlock API: mutex index is 0..(g_num_mutexes-1) and proc is target PE */
int comex_lock(int mutex, int proc);
int comex_unlock(int mutex, int proc);

/* Internal helpers for reserved per-PE lock access (used by comex_acc)
 * locks_set_internal(pe) / locks_clear_internal(pe) acquire/release the
 * reserved per-PE lock slot. */
void locks_set_internal(int pe);
void locks_clear_internal(int pe);

#endif /* COMEX_LOCKS_H */
