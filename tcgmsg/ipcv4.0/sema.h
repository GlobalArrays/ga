/** @file
 * Header file declaring stubs for semaphore routines.
 *
 * These routines simplify the interface to semaphores for use in mutual
 * exclusion and queuing. Hopefully I can also make this portable.
 *
 * Interruption by signals is not tested for.
 *
 * An external routine Error is assumed which is called upon an error
 * and tidies up by calling SemSetDestroyAll.
 *
 * In most cases errors cause an internal hard failure (by calling Error).
 */
#ifndef SEMA_H_
#define SEMA_H_

/**
 * 1) make an array of n_sem semaphores, returning the id associated
 *    with the entire set. All the semaphore values are initialized to value
 *    which should be a positve integer (queuing) or 0 (synchronization).
 *    The semaphores in the set are indexed from 0 to n_sem-1.
 */
extern Integer SemSetCreate(Integer n_sem, Integer value);

/**
 * 2) Decrement and test the value associated with the semaphore specified by 
 *    (sem_set_id, sem_num). In effect this:
 *
 *    decrement value
 *
 *    if (value >= 0) {
 *       continue execution
 *   }
 *    else {
 *       wait in queue for the semaphore
 *   }
 */
extern void SemWait(Integer sem_set_id, Integer sem_num);

/**
 * 3) Increment the value associated with the semaphore specified by
 *    (sem_set_id, sem_num). If value <= 0 (i.e. there are processes
 *    in the queue) this releases the next process.
 */
extern void SemPost(Integer sem_set_id, Integer sem_num);
     
/**
 * 4) Return the current value associated with the semaphore sepcified by
 *    (sem_set_id, sem_num).
 */
extern Integer SemValue(Integer sem_set_id, Integer sem_num);

/**
 * 5) Destroy the set of semaphores. Any other processes that are accessing
 *    or try to access the semaphore set should get an error.
 *    On the SUN (all system V machines?) the semaphore sets should
 *    be destroyed explicitly before the final process exits.
 *    0 is returned if OK. -1 implies an error.
 */
extern Integer SemSetDestroy(Integer sem_set_id);

/**
 * 6) Destroy all the semaphore sets that are known about. This is really
 *    meant for an error routine to call to try and tidy up. Though all
 *    applications could call it before the last process exits.
 *    0 is returned if OK. -1 implies an error.
 */
extern Integer SemSetDestroyAll();

#endif /* SEMA_H_ */
