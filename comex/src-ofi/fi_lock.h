/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _FI_LOCK_H_
#define _FI_LOCK_H_

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#if PT_LOCK_SPIN == 1

#define fastlock_t_ pthread_spinlock_t
#define fastlock_init_(lock) pthread_spin_init(lock, PTHREAD_PROCESS_PRIVATE)
#define fastlock_destroy_(lock) pthread_spin_destroy(lock)
#define fastlock_acquire_(lock) pthread_spin_lock(lock)
#define fastlock_tryacquire_(lock) pthread_spin_trylock(lock)
#define fastlock_release_(lock) pthread_spin_unlock(lock)

#else

#define fastlock_t_ pthread_mutex_t
#define fastlock_init_(lock) pthread_mutex_init(lock, NULL)
#define fastlock_destroy_(lock) pthread_mutex_destroy(lock)
#define fastlock_acquire_(lock) pthread_mutex_lock(lock)
#define fastlock_tryacquire_(lock) pthread_mutex_trylock(lock)
#define fastlock_release_(lock) pthread_mutex_unlock(lock)

#endif /* PT_LOCK_SPIN */

#if ENABLE_DEBUG

typedef struct {
	fastlock_t_ impl;
	int is_initialized;
} fastlock_t;

static inline int fastlock_init(fastlock_t *lock)
{
	int ret;

	ret = fastlock_init_(&lock->impl);
	lock->is_initialized = !ret;
	return ret;
}

static inline void fastlock_destroy(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	lock->is_initialized = 0;
	ret = fastlock_destroy_(&lock->impl);
	assert(!ret);
}

static inline void fastlock_acquire(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = fastlock_acquire_(&lock->impl);
	assert(!ret);
}

static inline int fastlock_tryacquire(fastlock_t *lock)
{
	assert(lock->is_initialized);
	return fastlock_tryacquire_(&lock->impl);
}

static inline void fastlock_release(fastlock_t *lock)
{
	int ret;

	assert(lock->is_initialized);
	ret = fastlock_release_(&lock->impl);
	assert(!ret);
}

#else /* !ENABLE_DEBUG */

#  define fastlock_t fastlock_t_
#  define fastlock_init(lock) fastlock_init_(lock)
#  define fastlock_destroy(lock) fastlock_destroy_(lock)
#  define fastlock_acquire(lock) fastlock_acquire_(lock)
#  define fastlock_tryacquire(lock) fastlock_tryacquire_(lock)
#  define fastlock_release(lock) fastlock_release_(lock)

#endif



#ifdef __cplusplus
}
#endif

#endif /* _FI_LOCK_H_ */
