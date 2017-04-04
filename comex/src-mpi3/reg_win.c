/**
 * Registration window.
 *
 * Defensive programming via copious COMEX_ASSERT statements is encouraged.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* C headers */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3rd party headers */
#include <mpi.h>

/* our headers */
#include "comex.h"
#include "comex_impl.h"
#include "reg_win.h"

#define STATIC static inline

/* the static members in this module */
static reg_entry_t **reg_win = NULL; /* list of windows on each process. This
                                        array contains the starting node in a
                                        linked list */
static int reg_nprocs = 0; /* number of windows (one per process) */


/* the static functions in this module */
static reg_return_t seg_cmp(void *reg_addr, size_t reg_len,
                            void *oth_addr, size_t oth_len, int op);
static reg_return_t seg_intersects(void *reg_addr, size_t reg_len,
                                   void *oth_addr, size_t oth_len);
static reg_return_t seg_contains(void *reg_addr, size_t reg_len,
                                 void *oth_addr, size_t oth_len);
static reg_return_t reg_entry_intersects(reg_entry_t *reg_entry,
                                         void *buf, int len);
static reg_return_t reg_entry_contains(reg_entry_t *reg_entry,
                                       void *buf, int len);

#define TEST_FOR_INTERSECTION 0
#define TEST_FOR_CONTAINMENT 1

/*#define TEST_DEBUG*/
#ifdef TEST_DEBUG
int reg_win_rank;
int reg_win_nprocs;
#endif


/**
 * Detects whether two memory segments intersect or one contains the other.
 *
 * @param[in] reg_addr  starting address of original segment
 * @param[in] reg_len   length of original segment
 * @param[in] oth_addr  starting address of other segment
 * @param[in] oth_len   length of other segment
 * @param[in] op        op to perform, either TEST_FOR_INTERSECTION or
 *                      TEST_FOR_CONTAINMENT
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
seg_cmp(void *reg_addr, size_t reg_len, void *oth_addr, size_t oth_len, int op)
{
    ptrdiff_t reg_beg = 0;
    ptrdiff_t reg_end = 0;
    ptrdiff_t oth_beg = 0;
    ptrdiff_t oth_end = 0;
    int result = 0;

    /* preconditions */
    COMEX_ASSERT(NULL != reg_addr);
    COMEX_ASSERT(NULL != oth_addr);

    /* if reg_len = 0 and oth_len = 0 do a direct comparison. This assumes that
     * registered region is zero length and we are just using a small buffer to
     * keep from running into problems associated with not allocating memory on
     * some processors */
    if (reg_len == 0 && oth_len == 0) {
      if (reg_addr == oth_addr) {
        return RR_SUCCESS;
      } else {
        return RR_FAILURE;
      }
    }


    /* casts to ptrdiff_t since arithmetic on void* is undefined */
    reg_beg = (ptrdiff_t)(reg_addr);
    reg_end = reg_beg + (ptrdiff_t)(reg_len);
    oth_beg = (ptrdiff_t)(oth_addr);
    oth_end = oth_beg + (ptrdiff_t)(oth_len);
    
    /* hack? we had problems with adjacent registered memory regions and
     * when the length of the query region was 0 */
    if (oth_beg == oth_end) {
        oth_end += 1;
    }

    switch (op) {
        case TEST_FOR_INTERSECTION:
            result = (reg_beg >= oth_beg && reg_beg <  oth_end) ||
                     (reg_end >  oth_beg && reg_end <= oth_end);
#if DEBUG
            printf("[%d] TEST_FOR_INTERSECTION "
                    "(%td >= %td [%d] && %td < %td [%d]) ||"
                    "(%td > %td [%d] && %td <= %td [%d])\n",
                    g_state.rank,
                    reg_beg, oth_beg, (reg_beg >= oth_beg),
                    reg_beg, oth_end, (reg_beg < oth_end),
                    reg_end, oth_beg, (reg_end > oth_beg),
                    reg_end, oth_end, (reg_end <= oth_end));
#endif
            break;
        case TEST_FOR_CONTAINMENT:
            result = reg_beg <= oth_beg && reg_end >= oth_end;
#if DEBUG
            printf("[%d] TEST_FOR_CONTAINMENT "
                    "%td <= %td [%d] && %td >= %td [%d]\n",
                    g_state.rank,
                    reg_beg, oth_beg, (reg_beg <= oth_beg),
                    reg_end, oth_end, (reg_end >= oth_end));
#endif
            break;
        default:
            COMEX_ASSERT(0);
    }

    if (result) {
        return RR_SUCCESS;
    }
    else {
        return RR_FAILURE;
    }
}


/**
 * Detects whether two memory segments intersect.
 *
 * @param[in] reg_addr starting address of original segment
 * @param[in] reg_len  length of original segment
 * @param[in] oth_addr starting address of other segment
 * @param[in] oth_len  length of other segment
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
seg_intersects(void *reg_addr, size_t reg_len, void *oth_addr, size_t oth_len)
{
    /* preconditions */
    COMEX_ASSERT(NULL != reg_addr);
    COMEX_ASSERT(NULL != oth_addr);

    return seg_cmp(
            reg_addr, reg_len,
            oth_addr, oth_len,
            TEST_FOR_INTERSECTION);
}


/**
 * Detects whether the first memory segment contains the other.
 *
 * @param[in] reg_addr starting address of original segment
 * @param[in] reg_len  length of original segment
 * @param[in] oth_addr starting address of other segment
 * @param[in] oth_len  length of other segment
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
seg_contains(void *reg_addr, size_t reg_len, void *oth_addr, size_t oth_len)
{
    /* preconditions */
    COMEX_ASSERT(NULL != reg_addr);
    COMEX_ASSERT(NULL != oth_addr);

    return seg_cmp(
            reg_addr, reg_len,
            oth_addr, oth_len,
            TEST_FOR_CONTAINMENT);
}


/**
 * Detects whether two memory segments intersect.
 *
 * @param[in] reg_entry the registration entry
 * @param[in] buf       starting address for the contiguous memory region
 * @param[in] len       length of the contiguous memory region
 *
 * @pre NULL != reg_entry
 * @pre NULL != buf
 * @pre len >= 0
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
reg_entry_intersects(reg_entry_t *reg_entry, void *buf, int len)
{
#if DEBUG
    printf("[%d] reg_entry_intersects(reg_entry=%p, buf=%p, len=%d)\n",
            g_state.rank, reg_entry, buf, len);
#endif
    /* preconditions */
    COMEX_ASSERT(NULL != reg_entry);
    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(len >= 0);

    return seg_intersects(
            reg_entry->buf, reg_entry->len,
            buf, len);
}


/**
 * Detects whether the first memory segment contains the other.
 *
 * @param[in] reg_entry the registration entry
 * @param[in] buf       starting address for the contiguous memory region
 * @param[in] len       length of the contiguous memory region
 *
 * @pre NULL != reg_entry
 * @pre NULL != buf
 * @pre len >= 0
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
reg_entry_contains(reg_entry_t *reg_entry, void *buf, int len)
{
#if DEBUG
    printf("[%d] reg_entry_contains(reg_entry=%p, buf=%p, len=%d)\n",
            g_state.rank, reg_entry, buf, len);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_entry);
    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(len >= 0);

    return seg_contains(
            reg_entry->buf, reg_entry->len,
            buf, len);
}


/**
 * Remove registration window entry without deregistration.
 *
 * @param[in] rank the rank where the entry came from
 * @param[in] reg_entry the entry
 *
 * @pre NULL != reg_entry
 * @pre 0 <= rank && rank < reg_nprocs
 *
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
reg_entry_destroy(int rank, reg_entry_t *reg_entry)
{
#if DEBUG
    printf("[%d] reg_entry_destroy(rank=%d, reg_entry=%p)\n"
            "buf=%p len=%zu name=%s mapped=%p\n",
            g_state.rank, rank, reg_entry,
            reg_entry->buf, reg_entry->len,
            reg_entry->name, reg_entry->mapped);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_entry);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);

    /* free window entry */
    free(reg_entry);

    return RR_SUCCESS;
}


/**
 * Create internal data structures for the registration window.
 *
 * @param[in] nprocs    number of registration windows to create i.e. one per
 *                      process
 *
 * @pre this function is called once to initialize the internal data
 * structures and cannot be called again until reg_win_destroy() has been
 * called
 *
 * @see reg_win_destroy()
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
reg_win_init(int nprocs)
{
    int i = 0;

#ifdef TEST_DEBUG
    reg_win_nprocs = nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&reg_win_rank);
#endif
#if DEBUG
    printf("[%d] reg_win_init(nprocs=%d)\n",
            g_state.rank, nprocs);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL == reg_win);
    COMEX_ASSERT(0 == reg_nprocs);

    /* keep the number of processors around for later use */
    reg_nprocs = nprocs;

    /* allocate the registration window list: */
    reg_win = (reg_entry_t **)malloc(sizeof(reg_entry_t*) * reg_nprocs); 
    COMEX_ASSERT(reg_win); 

    /* initialize the registration window list: */
    for (i = 0; i < reg_nprocs; ++i) {
        reg_win[i] = NULL;
    }

    return RR_SUCCESS;
}


/**
 * Deregister and destroy all window entries and associated buffers.
 *
 * @pre this function is called once to destroy the internal data structures
 * and cannot be called again until reg_win_init() has been called
 *
 * @see reg_win_init()
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
reg_win_destroy()
{
    int i = 0;

#if DEBUG
    printf("[%d] reg_win_destroy()\n", g_state.rank);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_win);
    COMEX_ASSERT(0 != reg_nprocs);

    for (i = 0; i < reg_nprocs; ++i) {
        reg_entry_t *runner = reg_win[i];

        while (runner) {
            reg_entry_t *previous = runner; /* pointer to previous runner */

            /* get next runner */
            runner = runner->next;
            /* destroy the entry */
            reg_entry_destroy(i, previous);
        }
    }

    /* free registration window list */
    free(reg_win);
    reg_win = NULL;

    /* reset the number of windows */
    reg_nprocs = 0;

    return RR_SUCCESS;
}


/**
 * Locate a registration window entry which contains the given segment
 * completely.
 *
 * @param[in] rank  rank of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * 
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre reg_win_init() was previously called
 *
 * @return the reg window entry, or NULL on failure
 */
reg_entry_t*
reg_win_find(int rank, void *buf, int len)
{
    reg_entry_t *entry = NULL;
    reg_entry_t *runner = NULL;

#ifdef TEST_DEBUG
    printf("[%d] reg_win_find(rank=%d, buf=%p, len=%d reg_win=%p)\n",
            reg_win_rank, rank, buf, len, reg_win);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_win);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);

    runner = reg_win[rank];

    while (runner && NULL == entry) {
#ifdef TEST_DEBUG
        printf("p[%d] searching on p[%d] runner rank: %d buf: %p len: %d\n",
            reg_win_rank,rank,
            runner->rank,runner->buf,runner->len);
#endif
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
            entry = runner;
#ifdef TEST_DEBUG
            printf("[%d] reg_win_find entry found "
                    "reg_entry=%p buf=%p len=%d "
                    "runner: rank=%d buf=%p len=%d\n",
                    reg_win_rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len);
#endif
        }
        runner = runner->next;
    }

#ifndef NDEBUG
    /* we COMEX_ASSERT that the found entry was unique */
    while (runner) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
#ifdef TEST_DEBUG
            printf("[%d] reg_win_find duplicate found "
                    "reg_entry=%p buf=%p len=%d "
                    "rank=%d buf=%p len=%d\n",
                    reg_win_rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len);
#endif
            COMEX_ASSERT(0);
        }
        runner = runner->next;
    }
#endif

    return entry;
}


/**
 * Locate a registration window entry which intersects the given segment.
 *
 * @param[in] rank  rank of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * 
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre reg_win_init() was previously called
 *
 * @return the reg window entry, or NULL on failure
 */
reg_entry_t*
reg_win_find_intersection(int rank, void *buf, int len)
{
    reg_entry_t *entry = NULL;
    reg_entry_t *runner = NULL;

#if DEBUG
    printf("[%d] reg_win_find_intersection(rank=%d, buf=%p, len=%d)\n",
            g_state.rank, rank, buf, len);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_win);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);

    runner = reg_win[rank];

    while (runner && NULL == entry) {
        if (RR_SUCCESS == reg_entry_intersects(runner, buf, len)) {
            entry = runner;
        }
        runner = runner->next;
    }

    /* we COMEX_ASSERT that the found entry was unique */
    while (runner) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
            COMEX_ASSERT(0);
        }
        runner = runner->next;
    }

    return entry;
}


/**
 * Create a new registration entry based on the given members. This new
 * entry contains the buffer location on the remote processor, size of the
 * buffer, the MPI window that the buffer belongs to and the group associated
 * with the window
 * @param rank processor rank for which buffer location applies
 * @param buf pointer to memory allocation
 * @param len size of memory allocation
 * @param win MPI window for memory allocation
 * @param group group associated with memory allocation
 *
 * @return return new entry in linked list
 */
reg_entry_t*
reg_win_insert(int rank, void *buf, int len, MPI_Win win, comex_igroup_t *group)
{
    reg_entry_t *node = NULL;

#if DEBUG
    printf("[%d] reg_win_insert(rank=%d, buf=%p, len=%d, name=%s, mapped=%p)\n",
            g_state.rank, rank, buf, len, name, mapped);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_win);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);
    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(len >= 0);
    COMEX_ASSERT(NULL != group);
    COMEX_ASSERT(NULL == reg_win_find(rank, buf, len));
    COMEX_ASSERT(NULL == reg_win_find_intersection(rank, buf, len));

    /* allocate the new entry */
    node = (reg_entry_t *)malloc(sizeof(reg_entry_t));
    COMEX_ASSERT(node);

    /* initialize the new entry */
    node->rank = rank;
    node->buf = buf;
    node->len = len;
    node->win = win;
    node->igroup = group;
    node->next = NULL;

    /* push new entry to tail of linked list */
    if (NULL == reg_win[rank]) {
        reg_win[rank] = node;
    } else {
        reg_entry_t *runner = reg_win[rank];
        while (runner->next) {
            runner = runner->next;
        }
        runner->next = node;
    }

    return node;
}


/**
 * Removes the reg window entry associated with the given rank and buffer.
 *
 * If this process owns the buffer, it will unregister the buffer, as well.
 *
 * @param[in] rank
 * @param[in] buf
 *
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre NULL != buf
 * @pre reg_win_init() was previously called
 * @pre NULL != reg_win_find(rank, buf, 0)
 *
 * @return RR_SUCCESS on success
 *         RR_FAILURE otherwise
 */
reg_return_t
reg_win_delete(int rank, void *buf)
{
    reg_return_t status = RR_FAILURE;
    reg_entry_t *runner = NULL;
    reg_entry_t *previous_runner = NULL;

#if DEBUG
    printf("[%d] reg_win_delete(rank=%d, buf=%p)\n",
            g_state.rank, rank, buf);
#endif

    /* preconditions */
    COMEX_ASSERT(NULL != reg_win);
    COMEX_ASSERT(0 <= rank && rank < reg_nprocs);
    COMEX_ASSERT(NULL != buf);
    COMEX_ASSERT(NULL != reg_win_find(rank, buf, 0));

    /* this is more restrictive than reg_win_find() in that we locate
     * exactlty the same region starting address */
    runner = reg_win[rank];
    while (runner) {
        if (runner->buf == buf) {
            break;
        }
        /* no match so match may be next runner */
        previous_runner = runner;
        runner = runner->next;
    }
    /* we should have found an entry */
    if (NULL == runner) {
        COMEX_ASSERT(0);
        return RR_FAILURE;
    }

    /* pop the entry out of the linked list */
    if (previous_runner) {
      /* runner is a match and it is not the first entry in list */
        previous_runner->next = runner->next;
    }
    else {
      /* buf corresponds to first entry in list */
        reg_win[rank] = reg_win[rank]->next;
    }

    status = reg_entry_destroy(rank, runner);

    return status;
}
