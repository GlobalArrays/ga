/**
 * Registration window.
 *
 * Defensive programming via copious CMX_ASSERT statements is encouraged.
 */
/*
#if HAVE_CONFIG_H
#   include "config.h"
#endif
*/

/* C headers */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3rd party headers */
#include <mpi.h>

/* our headers */
#include "cmx.h"
#include "reg_entry.h"

#define STATIC static inline

/* the static members in this module */
static reg_entry_t **reg_entries = NULL; /* list of allocations on each procss. This
                                          array contains the starting node in a
                                          linked list */
static int reg_nprocs = 0; /* number of allocations on this process */


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
int reg_entry_rank;
int reg_entry_nprocs;
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
    /*
    CMX_ASSERT(NULL != reg_addr);
    CMX_ASSERT(NULL != oth_addr);
    */

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
            CMX_ASSERT(0);
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
 * @return RR_SUCCESS on success
 */
STATIC reg_return_t
seg_intersects(void *reg_addr, size_t reg_len, void *oth_addr, size_t oth_len)
{
    /* preconditions */
    /*
    CMX_ASSERT(NULL != reg_addr);
    CMX_ASSERT(NULL != oth_addr);
    */

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
    /*
    CMX_ASSERT(NULL != reg_addr);
    CMX_ASSERT(NULL != oth_addr);
    */

    return seg_cmp(
            reg_addr, reg_len,
            oth_addr, oth_len,
            TEST_FOR_CONTAINMENT);
}


/**
 * Detects whether two memory segments intersect.
 * (Probably not used)
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
    /*
    CMX_ASSERT(NULL != reg_entry);
    CMX_ASSERT(NULL != buf);
    */
    CMX_ASSERT(len >= 0);

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
    CMX_ASSERT(NULL != reg_entry);
    /*CMX_ASSERT(NULL != buf);*/
    CMX_ASSERT(len >= 0);

    return seg_contains(
            reg_entry->buf, reg_entry->len,
            buf, len);
}


/**
 * Remove registration window entry without deregistration.
 *
 * @param[in] rank the local rank (on group) where the entry came from
 * @param[in] reg_entry the entry
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
    CMX_ASSERT(NULL != reg_entry);
    CMX_ASSERT(0 <= rank && rank < reg_nprocs);

    /* free window entry */
    free(reg_entry);

    return RR_SUCCESS;
}


/**
 * Create internal data structures for the registration window.
 * This function is called once to initialize the internal data
 * structures and cannot be called again until reg_entry_destroy() has been
 * called
 *
 * @param[in] nprocs    number of registration windows to create i.e. one per
 *                      process
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
reg_entry_init(int nprocs)
{
    int i = 0;

#ifdef TEST_DEBUG
    reg_entry_nprocs = nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&reg_entry_rank);
#endif
#if DEBUG
    printf("[%d] reg_entry_init(nprocs=%d)\n",
            g_state.rank, nprocs);
#endif

    /* preconditions */
    CMX_ASSERT(0 == reg_nprocs);

    /* keep the number of processors around for later use */
    reg_nprocs = nprocs;

    /* allocate the registration window list: */
    reg_entries = (reg_entry_t **)malloc(sizeof(reg_entry_t*) * reg_nprocs); 
    CMX_ASSERT(reg_entries); 

    /* initialize the registration window list: */
    for (i = 0; i < reg_nprocs; ++i) {
        reg_entries[i] = NULL;
    }

    return RR_SUCCESS;
}


/**
 * Deregister and destroy all window entries and associated buffers.
 *
 * @pre this function is called once to destroy the internal data structures
 * and cannot be called again until reg_entry_init() has been called
 *
 * @see reg_entry_init()
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
reg_entries_destroy()
{
    int i = 0;

#if DEBUG
    printf("[%d] reg_entries_destroy()\n", g_state.rank);
#endif

    /* preconditions */
    CMX_ASSERT(NULL != reg_entries);
    CMX_ASSERT(0 != reg_nprocs);

    for (i = 0; i < reg_nprocs; ++i) {
        reg_entry_t *runner = reg_entries[i];

        while (runner) {
            reg_entry_t *previous = runner; /* pointer to previous runner */

            /* get next runner */
            runner = runner->next;
            /* destroy the entry */
            reg_entry_destroy(i, previous);
        }
    }

    /* free registration window list */
    free(reg_entries);
    reg_entries = NULL;

    /* reset the number of windows */
    reg_nprocs = 0;

    return RR_SUCCESS;
}


/**
 * Locate a registration window entry which contains the given segment
 * completely.
 *
 * @param[in] rank  the local rank (on group) of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * 
 * @return the reg window entry, or NULL on failure
 */
reg_entry_t*
reg_entry_find(int rank, void *buf, int len)
{
    reg_entry_t *entry = NULL;
    reg_entry_t *runner = NULL;

#ifdef TEST_DEBUG
    printf("p[%d] reg_entry_find(rank=%d, buf=%p, len=%d reg_entries[%d]=%p)\n",
            reg_entry_rank, rank, buf, len, rank, reg_entries[rank]);
#endif

    /* preconditions */
    CMX_ASSERT(NULL != reg_entries);
    CMX_ASSERT(0 <= rank && rank < reg_nprocs);

    runner = reg_entries[rank];

    while (runner && NULL == entry) {
#ifdef TEST_DEBUG
        printf("p[%d] rank: %d runner: %p buf: %p len: %d\n",
            reg_entry_rank,rank,runner,runner->buf,runner->len);
#endif
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
            entry = runner;
#ifdef TEST_DEBUG
            printf("p[%d] reg_entry_find entry found "
                    "runner=%p buf=%p len=%d "
                    "runner: rank=%d buf=%p len=%d\n",
                    reg_entry_rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len);
#endif
        }
        runner = runner->next;
    }

#ifndef NDEBUG
    /* we CMX_ASSERT that the found entry was unique. This code checks all
     * entries after the one found in the previous loop to see if there
     * is another overlap. */
    while (runner) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
#ifdef TEST_DEBUG
            printf("[%d] reg_entry_find duplicate found "
                    "runner=%p buf=%p len=%d "
                    "rank=%d buf=%p len=%d\n",
                    reg_entry_rank, runner, buf, len,
                    runner->rank, runner->buf, runner->len);
#endif
            CMX_ASSERT(0);
        }
        runner = runner->next;
    }
#endif

    return entry;
}


/**
 * Locate a registration window entry which intersects the given segment.
 * (probably not used)
 *
 * @param[in] rank  local rank (on group) of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 *
 * @return the reg window entry, or NULL on failure
 */
reg_entry_t*
reg_entry_find_intersection(int rank, void *buf, int len)
{
    reg_entry_t *entry = NULL;
    reg_entry_t *runner = NULL;

#if DEBUG
    printf("[%d] reg_entry_find_intersection(rank=%d, buf=%p, len=%d)\n",
            g_state.rank, rank, buf, len);
#endif

    /* preconditions */
    CMX_ASSERT(NULL != reg_entries);
    CMX_ASSERT(0 <= rank && rank < reg_nprocs);

    runner = reg_entries[rank];

    while (runner && NULL == entry) {
        if (RR_SUCCESS == reg_entry_intersects(runner, buf, len)) {
            entry = runner;
        }
        runner = runner->next;
    }

    /* we CMX_ASSERT that the found entry was unique */
    while (runner) {
        if (RR_SUCCESS == reg_entry_contains(runner, buf, len)) {
            CMX_ASSERT(0);
        }
        runner = runner->next;
    }

    return entry;
}


/**
 * Create a new registration entry based on the given members.
 *
 * @param[in] rank local rank (on group) of processor inserting entry
 * @param[in] buf pointer to local memory allocation
 * @param[in] len length of local memory allocation in bytes
 * @param[in] cmx_hdl handle of CMX allocation
 * @return pointer to window registration entry for this cmx_handle_t
 */
reg_entry_t*
reg_entry_insert(int world_rank, void *buf, int len, cmx_handle_t *cmx_hdl)
{
  reg_entry_t *node = NULL;

#ifdef TEST_DEBUG
  printf("p[%d] reg_entry_insert(rank=%d, buf=%p, len=%d)\n",
      reg_entry_rank, world_rank, buf, len);
#endif

  /* preconditions */
  CMX_ASSERT(NULL != reg_entries);
  CMX_ASSERT(0 <= world_rank && world_rank < reg_nprocs);
  /* CMX_ASSERT(NULL != buf); */
  CMX_ASSERT(len >= 0);
  CMX_ASSERT(NULL == reg_entry_find(world_rank, buf, len));
  CMX_ASSERT(NULL == reg_entry_find_intersection(world_rank, buf, len));

  /* allocate the new entry */
  node = (reg_entry_t *)malloc(sizeof(reg_entry_t));
  CMX_ASSERT(node);

  /* initialize the new entry */
  node->rank = world_rank;
  node->buf = buf;
  node->len = len;
  node->hdl = cmx_hdl;
  node->next = NULL;

  /* push new entry to tail of linked list */
  if (NULL == reg_entries[world_rank]) {
    reg_entries[world_rank] = node;
  } else {
    reg_entry_t *runner = reg_entries[world_rank];
    while (runner->next) {
      runner = runner->next;
    }
    runner->next = node;
  }
  return node;
}

/**
 * Removes the reg window entry associated with the given rank and buffer.
 * This does not destroy the armci_handle_t object, this must be done
 * separately.
 *
 * @param[in] rank world rank of processor calling this function
 * @param[in] buf pointer to memory allocation
 *
 * @return RR_SUCCESS on success
 *         RR_FAILURE otherwise
 */
reg_return_t
reg_entry_delete(int rank, void *buf)
{
    reg_return_t status = RR_FAILURE;
    reg_entry_t *runner = NULL;
    reg_entry_t *previous_runner = NULL;

#ifdef TEST_DEBUG
    printf("p[%d] reg_entry_delete(rank=%d, buf=%p)\n",
            reg_entry_rank, rank, buf);
#endif

    /* preconditions */
    CMX_ASSERT(NULL != reg_entries);
    CMX_ASSERT(0 <= rank && rank < reg_nprocs);
    /* CMX_ASSERT(NULL != buf); */
    CMX_ASSERT(NULL != reg_entry_find(rank, buf, 0));

    /* this is more restrictive than reg_entry_find() in that we locate
     * exactly the same region starting address */
    runner = reg_entries[rank];
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
        CMX_ASSERT(0);
        return RR_FAILURE;
    }

    /* pop the entry out of the linked list */
    if (previous_runner) {
      /* runner is a match and it is not the first entry in list */
        previous_runner->next = runner->next;
    }
    else {
      /* buf corresponds to first entry in list */
        reg_entries[rank] = reg_entries[rank]->next;
    }

    status = reg_entry_destroy(rank, runner);

    return status;
}
