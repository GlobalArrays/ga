/* $Id: onesided.c,v 1.80.2.18 2007/12/18 22:22:27 d3g293 Exp $ */
/* 
 * module: onesided.c
 * author: Jarek Nieplocha
 * description: implements GA primitive communication operations --
 *              accumulate, scatter, gather, read&increment & synchronization 
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif
 
/*#define PERMUTE_PIDS */
/*#define USE_GATSCAT_NEW 1 */

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STDINT_H
#   include <stdint.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_ASSERT_H
#   include <assert.h>
#endif
#if HAVE_STDDEF_H
#include <stddef.h>
#endif

#include "global.h"
#include "globalp.h"
#include "base.h"
#include "armci.h"
#include "macdecls.h"
#include "ga-papi.h"
#include "ga-wapi.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

#ifdef PROFILE_OLD
#include "ga_profile.h"
#endif

#ifdef ENABLE_UNSAFE_PUT_NOTIFY

#define HANDLES_OUTSTANDING 100
/* Maximum number of outstanding put/notify handles */

typedef struct {
  Integer *orighdl;
  Integer firsthdl;
  Integer elementhdl;
  void *elem_copy;
} gai_putn_hdl_t;

static gai_putn_hdl_t putn_handles[HANDLES_OUTSTANDING]; /* RACE */

/**
 * (Non-blocking) Put an N-dimensional patch of data into a Global Array and notify the other
                  side with information on another Global Array
 */

static int putn_find_empty_slot(void)
{
  for (int i = 0; i < HANDLES_OUTSTANDING; i++)
    if (!putn_handles[i].orighdl)
      return i;

  return -1;
} /* putn_find_empty_slot */

static int putn_intersect_coords(Integer g_a, Integer *lo, Integer *hi, Integer *ecoords)
{
  int ndims = pnga_ndim(g_a);

  for (int i = 0; i < ndims; i++)
    if ((ecoords[i] < lo[i]) || (ecoords[i] > hi[i]))
      return 0;

  return 1;
} /* putn_intersect_coords */

static int putn_verify_element_in_buf(Integer g_a, Integer *lo, Integer *hi, void *buf,
				      Integer *ld, Integer *ecoords, void *bufn,
				      Integer elemSize)
{
#ifdef HAVE_STDDEF_H
  ptrdiff_t off = (char *)bufn - (char *)buf;
#else
  Integer off = (char *)bufn - (char *)buf;
#endif
  Integer eoff = 0;

  off /= elemSize; /* Offset in terms of elements */

  int ndims = pnga_ndim(g_a);
  eoff = ecoords[0] - lo[0];

  /* Check in Fortran ordering */
  for (int i = 1; i < ndims; i++)
    eoff += (ecoords[i] - lo[i]) * ld[i - 1];

  return (eoff == (Integer)off); /* Must be the same for a correct notify buffer */
} /* putn_verify_element_in_buf */

#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_nbput_notify = pnga_nbput_notify
#endif

void pnga_nbput_notify(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer g_b, Integer *ecoords, void *bufn, Integer *nbhandle)
{
  Integer ldn[MAXDIM] = { 1 };
  int pos, intersect;

  static int putn_handles_initted = 0; /* RACE */
  /* Make sure everything has been initialized */
  if (!putn_handles_initted) {
    memset(putn_handles, 0, sizeof(putn_handles));
    putn_handles_initted = 1;
  }

  pos = putn_find_empty_slot();
  if (pos == -1) /* no empty handles available */
    pnga_error("Too many outstanding put/notify operations!", 0);

  putn_handles[pos].orighdl = nbhandle; /* Store original handle for nbwait_notify */

  if (g_a == g_b)
    intersect = putn_intersect_coords(g_a, lo, hi, ecoords);
  else
    intersect = 0;

  if (!intersect) { /* Simpler case */
    ngai_put_common(g_a, lo, hi, buf, ld, 0, -1, &putn_handles[pos].firsthdl);
    ngai_put_common(g_b, ecoords, ecoords, bufn, ldn, 0, -1, &putn_handles[pos].elementhdl);

    putn_handles[pos].elem_copy = NULL;
  }
  else {
    int ret;
    Integer handle = GA_OFFSET + g_a, size;
    void *elem_copy;
    char *elem;

    size = GA[handle].elemsize;
    ret = putn_verify_element_in_buf(g_a, lo, hi, buf, ld, ecoords, bufn, size);

    if (!ret)
      pnga_error("Intersecting buffers, but notify element is not in buffer!", 0);

    elem_copy = malloc(size);
    memcpy(elem_copy, bufn, size);

    elem = bufn;
    for (int i = 0; i < size; i++)
      elem[i] += 1; /* Increment each byte by one, safe? */

    putn_handles[pos].elem_copy = elem_copy;

    ngai_put_common(g_a, lo, hi, buf, ld, 0, -1, &putn_handles[pos].firsthdl);
    ngai_put_common(g_a, ecoords, ecoords, elem_copy, ldn, 0, -1,
		    &putn_handles[pos].elementhdl);
  }
} /* pnga_nbput_notify */

/**
 *  Wait for a non-blocking put/notify to complete
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_nbwait_notify = pnga_nbwait_notify
#endif

void pnga_nbwait_notify(Integer *nbhandle)
{
  int i;

  for (i = 0; i < HANDLES_OUTSTANDING; i++)
    if (putn_handles[i].orighdl == nbhandle)
      break;

  if (i >= HANDLES_OUTSTANDING)
    return; /* Incorrect handle used or maybe wait was called multiple times? */

  nga_wait_internal(&putn_handles[i].firsthdl);
  nga_wait_internal(&putn_handles[i].elementhdl);

  if (putn_handles[i].elem_copy) {
    free(putn_handles[i].elem_copy);
    putn_handles[i].elem_copy = NULL;
  }

  putn_handles[i].orighdl = NULL;
} /* pnga_nbwait_notify */
#endif // ENABLE_UNSAFE_PUT_NOTIFY
