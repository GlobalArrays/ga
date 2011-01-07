#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_ASSERT_H
#   include <assert.h>
#endif

#include "gacommon.h"
#include "typesf2c.h"
#include "papi.h"
#include "gpbase.h"

gp_array_t *GP;

/**
 *  Initialize internal library structures for Global Pointer Arrays
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_initialize = pgp_initialize
#endif

void pgp_initialize()
{
  Integer i;
  GP = (gp_array_t*)malloc(sizeof(gp_array_t)*GP_MAX_ARRAYS);
  if (!GP) {
    pnga_error("gp_initialize: malloc GP failed",0);
  }
  for (i=0; i<GP_MAX_ARRAYS; i++) {
    GP[i].active = 0;
  }
}

/**
 *  Deallocate all arrays and clean up library
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_terminate = pgp_terminate
#endif

void pgp_terminate()
{
  Integer i;
  for (i=0; i<GP_MAX_ARRAYS; i++) {
    if (GP[i].active) {
      pnga_destroy(&GP[i].g_size_array);
      pnga_destroy(&GP[i].g_ptr_array);
      GP[i].active = 0;
    }
  }
}

/**
 *  Create a handle for a GP array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_create_handle = pgp_create_handle
#endif

Integer pgp_create_handle()
{
  Integer i, handle=-GP_OFFSET-1;
  for (i=0; i<GP_MAX_ARRAYS; i++) {
    if (!GP[i].active) {
      handle = i-GP_OFFSET;
      GP[i].g_size_array = pnga_create_handle();
      GP[i].g_ptr_array = pnga_create_handle();
      break;
    }
  }
  return handle;
}

/**
 *  Set array dimensions
 *  @param[in] g_p         pointer array handle
 *  @param[in] ndim        dimension of pointer array
 *  @param[in] dims[ndim]  dimension of array axes
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_set_dimensions = pgp_set_dimensions
#endif

void pgp_set_dimensions(Integer *g_p, Integer *ndim, Integer *dims)
{
  Integer handle, i, type;
  handle = *g_p + GP_OFFSET;

  /* Do some basic checks on parameters */
  if (!GP[handle].active) {
    pnga_error("gp_set_dimensions: Global Pointer handle is not active", 0);
  }
  if (*ndim < 0 || *ndim > GP_MAX_DIM) {
    pnga_error("gp_set_dimensions: dimension is not valid", *ndim);
  }
  for (i=0; i<*ndim; i++) {
    if (dims[i] < 0) {
      pnga_error("gp_set_dimensions: invalid dimension found", dims[i]);
    }
  }

  type = C_INT;
  pnga_set_data(&GP[handle].g_size_array, ndim, dims, &type);
  type = GP_POINTER_TYPE;
  pnga_set_data(&GP[handle].g_ptr_array, ndim, dims, &type);
  GP[handle].ndim = *ndim;
  for (i=0; i<*ndim; i++) {
    GP[handle].dims[i] = dims[i];
  }
}

/**
 *  Set chunk array dimensions. This determines the minimum dimension of a
 *  local block of data
 *  @param[in] g_p         pointer array handle
 *  @param[in] chunk[ndim] minimum dimensions of array blocks
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_set_chunk = pgp_set_chunk
#endif

void pgp_set_chunk(Integer *g_p, Integer *chunk)
{
  Integer handle;
  handle = *g_p + GP_OFFSET;
  pnga_set_chunk(&GP[handle].g_size_array, chunk);
  pnga_set_chunk(&GP[handle].g_ptr_array, chunk);
}

/**
 *  Allocate memory for a Global Pointer array.
 *  @param[in] g_p         pointer array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_allocate = pgp_allocate
#endif

logical pgp_allocate(Integer *g_p)
{
  logical status;
  Integer handle, me;
  handle = *g_p + GP_OFFSET;
  status = pnga_allocate(&GP[handle].g_size_array);
  status = status && pnga_allocate(&GP[handle].g_ptr_array);
  if (!status) {
     pnga_error("gp_allocate: unable to allocate GP array", 0);
  } else {
    me = pnga_nodeid();
    pnga_distribution(&GP[handle].g_ptr_array, &me, GP[handle].lo,
                      GP[handle].hi);
    GP[handle].active = 1;
    return status;
  }
  pnga_zero(&GP[handle].g_size_array);
  pnga_zero(&GP[handle].g_ptr_array);
  me = pnga_nodeid();
  pnga_distribution(&GP[handle].g_ptr_array, &me, GP[handle].lo,
          GP[handle].hi);
  return status;
}

/**
 *  Destroy Global Pointer array and free memory for reuse.
 *  @param[in] g_p         pointer array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_destroy = pgp_destroy
#endif

logical pgp_destroy(Integer *g_p)
{
  logical status;
  Integer handle;
  handle = *g_p + GP_OFFSET;
  status = pnga_destroy(&GP[handle].g_size_array);
  status = status && pnga_destroy(&GP[handle].g_ptr_array);
  if (!status) {
    pnga_error("gp_destroy: unable to destroy GP array", 0);
  } else {
    GP[handle].active = 0;
    return status;
  }
  return status;
}

/**
 *  Return coordinates of a GP patch associated with processor proc
 *  @param[in] g_p                pointer array handle
 *  @param[in] proc               processor for which patch coordinates
 *                                are being requested
 *  @param[out] lo[ndim],hi[ndim] bounding indices of patch
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_distribution = pgp_distribution
#endif
void pgp_distribution(Integer *g_p, Integer *proc, Integer *lo, Integer *hi)
{
  Integer handle, ndim, i;
  handle = *g_p + GP_OFFSET;
  if (pnga_nodeid() == *proc) {
  } else {
  }
}

/**
 *  Assign data object to a pointer array element. Pointer array element
 *  must be on the same processor as the data object.
 *  @param[in] g_p             pointer array handle
 *  @param[in] subscript[ndim] location of element in pointer array
 *  @param[in] ptr             ptr to local data ojbect
 *  @param[in] size            size of local data ojbect
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_assign_local_element = pgp_assign_local_element
#endif
void pgp_assign_local_element(Integer *g_p, Integer *subscript, void *ptr, Integer *size)
{
  void *gp_ptr;
  Integer handle, ld[GP_MAX_DIM-1], i;
  handle = *g_p + GP_OFFSET;
  /* check to make sure that element is located in local block of GP array */
  for (i=0; i<GP[handle].ndim; i++) {
    if (subscript[i]<GP[handle].lo[i] || subscript[i]>GP[handle].hi[i]) {
      pnga_error("gp_assign_local_element: subscript out of bounds", i);
    }
  }
  pnga_access_ptr(&GP[handle].g_ptr_array,subscript,subscript,&gp_ptr,ld);
  *((GP_INT*)gp_ptr) = (GP_INT)ptr;
/*  pnga_release_update(&GP[handle].g_ptr_array, subscript, subscript); */
}
