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
  GP = (gp_array_t*)malloc(sizeof(gp_array_t)*MAX_GP_ARRAYS);
  if (!GP) {
    pnga_error("gp_initialize: malloc GP failed",0);
  }
  for (i=0; i<MAX_GP_ARRAYS; i++) {
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
  for (i=0; i<MAX_GP_ARRAYS; i++) {
    if (GP[i].active) {
      pnga_destroy(GP[i].gp_size_array);
      pnga_destroy(GP[i].gp_ptr_array);
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
  for (i=0; i<MAX_GP_ARRAYS; i++) {
    if (!GP[i].active) {
      handle = i-GP_OFFSET;
      GP[i].gp_size_array = pnga_create_handle();
      GP[i].gp_ptr_array = pnga_create_handle();
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
  if (*ndim < 0 || *ndim > MAX_GP_DIM) {
    pnga_error("gp_set_dimensions: dimension is not valid", *ndim);
  }
  for (i=0; i<*ndim; i++) {
    if (dims[i] < 0) {
      pnga_error("gp_set_dimensions: invalid dimension found", dims[i]);
    }
  }

  type = C_INT;
  pnga_set_data(GP[handle].gp_size_array, *ndim, dims, type);
  type = GP_POINTER_TYPE;
  pnga_set_data(GP[handle].gp_ptr_array, *ndim, dims, type);
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
  pnga_set_chunk(GP[handle].gp_size_array, chunk);
  pnga_set_chunk(GP[handle].gp_ptr_array, chunk);
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
  status = pnga_allocate(GP[handle].gp_size_array);
  status = status && pnga_allocate(GP[handle].gp_ptr_array);
  if (!status) {
     pnga_error("gp_allocate: unable to allocate GP array", 0);
  }
  me = pnga_nodeid();
  pnga_distribution(GP[handle].gp_ptr_array, me, GP[handle].lo, GP[handle].hi);
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
  status = pnga_destroy(GP[handle].gp_size_array);
  status = status && pnga_destroy(GP[handle].gp_ptr_array);
  if (!status) {
     pnga_error("gp_destroy: unable to destroy GP array", 0);
  }
  return status;
}

/**
 * Assign data object to a pointer array element. Pointer array element
 * must be on the same processor as the data object.
 *  @param[in] g_p             pointer array handle
 *  @param[in] subscript[ndim] location of element in pointer array
 *  @param[in] ptr             ptr to local data ojbect
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_assign_element = pgp_assign_element
#endif
void pgp_assign_element(Integer *g_p, Integer *subscript, void *ptr)
{
  /* TODO: Implement this method */
}
