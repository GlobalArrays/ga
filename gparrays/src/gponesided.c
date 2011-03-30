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
#include "gpbase.h"
#include "armci.h"
#include "papi.h"

#define gpm_GetRangeFromMap(p, ndim, plo, phi){                           \
  Integer   _mloc = p* ndim *2;                                           \
            *plo  = (Integer*)_gp_map + _mloc;                            \
            *phi  = *plo + ndim;                                          \
}

/**
 *  Utility arrays used in onesided communication
 */
Integer *_gp_map;
Integer *_gp_proclist;

/**
 *  Initialize utility arrays
 */
void gpi_onesided_init()
{
  Integer nproc;
  nproc = pnga_pgroup_nnodes(pnga_pgroup_get_world());
  _gp_proclist = (Integer*)malloc((size_t)nproc*sizeof(Integer));
  _gp_map = (Integer*)malloc((size_t)(nproc*2*GP_MAX_DIM)*sizeof(Integer));
}

/**
 * Clean utility arrays
 */
void gpi_onesided_clean()
{
  free(_gp_map);
  free(_gp_proclist);
}

/**
 * Get sizes of element in GP array and return them to a local buffer. Also
 * evaluate the total size of the data to be copied and return that in the
 * variable size. intsize is an internal variable that can be used to
 * distinguish between 4 and 8 byte integers
 * @param[in] g_p                pointer array handle
 * @param[in] lo[ndim]           lower corner of pointer array block
 * @param[in] hi[ndim]           upper corner of pointer array block
 * @param[out] buf               buffer that holds element sizes
 * @param[in] ld[ndim-1]         physical dimensions of buffer
 * @param[out] size              total size of requested data
 * @param[in] intsize            parameter to distinguish between 4 and 8
 *                               byte integers
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_get_size = pgp_get_size
#endif

void pgp_get_size(Integer g_p, Integer *lo, Integer *hi,
                  Integer *size, Integer intsize)
{
  Integer handle, ndim, i, nelems;
  Integer block_ld[GP_MAX_DIM-1];
  int *int_ptr;
  handle = g_p + GP_OFFSET;
  if (!GP[handle].active) {
    pnga_error("gp_get_size: inactive array handle specified", g_p);
  }
  ndim = pnga_ndim(GP[handle].g_ptr_array);
  for (i=0; i<ndim; i++) {
    if (GP[handle].lo[i] > GP[handle].hi[i])
      pnga_error("gp_get_size: illegal block size specified", g_p);
  }

  /* Find total number of elements in block and get strides of requested block */
  ndim = GP[handle].ndim;
  nelems = 1;
  for (i=0; i<ndim; i++) {
    nelems *= (hi[i] - lo[i] + 1);
    if (i<ndim-1) {
      block_ld[i] = (hi[i] - lo[i] + 1);
    }
  }


  int_ptr = (int*)malloc(nelems*sizeof(int));
  pnga_get(GP[handle].g_size_array, lo, hi, int_ptr, block_ld);

  /* Find total size of elements in block */
  *size = 0;
  for (i=0; i<nelems; i++) {
    *size += (Integer)int_ptr[i];
  }

  free(int_ptr);
}

/**
 * Get data from a GP array and return it to a local buffer. Also return
 * an array of pointers to data in local buffer.
 * @param[in] g_p                pointer array handle
 * @param[in] lo[ndim]           lower corner of pointer array block
 * @param[in] hi[ndim]           upper corner of pointer array block
 * @param[out] buf               buffer that holds data
 * @param[out] buf_ptr           buffer that holds pointers to data
 * @param[in] ld[ndim-1]         physical dimensions of pointer buffer
 * @param[out] buf_size          buffer that holds size data
 * @param[in] ld_sz[ndim-1]      physical dimensions of size buffer
 * @param[out] size              total size of requested data
 * @param[in] intsize            parameter to distinguish between 4 and 8
 *                               byte integers
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wgp_get = pgp_get
#endif

void pgp_get(Integer g_p, Integer *lo, Integer *hi, void *buf,
             void **buf_ptr, Integer *ld, void *buf_size, Integer *ld_sz,
             Integer *size, Integer intsize)
{
  Integer handle, ndim, i, j, d, itmp, offset_sz, np;
  Integer idx, offset_d, offset_ptr, offset_rem;
  Integer nelems, index[GP_MAX_DIM];
  Integer block_ld[GP_MAX_DIM], block_ld_loc[GP_MAX_DIM];
  int *int_ptr;
  GP_INT *rem_ptr;
  int rc;
  armci_giov_t *desc;
  handle = g_p + GP_OFFSET;
  if (!GP[handle].active) {
    pnga_error("gp_get_size: inactive array handle specified", g_p);
  }
  ndim = pnga_ndim(GP[handle].g_ptr_array);
  for (i=0; i<ndim; i++) {
    if (GP[handle].lo[i] > GP[handle].hi[i])
      pnga_error("gp_get_size: illegal block size specified", g_p);
  }

  pnga_get(GP[handle].g_size_array, lo, hi, buf_size, ld_sz);
  
  /* Get strides of requested block */
  ndim = GP[handle].ndim;
  nelems = 1;
  for (i=0; i<ndim; i++) {
    block_ld[i] = (hi[i] - lo[i] + 1);
    nelems *= block_ld[i];
  }

  /* Based on sizes, construct buf_ptr array */
  int_ptr = (int*)buf;
  idx = 0;
  offset_ptr = 0;
  while(idx<nelems) {
    /* find corresponding index for idx */
    itmp = idx;
    for (j=0; j<ndim-1; j++) {
      index[j] = itmp%block_ld[j];
      itmp = (itmp - index[j])/block_ld[j];
    }
    index[ndim-1] = itmp;
    /* use index to evaluate offset in size array and
       buf_ptr array */
    offset_sz = index[ndim-1];
    offset_d = index[ndim-1];
    for (d=ndim-2; d>=0; d--) {
      offset_sz = offset_sz*ld_sz[d] + index[d];
      offset_d = offset_d*ld[d] + index[d];
    }
    /* evaluate offset in data buffer */
    buf_ptr[offset_d] = buf+offset_ptr;
    offset_ptr += ((int*)buf_size)[offset_sz];
    idx++;
  }
  *size = offset_ptr;

  /* locate the processors containing some portion of the patch represented by
   * lo and hi and return the results in _gp_map, gp_proclist, and np.
   * _gp_proclist contains a list of processors containing some portion of the
   * patch, _gp_map contains the lower and upper indices of the portion of the
   * patch held by a given processor and np contains the number of processors
   * tha contain some portion of the patch.
   */
  pnga_locate_region(GP[handle].g_size_array, lo, hi, _gp_map, _gp_proclist,
                     &np);
  /*printf("\n");*/
  /* Loop over processors containing data */
  for (idx=0; idx<np; idx++) {
    Integer p = _gp_proclist[idx];
    Integer *plo, *phi;
    Integer jcnt;
    gpm_GetRangeFromMap(idx, ndim, &plo, &phi);
    nelems = 1;
    /* Find out how big patch is */
    for (i=0; i<ndim; i++) {
      nelems *= (phi[i]-plo[i] + 1);
      block_ld_loc[i] = phi[i]-plo[i] + 1;
    }
    /* Allocate a buffer to hold remote pointers for patch and and
       array of descriptors for GetV operation */
    rem_ptr = (void**)malloc((size_t)(nelems)*sizeof(void*));
    desc = (armci_giov_t*)malloc((size_t)(nelems)*sizeof(armci_giov_t));

    /* Get remote pointers */
    pnga_get(GP[handle].g_ptr_array, plo, phi, rem_ptr, block_ld_loc);
    /* Construct descriptors */
    jcnt = 0;
    for (j=0; j<nelems; j++) {
      itmp = j;
      for (d=0; d<ndim-1; d++) {
        index[d] = itmp%block_ld_loc[d];
        itmp = (itmp - index[d])/block_ld_loc[d];
      }
      index[ndim-1] = itmp;
      offset_rem = index[ndim-1];
      offset_sz = index[ndim-1] + plo[ndim-1] - lo[ndim-1];
      offset_d = index[ndim-1] + plo[ndim-1] - lo[ndim-1];
      for (d=ndim-2; d>=0; d--) {
        offset_rem = offset_rem*block_ld_loc[d] + index[d];
        offset_sz = offset_sz*ld_sz[d] + index[d] + plo[d] - lo[d];
        offset_d = offset_d*ld[d] + index[d] + plo[d] - lo[d];
      }
      if (((int*)buf_size)[offset_sz] > 0) {
        desc[jcnt].src_ptr_array = (void*)rem_ptr[offset_rem];
        desc[jcnt].dst_ptr_array = buf_ptr[offset_d];
        if (intsize == 4) {
          desc[jcnt].bytes = (int)((int*)buf_size)[offset_sz];
        } else {
          desc[jcnt].bytes = (int)((long*)buf_size)[offset_sz];
        }
        desc[jcnt].ptr_array_len = 1;
        printf("p[%ld] nelems: %ld index[%ld,%ld] bytes: %d src_ptr: %p dst_ptr: %p\n",
                (long)pnga_nodeid(), (long)nelems,
                (long)(index[1]+plo[1]-1), (long)(index[0]+plo[0]-1),
                desc[jcnt].bytes, desc[jcnt].src_ptr_array,
                desc[jcnt].dst_ptr_array);
        jcnt++;
      } else {
        printf("p[%ld] null pointer at i: %ld j: %ld\n", (long)pnga_nodeid(),
                (long)(index[0]+plo[0]), (long)(index[1]+plo[1]));
      }
    }
  printf("p[%ld] Got to 5\n",(long)pnga_nodeid());
  if (jcnt > 0) {
    rc = ARMCI_GetV(desc, (int)jcnt, (int)p);
printf("p[%ld] Got to 6\n",(long)pnga_nodeid());
    if (rc) pnga_error("ARMCI_GetV failure in gp_get",rc);
  }
#if 0
#endif
    /* Free temporary buffers */
    free(rem_ptr);
    free(desc);
  }
  printf("p[%ld] Got to 7\n",(long)pnga_nodeid());
}
