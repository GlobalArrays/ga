/* $Id: ghosts.c,v 1.25 2002-08-29 22:49:18 manoj Exp $ */
/* 
 * module: ghosts.c
 * author: Bruce Palmer
 * description: implements GA collective communication operations to
 * update ghost cell regions.
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

 
/*#define PERMUTE_PIDS */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "globalp.h"
#include "base.h"
#include "armci.h"
#include "macdecls.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

#if !defined(CRAY_YMP)
#define BYTE_ADDRESSABLE_MEMORY
#endif

static global_array_t *GA = _ga_main_data_structure;

/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

/***************************************************************************/

/*\ Return a pointer to the location indicated by subscript and and an array
 * of leading dimensions (ld). Assume that subscript refers to a set of local
 * coordinates relative to the origin of the array and account for the
 * presence of ghost cells.
\*/
#define gam_LocationWithGhosts(proc, handle, subscript, ptr_loc, ld)           \
{                                                                              \
Integer _d, _factor = 1, _last=GA[handle].ndim - 1, _offset=0;                 \
Integer _lo[MAXDIM], _hi[MAXDIM];                                              \
  ga_ownsM(handle, proc, _lo, _hi);                                            \
  if (_last == 0) ld[0] = _hi[0] - _lo[0] + 1 + 2*GA[handle].width[0];         \
  for (_d = 0; _d < _last; _d++) {                                             \
    _offset += subscript[_d] * _factor;                                        \
    ld[_d] = _hi[_d] - _lo[_d] + 1 + 2*GA[handle].width[_d];                   \
    _factor *= ld[_d];                                                         \
  }                                                                            \
  _offset += subscript[_last] * _factor;                                       \
  *(ptr_loc) = GA[handle].ptr[proc] + _offset*GA[handle].elemsize;             \
}

void nga_access_ghost_ptr(Integer* g_a, Integer dims[],
                      void* ptr, Integer ld[])

{
char *lptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  i, lo[MAXDIM], hi[MAXDIM];
Integer ndim = GA[handle].ndim;

   GA_PUSH_NAME("nga_access_ghost_ptr");

   nga_distribution_(g_a, &GAme, lo, hi);

   for (i=0; i < ndim; i++) {
     dims[i] = 0;
   }

   gam_LocationWithGhosts(GAme, handle, dims, &lptr, ld);
   *(char**)ptr = lptr; 
   for (i=0; i < ndim; i++)
     dims[i] = hi[i] - lo[i] + 1 + 2*GA[handle].width[i];
   GA_POP_NAME;
}

/*\  PROVIDE POINTER TO LOCALLY HELD DATA, ACCOUNTING FOR
 *   PRESENCE OF GHOST CELLS
\*/
void nga_access_ghost_element_(Integer* g_a, Integer* index,
                        Integer subscript[], Integer ld[])
{
char *ptr;
Integer  handle = GA_OFFSET + *g_a;
Integer i;
unsigned long    elemsize;
unsigned long    lref, lptr;
   GA_PUSH_NAME("nga_access_ghost_element");
   /* Indices conform to Fortran convention. Shift them down 1 so that
      gam_LocationWithGhosts works. */
   for (i=0; i<GA[handle].ndim; i++) subscript[i]--;
   gam_LocationWithGhosts(GAme, handle, subscript, &ptr, ld);
   /*
    * return patch address as the distance elements from the reference address
    *
    * .in Fortran we need only the index to the type array: dbl_mb or int_mb
    *  that are elements of COMMON in the the mafdecls.h include file
    * .in C we need both the index and the pointer
    */

   elemsize = (unsigned long)GA[handle].elemsize;

   /* compute index and check if it is correct */
   switch (ga_type_c2f(GA[handle].type)){
     case MT_F_DBL:
        *index = (Integer) ((DoublePrecision*)ptr - DBL_MB);
        lref = (unsigned long)DBL_MB;
        break;

     case MT_F_DCPL:
        *index = (Integer) ((DoubleComplex*)ptr - DCPL_MB);
        lref = (unsigned long)DCPL_MB;
        break;

     case MT_F_INT:
        *index = (Integer) ((Integer*)ptr - INT_MB);
        lref = (unsigned long)INT_MB;
        break;

     case MT_F_REAL:
        *index = (Integer) ((float*)ptr - FLT_MB);
        lref = (unsigned long)FLT_MB;
        break;        
   }

#ifdef BYTE_ADDRESSABLE_MEMORY
   /* check the allignment */
   lptr = (unsigned long)ptr;
   if( lptr%elemsize != lref%elemsize ){ 
       printf("%d: lptr=%lu(%lu) lref=%lu(%lu)\n",(int)GAme,lptr,lptr%elemsize,
                                                    lref,lref%elemsize);
       ga_error("nga_access: MA addressing problem: base address misallignment",
                 handle);
   }
#endif

   /* adjust index for Fortran addressing */
   (*index) ++ ;

   FLUSH_CACHE;
   GA_POP_NAME;
}

/*\ PROVIDE ACCESS TO LOCAL PATCH OF A GLOBAL ARRAY WITH GHOST CELLS
\*/
void FATR nga_access_ghosts_(Integer* g_a, Integer dims[],
                      Integer* index, Integer ld[])
{
char     *ptr;
Integer  handle = GA_OFFSET + *g_a;
unsigned long    elemsize;
unsigned long    lref, lptr;

   GA_PUSH_NAME("nga_access_ghosts");
   nga_access_ghost_ptr(g_a, dims, &ptr, ld);

   /*
    * return patch address as the distance elements from the reference address
    *
    * .in Fortran we need only the index to the type array: dbl_mb or int_mb
    *  that are elements of COMMON in the the mafdecls.h include file
    * .in C we need both the index and the pointer
    */

   elemsize = (unsigned long)GA[handle].elemsize;

   /* compute index and check if it is correct */
   switch (ga_type_c2f(GA[handle].type)){
     case MT_F_DBL:
        *index = (Integer) ((DoublePrecision*)ptr - DBL_MB);
        lref = (unsigned long)DBL_MB;
        break;

     case MT_F_DCPL:
        *index = (Integer) ((DoubleComplex*)ptr - DCPL_MB);
        lref = (unsigned long)DCPL_MB;
        break;

     case MT_F_INT:
        *index = (Integer) ((Integer*)ptr - INT_MB);
        lref = (unsigned long)INT_MB;
        break;

     case MT_F_REAL:
        *index = (Integer) ((float*)ptr - FLT_MB);
        lref = (unsigned long)FLT_MB;
        break;        

   }

#ifdef BYTE_ADDRESSABLE_MEMORY
   /* check the allignment */
   lptr = (unsigned long)ptr;
   if( lptr%elemsize != lref%elemsize ){ 
       printf("%d: lptr=%lu(%lu) lref=%lu(%lu)\n",(int)GAme,lptr,lptr%elemsize,
                                                    lref,lref%elemsize);
       ga_error("nga_access: MA addressing problem: base address misallignment",
                 handle);
   }
#endif

   /* adjust index for Fortran addressing */
   (*index) ++ ;
   FLUSH_CACHE;

   GA_POP_NAME;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM
\*/
void FATR ga_update1_ghosts_(Integer *g_a)
{
  Integer idx, ipx, inx, i, np, handle=GA_OFFSET + *g_a, proc_rem;
  Integer size, ndim, nwidth, offset, slice, increment[MAXDIM];
  Integer width[MAXDIM];
  Integer dims[MAXDIM], imax;
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer plo_loc[MAXDIM], phi_loc[MAXDIM];
  Integer lo_rem[MAXDIM], hi_rem[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer slo_rem[MAXDIM], shi_rem[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  int stride_loc[MAXDIM], stride_rem[MAXDIM],count[MAXDIM];
  char *ptr_loc, *ptr_rem;
  logical hasData = TRUE;

  /* This routine makes use of the shift algorithm to update data in the
   * ghost cells bounding the local block of visible data. The shift
   * algorithm starts by updating the blocks of data along the first
   * dimension by grabbing a block of data that is width[0] deep but
   * otherwise matches the  dimensions of the data residing on the
   * calling processor. The update of the second dimension, however,
   * grabs a block that is width[1] deep in the second dimension but is
   * ldim0 + 2*width[0] in the first dimensions where ldim0 is the
   * size of the visible data along the first dimension. The remaining
   * dimensions are left the same. For the next update, the width of the
   * second dimension is also increased by 2*width[1] and so on. This
   * algorith makes use of the fact that data for the dimensions that
   * have already been updated is available on each processor and can be
   * used in the updates of subsequent dimensions. The total number of
   * separate updates is 2*ndim, an update in the negative and positive
   * directions for each dimension.
   *
   * To perform the update, this routine makes use of several copies of
   * indices marking the upper and lower limits of data. Indices
   * beginning with the character "p" are relative indices marking the
   * location of the data set relative to the origin the local patch of
   * the global array, all other indices are in absolute coordinates and
   * mark locations in the total global array. The indices used by this
   * routine are described below.
   *
   *       lo_loc[], hi_loc[]: The lower and upper indices of the visible
   *       block of data held by the calling processor.
   *
   *       lo_rem[], hi_rem[]: The lower and upper indices of the block
   *       of data on a remote processor or processors that is needed to
   *       fill in the calling processors ghost cells. These indices are
   *       NOT corrected for wrap-around (periodic) boundary conditions
   *       so they can be negative or greater than the array dimension
   *       values held in dims[].
   *
   *       slo_rem[], shi_rem[]: Similar to lo_rem[] and hi_rem[], except
   *       that these indices have been corrected for wrap-around
   *       boundary conditions. If lo_rem[] and hi_rem[] cross a global
   *        array boundary, as opposed to being entirely located on one
   *       side or the other of the array, then two sets of slo_rem[] and
   *       shi_rem[] will be created. One set will correspond to the
   *       block of data on one side of the global array boundary and the
   *       other set will correspond to the remaining block. This
   *       situation will only occur if the value of the ghost cell width
   *       is greater than the dimension of the visible global array
   *       data on a single processor.
   *
   *       thi_rem[], thi_rem[]: The lower and upper indices of the visible
   *       data on a remote processor.
   *
   *       plo_loc[], phi_loc[]: The indices of the local data patch that
   *       is going to be updated.
   *
   *       plo_rem[], phi_rem[]: The indices of the data patch on the
   *       remote processor that will be used to update the data on the
   *       calling processor. Note that the dimensions of the patches
   *       represented by plo_loc[], plo_rem[] and plo_loc[], phi_loc[]
   *       must be the same.
   *
   * For the case where the width of the ghost cells is more than the
   * width of the visible data held on a processor, special problems
   * arise. It now takes several updates to fill in one block of boundary
   * data and it is now necessary to keep track of where each of these
   * blocks of data go in the ghost cell region. To do this two extra
   * variables are needed. These are offset and slice. Slice is equal to
   * the width of the visible data along the dimension being updated
   * minus one coming from the remote processor. Offset is the amount
   * that this data must be moved inward from the lower boundary of the
   * ghost cell region. Another variable that is also used to handle
   * this case is imax. If this variable is set to 2, then this means
   * that the block of data that is needed to update the ghost cells
   * crosses a global array boundary and the block needs to be broken
   * up into two pieces. */

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) return;

  GA_PUSH_NAME("ga_update1_ghosts");

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);
  if (DEBUG) {
    fprintf(stderr,"p[%d] lo(1) %d hi(1) %d\n",(int)GAme,
        (int)lo_loc[0],(int)hi_loc[0]);
    fprintf(stderr,"p[%d] lo(2) %d hi(2) %d\n",(int)GAme,
        (int)lo_loc[1],(int)hi_loc[1]);
  }
  /* initialize range increments and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    increment[idx] = 0;
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
    if (lo_loc[idx] == 0 && hi_loc[idx] == -1) hasData = FALSE;
  }

  /* loop over dimensions for sequential update using shift algorithm */
  for (idx=0; idx < ndim; idx++) {
    nwidth = width[idx];

    /* Do not bother with update if nwidth is zero or processor has
       no data */
    if (nwidth != 0 && hasData) {

      /* Perform update in negative direction. Start by getting rough
         estimate of block of needed data*/
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = lo_loc[i] - nwidth;
          hi_rem[i] = lo_loc[i] - 1;
          /* Check to see if we will need to update ghost cells using
             one or two major patches of the global array. */
          if (lo_rem[i] < 1) {
            if (hi_rem[i] > 0) {
              imax = 2;
            } else {
              imax = 1;
            }
          } else {
            imax = 1;
          }
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      for (inx = 0; inx < imax; inx++) {
        /* Check to see if boundary is being updated in one patch or two,
           adjust lower boundary accordingly. */
        if (DEBUG) {
          fprintf(stderr,"\n Value of inx is %d\n\n",(int)inx);
        }
        for (i=0; i<ndim; i++) {
          if (imax == 2 && i == idx) {
            if (inx == 0) {
              slo_rem[i] = 1;
              shi_rem[i] = hi_rem[i];
            } else {
              slo_rem[i] = lo_rem[i] + dims[i];
              shi_rem[i] = dims[i];
            }
          } else if (i == idx) {
            if (lo_rem[i] < 1) {
              slo_rem[i] = dims[i] - nwidth + 1;
              shi_rem[i] = dims[i];
            } else {
              slo_rem[i] = lo_rem[i];
              shi_rem[i] = hi_rem[i];
            }
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
          if (DEBUG) {
            fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
                "  i=%d idx=%d imax=%d\n",(int)GAme,(int)i+1,
                (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
                (int)idx,(int)imax);
          }
        }
        /* locate processor with this data */
        if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
            GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
            slo_rem, shi_rem, *g_a);

        if (DEBUG) {
          fprintf(stderr,"\np[%d] Value of np is %d Value of imax is %d\n",
              (int)GAme,(int)np,(int)imax);
        }
        for (ipx = 0; ipx < np; ipx++) {
          /* Get actual coordinates of desired chunk of remote
             data as well as the actual coordinates of the local chunk
             of data that will receive the remote data (these
             coordinates take into account the presence of ghost
             cells). Start by finding out what data is actually held by
             remote processor. */
          proc_rem = GA_proclist[ipx];
          nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
          if (DEBUG) {
            fprintf(stderr,"\np[%d] Checking second step\n",(int)GAme);
          }
          for (i = 0; i < ndim; i++) {
            if (increment[i] == 0) {
              if (i == idx) {
                if (np == 1 && imax == 1) {
                  plo_rem[i] = thi_rem[i] - tlo_rem[i] + 1;
                  phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
                  plo_loc[i] = 0;
                  phi_loc[i] = width[i] - 1;
                } else {
                  if (tlo_rem[i] >= slo_rem[i]) {
                    offset = tlo_rem[i] - lo_rem[i];
                    slice = thi_rem[i] - tlo_rem[i];
                  } else {
                    offset = 0;
                    slice = thi_rem[i] - slo_rem[i];
                  }
                  if (offset < 0) offset = offset + dims[i];
                  if (offset >= dims[i]) offset = offset - dims[i];
                  plo_rem[i] = thi_rem[i] - tlo_rem[i] + width[i] - slice;
                  phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
                  plo_loc[i] = offset;
                  phi_loc[i] = offset + slice;
                }
              } else {
                plo_rem[i] = width[i];
                phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
                plo_loc[i] = width[i];
                phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
              }
            } else {
              plo_rem[i] = 0;
              phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
              plo_loc[i] = 0;
              phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
            }
            if (DEBUG) {
              if (i == idx && (np > 1 || imax > 1)) {
                fprintf(stderr,"\np[%d] offset %d slice %d increment(%d) %d\n",
                    (int)GAme,(int)offset,(int)slice,(int)i+1,(int)increment[i]);
              } else {
                fprintf(stderr,"\n");
              }
              fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
                  (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
              fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
                  (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
              fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
                  (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
            }
          }

          /* Get pointer to local data buffer and remote data
             buffer as well as lists of leading dimenstions */
          gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
          gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
          if (DEBUG) {
            for (i=0; i<ndim-1; i++) {
              fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
                  (int)ld_loc[i]);
              fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
                  (int)ld_rem[i]);
            }
          }

          /* Evaluate strides on local and remote processors */
          gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
              stride_loc);

          /* Compute the number of elements in each dimension and store
             result in count. Scale the first element in count by the
             element size. */
          gam_ComputeCount(ndim, plo_rem, phi_rem, count);
          count[0] *= size;
 
          /* get remote data */
          ARMCI_GetS(ptr_rem, stride_rem, ptr_loc, stride_loc, count,
              (int)(ndim - 1), (int)proc_rem);
        }
      }

      /* Perform update in positive direction. Start by getting rough
         estimate of block of needed data*/
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = hi_loc[i] + 1;
          hi_rem[i] = hi_loc[i] + nwidth;
          /* Check to see if we will need to update ghost cells using
             one or two major patches of the global array. */
          if (hi_rem[i] > dims[i]) {
            if (lo_rem[i] <= dims[i]) {
              imax = 2;
            } else {
              imax = 1;
            }
          } else {
            imax = 1;
          }
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      for (inx = 0; inx < imax; inx++) {
        /* Check to see if boundary is being updated in one patch or two,
           adjust lower boundary accordingly. */
        if (DEBUG) {
          fprintf(stderr,"\n Value of inx is %d\n\n",(int)inx);
        }
        for (i=0; i<ndim; i++) {
          if (imax == 2 && i == idx) {
            if (inx == 0) {
              slo_rem[i] = lo_rem[i];
              shi_rem[i] = dims[i];
            } else {
              slo_rem[i] = 1;
              shi_rem[i] = hi_rem[i] - dims[i];
            }
          } else if (i == idx) {
            if (hi_rem[i] > dims[i]) {
              slo_rem[i] = 1;
              shi_rem[i] = nwidth;
            } else {
              slo_rem[i] = lo_rem[i];
              shi_rem[i] = hi_rem[i];
            }
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
          if (DEBUG) {
            fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
                "  i=%d idx=%d imax=%d\n",(int)GAme,(int)i+1,
                (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
                (int)idx,(int)imax);
          }
        }
        /* locate processor with this data */
        if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
            GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
            slo_rem, shi_rem, *g_a);

        if (DEBUG) {
          fprintf(stderr,"\np[%d] Value of np is %d Value of imax is %d\n",
              (int)GAme,(int)np,(int)imax);
        }
        for (ipx = 0; ipx < np; ipx++) {
          /* Get actual coordinates of desired chunk of remote
             data as well as the actual coordinates of the local chunk
             of data that will receive the remote data (these
             coordinates take into account the presence of ghost
             cells). Start by finding out what data is actually held by
             remote processor. */
          proc_rem = GA_proclist[ipx];
          nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
          if (DEBUG) {
            fprintf(stderr,"\np[%d] Checking second step\n",(int)GAme);
          }
          for (i = 0; i < ndim; i++) {
            if (increment[i] == 0) {
              if (i == idx) {
                if (np == 1 && imax == 1) {
                  plo_rem[i] = width[i];
                  phi_rem[i] = 2*width[i] - 1;
                  plo_loc[i] = hi_loc[i] - lo_loc[i] + 1 + width[i];
                  phi_loc[i] = hi_loc[i] - lo_loc[i] + 2*width[i];
                } else {
                  offset = tlo_rem[i] - hi_loc[i] - 1;
                  if (thi_rem[i] <= shi_rem[i]) {
                    slice = thi_rem[i] - tlo_rem[i];
                  } else {
                    slice = shi_rem[i] - tlo_rem[i];
                  }
                  if (offset < 0) offset = offset + dims[i];
                  if (offset >= dims[i]) offset = offset - dims[i];
                  plo_rem[i] = width[i];
                  phi_rem[i] = width[i] + slice;
                  plo_loc[i] = hi_loc[i] - lo_loc[i] + width[i] + 1 + offset;
                  phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i] + 1
                    + offset + slice;
                }
              } else {
                plo_rem[i] = width[i];
                phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
                plo_loc[i] = width[i];
                phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
              }
            } else {
              plo_rem[i] = 0;
              phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
              plo_loc[i] = 0;
              phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
            }
            if (DEBUG) {
              if (i == idx && (np > 1 || imax > 1)) {
                fprintf(stderr,"\np[%d] offset %d slice %d increment(%d) %d\n",
                    (int)GAme,(int)offset,(int)slice,(int)i+1,(int)increment[i]);
              } else {
                fprintf(stderr,"\n");
              }
              fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
                  (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
              fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
                  (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
              fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
                  (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
            }
          }

          /* Get pointer to local data buffer and remote data
             buffer as well as lists of leading dimenstions */
          gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
          gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
          if (DEBUG) {
            for (i=0; i<ndim-1; i++) {
              fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
                  (int)ld_loc[i]);
              fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
                  (int)ld_rem[i]);
            }
          }

          /* Evaluate strides on local and remote processors */
          gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
              stride_loc);

          /* Compute the number of elements in each dimension and store
             result in count. Scale the first element in count by the
             element size. */
          gam_ComputeCount(ndim, plo_rem, phi_rem, count);
          count[0] *= size;
 
          /* get remote data */
          ARMCI_GetS(ptr_rem, stride_rem, ptr_loc, stride_loc, count,
              (int)(ndim - 1), (int)proc_rem);
        }
      }
    }
    /* synchronize all processors and update increment array */
    if (idx < ndim-1) ga_sync_();
    increment[idx] = 2*nwidth;
  }

  GA_POP_NAME;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING PUT CALLS
\*/
logical FATR ga_update2_ghosts_(Integer *g_a)
{
  Integer idx, ipx, np, handle=GA_OFFSET + *g_a, proc_rem;
  Integer ntot, mask[MAXDIM];
  Integer size, ndim, i, itmp;
  Integer width[MAXDIM], dims[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer tlo_loc[MAXDIM], thi_loc[MAXDIM];
  Integer plo_loc[MAXDIM], phi_loc[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer plo_rem[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  logical mask0;
  int stride_loc[MAXDIM], stride_rem[MAXDIM],count[MAXDIM];
  char *ptr_loc, *ptr_rem;

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) {
    return TRUE;
  }

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;
  /* initialize ghost cell widths and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater than
     the corresponding value in width[]). */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1<width[idx]) {
          if (DEBUG) {
            fprintf(stderr,"ERR1 p[%d]  ipx = %d mapc[%d] = %d\n",
                (int)GAme,(int)ipx,(int)ipx,GA[handle].mapc[ipx]);
          }
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1<width[idx]) {
          if (DEBUG) {
            fprintf(stderr,"ERR2 p[%d] dims[%d] = %d  ipx = %d mapc[%d] = %d\n",
                (int)GAme,(int)idx,GA[handle].dims[idx],
                (int)ipx,(int)ipx,GA[handle].mapc[ipx]);
          }
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("ga_update2_ghosts");
  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);
  if (DEBUG) {
    fprintf(stderr,"p[%d] lo(1) %d hi(1) %d\n",(int)GAme,
        (int)lo_loc[0],(int)hi_loc[0]);
    fprintf(stderr,"p[%d] lo(2) %d hi(2) %d\n",(int)GAme,
        (int)lo_loc[1],(int)hi_loc[1]);
  }

  /* evaluate total number of PUT operations that will be required */
  ntot = 1;
  for (idx=0; idx < ndim; idx++) ntot *= 3;

  /* Loop over all PUT operations. The operation corresponding to the
     mask of all zeros is left out. */
  for (ipx=0; ipx < ntot; ipx++) {
    /* Convert ipx to corresponding mask values */
    itmp = ipx;
    mask0 = TRUE;
    for (idx = 0; idx < ndim; idx++) {
      i = itmp%3;
      mask[idx] = i-1;
      if (mask[idx] != 0) mask0 = FALSE;
      itmp = (itmp-i)/3;
    }
    if (mask0) continue;

    /* check to see if ghost cell block has zero elements*/
    mask0 = FALSE;
    for (idx = 0; idx < ndim; idx++) {
      if (mask[idx] != 0 && width[idx] == 0) mask0 = TRUE;
    }
    if (mask0) continue;
    if (DEBUG) {
      fprintf(stderr,"\n");
      for (idx=0; idx<ndim; idx++) {
        fprintf(stderr,"p[%d] ipx = %d  mask[%d] = %d\n",
            (int)GAme,(int)ipx,(int)idx,(int)mask[idx]);
      }
    }
    /* Now that mask has been determined, find data that is to be moved
     * and identify processor to which it is going. Wrap boundaries
     * around, if necessary */
    for (idx = 0; idx < ndim; idx++) {
      if (mask[idx] == 0) {
        tlo_loc[idx] = lo_loc[idx];
        thi_loc[idx] = hi_loc[idx];
        tlo_rem[idx] = lo_loc[idx];
        thi_rem[idx] = hi_loc[idx];
      } else if (mask[idx] == -1) {
        tlo_loc[idx] = lo_loc[idx];
        thi_loc[idx] = lo_loc[idx]+width[idx]-1;
        if (lo_loc[idx] > 1) {
          tlo_rem[idx] = lo_loc[idx]-width[idx];
          thi_rem[idx] = lo_loc[idx]-1;
        } else {
          tlo_rem[idx] = dims[idx]-width[idx]+1;
          thi_rem[idx] = dims[idx];
        }
      } else if (mask[idx] == 1) {
        tlo_loc[idx] = hi_loc[idx]-width[idx]+1;
        thi_loc[idx] = hi_loc[idx];
        if (hi_loc[idx] < dims[idx]) {
          tlo_rem[idx] = hi_loc[idx] + 1;
          thi_rem[idx] = hi_loc[idx] + width[idx];
        } else {
          tlo_rem[idx] = 1;
          thi_rem[idx] = width[idx];
        }
      } else {
        fprintf(stderr,"Illegal mask value found\n");
      }
      if (DEBUG) {
        fprintf(stderr,"p[%d] ipx = %d tlo_loc[%d] = %d thi_loc[%d] = %d\n",
            (int)GAme,(int)ipx,(int)idx,(int)tlo_loc[idx],(int)idx,
            (int)thi_loc[idx]);
        fprintf(stderr,"p[%d] ipx = %d tlo_rem[%d] = %d thi_rem[%d] = %d\n",
            (int)GAme,(int)ipx,(int)idx,(int)tlo_rem[idx],(int)idx,
            (int)thi_rem[idx]);
      }
    }
    /* Locate remote processor to which data must be sent */
    if (!nga_locate_region_(g_a, tlo_rem, thi_rem, _ga_map,
       GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
       tlo_rem, thi_rem, *g_a);
    if (np > 1) {
      fprintf(stderr,"More than one remote processor found\n");
    }
    /* Remote processor has been identified, now get ready to send
       data to it. Start by getting distribution on remote
       processor.*/
    proc_rem = GA_proclist[0];
    nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
    for (idx = 0; idx < ndim; idx++) {
      if (mask[idx] == 0) {
        plo_loc[idx] = width[idx];
        phi_loc[idx] = hi_loc[idx]-lo_loc[idx]+width[idx];
        plo_rem[idx] = plo_loc[idx];
      } else if (mask[idx] == -1) {
        plo_loc[idx] = width[idx];
        phi_loc[idx] = 2*width[idx]-1;
        plo_rem[idx] = thi_rem[idx]-tlo_rem[idx]+width[idx]+1;
      } else if (mask[idx] == 1) {
        plo_loc[idx] = hi_loc[idx]-lo_loc[idx]+1;
        phi_loc[idx] = hi_loc[idx]-lo_loc[idx]+width[idx];
        plo_rem[idx] = 0;
      }
    }
    /* Get pointer to local data buffer and remote data
       buffer as well as lists of leading dimenstions */
    gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
    gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);

    /* Evaluate strides on local and remote processors */
    gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
                  stride_loc);

    /* Compute the number of elements in each dimension and store
       result in count. Scale the first element in count by the
       element size. */
    gam_ComputeCount(ndim, plo_loc, phi_loc, count);
    count[0] *= size;
 
    /* put data on remote processor */
    ARMCI_PutS(ptr_loc, stride_loc, ptr_rem, stride_rem, count,
          (int)(ndim - 1), (int)proc_rem);
  }

  GA_POP_NAME;
  return TRUE;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM AND PUT CALLS
\*/
logical FATR ga_update3_ghosts_(Integer *g_a)
{
  Integer idx, ipx, inx, i, np, handle=GA_OFFSET + *g_a, proc_rem;
  Integer size, ndim, nwidth, increment[MAXDIM];
  Integer width[MAXDIM];
  Integer dims[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer plo_loc[MAXDIM], phi_loc[MAXDIM];
  Integer lo_rem[MAXDIM], hi_rem[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer slo_rem[MAXDIM], shi_rem[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  int stride_loc[MAXDIM], stride_rem[MAXDIM],count[MAXDIM];
  char *ptr_loc, *ptr_rem;

  /* This routine makes use of the shift algorithm to update data in the
   * ghost cells bounding the local block of visible data. The shift
   * algorithm starts by updating the blocks of data along the first
   * dimension by grabbing a block of data that is width[0] deep but
   * otherwise matches the  dimensions of the data residing on the
   * calling processor. The update of the second dimension, however,
   * grabs a block that is width[1] deep in the second dimension but is
   * ldim0 + 2*width[0] in the first dimensions where ldim0 is the
   * size of the visible data along the first dimension. The remaining
   * dimensions are left the same. For the next update, the width of the
   * second dimension is also increased by 2*width[1] and so on. This
   * algorith makes use of the fact that data for the dimensions that
   * have already been updated is available on each processor and can be
   * used in the updates of subsequent dimensions. The total number of
   * separate updates is 2*ndim, an update in the negative and positive
   * directions for each dimension. This implementation uses simple put
   * operations to perform the updates along each dimension with an
   * intervening synchronization call being used to make sure that the
   * necessary data is available on each processor before starting the
   * update along the next dimension.
   *
   * To perform the update, this routine makes use of several copies of
   * indices marking the upper and lower limits of data. Indices
   * beginning with the character "p" are relative indices marking the
   * location of the data set relative to the origin the local patch of
   * the global array, all other indices are in absolute coordinates and
   * mark locations in the total global array. The indices used by this
   * routine are described below.
   *
   *       lo_loc[], hi_loc[]: The lower and upper indices of the visible
   *       block of data held by the calling processor.
   *
   *       lo_rem[], hi_rem[]: The lower and upper indices of the block
   *       of data on a remote processor or processors that is needed to
   *       fill in the calling processors ghost cells. These indices are
   *       NOT corrected for wrap-around (periodic) boundary conditions
   *       so they can be negative or greater than the array dimension
   *       values held in dims[].
   *
   *       slo_rem[], shi_rem[]: Similar to lo_rem[] and hi_rem[], except
   *       that these indices have been corrected for wrap-around
   *       boundary conditions. 
   *
   *       thi_rem[], thi_rem[]: The lower and upper indices of the visible
   *       data on a remote processor.
   *
   *       plo_loc[], phi_loc[]: The indices of the local data patch that
   *       is going to be updated.
   *
   *       plo_rem[], phi_rem[]: The indices of the data patch on the
   *       remote processor that will be used to update the data on the
   *       calling processor. Note that the dimensions of the patches
   *       represented by plo_loc[], plo_rem[] and plo_loc[], phi_loc[]
   *       must be the same.
   */

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) return TRUE;

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* initialize range increments and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    increment[idx] = 0;
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
    if (lo_loc[idx] == 0 && hi_loc[idx] == -1) return FALSE;
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater
     than the corresponding value in width[]. */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("ga_update3_ghosts");

  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);

  /* loop over dimensions for sequential update using shift algorithm */
  for (idx=0; idx < ndim; idx++) {
    nwidth = width[idx];

    /* Do not bother with update if nwidth is zero */
    if (nwidth != 0) {

      /* Perform update in negative direction. Start by getting rough
         idea of where data needs to go. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = lo_loc[i] - width[i];
          hi_rem[i] = lo_loc[i] - 1;
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (lo_rem[i] < 1) {
            slo_rem[i] = dims[i] - width[i] + 1;
            shi_rem[i] = dims[i];
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
        } else {
          slo_rem[i] = lo_rem[i];
          shi_rem[i] = hi_rem[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rem, shi_rem, *g_a);

      if (DEBUG) {
        fprintf(stderr,"\np[%d] Value of np is %d\n",
            (int)GAme,(int)np);
      }
      /* Get actual coordinates of desired location of remote
         data as well as the actual coordinates of the local chunk
         of data that will be sent to remote processor (these
         coordinates take into account the presence of ghost
         cells). Start by finding out what data is actually held by
         remote processor. */
      proc_rem = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_rem[i] = thi_rem[i] - tlo_rem[i] + width[i] + 1;
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + 2*width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = 2*width[i] - 1;
          } else {
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
          plo_loc[i] = 0;
          phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
          fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
          fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
      gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
          fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_rem[i]);
        }
      }

      /* Evaluate strides on local and remote processors */
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_loc);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rem, phi_rem, count);
      count[0] *= size;

      /* Put local data on remote processor */
      ARMCI_PutS(ptr_loc, stride_loc, ptr_rem, stride_rem, count,
          (int)(ndim - 1), (int)proc_rem);

      /* Perform update in positive direction. Start by getting rough
         idea of where data needs to go. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = hi_loc[i] + 1;
          hi_rem[i] = hi_loc[i] + nwidth;
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      /* Check to see if boundary is being updated in one patch or two,
         adjust lower boundary accordingly. */
      if (DEBUG) {
        fprintf(stderr,"\n Value of inx is %d\n\n",(int)inx);
      }
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rem[i] > dims[i]) {
            slo_rem[i] = 1;
            shi_rem[i] = width[i];
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
        } else {
          slo_rem[i] = lo_rem[i];
          shi_rem[i] = hi_rem[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rem, shi_rem, *g_a);

      if (DEBUG) {
        fprintf(stderr,"\np[%d] Value of np is %d\n",
            (int)GAme,(int)np);
      }
      /* Get actual coordinates of desired chunk of remote
         data as well as the actual coordinates of the local chunk
         of data that will receive the remote data (these
         coordinates take into account the presence of ghost
         cells). Start by finding out what data is actually held by
         remote processor. */
      proc_rem = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Checking second step\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_rem[i] = 0;
            phi_rem[i] = width[i] - 1;
            plo_loc[i] = hi_loc[i] - lo_loc[i] + width[i] - 1;
            phi_loc[i] = hi_loc[i] - lo_loc[i] + 2*width[i] - 1;
          } else {
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
          plo_loc[i] = 0;
          phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
          fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
          fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
      gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
          fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_rem[i]);
        }
      }

      /* Evaluate strides on local and remote processors */
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_loc);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rem, phi_rem, count);
      count[0] *= size;

      /* get remote data */
      ARMCI_PutS(ptr_loc, stride_loc, ptr_rem, stride_rem, count,
          (int)(ndim - 1), (int)proc_rem);
    }
    /* synchronize all processors and update increment array */
    if (idx < ndim-1) ga_sync_();
    increment[idx] = 2*nwidth;
  }

  GA_POP_NAME;
  return TRUE;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM AND
 *  MESSAGE PASSING
\*/
logical FATR ga_update4_ghosts_(Integer *g_a)
{
  Integer idx, ipx, idir, i, np, handle=GA_OFFSET + *g_a;
  Integer size, buflen, buftot, bufsize, ndim, increment[MAXDIM];
  Integer send_buf = INVALID_MA_HANDLE, rcv_buf = INVALID_MA_HANDLE;
  Integer proc_rem_snd, proc_rem_rcv, pmax;
  Integer msgcnt, length, msglen;
  Integer width[MAXDIM], dims[MAXDIM], index[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer plo_snd[MAXDIM], phi_snd[MAXDIM];
  Integer lo_rcv[MAXDIM], hi_rcv[MAXDIM];
  Integer slo_rcv[MAXDIM], shi_rcv[MAXDIM];
  Integer plo_rcv[MAXDIM], phi_rcv[MAXDIM];
  Integer ld_loc[MAXDIM];
  int stride_snd[MAXDIM], stride_rcv[MAXDIM],count[MAXDIM];
  char *ptr_snd, *ptr_rcv;
  char send_name[32], rcv_name[32];
  void *snd_ptr, *rcv_ptr;

  /* This routine makes use of the shift algorithm to update data in the
   * ghost cells bounding the local block of visible data. The shift
   * algorithm starts by updating the blocks of data along the first
   * dimension by grabbing a block of data that is width[0] deep but
   * otherwise matches the  dimensions of the data residing on the
   * calling processor. The update of the second dimension, however,
   * grabs a block that is width[1] deep in the second dimension but is
   * ldim0 + 2*width[0] in the first dimensions where ldim0 is the
   * size of the visible data along the first dimension. The remaining
   * dimensions are left the same. For the next update, the width of the
   * second dimension is also increased by 2*width[1] and so on. This
   * algorith makes use of the fact that data for the dimensions that
   * have already been updated is available on each processor and can be
   * used in the updates of subsequent dimensions. The total number of
   * separate updates is 2*ndim, an update in the negative and positive
   * directions for each dimension.
   *
   * This implementation make use of explicit message passing to perform
   * the update. Separate message types for the updates in each coordinate
   * direction are used to maintain synchronization locally and to
   * guarantee that the data is present before the updates in a new
   * coordinate direction take place.
   *
   * To perform the update, this routine makes use of several copies of
   * indices marking the upper and lower limits of data. Indices
   * beginning with the character "p" are relative indices marking the
   * location of the data set relative to the origin the local patch of
   * the global array, all other indices are in absolute coordinates and
   * mark locations in the total global array. The indices used by this
   * routine are described below.
   *
   *       lo_loc[], hi_loc[]: The lower and upper indices of the visible
   *       block of data held by the calling processor.
   *
   *       lo_rcv[], hi_rcv[]: The lower and upper indices of the blocks
   *       of data that will be either sent to or received from a remote
   *       processor. These indices are NOT corrected for wrap-around
   *       (periodic) boundary conditions so they can be negative or greater
   *       than the array dimension values held in dims[].
   *
   *       slo_rcv[], shi_rcv[]: Similar to lo_rcv[] and hi_rcv[], except
   *       that these indices have been corrected for wrap-around
   *       boundary conditions.
   *
   *       plo_rcv[], phi_rcv[]: The local indices of the local data patch
   *       that receive that message from the remote processor.
   *
   *       plo_snd[], phi_snd[]: The local indices of the data patch
   *       that will be sent to the remote processor. Note that the
   *       dimensions of the patches represented by plo_rec[], plo_rec[] and
   *       plo_snd[], phi_snd[] must be the same.
   */

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) return TRUE;

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* initialize range increments and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    increment[idx] = 0;
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater
     than the corresponding value in width[]. */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("ga_update4_ghosts");
  msgcnt = 0;

  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);
  /* Get indices of processor in virtual grid */
  nga_proc_topology_(g_a, &GAme, index);

  /* Try to find maximum size of message that will be sent during
   * update operations and use this to allocate memory for message
   * passing buffers. */
  buftot = 1;
  for (i=0; i<ndim; i++) {
    buftot *= (hi_loc[i]-lo_loc[i] + 1 + 2*width[i]);
  }
  buflen = 1;
  for (i = 0; i < ndim; i++) {
    idir =  hi_loc[i] - lo_loc[i] + 1;
    if (buflen < (buftot/(idir + 2*width[i]))*width[i]) {
      buflen = (buftot/(idir + 2*width[i]))*width[i];
    }
  }
  bufsize = size*buflen;
  strcpy(send_name,"send_buffer");
  strcpy(rcv_name,"receive_buffer");
  if (!MA_push_stack(GA[handle].type, buflen, send_name, &send_buf))
      return FALSE;
  if (!MA_get_pointer(send_buf, &snd_ptr)) return FALSE;
  if (!MA_push_stack(GA[handle].type, buflen, rcv_name, &rcv_buf))
      return FALSE;
  if (!MA_get_pointer(rcv_buf, &rcv_ptr)) return FALSE;

  /* loop over dimensions for sequential update using shift algorithm */
  for (idx=0; idx < ndim; idx++) {

    /* Do not bother with update if nwidth is zero */
    if (width[idx] != 0) {

      /* Find parameters for message in negative direction. Start by
       * finding processor to which data will be sent. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = lo_loc[i] - width[i];
          hi_rcv[i] = lo_loc[i] - 1;
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (lo_rcv[i] < 1) {
            slo_rcv[i] = dims[i] - width[i] + 1;
            shi_rcv[i] = dims[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data sent to\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      proc_rem_snd = GA_proclist[0];

      /* Find processor from which data will be recieved */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = hi_loc[i] + 1;
          hi_rcv[i] = hi_loc[i] + width[i];
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] > dims[i]) {
            slo_rcv[i] = 1;
            shi_rcv[i] = width[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data recieved from\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      proc_rem_rcv = GA_proclist[0];

      /* Get actual coordinates of chunk of data that will be sent to
       * remote processor as well as coordinates of the array space that
       * will receive data from remote processor. */
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_snd[i] = width[i];
            phi_snd[i] = 2*width[i] - 1;
            plo_rcv[i] = hi_loc[i] - lo_loc[i] + width[i] + 1;
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + 2*width[i];
          } else {
            plo_snd[i] = width[i];
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = width[i];
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rcv[i] = 0;
          phi_rcv[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_snd[i] = 0;
          phi_snd[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] plo_rcv(%d) %d phi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rcv[i],(int)i+1,(int)phi_rcv[i]);
          fprintf(stderr,"p[%d] plo_snd(%d) %d phi_snd(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_snd[i],(int)i+1,(int)phi_snd[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_snd, &ptr_snd, ld_loc);
      gam_LocationWithGhosts(GAme, handle, plo_rcv, &ptr_rcv, ld_loc);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
        }
      }

      /* Evaluate strides for send and recieve */
      gam_setstride(ndim, size, ld_loc, ld_loc, stride_rcv,
          stride_snd);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rcv, phi_rcv, count);
      gam_CountElems(ndim, plo_snd, phi_snd, &length);
      length *= size;
      count[0] *= size;

      /* Fill send buffer with data. */
      armci_write_strided(ptr_snd, (int)ndim-1, stride_snd, count, snd_ptr);

      /* Send Messages. If processor has odd index in direction idx, it
       * sends message first, if processor has even index it receives
       * message first. Then process is reversed. Also need to account
       * for whether or not there are an odd number of processors along
       * update direction. */

      if (GAme != proc_rem_snd) {
        if (GA[handle].nblock[idx]%2 == 0) {
          if (index[idx]%2 != 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          } else {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
          if (index[idx]%2 != 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          } else {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
        } else {
          pmax = GA[handle].nblock[idx] - 1;
          if (index[idx]%2 != 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          } else if (index[idx] != pmax) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
          if (index[idx]%2 != 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          } else if (index[idx] != 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
          /* make up for odd processor at end of string */
          if (index[idx] == 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
          if (index[idx] == pmax) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
        }
      } else {
        rcv_ptr = snd_ptr;
      }
      msgcnt++;
      /* copy data back into global array */
      armci_read_strided(ptr_rcv, (int)ndim-1, stride_rcv, count, rcv_ptr);

      /* Find parameters for message in positive direction. Start by
       * finding processor to which data will be sent. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = hi_loc[i] + 1;
          hi_rcv[i] = hi_loc[i] + width[i];
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] > dims[i]) {
            slo_rcv[i] = 1;
            shi_rcv[i] = width[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data sent to\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      proc_rem_snd = GA_proclist[0];

      /* Find processor from which data will be recieved */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = lo_loc[i] - width[i];
          hi_rcv[i] = lo_loc[i] - 1;
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] < 1) {
            slo_rcv[i] = dims[i] - width[i] + 1;
            shi_rcv[i] = dims[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data recieved from\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      proc_rem_rcv = GA_proclist[0];
      /* Get actual coordinates of chunk of data that will be sent to
       * remote processor as well as coordinates of the array space that
       * will receive data from remote processor. */
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_snd[i] = hi_loc[i] - lo_loc[i] + 1;
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = 0;
            phi_rcv[i] = width[i] - 1;
          } else {
            plo_snd[i] = width[i];
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = width[i];
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rcv[i] = 0;
          phi_rcv[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_snd[i] = 0;
          phi_snd[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] plo_rcv(%d) %d phi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rcv[i],(int)i+1,(int)phi_rcv[i]);
          fprintf(stderr,"p[%d] plo_snd(%d) %d phi_snd(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_snd[i],(int)i+1,(int)phi_snd[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_snd, &ptr_snd, ld_loc);
      gam_LocationWithGhosts(GAme, handle, plo_rcv, &ptr_rcv, ld_loc);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
        }
      }

      /* Evaluate strides for send and recieve */
      gam_setstride(ndim, size, ld_loc, ld_loc, stride_rcv,
          stride_snd);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rcv, phi_rcv, count);
      gam_CountElems(ndim, plo_snd, phi_snd, &length);
      length *= size;
      count[0] *= size;

      /* Need to reallocate memory if length > buflen */
      /* TO DO */

      /* Fill send buffer with data. */
      armci_write_strided(ptr_snd, (int)ndim-1, stride_snd, count, snd_ptr);

      /* Send Messages. If processor has odd index in direction idx, it
       * sends message first, if processor has even index it receives
       * message first. Then process is reversed. Also need to account
       * for whether or not there are an odd number of processors along
       * update direction. */

      if (GAme != proc_rem_rcv) {
        if (GA[handle].nblock[idx]%2 == 0) {
          if (index[idx]%2 != 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          } else {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
          if (index[idx]%2 != 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          } else {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
        } else {
          pmax = GA[handle].nblock[idx] - 1;
          if (index[idx]%2 != 0) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          } else if (index[idx] != 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
          if (index[idx]%2 != 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          } else if (index[idx] != pmax) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
          /* make up for odd processor at end of string */
          if (index[idx] == pmax) {
            armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
          }
          if (index[idx] == 0) {
            armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
          }
        }
      } else {
        rcv_ptr = snd_ptr;
      }
      /* copy data back into global array */
      armci_read_strided(ptr_rcv, (int)ndim-1, stride_rcv, count, rcv_ptr);
      msgcnt++;
    }
    increment[idx] = 2*width[idx];
  }

  (void)MA_pop_stack(rcv_buf);
  (void)MA_pop_stack(send_buf);
  GA_POP_NAME;
  return TRUE;
}

/* Utility function for ga_update5_ghosts routine */
double waitforflags (int *ptr1, int *ptr2) {
  int i = 1;
  double val;
  while (gai_getval(ptr1) ==  0 || gai_getval(ptr2) == 0) {
    val = exp(-(double)i++);
  }
#if 0
  printf("%d: flags set at %p and %p\n",GAme,ptr1,ptr2); fflush(stdout);
#endif
  return(val);
}

/* Stub in new ARMCI_PutS_flag call until actual implementation is
   available */
int ARMCI_PutS_flag__(
      void* src_ptr,        /* pointer to 1st segment at source */
      int src_stride_arr[], /* array of strides at source */
      void* dst_ptr,        /* pointer to 1st segment at destination */
      int dst_stride_arr[], /* array of strides at destination */
      int count[],          /* number of units at each stride level,
                               count[0] = #bytes */
      int stride_levels,    /* number of stride levels */
      int *flag,            /* pointer to remote flag */
      int val,              /* value to set flag upon completion of
                               data transfer */
      int proc              /* remote process(or) ID */
      )
{
  int bytes;
  /* Put local data on remote processor */
  ARMCI_PutS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
             count, stride_levels, proc);

  /* Send signal to remote processor that data transfer has
   * been completed. */
  bytes = sizeof(int);
  ARMCI_Put(&val, flag, bytes, proc);
  return 1;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM AND PUT CALLS
 *  WITHOUT ANY BARRIERS
\*/
logical FATR ga_update5_ghosts_(Integer *g_a)
{
  Integer idx, ipx, inx, i, np, handle=GA_OFFSET + *g_a, proc_rem;
  Integer size, ndim, nwidth, increment[MAXDIM];
  Integer width[MAXDIM];
  Integer dims[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer plo_loc[MAXDIM], phi_loc[MAXDIM];
  Integer lo_rem[MAXDIM], hi_rem[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer slo_rem[MAXDIM], shi_rem[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  int stride_loc[MAXDIM], stride_rem[MAXDIM],count[MAXDIM];
  int msgcnt, signal, bytes;
  char *ptr_loc, *ptr_rem;

  /* This routine makes use of the shift algorithm to update data in the
   * ghost cells bounding the local block of visible data. The shift
   * algorithm starts by updating the blocks of data along the first
   * dimension by grabbing a block of data that is width[0] deep but
   * otherwise matches the  dimensions of the data residing on the
   * calling processor. The update of the second dimension, however,
   * grabs a block that is width[1] deep in the second dimension but is
   * ldim0 + 2*width[0] in the first dimensions where ldim0 is the
   * size of the visible data along the first dimension. The remaining
   * dimensions are left the same. For the next update, the width of the
   * second dimension is also increased by 2*width[1] and so on. This
   * algorith makes use of the fact that data for the dimensions that
   * have already been updated is available on each processor and can be
   * used in the updates of subsequent dimensions. The total number of
   * separate updates is 2*ndim, an update in the negative and positive
   * directions for each dimension.
   *
   * This operation is implemented using put calls to place the
   * appropriate data on remote processors. To signal the remote
   * processor that it has received the data, a second put call
   * consisting of a single integer is sent after the first put call and
   * used to update a signal buffer on the remote processor. Each
   * processor can determine how much data it has received by checking
   * its signal buffer. 
   *
   * To perform the update, this routine makes use of several copies of
   * indices marking the upper and lower limits of data. Indices
   * beginning with the character "p" are relative indices marking the
   * location of the data set relative to the origin the local patch of
   * the global array, all other indices are in absolute coordinates and
   * mark locations in the total global array. The indices used by this
   * routine are described below.
   *
   *       lo_loc[], hi_loc[]: The lower and upper indices of the visible
   *       block of data held by the calling processor.
   *
   *       lo_rem[], hi_rem[]: The lower and upper indices of the block
   *       of data on a remote processor or processors that is needed to
   *       fill in the calling processors ghost cells. These indices are
   *       NOT corrected for wrap-around (periodic) boundary conditions
   *       so they can be negative or greater than the array dimension
   *       values held in dims[].
   *
   *       slo_rem[], shi_rem[]: Similar to lo_rem[] and hi_rem[], except
   *       that these indices have been corrected for wrap-around
   *       boundary conditions. 
   *
   *       thi_rem[], thi_rem[]: The lower and upper indices of the visible
   *       data on a remote processor.
   *
   *       plo_loc[], phi_loc[]: The indices of the local data patch that
   *       is going to be updated.
   *
   *       plo_rem[], phi_rem[]: The indices of the data patch on the
   *       remote processor that will be used to update the data on the
   *       calling processor. Note that the dimensions of the patches
   *       represented by plo_loc[], plo_rem[] and plo_loc[], phi_loc[]
   *       must be the same.
   */

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) return TRUE;

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* initialize range increments and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    increment[idx] = 0;
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
    if (lo_loc[idx] == 0 && hi_loc[idx] == -1) return FALSE;
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater
     than the corresponding value in width[]. */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("ga_update5_ghosts");

  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);

  /* loop over dimensions for sequential update using shift algorithm */
  msgcnt = 0;
  signal = 1;
  for (idx=0; idx < ndim; idx++) {
    nwidth = width[idx];

    /* Do not bother with update if nwidth is zero */
    if (nwidth != 0) {

      /* Perform update in negative direction. Start by getting rough
         idea of where data needs to go. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = lo_loc[i] - width[i];
          hi_rem[i] = lo_loc[i] - 1;
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (lo_rem[i] < 1) {
            slo_rem[i] = dims[i] - width[i] + 1;
            shi_rem[i] = dims[i];
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
        } else {
          slo_rem[i] = lo_rem[i];
          shi_rem[i] = hi_rem[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rem, shi_rem, *g_a);

      if (DEBUG) {
        fprintf(stderr,"\np[%d] Value of np is %d\n",
            (int)GAme,(int)np);
      }
      /* Get actual coordinates of desired location of remote
         data as well as the actual coordinates of the local chunk
         of data that will be sent to remote processor (these
         coordinates take into account the presence of ghost
         cells). Start by finding out what data is actually held by
         remote processor. */
      proc_rem = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_rem[i] = thi_rem[i] - tlo_rem[i] + width[i] + 1;
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + 2*width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = 2*width[i] - 1;
          } else {
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
          plo_loc[i] = 0;
          phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
          fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
          fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
      gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
          fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_rem[i]);
        }
      }

      /* Evaluate strides on local and remote processors */
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_loc);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rem, phi_rem, count);
      count[0] *= size;

      /* Put local data on remote processor */
#if 0
      ARMCI_PutS(ptr_loc, stride_loc, ptr_rem, stride_rem, count, ndim- 1, proc_rem);
      /* Send signal to remote processor that data transfer has been completed. */
      bytes = sizeof(int);
      ARMCI_Put(&signal, GA_Update_Flags[proc_rem]+msgcnt, bytes, proc_rem);
#else
      ARMCI_PutS_flag(ptr_loc, stride_loc, ptr_rem, stride_rem, count,
          (int)(ndim - 1), GA_Update_Flags[proc_rem]+msgcnt,
          signal, (int)proc_rem);
#endif
      msgcnt++;

      /* Perform update in positive direction. Start by getting rough
         idea of where data needs to go. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rem[i] = hi_loc[i] + 1;
          hi_rem[i] = hi_loc[i] + nwidth;
        } else {
          lo_rem[i] = lo_loc[i];
          hi_rem[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rem(%d) %d hi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rem[i],(int)i+1,(int)hi_rem[i]);
        }
      }

      /* Check to see if boundary is being updated in one patch or two,
         adjust lower boundary accordingly. */
      if (DEBUG) {
        fprintf(stderr,"\n Value of inx is %d\n\n",(int)inx);
      }
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rem[i] > dims[i]) {
            slo_rem[i] = 1;
            shi_rem[i] = width[i];
          } else {
            slo_rem[i] = lo_rem[i];
            shi_rem[i] = hi_rem[i];
          }
        } else {
          slo_rem[i] = lo_rem[i];
          shi_rem[i] = hi_rem[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] slo_rem(%d) %d shi_rem(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rem[i],(int)i+1,(int)shi_rem[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rem, shi_rem, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rem, shi_rem, *g_a);

      if (DEBUG) {
        fprintf(stderr,"\np[%d] Value of np is %d\n",
            (int)GAme,(int)np);
      }
      /* Get actual coordinates of desired chunk of remote
         data as well as the actual coordinates of the local chunk
         of data that will receive the remote data (these
         coordinates take into account the presence of ghost
         cells). Start by finding out what data is actually held by
         remote processor. */
      proc_rem = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Checking second step\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_rem[i] = 0;
            phi_rem[i] = width[i] - 1;
            plo_loc[i] = hi_loc[i] - lo_loc[i] + width[i] - 1;
            phi_loc[i] = hi_loc[i] - lo_loc[i] + 2*width[i] - 1;
          } else {
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
            plo_loc[i] = width[i];
            phi_loc[i] = hi_loc[i] - lo_loc[i] + width[i];
          }
        } else {
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
          plo_loc[i] = 0;
          phi_loc[i] = hi_loc[i] - lo_loc[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] tlo_rem(%d) %d thi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)tlo_rem[i],(int)i+1,(int)thi_rem[i]);
          fprintf(stderr,"p[%d] plo_rem(%d) %d phi_rem(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rem[i],(int)i+1,(int)phi_rem[i]);
          fprintf(stderr,"p[%d] plo_loc(%d) %d phi_loc(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_loc[i],(int)i+1,(int)phi_loc[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
      gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
          fprintf(stderr,"p[%d]   ld_rem[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_rem[i]);
        }
      }

      /* Evaluate strides on local and remote processors */
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_loc);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rem, phi_rem, count);
      count[0] *= size;

      /* Put local data on remote processor */
#if 0
      ARMCI_PutS(ptr_loc, stride_loc, ptr_rem, stride_rem, count, ndim- 1, proc_rem);
      /* Send signal to remote processor that data transfer has been completed. */
      bytes = sizeof(int);
      ARMCI_Put(&signal, GA_Update_Flags[proc_rem]+msgcnt, bytes, proc_rem);

#else
      ARMCI_PutS_flag(ptr_loc, stride_loc, ptr_rem, stride_rem, count,
          (int)(ndim - 1), GA_Update_Flags[proc_rem]+msgcnt,
          signal, (int)proc_rem);
#endif
      msgcnt++;
    }
    /* check to make sure that all messages have been recieved before
       starting update along new dimension */
    waitforflags((GA_Update_Flags[GAme]+msgcnt-2),
        (GA_Update_Flags[GAme]+msgcnt-1));
    /* update increment array */
    increment[idx] = 2*nwidth;
  }

  /* set GA_Update_Flags array to zero for next update operation. */
  for (idx=0; idx < 2*ndim; idx++) {
    GA_Update_Flags[GAme][idx] = 0;
  }

  GA_POP_NAME;
  return TRUE;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY ALONG ONE SIDE OF ARRAY
\*/
logical nga_update_ghost_dir_(Integer *g_a,    /* GA handle */
                                   Integer *pdim,   /* Dimension of update */
                                   Integer *pdir,   /* Direction of update (+/-1) */
                                   logical *pflag)  /* include corner cells */
{
  Integer idx, ipx, inx, np, handle=GA_OFFSET + *g_a, proc_rem;
  Integer ntot, mask[MAXDIM],lmask[MAXDIM];
  Integer size, ndim, i, itmp, idim, idir;
  Integer width[MAXDIM], dims[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer tlo_loc[MAXDIM], thi_loc[MAXDIM];
  Integer plo_loc[MAXDIM], phi_loc[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  logical flag;
  int stride_loc[MAXDIM], stride_rem[MAXDIM],count[MAXDIM];
  char *ptr_loc, *ptr_rem;

  int ijx;
  char *ptr, *iptr;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) 
    return TRUE;
  
  if(local_sync_begin)ga_sync_();
  idim = *pdim;
  idir = *pdir;
  flag = *pflag;

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;
  /* initialize ghost cell widths and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater than
     the corresponding value in width[]). */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1<width[idx]) {
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1<width[idx]) {
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("nga_update_ghost_dir");
  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);
  if (DEBUG) {
    fprintf(stderr,"p[%d] lo(1) %d hi(1) %d\n",(int)GAme,
        (int)lo_loc[0],(int)hi_loc[0]);
    fprintf(stderr,"p[%d] lo(2) %d hi(2) %d\n",(int)GAme,
        (int)lo_loc[1],(int)hi_loc[1]);
  }

  /* evaluate total number of GET operations */
  ntot = 1;
  if (flag) {
    for (idx=0; idx < ndim-1; idx++) ntot *= 3;
  }

  /* Loop over all GET operations. */
  for (ipx=0; ipx < ntot; ipx++) {
    /* Convert ipx to corresponding mask values */
    if (flag) {
      itmp = ipx;
      for (idx = 0; idx < ndim-1; idx++) {
        i = itmp%3;
        lmask[idx] = i-1;
        itmp = (itmp-i)/3;
      }
    } else {
      for (idx = 0; idx < ndim-1; idx++) lmask[idx] = 0;
    }
    inx = 0;
    for (idx = 0; idx < ndim; idx++) {
      if (idx == idim-1) {
        mask[idx] = idir;
      } else {
        mask[idx] = lmask[inx];
        inx++;
      }
    }
    /* Now that mask has been determined, find processor that contains
     * data needed by the corresponding block of ghost cells */
    for (idx = 0; idx < ndim; idx++) {
      if (mask[idx] == 0) {
        tlo_rem[idx] = lo_loc[idx];
        thi_rem[idx] = hi_loc[idx];
      } else if (mask[idx] == -1) {
        if (lo_loc[idx] > 1) {
          tlo_rem[idx] = lo_loc[idx]-width[idx];
          thi_rem[idx] = lo_loc[idx]-1;
        } else {
          tlo_rem[idx] = dims[idx]-width[idx]+1;
          thi_rem[idx] = dims[idx];
        }
      } else if (mask[idx] == 1) {
        if (hi_loc[idx] < dims[idx]) {
          tlo_rem[idx] = hi_loc[idx] + 1;
          thi_rem[idx] = hi_loc[idx] + width[idx];
        } else {
          tlo_rem[idx] = 1;
          thi_rem[idx] = width[idx];
        }
      } else {
        fprintf(stderr,"Illegal mask value found\n");
      }
      if (DEBUG) {
        fprintf(stderr,"\np[%d] ipx = %d tlo_rem[%d] = %d thi_rem[%d] = %d\n",
            (int)GAme,(int)ipx,(int)idx,(int)tlo_rem[idx],(int)idx,
            (int)thi_rem[idx]);
      }
    }
    /* Locate remote processor to which data must be sent */
    if (!nga_locate_region_(g_a, tlo_rem, thi_rem, _ga_map,
       GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
       tlo_rem, thi_rem, *g_a);
    if (np > 1) {
      fprintf(stderr,"More than one remote processor found\n");
    }
    /* Remote processor has been identified, now get ready to get
       data from it. Start by getting distribution on remote
       processor.*/
    proc_rem = GA_proclist[0];
    nga_distribution_(g_a, &proc_rem, tlo_rem, thi_rem);
    for (idx = 0; idx < ndim; idx++) {
      if (mask[idx] == 0) {
        plo_loc[idx] = width[idx];
        phi_loc[idx] = hi_loc[idx]-lo_loc[idx]+width[idx];
        plo_rem[idx] = plo_loc[idx];
        phi_rem[idx] = phi_loc[idx];
      } else if (mask[idx] == -1) {
        plo_loc[idx] = 0;
        phi_loc[idx] = width[idx]-1;
        plo_rem[idx] = thi_rem[idx]-tlo_rem[idx]+1;
        phi_rem[idx] = thi_rem[idx]-tlo_rem[idx]+width[idx];
      } else if (mask[idx] == 1) {
        plo_loc[idx] = hi_loc[idx]-lo_loc[idx]+width[idx]+1;
        phi_loc[idx] = hi_loc[idx]-lo_loc[idx]+2*width[idx];
        plo_rem[idx] = width[idx];
        phi_rem[idx] = 2*width[idx]-1;
      }
      if (DEBUG) {
        fprintf(stderr,"\np[%d] plo_loc[%d] = %d phi_loc[%d] = %d\n",
            (int)GAme,(int)idx,(int)plo_loc[idx],(int)idx,
            (int)phi_loc[idx]);
        fprintf(stderr,"p[%d] plo_rem[%d] = %d phi_rem[%d] = %d\n",
            (int)GAme,(int)idx,(int)plo_rem[idx],(int)idx,
            (int)phi_rem[idx]);
      }
    }
    /* Get pointer to local data buffer and remote data
       buffer as well as lists of leading dimenstions */
    gam_LocationWithGhosts(GAme, handle, plo_loc, &ptr_loc, ld_loc);
    gam_LocationWithGhosts(proc_rem, handle, plo_rem, &ptr_rem, ld_rem);

    /* Evaluate strides on local and remote processors */
    gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
                  stride_loc);

    /* Compute the number of elements in each dimension and store
       result in count. Scale the first element in count by the
       element size. */
    gam_ComputeCount(ndim, plo_loc, phi_loc, count);
    count[0] *= size;
 
    /* get data from remote processor */
    ARMCI_GetS(ptr_rem, stride_rem, ptr_loc, stride_loc, count,
          (int)(ndim - 1), (int)proc_rem);
  }

  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
  return TRUE;
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM
\*/
void ga_update_ghosts_(Integer *g_a)
{
  /* Wrapper program for ghost cell update operations. If optimized
     update operation fails then use slow but robust version of
     update operation */
   int local_sync_begin,local_sync_end;

   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(local_sync_begin)ga_sync_();

#ifdef CRAY_T3D
   if (!ga_update5_ghosts_(g_a)) {
#else
   if (!ga_update4_ghosts_(g_a)) {
#endif
     ga_update1_ghosts_(g_a);
   }

   if(local_sync_end)ga_sync_();
}

/* Utility function for ga_update6_ghosts routine */
double waitformixedflags (int flag1, int flag2, int *ptr1, int *ptr2) {
  int i = 1;
  double val;
  while ((flag1 && gai_getval(ptr1) ==  0) ||
         (flag2 && gai_getval(ptr2) == 0)) {
    val = exp(-(double)i++);
  }
  return(val);
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY USING SHIFT ALGORITHM AND
 *  MESSAGE PASSING
\*/
logical FATR ga_update6_ghosts_(Integer *g_a)
{
  Integer idx, ipx, idir, i, np, handle=GA_OFFSET + *g_a;
  Integer size, buflen, buftot, bufsize, ndim, increment[MAXDIM];
  Integer send_buf = INVALID_MA_HANDLE, rcv_buf = INVALID_MA_HANDLE;
  Integer proc_rem_snd, proc_rem_rcv, pmax;
  Integer msgcnt, length, msglen;
  Integer width[MAXDIM], dims[MAXDIM], index[MAXDIM];
  Integer lo_loc[MAXDIM], hi_loc[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer plo_snd[MAXDIM], phi_snd[MAXDIM];
  Integer lo_rcv[MAXDIM], hi_rcv[MAXDIM];
  Integer slo_rcv[MAXDIM], shi_rcv[MAXDIM];
  Integer plo_rcv[MAXDIM], phi_rcv[MAXDIM];
  Integer ld_loc[MAXDIM], ld_rem[MAXDIM];
  int stride_snd[MAXDIM], stride_rcv[MAXDIM],count[MAXDIM];
  int stride_loc[MAXDIM], stride_rem[MAXDIM];
  int signal, bytes, flag1, flag2, sprocflag, rprocflag;
  char *ptr_snd, *ptr_rcv;
  char *ptr_loc, *ptr_rem;
  char send_name[32], rcv_name[32];
  void *snd_ptr, *rcv_ptr;

  /* This routine makes use of the shift algorithm to update data in the
   * ghost cells bounding the local block of visible data. The shift
   * algorithm starts by updating the blocks of data along the first
   * dimension by grabbing a block of data that is width[0] deep but
   * otherwise matches the  dimensions of the data residing on the
   * calling processor. The update of the second dimension, however,
   * grabs a block that is width[1] deep in the second dimension but is
   * ldim0 + 2*width[0] in the first dimensions where ldim0 is the
   * size of the visible data along the first dimension. The remaining
   * dimensions are left the same. For the next update, the width of the
   * second dimension is also increased by 2*width[1] and so on. This
   * algorith makes use of the fact that data for the dimensions that
   * have already been updated is available on each processor and can be
   * used in the updates of subsequent dimensions. The total number of
   * separate updates is 2*ndim, an update in the negative and positive
   * directions for each dimension.
   *
   * This implementation make use of a combination of explicit message
   * passing between processors on different nodes and shared memory
   * copies with and additional flag between processors on the same node
   * to perform the update. Separate message types for the messages and
   * the use of the additional flag are for the updates in each
   * coordinate direction are used to maintain synchronization locally
   * and to guarantee that the data is present before the updates in a
   * new coordinate direction take place.
   *
   * To perform the update, this routine makes use of several copies of
   * indices marking the upper and lower limits of data. Indices
   * beginning with the character "p" are relative indices marking the
   * location of the data set relative to the origin the local patch of
   * the global array, all other indices are in absolute coordinates and
   * mark locations in the total global array. The indices used by this
   * routine are described below.
   *
   *       lo_loc[], hi_loc[]: The lower and upper indices of the visible
   *       block of data held by the calling processor.
   *
   *       lo_rcv[], hi_rcv[]: The lower and upper indices of the blocks
   *       of data that will be either sent to or received from a remote
   *       processor. These indices are NOT corrected for wrap-around
   *       (periodic) boundary conditions so they can be negative or greater
   *       than the array dimension values held in dims[].
   *
   *       slo_rcv[], shi_rcv[]: Similar to lo_rcv[] and hi_rcv[], except
   *       that these indices have been corrected for wrap-around
   *       boundary conditions.
   *
   *       plo_rcv[], phi_rcv[]: The local indices of the local data patch
   *       that receive that message from the remote processor.
   *
   *       plo_snd[], phi_snd[]: The local indices of the data patch
   *       that will be sent to the remote processor. Note that the
   *       dimensions of the patches represented by plo_rec[], plo_rec[] and
   *       plo_snd[], phi_snd[] must be the same.
   *
   *       tlo_rem[], thi_rem[]: The indices of the locally held visible
   *       portion of the global array on the remote processor that will be
   *       receiving the data using a shared memory copy.
   *
   *       plo_rem[], phi_rem[]: The local indices of the coordinate patch
   *       that will be put on the remote processor using a shared memory
   *       copy.
   */

  /* if global array has no ghost cells, just return */
  if (!ga_has_ghosts_(g_a)) return TRUE;

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* initialize range increments and get array dimensions */
  for (idx=0; idx < ndim; idx++) {
    increment[idx] = 0;
    width[idx] = GA[handle].width[idx];
    dims[idx] = GA[handle].dims[idx];
  }

  /* Check to make sure that global array is well-behaved (all processors
     have data and the width of the data in each dimension is greater
     than the corresponding value in width[]. */
  ipx = 0;
  for (idx = 0; idx < ndim; idx++) {
    for (np = 0; np < GA[handle].nblock[idx]; np++) {
      if (np < GA[handle].nblock[idx] - 1) {
        if (GA[handle].mapc[ipx+1]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1 < width[idx]) {
          return FALSE;
        }
      }
      ipx++;
    }
  }

  GA_PUSH_NAME("ga_update6_ghosts");
  msgcnt = 0;

  /* Get pointer to local memory */
  ptr_loc = GA[handle].ptr[GAme];
  /* obtain range of data that is held by local processor */
  nga_distribution_(g_a,&GAme,lo_loc,hi_loc);
  /* Get indices of processor in virtual grid */
  nga_proc_topology_(g_a, &GAme, index);

  /* Try to find maximum size of message that will be sent during
   * update operations and use this to allocate memory for message
   * passing buffers. */
  buftot = 1;
  for (i=0; i<ndim; i++) {
    buftot *= (hi_loc[i]-lo_loc[i] + 1 + 2*width[i]);
  }
  buflen = 1;
  for (i = 0; i < ndim; i++) {
    idir =  hi_loc[i] - lo_loc[i] + 1;
    if (buflen < (buftot/(idir + 2*width[i]))*width[i]) {
      buflen = (buftot/(idir + 2*width[i]))*width[i];
    }
  }
  bufsize = size*buflen;
  strcpy(send_name,"send_buffer");
  strcpy(rcv_name,"receive_buffer");
  if (!MA_push_stack(GA[handle].type, buflen, send_name, &send_buf))
      return FALSE;
  if (!MA_get_pointer(send_buf, &snd_ptr)) return FALSE;
  if (!MA_push_stack(GA[handle].type, buflen, rcv_name, &rcv_buf))
      return FALSE;
  if (!MA_get_pointer(rcv_buf, &rcv_ptr)) return FALSE;

  /* loop over dimensions for sequential update using shift algorithm */
  msgcnt = 0;
  signal = 1;
  for (idx=0; idx < ndim; idx++) {

    /* Do not bother with update if nwidth is zero */
    if (width[idx] != 0) {

      /* Find parameters for message in negative direction. Start by
       * finding processor to which data will be sent. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = lo_loc[i] - width[i];
          hi_rcv[i] = lo_loc[i] - 1;
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (lo_rcv[i] < 1) {
            slo_rcv[i] = dims[i] - width[i] + 1;
            shi_rcv[i] = dims[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data sent to\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      /* find out if this processor is on the same node */
      rprocflag = ARMCI_Same_node(GA_proclist[0]);
      proc_rem_snd = GA_proclist[0];

      /* Find processor from which data will be received */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in negative direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = hi_loc[i] + 1;
          hi_rcv[i] = hi_loc[i] + width[i];
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] > dims[i]) {
            slo_rcv[i] = 1;
            shi_rcv[i] = width[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data recieved from\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      sprocflag = ARMCI_Same_node(GA_proclist[0]);
      proc_rem_rcv = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem_rcv, tlo_rem, thi_rem);

      /* Get actual coordinates of chunk of data that will be sent to
       * remote processor as well as coordinates of the array space that
       * will receive data from remote processor. */
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_snd[i] = width[i];
            phi_snd[i] = 2*width[i] - 1;
            plo_rcv[i] = hi_loc[i] - lo_loc[i] + width[i] + 1;
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + 2*width[i];
            plo_rem[i] = thi_rem[i] - tlo_rem[i] + width[i] + 1;
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + 2*width[i];
          } else {
            plo_snd[i] = width[i];
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = width[i];
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
          }
        } else {
          plo_snd[i] = 0;
          phi_snd[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_rcv[i] = 0;
          phi_rcv[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] plo_rcv(%d) %d phi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rcv[i],(int)i+1,(int)phi_rcv[i]);
          fprintf(stderr,"p[%d] plo_snd(%d) %d phi_snd(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_snd[i],(int)i+1,(int)phi_snd[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_snd, &ptr_snd, ld_loc);
      gam_LocationWithGhosts(GAme, handle, plo_rcv, &ptr_rcv, ld_loc);
      gam_LocationWithGhosts(proc_rem_snd, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
        }
      }

      /* Evaluate strides for send and receive */
      gam_setstride(ndim, size, ld_loc, ld_loc, stride_rcv,
          stride_snd);
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_snd);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rcv, phi_rcv, count);
      gam_CountElems(ndim, plo_snd, phi_snd, &length);
      length *= size;
      count[0] *= size;

      /* If we are sending data to another node, then use message passing */
      if (!rprocflag) {
        /* Fill send buffer with data. */
        armci_write_strided(ptr_snd, (int)ndim-1, stride_snd, count, snd_ptr);
      }

      /* Send Messages. If processor has odd index in direction idx, it
       * sends message first, if processor has even index it receives
       * message first. Then process is reversed. Also need to account
       * for whether or not there are an odd number of processors along
       * update direction. */

      if (GA[handle].nblock[idx]%2 == 0) {
        if (index[idx]%2 != 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        } else if (index[idx]%2 == 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } 
        if (rprocflag) {
          ARMCI_PutS_flag(ptr_snd, stride_snd, ptr_rem, stride_rem, count,
                          (int)(ndim-1), GA_Update_Flags[proc_rem_snd]+msgcnt,
                          signal, (int)proc_rem_snd);
        }
        if (index[idx]%2 != 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } else if (index[idx]%2 == 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
      } else {
        /* account for wrap-around boundary condition, if necessary */
        pmax = GA[handle].nblock[idx] - 1;
        if (index[idx]%2 != 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        } else if (index[idx]%2 == 0 && index[idx] != pmax && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        }
        if (rprocflag) {
          ARMCI_PutS_flag(ptr_snd, stride_snd, ptr_rem, stride_rem, count,
                          (int)(ndim-1), GA_Update_Flags[proc_rem_snd]+msgcnt,
                          signal, (int)proc_rem_snd);
        }
        if (index[idx]%2 != 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } else if (index[idx]%2 == 0 && index[idx] != 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
        /* make up for odd processor at end of string */
        if (index[idx] == 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
        if (index[idx] == pmax && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        }
      }
      if (sprocflag) {
        flag1 = 1;
      } else {
        flag1 = 0;
      }
      msgcnt++;
      /* copy data back into global array */
      if (!sprocflag) {
        armci_read_strided(ptr_rcv, (int)ndim-1, stride_rcv, count, rcv_ptr);
      }

      /* Find parameters for message in positive direction. Start by
       * finding processor to which data will be sent. */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = hi_loc[i] + 1;
          hi_rcv[i] = hi_loc[i] + width[i];
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] > dims[i]) {
            slo_rcv[i] = 1;
            shi_rcv[i] = width[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data sent to\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      rprocflag = ARMCI_Same_node(GA_proclist[0]);
      proc_rem_snd = GA_proclist[0];

      /* Find processor from which data will be recieved */
      if (DEBUG) {
        fprintf(stderr,"\np[%d] Update in positive direction\n\n",(int)GAme);
      }
      for (i = 0; i < ndim; i++) {
        if (i == idx) {
          lo_rcv[i] = lo_loc[i] - width[i];
          hi_rcv[i] = lo_loc[i] - 1;
        } else {
          lo_rcv[i] = lo_loc[i];
          hi_rcv[i] = hi_loc[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] lo_rcv(%d) %d hi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)lo_rcv[i],(int)i+1,(int)hi_rcv[i]);
        }
      }

      /* Account for boundaries, if necessary. */
      for (i=0; i<ndim; i++) {
        if (i == idx) {
          if (hi_rcv[i] < 1) {
            slo_rcv[i] = dims[i] - width[i] + 1;
            shi_rcv[i] = dims[i];
          } else {
            slo_rcv[i] = lo_rcv[i];
            shi_rcv[i] = hi_rcv[i];
          }
        } else {
          slo_rcv[i] = lo_rcv[i];
          shi_rcv[i] = hi_rcv[i];
        }
        if (DEBUG) {
          fprintf(stderr,"p[%d] Data recieved from\n slo_rcv(%d) %d shi_rcv(%d) %d"
              "  i=%d idx=%d\n",(int)GAme,(int)i+1,
              (int)slo_rcv[i],(int)i+1,(int)shi_rcv[i],(int)i,
              (int)idx);
        }
      }
      /* locate processor with this data */
      if (!nga_locate_region_(g_a, slo_rcv, shi_rcv, _ga_map,
          GA_proclist, &np)) ga_RegionError(ga_ndim_(g_a),
          slo_rcv, shi_rcv, *g_a);
      sprocflag = ARMCI_Same_node(GA_proclist[0]);
      proc_rem_rcv = GA_proclist[0];
      nga_distribution_(g_a, &proc_rem_rcv, tlo_rem, thi_rem);

      /* Get actual coordinates of chunk of data that will be sent to
       * remote processor as well as coordinates of the array space that
       * will receive data from remote processor. */
      for (i = 0; i < ndim; i++) {
        if (increment[i] == 0) {
          if (i == idx) {
            plo_snd[i] = hi_loc[i] - lo_loc[i] + 1;
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = 0;
            phi_rcv[i] = width[i] - 1;
            plo_rem[i] = 0;
            phi_rem[i] = width[i] - 1;
          } else {
            plo_snd[i] = width[i];
            phi_snd[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rcv[i] = width[i];
            phi_rcv[i] = hi_loc[i] - lo_loc[i] + width[i];
            plo_rem[i] = width[i];
            phi_rem[i] = thi_rem[i] - tlo_rem[i] + width[i];
          }
        } else {
          plo_snd[i] = 0;
          phi_snd[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_rcv[i] = 0;
          phi_rcv[i] = hi_loc[i] - lo_loc[i] + increment[i];
          plo_rem[i] = 0;
          phi_rem[i] = thi_rem[i] - tlo_rem[i] + increment[i];
        }
        if (DEBUG) {
          fprintf(stderr,"\n");
          fprintf(stderr,"p[%d] plo_rcv(%d) %d phi_rcv(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_rcv[i],(int)i+1,(int)phi_rcv[i]);
          fprintf(stderr,"p[%d] plo_snd(%d) %d phi_snd(%d) %d\n",
              (int)GAme,(int)i+1,(int)plo_snd[i],(int)i+1,(int)phi_snd[i]);
        }
      }

      /* Get pointer to local data buffer and remote data
         buffer as well as lists of leading dimenstions */
      gam_LocationWithGhosts(GAme, handle, plo_snd, &ptr_snd, ld_loc);
      gam_LocationWithGhosts(GAme, handle, plo_rcv, &ptr_rcv, ld_loc);
      gam_LocationWithGhosts(proc_rem_snd, handle, plo_rem, &ptr_rem, ld_rem);
      if (DEBUG) {
        for (i=0; i<ndim-1; i++) {
          fprintf(stderr,"\np[%d]   ld_loc[%d] = %d\n",(int)GAme,(int)i,
              (int)ld_loc[i]);
        }
      }

      /* Evaluate strides for send and recieve */
      gam_setstride(ndim, size, ld_loc, ld_loc, stride_rcv,
          stride_snd);
      gam_setstride(ndim, size, ld_loc, ld_rem, stride_rem,
          stride_snd);

      /* Compute the number of elements in each dimension and store
         result in count. Scale the first element in count by the
         element size. */
      gam_ComputeCount(ndim, plo_rcv, phi_rcv, count);
      gam_CountElems(ndim, plo_snd, phi_snd, &length);
      length *= size;
      count[0] *= size;

      /* if we are sending data to another node, use message passing */
      if (!rprocflag) {
        /* Fill send buffer with data. */
        armci_write_strided(ptr_snd, (int)ndim-1, stride_snd, count, snd_ptr);
      }

      /* Send Messages. If processor has odd index in direction idx, it
       * sends message first, if processor has even index it receives
       * message first. Then process is reversed. Also need to account
       * for whether or not there are an odd number of processors along
       * update direction. */

      if (GA[handle].nblock[idx]%2 == 0) {
        if (index[idx]%2 != 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        } else if (index[idx]%2 == 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } 
        if (rprocflag) {
          ARMCI_PutS_flag(ptr_snd, stride_snd, ptr_rem, stride_rem, count,
                          (int)(ndim-1), GA_Update_Flags[proc_rem_snd]+msgcnt,
                          signal, (int)proc_rem_snd);
        }
        if (index[idx]%2 != 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } else if (index[idx]%2 == 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
      } else {
        /* account for wrap-around boundary condition, if necessary */
        pmax = GA[handle].nblock[idx] - 1;
        if (index[idx]%2 != 0 && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        } else if (index[idx]%2 == 0 && index[idx] != 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        }
        if (rprocflag) {
          ARMCI_PutS_flag(ptr_snd, stride_snd, ptr_rem, stride_rem, count,
                          (int)(ndim-1), GA_Update_Flags[proc_rem_snd]+msgcnt,
                          signal, (int)proc_rem_snd);
        }
        if (index[idx]%2 != 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        } else if (index[idx]%2 == 0 && index[idx] != pmax && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
        /* make up for odd processor at end of string */
        if (index[idx] == pmax && !rprocflag) {
          armci_msg_snd(msgcnt, snd_ptr, length, proc_rem_snd);
        }
        if (index[idx] == 0 && !sprocflag) {
          armci_msg_rcv(msgcnt, rcv_ptr, bufsize, &msglen, proc_rem_rcv);
        }
      }
      /* copy data back into global array */
      if (!sprocflag) {
        armci_read_strided(ptr_rcv, (int)ndim-1, stride_rcv, count, rcv_ptr);
      }
      if (sprocflag) {
        flag2 = 1;
      } else {
        flag2 = 0;
      }
      msgcnt++;
    }
    /* check to make sure any outstanding puts have showed up */
    waitformixedflags(flag1, flag2, GA_Update_Flags[GAme]+msgcnt-2,
                      GA_Update_Flags[GAme]+msgcnt-1);
    /* update increment array */
    increment[idx] = 2*width[idx];
  }

  (void)MA_pop_stack(rcv_buf);
  (void)MA_pop_stack(send_buf);
  /* set update flags to zero for next operation */
  for (idx=0; idx < 2*ndim; idx++) {
    GA_Update_Flags[GAme][idx] = 0;
  }

  GA_POP_NAME;
  return TRUE;
}

void FATR ga_ghost_barrier_()
{
#ifdef LAPI
  int signal = 1, n = 1;
  int *ptr;
  ptr = &signal;
  armci_msg_igop(ptr,n,"+");
#else
  armci_msg_barrier();
#endif
}
