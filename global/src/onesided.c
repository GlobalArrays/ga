/* $Id: onesided.c,v 1.69 2006-09-22 22:28:09 d3g293 Exp $ */
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

#ifdef GA_USE_VAMPIR
#include "ga_vampir.h"
#endif
#ifdef GA_PROFILE
#include "ga_profile.h"
#endif

int    ProcListPerm[MAX_NPROC];            /* permuted list of processes */


/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

char *fence_array;
static int GA_fence_set=0;
Integer *_ga_map;       /* used in get/put/acc */

extern void ga_sort_scat(Integer*,Void*,Integer*,Integer*,Integer*, Integer);
extern void ga_sort_gath_(Integer*, Integer*, Integer*, Integer*);
extern void ga_msg_sync_();
extern void ga_msg_pgroup_sync_(Integer *grp_id);

extern void armci_read_strided(void*, int, int*, int*, char*);
extern void armci_write_strided(void*, int, int*, int*, char*);
extern armci_hdl_t* get_armci_nbhandle(Integer *);

/***************************************************************************/

/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void FATR ga_pgroup_sync_(Integer *grp_id)
{
#ifdef CHECK_MA
    Integer status;
#endif
#ifdef GA_USE_VAMPIR
    vampir_begin(GA_PGROUP_SYNC,__FILE__,__LINE__);
#endif
 
    if (*grp_id > 0) {
#   ifdef MPI
       /* fence on all processes in this group */
       { int p; for(p=0;p<PGRP_LIST[*grp_id].map_nproc;p++)
          ARMCI_Fence(ARMCI_Absolute_id(&PGRP_LIST[*grp_id].group, p)); }
       ga_msg_pgroup_sync_(grp_id);
       if(GA_fence_set)bzero(fence_array,(int)GAnproc);
       GA_fence_set=0;
#   else
       ga_error("ga_pgroup_sync_(): MPI not defined. ga_msg_pgroup_sync_()  can be called only if GA is built with MPI", 0);
#   endif
    } else {
       ARMCI_AllFence();
       ga_msg_sync_();
       if(GA_fence_set)bzero(fence_array,(int)GAnproc);
       GA_fence_set=0;
    }
#ifdef CHECK_MA
    status = MA_verify_allocator_stuff();
#endif
#ifdef GA_USE_VAMPIR
    vampir_end(GA_PGROUP_SYNC,__FILE__,__LINE__);
#endif
}

/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void FATR ga_sync_()
{
#ifdef CHECK_MA
Integer status;
#endif
#ifdef GA_USE_VAMPIR
       vampir_begin(GA_SYNC,__FILE__,__LINE__);
#endif
       
       if (GA_Default_Proc_Group == -1) {
	  ARMCI_AllFence();
	  ga_msg_sync_();
	  if(GA_fence_set)bzero(fence_array,(int)GAnproc);
	  GA_fence_set=0;
       } else {
         Integer grp_id = (Integer)GA_Default_Proc_Group;
         ga_pgroup_sync_(&grp_id);
       }
#ifdef CHECK_MA
       status = MA_verify_allocator_stuff();
#endif
#ifdef GA_USE_VAMPIR
       vampir_end(GA_SYNC,__FILE__,__LINE__);
#endif
}


/*\ wait until requests intiated by calling process are completed
\*/
void FATR ga_fence_()
{
    int proc;
#ifdef GA_USE_VAMPIR
    vampir_begin(GA_FENCE,__FILE__,__LINE__);
#endif
    if(GA_fence_set<1)ga_error("ga_fence: fence not initialized",0);
    GA_fence_set--;
    for(proc=0;proc<GAnproc;proc++)if(fence_array[proc])ARMCI_Fence(proc);
    bzero(fence_array,(int)GAnproc);
#ifdef GA_USE_VAMPIR
    vampir_end(GA_FENCE,__FILE__,__LINE__);
#endif
}

/*\ initialize tracing of request completion
\*/
void FATR ga_init_fence_()
{
#ifdef GA_USE_VAMPIR
    vampir_begin(GA_INIT_FENCE,__FILE__,__LINE__);
#endif
    GA_fence_set++;
#ifdef GA_USE_VAMPIR
    vampir_end(GA_INIT_FENCE,__FILE__,__LINE__);
#endif
}

void gai_init_onesided()
{
    _ga_map = (Integer*)malloc((size_t)(GAnproc*2*MAXDIM +1)*sizeof(Integer));
    if(!_ga_map) ga_error("ga_init:malloc failed (_ga_map)",0);
    fence_array = calloc((size_t)GAnproc,1);
    if(!fence_array) ga_error("ga_init:calloc failed",0);
}


/*\ prepare permuted list of processes for remote ops
\*/
#define gaPermuteProcList(nproc)\
{\
  if((nproc) ==1) ProcListPerm[0]=0; \
  else{\
    int _i, iswap, temp;\
    if((nproc) > GAnproc) ga_error("permute_proc: error ", (nproc));\
\
    /* every process generates different random sequence */\
    (void)srand((unsigned)GAme); \
\
    /* initialize list */\
    for(_i=0; _i< (nproc); _i++) ProcListPerm[_i]=_i;\
\
    /* list permutation generated by random swapping */\
    for(_i=0; _i< (nproc); _i++){ \
      iswap = (int)(rand() % (nproc));  \
      temp = ProcListPerm[iswap]; \
      ProcListPerm[iswap] = ProcListPerm[_i]; \
      ProcListPerm[_i] = temp; \
    } \
  }\
}
     




/*\ internal malloc that bypasses MA and uses internal buf when possible
\*/
#define MBUFLEN 256
#define MBUF_LEN MBUFLEN+2
static double ga_int_malloc_buf[MBUF_LEN];
static int mbuf_used=0;
#define MBUF_GUARD -1998.1998
void *gai_malloc(int bytes)
{
    void *ptr;
    if(!mbuf_used && bytes <= MBUF_LEN){
       if(DEBUG){
          ga_int_malloc_buf[0]= MBUF_GUARD;
          ga_int_malloc_buf[MBUFLEN]= MBUF_GUARD;
       }
       ptr = ga_int_malloc_buf+1;
       mbuf_used++;
    }else{
        Integer elems = (bytes+sizeof(double)-1)/sizeof(double)+1;
	ptr=ga_malloc(elems, MT_DBL, "GA malloc temp");
	/* *((Integer*)ptr)= handle;
	   ptr = ((double*)ptr)+ 1;*/ /*needs sizeof(double)>=sizeof(Integer)*/
    }
    return ptr;
}

void gai_free(void *ptr)
{
    if(ptr == (ga_int_malloc_buf+1)){
        if(DEBUG){
          assert(ga_int_malloc_buf[0]== MBUF_GUARD);
          assert(ga_int_malloc_buf[MBUFLEN]== MBUF_GUARD);
          assert(mbuf_used ==1);
        }
        mbuf_used =0;
    }else{
        /* Integer handle= *( (Integer*) (-1 + (double*)ptr)); */
      ga_free(ptr);
    }
}

        


#define gaShmemLocation(proc, g_a, _i, _j, ptr_loc, _pld)                      \
{                                                                              \
Integer _ilo, _ihi, _jlo, _jhi, offset, proc_place, g_handle=(g_a)+GA_OFFSET;  \
Integer _lo[2], _hi[2], _p_handle;                                             \
Integer _iw = GA[g_handle].width[0];                                           \
Integer _jw = GA[g_handle].width[1];                                           \
                                                                               \
      ga_ownsM(g_handle, (proc),_lo,_hi);                                      \
      _p_handle = GA[g_handle].p_handle;                                       \
      if (_p_handle != 0) {                                                    \
        proc_place =  proc;                                                    \
      } else {                                                                 \
        proc_place = PGRP_LIST[_p_handle].inv_map_proc_list[proc];             \
      }                                                                        \
      _ilo = _lo[0]; _ihi=_hi[0];                                              \
      _jlo = _lo[1]; _jhi=_hi[1];                                              \
                                                                               \
      if((_i)<_ilo || (_i)>_ihi || (_j)<_jlo || (_j)>_jhi){                    \
       sprintf(err_string,"%s:p=%ld invalid i/j (%ld,%ld)><(%ld:%ld,%ld:%ld)", \
                 "gaShmemLocation", (long)proc, (long)(_i),(long)(_j),         \
                             (long)_ilo, (long)_ihi, (long)_jlo, (long)_jhi);  \
          ga_error(err_string, g_a );                                          \
      }                                                                        \
      offset = ((_i)-_ilo+_iw) + (_ihi-_ilo+1+2*_iw)*((_j)-_jlo+_jw);          \
                                                                               \
      /* find location of the proc in current cluster pointer array */         \
      *(ptr_loc) = GA[g_handle].ptr[proc_place] +                              \
                   offset*GAsizeofM(GA[g_handle].type);                        \
      *(_pld) = _ihi-_ilo+1+2*_iw;                                             \
}

/*\ Return pointer (ptr_loc) to location in memory of element with subscripts
 *  (subscript). Also return physical dimensions of array in memory in ld.
\*/
#define gam_Location(proc, g_handle,  subscript, ptr_loc, ld)                  \
{                                                                              \
Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;               \
Integer _lo[MAXDIM], _hi[MAXDIM], _pinv, _p_handle;                            \
                                                                               \
      ga_ownsM(g_handle, proc, _lo, _hi);                                      \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);               \
      if(_last==0) ld[0]=_hi[0]- _lo[0]+1+2*(Integer)GA[g_handle].width[0];    \
      __CRAYX1_PRAGMA("_CRI shortloop");                                       \
      for(_d=0; _d < _last; _d++)            {                                 \
          _w = (Integer)GA[g_handle].width[_d];                                \
          _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                     \
          ld[_d] = _hi[_d] - _lo[_d] + 1 + 2*_w;                               \
          _factor *= ld[_d];                                                   \
      }                                                                        \
      _offset += (subscript[_last]-_lo[_last]                                  \
               + (Integer)GA[g_handle].width[_last])                           \
               * _factor;                                                      \
      _p_handle = GA[g_handle].p_handle;                                       \
      if (_p_handle != 0) {                                                    \
        _pinv = proc;                                                          \
      } else {                                                                 \
        _pinv = PGRP_LIST[_p_handle].inv_map_proc_list[proc];                  \
      }                                                                        \
      *(ptr_loc) =  GA[g_handle].ptr[_pinv]+_offset*GA[g_handle].elemsize;     \
}


#define gam_GetRangeFromMap(p, ndim, plo, phi){\
Integer   _mloc = p* ndim *2;\
          *plo  = (Integer*)_ga_map + _mloc;\
          *phi  = *plo + ndim;\
}

/* compute index of point subscripted by plo relative to point
   subscripted by lo, for a block with dimensions dims */
#define gam_ComputePatchIndex(ndim, lo, plo, dims, pidx){\
Integer _d, _factor;\
          *pidx = plo[0] -lo[0];\
          __CRAYX1_PRAGMA("_CRI shortloop");\
          for(_d= 0,_factor=1; _d< ndim -1; _d++){\
             _factor *= (dims[_d]);\
             *pidx += _factor * (plo[_d+1]-lo[_d+1]);\
          }\
}

#define gam_GetBlockPatch(plo,phi,lo,hi,blo,bhi,ndim) {                        \
  Integer _d;                                                                  \
  for (_d=0; _d<ndim; _d++) {                                                  \
    if (lo[_d] <= phi[_d] && lo[_d] >= plo[_d]) blo[_d] = lo[_d];              \
    else blo[_d] = plo[_d];                                                    \
    if (hi[_d] <= phi[_d] && hi[_d] >= plo[_d]) bhi[_d] = hi[_d];              \
    else bhi[_d] = phi[_d];                                                    \
  }                                                                            \
}

/*\ A routine to wait for a non-blocking call to complete
\*/
void FATR nga_nbwait_(Integer *nbhandle) 
{
    nga_wait_internal((Integer *)nbhandle);
} 

/*\ A routine to wait for a non-blocking call to complete
\*/
void FATR ga_nbwait_(Integer *nbhandle) 
{
    nga_wait_internal((Integer *)nbhandle);
} 

/*\ A common routine called by both non-blocking and blocking GA put calls.
\*/
#ifdef __crayx1
#pragma _CRI inline nga_locate_region_
#endif
void nga_put_common(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   Integer *nbhandle)
{
  Integer  p, np, handle=GA_OFFSET + *g_a;
  Integer  idx, elems, size, p_handle, ga_nbhandle;
  int proc, ndim, loop, cond, counter=0;
  int num_loops=2; /* 1st loop for remote procs; 2nd loop for local procs */
  Integer use_blocks;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_NBPUT,__FILE__,__LINE__);
#endif
  GA_PUSH_NAME("nga_put_common");

  ga_check_handleM(g_a, "nga_put_common");

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;
  use_blocks = GA[handle].block_flag;
  p_handle = GA[handle].p_handle;

  if (!use_blocks) {
    if(!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np ))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

#ifndef NO_GA_STATS
    gam_CountElems(ndim, lo, hi, &elems);
    GAbytes.puttot += (double)size*elems;
    GAstat.numput++;
    GAstat.numput_procs += np;
#endif

    if(nbhandle)ga_init_nbhandle(nbhandle);
#ifndef __crayx1
    else ga_init_nbhandle(&ga_nbhandle);
#endif

#ifdef GA_PROFILE
    ga_profile_start((int)handle, (long)size*elems, ndim, lo, hi, 
        GA_PROFILE_PUT);
#endif

    gaPermuteProcList(np);

#ifndef __crayx1
    for(loop=0; loop<num_loops; loop++) {
      __CRAYX1_PRAGMA("_CRI novector");
#endif
      for(idx=0; idx<np; idx++){
        Integer ldrem[MAXDIM];
        int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
        Integer idx_buf, *plo, *phi;
        char *pbuf, *prem;

        p = (Integer)ProcListPerm[idx];
        proc = (int)GA_proclist[p];
        if (p_handle >= 0) {
          proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
        }
#ifdef PERMUTE_PIDS
        if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

        /* check if it is local to SMP */

#ifndef __crayx1
        cond = armci_domain_same_id(ARMCI_DOMAIN_SMP,(int)proc);
        if(loop==0) cond = !cond;
        if(cond) {
#endif
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = (int)GA_proclist[p];

          gam_Location(proc,handle, plo, &prem, ldrem); 

          /* find the right spot in the user buffer */
          gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;        

          gam_ComputeCount(ndim, plo, phi, count); 

          /* scale number of rows by element size */
          count[0] *= size; 
          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

          if (p_handle >= 0) {
            proc = (int)GA_proclist[p];
            /* BJP */
            proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
          }
          if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

#ifndef NO_GA_STATS	    
          if(proc == GAme){
            gam_CountElems(ndim, plo, phi, &elems);
            GAbytes.putloc += (double)size*elems;
          }
#endif

          /*casting what ganb_get_armci_handle function returns to armci_hdl is 
            very crucial here as on 64 bit platforms, pointer is 64 bits where 
            as temporary is only 32 bits*/ 
#ifdef __crayx1
          ARMCI_PutS(pbuf,stride_loc,prem,stride_rem,count,ndim-1,proc);
#else
          if(nbhandle) 
            ARMCI_NbPutS(pbuf, stride_loc, prem, stride_rem, count, ndim -1,
                proc,(armci_hdl_t*)get_armci_nbhandle(nbhandle));
          else {
            /* do blocking put for local processes. If all processes
               are remote processes then do blocking put for the last one */
            if((loop==0 && counter==(int)np-1) || loop==1)
              ARMCI_PutS(pbuf,stride_loc,prem,stride_rem,count,ndim-1,proc);
            else {
              ++counter;
              ARMCI_NbPutS(pbuf,stride_loc,prem,stride_rem,count, ndim-1,
                  proc,(armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
            }
          }
        } /* end if(cond) */
      }
#endif
    }
#ifndef __crayx1
    if(!nbhandle) nga_wait_internal(&ga_nbhandle);  
#endif
  } else {
    Integer offset = 0, last, pinv;
    Integer blo[MAXDIM],bhi[MAXDIM];
    Integer plo[MAXDIM],phi[MAXDIM];
    Integer idx, i, j, jtot, chk;
    Integer idx_buf, ldrem[MAXDIM];
    Integer blk_tot = GA[handle].block_total;
    int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
    char *pbuf, *prem;

    /* Loop through all blocks owned by this processor */
    for (idx=0; idx<blk_tot; idx++) {

      /* get the block corresponding to the virtual processor proc */
      ga_ownsM(handle, idx, blo, bhi);
    /*  printf("p[%d] idx: %d\n",GAme,idx);
      printf("p[%d] blo[0]: %d bhi[0]: %d\n",GAme,blo[0],bhi[0]);
      printf("p[%d] blo[1]: %d bhi[1]: %d\n",GAme,blo[1],bhi[1]); */

      /* check to see if this block overlaps with requested block
       * defined by lo and hi */
      chk = 1;
      for (j=0; j<ndim; j++) {
        /* check to see if at least one end point of the interval
         * represented by blo and bhi falls in the interval
         * represented by lo and hi */
        if (!((blo[j] >= lo[j] && blo[j] <= hi[j]) ||
              (bhi[j] >= lo[j] && bhi[j] <= hi[j]))) {
          chk = 0;
        }
      }

      if (chk) {

        /* evaluate offset for block idx */
        offset = 0;
        for (i=idx%GAnproc; i<idx; i += GAnproc) {
          ga_ownsM(handle, i, plo, phi);
          jtot = 1;
          for (j=0; j<ndim; j++) {
            jtot *= phi[j]-plo[j]+1;
          }
          offset += jtot;
        }
        /*printf("p[%d] idx: %d offset: %d\n",GAme,idx,offset); */

        /* get the patch of block that overlaps requested region */
        gam_GetBlockPatch(blo,bhi,lo,hi,plo,phi,ndim);
/*      printf("p[%d] idx: %d\n",GAme,idx);
      printf("p[%d] blo[0]: %d phi[0]: %d\n",GAme,plo[0],phi[0]);
      printf("p[%d] blo[1]: %d phi[1]: %d\n",GAme,plo[1],phi[1]); */

        /* evaluate offset within block */
        last = ndim - 1;
        jtot = 1;
        if (last == 0) ldrem[0] = bhi[0] - blo[0] + 1;
        for (j=0; j<last; j++) {
          offset += (plo[j]-blo[j])*jtot;
          ldrem[j] = bhi[j]-blo[j]+1;
          jtot = ldrem[j];
        }
        offset += (plo[last]-blo[last])*jtot;

        /* get pointer to data on remote block */
        pinv = idx%GAnproc;
        if (p_handle > 0) {
          pinv = PGRP_LIST[p_handle].inv_map_proc_list[pinv];
        }
        prem =  GA[handle].ptr[pinv]+offset*GA[handle].elemsize;

        gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
        pbuf = size*idx_buf + (char*)buf;        

        gam_ComputeCount(ndim, plo, phi, count); 
        /* scale number of rows by element size */
        count[0] *= size; 
        gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

        proc = pinv;
        if(GA_fence_set)fence_array[proc]=1;

#ifndef NO_GA_STATS	    
        if(proc == GAme){
          gam_CountElems(ndim, plo, phi, &elems);
          GAbytes.putloc += (double)size*elems;
        }
#endif

        /*casting what ganb_get_armci_handle function returns to armci_hdl is 
          very crucial here as on 64 bit platforms, pointer is 64 bits where 
          as temporary is only 32 bits*/ 
#ifdef __crayx1
        ARMCI_PutS(pbuf,stride_loc,prem,stride_rem,count,ndim-1,proc);
#else
        if(nbhandle) 
          ARMCI_NbPutS(pbuf, stride_loc, prem, stride_rem, count, ndim -1,
              proc,(armci_hdl_t*)get_armci_nbhandle(nbhandle));
        else {
          /* do blocking put for local processes. If all processes
             are remote processes then do blocking put for the last one */
          /*if((loop==0 && counter==(int)np-1) || loop==1) */
          ARMCI_PutS(pbuf,stride_loc,prem,stride_rem,count,ndim-1,proc);
          /*else {
            ++counter;
            ARMCI_NbPutS(pbuf,stride_loc,prem,stride_rem,count, ndim-1,
            proc,(armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
            } */
        }
      }
    }
#endif
#ifndef __crayx1
    if(!nbhandle) nga_wait_internal(&ga_nbhandle);  
#endif
  }

  GA_POP_NAME;
#ifdef GA_PROFILE
  ga_profile_stop();
#endif
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_NBPUT,__FILE__,__LINE__);
#endif
}

/*\ (NON-BLOCKING) PUT AN N-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY
\*/
void FATR nga_nbput_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   Integer *nbhandle)
{
    nga_put_common(g_a,lo,hi,buf,ld,nbhandle); 
}


/*\ PUT AN N-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY
\*/
void FATR nga_put_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
    nga_put_common(g_a,lo,hi,buf,ld,NULL); 
}


void FATR  ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];
#ifdef GA_USE_VAMPIR
   vampir_begin(GA_PUT,__FILE__,__LINE__);
#endif
#ifdef GA_TRACE
   trace_stime_();
#endif
   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_put_common(g_a, lo, hi, buf, ld,NULL);
#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_PUT,__FILE__,__LINE__);
#endif
}

#ifdef __crayx1 
#  pragma _CRI inline ga_put_
#  pragma _CRI inline nga_put_common
#endif

void FATR  ga_nbput_(g_a, ilo, ihi, jlo, jhi, buf, ld, nbhdl)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld, *nbhdl;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_PUT,__FILE__,__LINE__);
#endif
#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_put_common(g_a, lo, hi, buf, ld,nbhdl);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_PUT,__FILE__,__LINE__);
#endif
}


/*\ A common routine called by both non-blocking and blocking GA Get calls.
\*/
void FATR nga_get_common(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   Integer *nbhandle)
{
                   /* g_a:   Global array handle
                      lo[]:  Array of lower indices of patch of global array
                      hi[]:  Array of upper indices of patch of global array
                      buf[]: Local buffer that array patch will be copied into
                      ld[]:  Array of physical ndim-1 dimensions of local buffer */

  Integer  p, np, handle=GA_OFFSET + *g_a;
  Integer  idx, elems, size, p_handle, ga_nbhandle;
  int proc, ndim, loop, cond, counter=0;
  int num_loops=2; /* 1st loop for remote procs; 2nd loop for local procs */
  Integer use_blocks;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_GET,__FILE__,__LINE__);
#endif

  GA_PUSH_NAME("nga_get_common");

  ga_check_handleM(g_a, "nga_get_common");

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;
  use_blocks = GA[handle].block_flag;
  p_handle = (Integer)GA[handle].p_handle;

  if (!use_blocks) {

    /* Locate the processors containing some portion of the patch
       specified by lo and hi and return the results in _ga_map,
       GA_proclist, and np. GA_proclist contains a list of processors
       containing some portion of the patch, _ga_map contains
       the lower and upper indices of the portion of the patch held
       by a given processor, and np contains the total number of
       processors that contain some portion of the patch.
     */
    if(!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np ))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

    /* get total size of patch */
#ifndef NO_GA_STATS
    gam_CountElems(ndim, lo, hi, &elems);
    GAbytes.gettot += (double)size*elems;
    GAstat.numget++;
    GAstat.numget_procs += np;
#endif

    if(nbhandle)ga_init_nbhandle(nbhandle);
#ifndef __crayx1
    else ga_init_nbhandle(&ga_nbhandle);
#endif

#ifdef GA_PROFILE
    ga_profile_start((int)handle, (long)size*elems, ndim, lo, hi,
        GA_PROFILE_GET);
#endif

    gaPermuteProcList(np);

#ifndef __crayx1
    for(loop=0; loop<num_loops; loop++) {
      __CRAYX1_PRAGMA("_CRI novector");
#endif
      for(idx=0; idx< np; idx++){
        Integer ldrem[MAXDIM];
        int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
        Integer idx_buf, *plo, *phi;
        char *pbuf, *prem;

        p = (Integer)ProcListPerm[idx];
        proc = (int)GA_proclist[p];
        if (p_handle >= 0) {
          proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
        }
#ifdef PERMUTE_PIDS
        if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

        /* check if it is local to SMP */
#ifndef __crayx1
        cond = armci_domain_same_id(ARMCI_DOMAIN_SMP,(int)proc);
        if(loop==0) cond = !cond;
        if(cond) {
#endif

          /* Find  visible portion of patch held by processor p and
             return the result in plo and phi. Also get actual processor
             index corresponding to p and store the result in proc. */
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = (int)GA_proclist[p];

          /* get pointer prem to location indexed by plo. Also get
             leading physical dimensions in memory in ldrem */
          gam_Location(proc,handle, plo, &prem, ldrem);

          /* find the right spot in the user buffer for the point
             subscripted by plo given that the corner of the user
             buffer is subscripted by lo */
          gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;

          /* compute number of elements in each dimension and store the
             result in count */
          gam_ComputeCount(ndim, plo, phi, count);

          /* Scale first element in count by element size. The ARMCI_GetS
             routine uses this convention to figure out memory sizes.*/
          count[0] *= size; 

          /* Return strides for memory containing global array on remote
             processor indexed by proc (stride_rem) and for local buffer
             buf (stride_loc) */
          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

          if (p_handle >= 0) {
            proc = (int)GA_proclist[p];
            /* BJP */
            proc = (int)PGRP_LIST[p_handle].inv_map_proc_list[proc];
          }
#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

#ifndef NO_GA_STATS	    
          if(proc == GAme){
            gam_CountElems(ndim, plo, phi, &elems);
            GAbytes.getloc += (double)size*elems;
          }
#endif
#ifdef __crayx1
          ARMCI_GetS(prem,stride_rem,pbuf,stride_loc,count,ndim-1,proc);
#else
          if(nbhandle) 
            ARMCI_NbGetS(prem, stride_rem, pbuf, stride_loc, count, ndim -1,
                proc,(armci_hdl_t*)get_armci_nbhandle(nbhandle));
          else {
            if((loop==0 && counter==(int)np-1) || loop==1)
              ARMCI_GetS(prem,stride_rem,pbuf,stride_loc,count,ndim-1,proc);
            else {
              ++counter;
              ARMCI_NbGetS(prem,stride_rem,pbuf,stride_loc,count,ndim-1,
                  proc,(armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
            }
          }
        } /* end if(cond) */
      }
#endif
    }

#ifndef __crayx1
    if(!nbhandle) nga_wait_internal(&ga_nbhandle);  
#endif
  } else {
    Integer offset = 0, last, pinv;
    Integer blo[MAXDIM],bhi[MAXDIM];
    Integer plo[MAXDIM],phi[MAXDIM];
    Integer idx, i, j, jtot, chk;
    Integer idx_buf, ldrem[MAXDIM];
    Integer blk_tot = GA[handle].block_total;
    int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
    char *pbuf, *prem;

    /* Loop through all blocks owned by this processor */
    for (idx=0; idx<blk_tot; idx++) {

      /* get the block corresponding to the virtual processor proc */
      ga_ownsM(handle, idx, blo, bhi);

      /* check to see if this block overlaps with requested block
       * defined by lo and hi */
      chk = 1;
      for (j=0; j<ndim; j++) {
        /* check to see if at least one end point of the interval
         * represented by blo and bhi falls in the interval
         * represented by lo and hi */
        if (!((blo[j] >= lo[j] && blo[j] <= hi[j]) ||
              (bhi[j] >= lo[j] && bhi[j] <= hi[j]))) {
          chk = 0;
        }
      }

      if (chk) {

        /* evaluate offset for block idx */
        offset = 0;
        for (i=idx%GAnproc; i<idx; i += GAnproc) {
          ga_ownsM(handle, i, plo, phi);
          jtot = 1;
          for (j=0; j<ndim; j++) {
            jtot *= phi[j]-plo[j]+1;
          }
          offset += jtot;
        }

        /* get the patch of block that overlaps requested region */
        gam_GetBlockPatch(blo,bhi,lo,hi,plo,phi,ndim);

        /* evaluate offset within block */
        last = ndim - 1;
        jtot = 1;
        if (last == 0) ldrem[0] = bhi[0] - blo[0] + 1;
        for (j=0; j<last; j++) {
          offset += (plo[j]-blo[j])*jtot;
          ldrem[j] = bhi[j]-blo[j]+1;
          jtot = ldrem[j];
        }
        offset += (plo[last]-blo[last])*jtot;

        /* get pointer to data on remote block */
        pinv = idx%GAnproc;
        if (p_handle > 0) {
          pinv = PGRP_LIST[p_handle].inv_map_proc_list[pinv];
        }
        prem =  GA[handle].ptr[pinv]+offset*GA[handle].elemsize;

        gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
        pbuf = size*idx_buf + (char*)buf;        

        gam_ComputeCount(ndim, plo, phi, count); 
        /* scale number of rows by element size */
        count[0] *= size; 
        gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

        proc = pinv;
        if(GA_fence_set)fence_array[proc]=1;
        /*printf("p[%d] Got to 1\n",GAme); */

#ifndef NO_GA_STATS	    
        if(proc == GAme){
          gam_CountElems(ndim, plo, phi, &elems);
          GAbytes.putloc += (double)size*elems;
        }
#endif

        /*casting what ganb_get_armci_handle function returns to armci_hdl is 
          very crucial here as on 64 bit platforms, pointer is 64 bits where 
          as temporary is only 32 bits*/ 
#ifdef __crayx1
        ARMCI_GetS(prem,stride_rem,pbuf,stride_loc,count,ndim-1,proc);
#else
        if(nbhandle) 
          ARMCI_NbGetS(prem, stride_rem, pbuf, stride_loc, count, ndim -1,
              proc,(armci_hdl_t*)get_armci_nbhandle(nbhandle));
        else {
          /* do blocking put for local processes. If all processes
             are remote processes then do blocking put for the last one */
          /*if((loop==0 && counter==(int)np-1) || loop==1) */
        /*printf("p[%d] Got to 2\n",GAme); */
          ARMCI_GetS(prem,stride_rem,pbuf,stride_loc,count,ndim-1,proc);
        /*printf("p[%d] Got to 3\n",GAme); */
          /*else {
            ++counter;
            ARMCI_NbGetS(prem,stride_rem,pbuf,stride_loc,count, ndim-1,
            proc,(armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
            } */
        }
      }
    }
#endif
#ifndef __crayx1
    if(!nbhandle) nga_wait_internal(&ga_nbhandle);  
#endif
  }


  GA_POP_NAME;
#ifdef GA_PROFILE
  ga_profile_stop();
#endif
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_GET,__FILE__,__LINE__);
#endif
}

void FATR nga_get_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
    nga_get_common(g_a,lo,hi,buf,ld,(Integer *)NULL);
}


void FATR nga_nbget_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   Integer *nbhandle)
{
    nga_get_common(g_a,lo,hi,buf,ld,nbhandle);
}

void FATR  ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_GET,__FILE__,__LINE__);
#endif
#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_get_common(g_a, lo, hi, buf, ld,(Integer *)NULL);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_GET,__FILE__,__LINE__);
#endif
}

#ifdef __crayx1 
#  pragma _CRI inline ga_get_
#  pragma _CRI inline nga_get_common
#endif

void FATR  ga_nbget_(g_a, ilo, ihi, jlo, jhi, buf, ld,nbhdl)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld, *nbhdl;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_GET,__FILE__,__LINE__);
#endif
#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_get_common(g_a, lo, hi, buf, ld,nbhdl);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_GET,__FILE__,__LINE__);
#endif
}

void FATR nga_acc_common(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   void    *alpha,
                   Integer *nbhandle)
{
  Integer  p, np, handle=GA_OFFSET + *g_a;
  Integer  idx, elems, size, type, p_handle, ga_nbhandle;
  int optype=-1, proc, loop, ndim, cond;
  int num_loops=2; /* 1st loop for remote procs; 2nd loop for local procs */
  Integer use_blocks;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_ACC,__FILE__,__LINE__);
#endif
  GA_PUSH_NAME("nga_acc_common");

  ga_check_handleM(g_a, "nga_acc_common");

  size = GA[handle].elemsize;
  type = GA[handle].type;
  ndim = GA[handle].ndim;
  use_blocks = GA[handle].block_flag;
  p_handle = GA[handle].p_handle;

  if(type==C_DBL) optype= ARMCI_ACC_DBL;
  else if(type==C_FLOAT) optype= ARMCI_ACC_FLT;
  else if(type==C_DCPL)optype= ARMCI_ACC_DCP;
  else if(type==C_INT)optype= ARMCI_ACC_INT;
  else if(type==C_LONG)optype= ARMCI_ACC_LNG;
  else ga_error("type not supported",type);

  if (!use_blocks) {
    if(!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np ))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

#ifndef NO_GA_STATS
    gam_CountElems(ndim, lo, hi, &elems);
    GAbytes.acctot += (double)size*elems;
    GAstat.numacc++;
    GAstat.numacc_procs += np;
#endif

    if(nbhandle)ga_init_nbhandle(nbhandle);
    else ga_init_nbhandle(&ga_nbhandle);

#ifdef GA_PROFILE
    ga_profile_start((int)handle, (long)size*elems, ndim, lo, hi,
        GA_PROFILE_ACC);
#endif

    gaPermuteProcList(np);

    for(loop=0; loop<num_loops; loop++) {
      for(idx=0; idx< np; idx++){
        Integer ldrem[MAXDIM];
        int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
        Integer idx_buf, *plo, *phi;
        char *pbuf, *prem;

        p = (Integer)ProcListPerm[idx];
        proc = (int)GA_proclist[p];
        if (p_handle >= 0) {
          proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
        }
#ifdef PERMUTE_PIDS
        if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

        /* check if it is local to SMP */
        cond = armci_domain_same_id(ARMCI_DOMAIN_SMP,(int)proc);
        if(loop==0) cond = !cond;

        if(cond) {
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = (int)GA_proclist[p];

          gam_Location(proc,handle, plo, &prem, ldrem);

          /* find the right spot in the user buffer */
          gam_ComputePatchIndex(ndim,lo, plo, ld, &idx_buf);
          pbuf = size*idx_buf + (char*)buf;

          gam_ComputeCount(ndim, plo, phi, count);

          /* scale number of rows by element size */
          count[0] *= size;
          gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

          if (p_handle >= 0) {
            proc = (int)GA_proclist[p];
            /* BJP */
            proc = (int)PGRP_LIST[p_handle].inv_map_proc_list[proc];
          }

          if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif
#ifndef NO_GA_STATS
          if(proc == GAme){
            gam_CountElems(ndim, plo, phi, &elems);
            GAbytes.accloc += (double)size*elems;
          }
#endif

          if(nbhandle) 
            ARMCI_NbAccS(optype, alpha, pbuf, stride_loc, prem,
                stride_rem, count, ndim-1, proc,
                (armci_hdl_t*)get_armci_nbhandle(nbhandle));
          else {
#  if 0 /* disabled, as nbacc fails in quadrics */
            if((loop==0 && counter==(int)np-1) || loop==1)
              ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem, 
                  count, ndim-1, proc);
            else {
              ++counter;
              ARMCI_NbAccS(optype, alpha, pbuf, stride_loc, prem, 
                  stride_rem, count, ndim-1, proc,
                  (armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
            }
#  else
            ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem,
                count, ndim-1, proc);
#  endif
          }
        } /* end if(cond) */
      }
    }
#if 0
  if(!nbhandle) nga_wait_internal(&ga_nbhandle);
#endif
  } else {
    Integer offset = 0, last, pinv;
    Integer blo[MAXDIM],bhi[MAXDIM];
    Integer plo[MAXDIM],phi[MAXDIM];
    Integer idx, i, j, jtot, chk;
    Integer idx_buf, ldrem[MAXDIM];
    Integer blk_tot = GA[handle].block_total;
    int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
    char *pbuf, *prem;

    /* Loop through all blocks owned by this processor */
    for (idx=0; idx<blk_tot; idx++) {

      /* get the block corresponding to the virtual processor proc */
      ga_ownsM(handle, idx, blo, bhi);

      /* check to see if this block overlaps with requested block
       * defined by lo and hi */
      chk = 1;
      for (j=0; j<ndim; j++) {
        /* check to see if at least one end point of the interval
         * represented by blo and bhi falls in the interval
         * represented by lo and hi */
        if (!((blo[j] >= lo[j] && blo[j] <= hi[j]) ||
              (bhi[j] >= lo[j] && bhi[j] <= hi[j]))) {
          chk = 0;
        }
      }

      if (chk) {

        /* evaluate offset for block idx */
        offset = 0;
        for (i=idx%GAnproc; i<idx; i += GAnproc) {
          ga_ownsM(handle, i, plo, phi);
          jtot = 1;
          for (j=0; j<ndim; j++) {
            jtot *= phi[j]-plo[j]+1;
          }
          offset += jtot;
        }

        /* get the patch of block that overlaps requested region */
        gam_GetBlockPatch(blo,bhi,lo,hi,plo,phi,ndim);

        /* evaluate offset within block */
        last = ndim - 1;
        jtot = 1;
        if (last == 0) ldrem[0] = bhi[0] - blo[0] + 1;
        for (j=0; j<last; j++) {
          offset += (plo[j]-blo[j])*jtot;
          ldrem[j] = bhi[j]-blo[j]+1;
          jtot = ldrem[j];
        }
        offset += (plo[last]-blo[last])*jtot;

        /* get pointer to data on remote block */
        pinv = idx%GAnproc;
        if (p_handle > 0) {
          pinv = PGRP_LIST[p_handle].inv_map_proc_list[pinv];
        }
        prem =  GA[handle].ptr[pinv]+offset*GA[handle].elemsize;

        gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
        pbuf = size*idx_buf + (char*)buf;        

        gam_ComputeCount(ndim, plo, phi, count); 
        /* scale number of rows by element size */
        count[0] *= size; 
        gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc);

        proc = pinv;
        if(GA_fence_set)fence_array[proc]=1;

#ifndef NO_GA_STATS	    
        if(proc == GAme){
          gam_CountElems(ndim, plo, phi, &elems);
          GAbytes.putloc += (double)size*elems;
        }
#endif
        if(nbhandle) 
          ARMCI_NbAccS(optype, alpha, pbuf, stride_loc, prem,
              stride_rem, count, ndim-1, proc,
              (armci_hdl_t*)get_armci_nbhandle(nbhandle));
        else {
#  if 0 /* disabled, as nbacc fails in quadrics */
          if((loop==0 && counter==(int)np-1) || loop==1)
            ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem, 
                count, ndim-1, proc);
          else {
            ++counter;
            ARMCI_NbAccS(optype, alpha, pbuf, stride_loc, prem, 
                stride_rem, count, ndim-1, proc,
                (armci_hdl_t*)get_armci_nbhandle(&ga_nbhandle));
          }
#  else
          ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem,
              count, ndim-1, proc);
#  endif
        }
      }
    }
#if 0
  if(!nbhandle) nga_wait_internal(&ga_nbhandle);
#endif
  }

  GA_POP_NAME;
#ifdef GA_PROFILE
  ga_profile_stop();
#endif
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_ACC,__FILE__,__LINE__);
#endif
}

/*\ ACCUMULATE OPERATION FOR A N-DIMENSIONAL PATCH OF GLOBAL ARRAY
 *
 *  g_a += alpha * patch
\*/
void FATR nga_acc_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   void    *alpha)
{
    nga_acc_common(g_a,lo,hi,buf,ld,alpha,NULL);
}

void FATR nga_nbacc_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld,
                   void    *alpha,
                   Integer *nbhndl)
{
    nga_acc_common(g_a,lo,hi,buf,ld,alpha,nbhndl);
}


void FATR  ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   void *buf, *alpha;
{
Integer lo[2], hi[2];
#ifdef GA_TRACE
   trace_stime_();
#endif

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_ACC,__FILE__,__LINE__);
#endif
   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_acc_common(g_a,lo,hi,buf,ld,alpha,NULL);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_ACC,__FILE__,__LINE__);
#endif
}

void FATR  ga_nbacc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha,nbhndl)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld, *nbhndl;
   void *buf, *alpha;
{
Integer lo[2], hi[2];
#ifdef GA_TRACE
   trace_stime_();
#endif

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_ACC,__FILE__,__LINE__);
#endif
   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_acc_common(g_a,lo,hi,buf,ld,alpha,nbhndl);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_ACC,__FILE__,__LINE__);
#endif
}

/*\ RETURN A POINTER TO LOCAL DATA
\*/
void nga_access_ptr(Integer* g_a, Integer lo[], Integer hi[],
                      void* ptr, Integer ld[])
{
char *lptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i,p_handle;

   GA_PUSH_NAME("nga_access_ptr");
   p_handle = GA[handle].p_handle;
   if (!nga_locate_(g_a,lo,&ow)) ga_error("locate top failed",0);
   if (p_handle != -1)
      ow = PGRP_LIST[p_handle].inv_map_proc_list[ow];
   if ((GA_Cluster_proc_nodeid(ow) != GA_Cluster_nodeid()) && (ow != GAme)) 
      ga_error("cannot access top of the patch",ow);
   if (!nga_locate_(g_a,hi, &ow)) ga_error("locate bottom failed",0);
   if (p_handle != -1)
      ow = PGRP_LIST[p_handle].inv_map_proc_list[ow];
   if ((GA_Cluster_proc_nodeid(ow) != GA_Cluster_nodeid()) && (ow != GAme))
      ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }

   if (p_handle >= 0) {
     ow = PGRP_LIST[p_handle].map_proc_list[ow];
   }
   gam_Location(ow,handle, lo, &lptr, ld);
   *(char**)ptr = lptr; 
   GA_POP_NAME;
}


/*\ RETURN A POINTER TO BEGINNING OF LOCAL DATA BLOCK
\*/
void nga_access_block_ptr(Integer* g_a, Integer *idx, void* ptr, Integer *ld)
            /* g_a: array handle [input]
             * idx: block index  [input]
             * ptr: pointer to data in block [output]
             * ld:  array of strides for block data [output]
             */
{
  char *lptr;
  Integer  handle = GA_OFFSET + *g_a;
  Integer  i, j, p_handle, nblocks, offset, tsum, inode;
  Integer ndim, lo[MAXDIM], hi[MAXDIM], index;

  GA_PUSH_NAME("nga_access_block_ptr");
  p_handle = GA[handle].p_handle;
  nblocks = GA[handle].block_total;
  ndim = GA[handle].ndim;
  index = *idx;
  if (index < 0 || index >= nblocks)
    ga_error("block index outside allowed values",index);

  offset = 0;
  inode = index%GAnproc;
  for (i=inode; i<index; i += GAnproc) {
    ga_ownsM(handle,i,lo,hi); 
    tsum = 1;
    for (j=0; j<ndim; j++) {
      tsum *= (hi[j]-lo[j]+1);
    }
    offset += tsum;
  }
  lptr = GA[handle].ptr[inode]+offset*GA[handle].elemsize;

  ga_ownsM(handle,index,lo,hi); 
  for (i=0; i<ndim-1; i++) {
    ld[i] = hi[i]-lo[i]+1;
  }

  *(char**)ptr = lptr; 
  GA_POP_NAME;
}

/*\ RETURN A POINTER TO BEGINNING OF LOCAL DATA ON A PROCESSOR CONTAINING
 *  BLOCK-CYCLIC DATA
\*/
void ga_access_block_segment_ptr(Integer* g_a, Integer *proc, void* ptr, Integer *len)
            /* g_a:  array handle [input]
             * proc: processor for data [input]
             * ptr:  pointer to data start of data on processor [output]
             * len:  length of data contained on processor [output]
             */
{
  char *lptr;
  Integer  handle = GA_OFFSET + *g_a;
  Integer  i, j, p_handle, nblocks, offset, tsum;
  Integer ndim, lo[MAXDIM], hi[MAXDIM], index;

  GA_PUSH_NAME("ga_access_block_segment_ptr");
  p_handle = GA[handle].p_handle;
  nblocks = GA[handle].block_total;
  ndim = GA[handle].ndim;
  index = *proc;
  if (index < 0 || index >= GAnproc)
    ga_error("processor index outside allowed values",index);

  offset = 0;
  for (i=index; i<nblocks; i += GAnproc) {
    ga_ownsM(handle,i,lo,hi); 
    tsum = 1;
    for (j=0; j<ndim; j++) {
      tsum *= (hi[j]-lo[j]+1);
    }
    offset += tsum;
  }
  lptr = GA[handle].ptr[index];

  *len = offset;
  *(char**)ptr = lptr; 
  GA_POP_NAME;
}

/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR nga_access_(Integer* g_a, Integer lo[], Integer hi[],
                      Integer* index, Integer ld[])
{
char     *ptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i,p_handle;
unsigned long    elemsize;
unsigned long    lref=0, lptr;

#ifdef GA_USE_VAMPIR
   vampir_begin(NGA_ACCESS,__FILE__,__LINE__);
#endif
   GA_PUSH_NAME("nga_access");
   p_handle = GA[handle].p_handle;
   if(!nga_locate_(g_a,lo,&ow))ga_error("locate top failed",0);
   if (p_handle != -1)
      ow = PGRP_LIST[p_handle].inv_map_proc_list[ow];
   if ((GA_Cluster_proc_nodeid(ow) != GA_Cluster_nodeid()) && (ow != GAme)) 
      ga_error("cannot access top of the patch",ow);
   if(!nga_locate_(g_a,hi, &ow))ga_error("locate bottom failed",0);
   if (p_handle != -1)
      ow = PGRP_LIST[p_handle].inv_map_proc_list[ow];
   if ((GA_Cluster_proc_nodeid(ow) != GA_Cluster_nodeid()) && (ow != GAme)) 
      ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }

   
   if (p_handle != -1)
      ow = PGRP_LIST[p_handle].map_proc_list[ow];

   gam_Location(ow,handle, lo, &ptr, ld);

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
#ifdef GA_USE_VAMPIR
   vampir_end(NGA_ACCESS,__FILE__,__LINE__);
#endif
}

/*\ PROVIDE ACCESS TO AN INDIVIDUAL DATA BLOCK OF A GLOBAL ARRAY
\*/
void FATR nga_access_block_(Integer* g_a, Integer* idx, Integer* index, Integer *ld)
{
char     *ptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  i,p_handle, iblock;
unsigned long    elemsize;
unsigned long    lref=0, lptr;

#ifdef GA_USE_VAMPIR
   vampir_begin(NGA_ACCESS_BLOCK,__FILE__,__LINE__);
#endif
   GA_PUSH_NAME("nga_access_block");
   p_handle = GA[handle].p_handle;
   iblock = *idx;
   if (iblock < 0 || iblock >= GA[handle].block_total)
     ga_error("block index outside allowed values",iblock);

   nga_access_block_ptr(g_a,&iblock,&ptr,ld);
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
#ifdef GA_USE_VAMPIR
   vampir_end(NGA_ACCESS_BLOCK,__FILE__,__LINE__);
#endif
}

/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR ga_access_(g_a, ilo, ihi, jlo, jhi, index, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *index, *ld;
{
Integer lo[2], hi[2],ndim=ga_ndim_(g_a);

     if(ndim != 2) 
        ga_error("ga_access: 2D API cannot be used for array dimension",ndim);

#ifdef GA_USE_VAMPIR
     vampir_begin(GA_ACCESS,__FILE__,__LINE__);
#endif
     lo[0]=*ilo;
     lo[1]=*jlo;
     hi[0]=*ihi;
     hi[1]=*jhi;
     nga_access_(g_a,lo,hi,index,ld);
#ifdef GA_USE_VAMPIR
     vampir_end(GA_ACCESS,__FILE__,__LINE__);
#endif
} 



/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_(Integer *g_a, 
                       Integer *ilo, Integer *ihi, Integer *jlo, Integer *jhi)
{}


/*\ RELEASE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR  nga_release_(Integer *g_a, Integer *lo, Integer *hi)
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void FATR  ga_release_update_(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
{}


/*\ RELEASE ACCESS & UPDATE A PATCH OF A GLOBAL ARRAY
\*/
void FATR  nga_release_update_(Integer *g_a, Integer *lo, Integer *hi)
{}




void ga_scatter_acc_local(Integer g_a, Void *v,Integer *i,Integer *j,
                          Integer nv, void* alpha, Integer proc) 
{
void **ptr_src, **ptr_dst;
char *ptr_ref;
Integer ldp, item_size, ilo, ihi, jlo, jhi, type;
Integer handle,p_handle,iproc;
armci_giov_t desc;
register Integer k, offset;
int rc=0;

  if (nv < 1) return;

  GA_PUSH_NAME("ga_scatter_local");
  handle = GA_OFFSET + g_a;
  p_handle = GA[handle].p_handle;

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  iproc = proc;
  gaShmemLocation(iproc, g_a, ilo, jlo, &ptr_ref, &ldp);

  type = GA[handle].type;
  item_size = GAsizeofM(type);

  ptr_src = gai_malloc((int)nv*2*sizeof(void*));
  if(ptr_src==NULL)ga_error("gai_malloc failed",nv);
  ptr_dst=ptr_src+ nv;

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)>< [%ld:%ld,%ld:%ld]",
               (int)proc, (long)i[k], (long)j[k], (long)ilo, 
               (long)ihi, (long)jlo, (long)jhi); 
       ga_error(err_string,g_a);
     }

     offset  = (j[k] - jlo)* ldp + i[k] - ilo;
     ptr_dst[k] = ptr_ref + item_size * offset;
     ptr_src[k] = ((char*)v) + k*item_size;
  }
  desc.bytes = (int)item_size;
  desc.src_ptr_array = ptr_src;
  desc.dst_ptr_array = ptr_dst;
  desc.ptr_array_len = (int)nv;

  if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

  if (p_handle >= 0) {
    proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
  }
  if(alpha != NULL) {
    int optype=-1;
    if(type==C_DBL) optype= ARMCI_ACC_DBL;
    else if(type==C_DCPL)optype= ARMCI_ACC_DCP;
    else if(type==C_INT)optype= ARMCI_ACC_INT;
    else if(type==C_LONG)optype= ARMCI_ACC_LNG;
    else if(type==C_FLOAT)optype= ARMCI_ACC_FLT;  
    else ga_error("type not supported",type);
    rc= ARMCI_AccV(optype, alpha, &desc, 1, (int)proc);
  }

  if(rc) ga_error("scatter/_acc failed in armci",rc);

  gai_free(ptr_src);

  GA_POP_NAME;
}


/*\ based on subscripts compute pointers
\*/
void gai_sort_proc(Integer* g_a, Integer* sbar, Integer *nv, Integer list[], Integer proc[])
{
int k, ndim;
extern void ga_sort_permutation();

   if (*nv < 1) return;

   ga_check_handleM(g_a, "gai_get_pointers");
   ndim = GA[*g_a+GA_OFFSET].ndim;

   for(k=0; k< *nv; k++)if(!nga_locate_(g_a, sbar+k*ndim, proc+k)){
         gai_print_subscript("invalid subscript",ndim, sbar +k*ndim,"\n");
         ga_error("failed -element:",k);
   }
         
   /* Sort the entries by processor */
   ga_sort_permutation(nv, list, proc);
}
 

/*\ permutes input index list using sort routine used in scatter/gather
\*/
void FATR nga_sort_permut_(Integer* g_a, Integer index[], 
                           Integer* subscr_arr, Integer *nv)
{
    /* The new implementation doesn't change the order of the elements
     * They are identical
     */
    /*
Integer pindex, phandle;

  if (*nv < 1) return;

  if(!MA_push_get(MT_F_INT,*nv, "nga_sort_permut--p", &phandle, &pindex))
              ga_error("MA alloc failed ", *g_a);

  gai_sort_proc(g_a, subscr_arr, nv, index, INT_MB+pindex);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    */
}


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_(Integer *g_a, Void *v, Integer *i, Integer *j,
                       Integer *nv)
{
    register Integer k;
    Integer kk;
    Integer item_size;
    Integer proc, type=GA[GA_OFFSET + *g_a].type;
    Integer nproc, p_handle, iproc;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    Integer handle = *g_a + GA_OFFSET;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp, *owner;
    char **ptr_ref;
    
    if (*nv < 1) return;
#ifdef GA_USE_VAMPIR
    vampir_begin(GA_SCATTER,__FILE__,__LINE__);
#endif
    
    ga_check_handleM(g_a, "ga_scatter");
    GA_PUSH_NAME("ga_scatter");
    GAstat.numsca++;
    /* determine how many processors are associated with array */
    p_handle = GA[handle].p_handle;
    if (p_handle < 0) {
      nproc = GAnproc;
    } else {
      nproc = PGRP_LIST[p_handle].map_nproc;
    }
    
    /* allocate temp memory */
    buf1 = gai_malloc((int) (nproc *4 +*nv)* (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*nproc);
   
    owner = (Integer *)buf1;  
    count = owner+ *nv;  
    nelem =  count + nproc;  
    aproc = count + 2 * nproc;  
    map = count + 3 * nproc;  
    
    /* initialize the counters and nelem */
    for(kk=0; kk<nproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp:  */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, owner+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", (long)i[k], (long)j[k]);
            ga_error(err_string,*g_a);
        }
        iproc = owner[k];
        nelem[iproc]++;
    }

    naproc = 0;
    for(k=0; k<nproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    
    GAstat.numsca_procs += naproc;

    buf2 = gai_malloc((int)(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *) +
                      5*naproc*sizeof(Integer) + naproc*sizeof(char*)));
    if(buf2 == NULL) ga_error("gai_malloc failed", naproc);
 
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc*sizeof(void **));
    ptr_org = (void **)(buf2 + 2*naproc*sizeof(void **));
    ptr_ref = (char **)(buf2+2*naproc*sizeof(void **)+2*(*nv)*sizeof(void *));
    ilo = (Integer *)(((char*)ptr_ref) + naproc*sizeof(char*));
    ihi = ilo + naproc;
    jlo = ihi + naproc;
    jhi = jlo + naproc;
    ldp = jhi + naproc;

    for(kk=0; kk<naproc; kk++) {
        iproc = aproc[kk];
        ga_distribution_(g_a, &iproc,
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));
    }
    
    /* determine limit for message size --  v,i, & j will travel together */
    item_size = GAsizeofM(type);
    GAbytes.scatot += (double)item_size**nv ;
    iproc = owner[GAme];
    GAbytes.scaloc += (double)item_size* nelem[iproc];
    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        Integer this_count;
        proc = owner[k];
        this_count = count[proc]; 
        count[proc]++;
        proc = map[proc];
        ptr_src[proc][this_count] = ((char*)v) + k * item_size;

        if(i[k] < ilo[proc] || i[k] > ihi[proc]  ||
           j[k] < jlo[proc] || j[k] > jhi[proc]){
          sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)><[%ld:%ld,%ld:%ld]",
             (int)proc, (long)i[k], (long)j[k], (long)ilo[proc], 
             (long)ihi[proc], (long)jlo[proc], (long)jhi[proc]);
            ga_error(err_string, *g_a);
        }
        ptr_dst[proc][this_count] = ptr_ref[proc] + item_size *
            ((j[k] - jlo[proc])* ldp[proc] + i[k] - ilo[proc]);
    }
    
    /* source and destination pointers are ready for all processes */
    for(k=0; k<naproc; k++) {
        int rc;

        desc.bytes = (int)item_size;
        desc.src_ptr_array = ptr_src[k];
        desc.dst_ptr_array = ptr_dst[k];
        desc.ptr_array_len = (int)nelem[aproc[k]];
        if (p_handle < 0) {
          iproc = aproc[k];
        } else {
          iproc = PGRP_LIST[p_handle].inv_map_proc_list[aproc[k]];
        }
        
        rc = ARMCI_PutV(&desc, 1, (int)iproc);
        if(rc) ga_error("scatter failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);

    GA_POP_NAME;
#ifdef GA_USE_VAMPIR
    vampir_end(GA_SCATTER,__FILE__,__LINE__);
#endif
}
      


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_acc_(g_a, v, i, j, nv, alpha)
     Integer *g_a, *nv, *i, *j;
     Void *v, *alpha;
{
register Integer k;
Integer item_size;
Integer first, nelem, proc, type=GA[GA_OFFSET + *g_a].type;
Integer *int_ptr;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_scatter_acc");
  GA_PUSH_NAME("ga_scatter_acc");
  GAstat.numsca++;

  int_ptr = (Integer*) ga_malloc(*nv, MT_F_INT, "ga_scatter_acc--p");

  /* find proc that owns the (i,j) element; store it in temp: int_ptr */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, int_ptr+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", (long)i[k], (long)j[k]);
         ga_error(err_string,*g_a);
  }

  /* determine limit for message size --  v,i, & j will travel together */
  item_size = GAsizeofM(type);
  GAbytes.scatot += (double)item_size**nv ;

  /* Sort the entries by processor */
  ga_sort_scat(nv, v, i, j, int_ptr, type );

  /* go through the list again executing scatter for each processor */

  first = 0;
  do {
      proc = int_ptr[first];
      nelem = 0;

      /* count entries for proc from "first" to last */
      for(k=first; k< *nv; k++){
        if(proc == int_ptr[k]) nelem++;
        else break;
      }

      if(proc == GAme){
             GAbytes.scaloc += (double)item_size* nelem ;
      }

      ga_scatter_acc_local(*g_a, ((char*)v)+item_size*first, i+first,
                           j+first, nelem, alpha, proc);
      first += nelem;

  }while (first< *nv);

  ga_free(int_ptr);

  GA_POP_NAME;
}



/*\ permutes input index list using sort routine used in scatter/gather
\*/
void FATR  ga_sort_permut_(g_a, index, i, j, nv)
     Integer *g_a, *nv, *i, *j, *index;
{
    /* The new implementation doesn't change the order of the elements
     * They are identical
     */

#if 0
register Integer k;
Integer *int_ptr;
extern void ga_sort_permutation();

  if (*nv < 1) return;

  int_ptr = (Integer*) ga_malloc(*nv, MT_F_INT, "ga_sort_permut--p");

  /* find proc that owns the (i,j) element; store it in temp: int_ptr */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, int_ptr+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* Sort the entries by processor */
  ga_sort_permutation(nv, index, int_ptr);
  ga_free(int_ptr);
#endif
}




#define SCATTER -99
#define GATHER -98
#define SCATTER_ACC -97

/*\ GATHER OPERATION elements from the global array into v
\*/
void gai_gatscat(int op, Integer* g_a, void* v, Integer subscript[], 
                 Integer* nv, double *locbytes, double* totbytes, void *alpha)
{
    Integer k, handle=*g_a+GA_OFFSET;
    int  ndim, item_size, type;
    Integer *proc;
    Integer nproc, p_handle, iproc;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;        /* counters for each process */
    Integer *nelem;        /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    
    GA_PUSH_NAME("gai_gatscat");

    proc=(Integer *)ga_malloc(*nv, MT_F_INT, "ga_gat-p");

    ndim = GA[handle].ndim;
    type = GA[handle].type;
    item_size = GA[handle].elemsize;
    *totbytes += (double)item_size**nv;

    /* determine how many processors are associated with array */
    p_handle = GA[handle].p_handle;
    if (p_handle < 0) {
      nproc = GAnproc;
    } else {
      nproc = PGRP_LIST[p_handle].map_nproc;
    }

    /* allocate temp memory */
    buf1 = gai_malloc((int) nproc * 4 * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*nproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + nproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * nproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * nproc * sizeof(Integer));
    
    /* initialize the counters and nelem */
    for(k=0; k<nproc; k++) count[k] = 0; 
    for(k=0; k<nproc; k++) nelem[k] = 0;

    /* get the process id that the element should go and count the
     * number of elements for each process
     */
    for(k=0; k<*nv; k++) {
        if(!nga_locate_(g_a, subscript+k*ndim, proc+k)) {
            gai_print_subscript("invalid subscript",ndim, subscript+k*ndim,"\n");
            ga_error("failed -element:",k);
        }
        iproc = proc[k];
        nelem[iproc]++;
    }

    /* find the active processes (with which transfer data) */
    naproc = 0;
    for(k=0; k<nproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }

    buf2 = gai_malloc((int)(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *)));
    if(buf2 == NULL) ga_error("gai_malloc failed", 2*naproc);
    
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc * sizeof(void **));
    ptr_org = (void **)(buf2 + 2 * naproc * sizeof(void **));
    
    /* set the pointers as
     *    P0            P1                  P0          P1        
     * ptr_src[0]   ptr_src[1] ...       ptr_dst[0]  ptr_dst[1] ...
     *        \          \                    \          \
     * ptr_org |-------------------------------|---------------------------|
     *         |              (*nv)            |            (*nv)          |
     *         | nelem[0] | nelem[1] |...      | nelem[0] | nelem[1] |...
     */
    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }

    *locbytes += (double)item_size* nelem[GAme];
    
/*
#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif
*/    
    switch(op) { 
      case GATHER:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = subscript + offset0...
         *                ptr_dst[0][0, 1, ...] = v + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */  
        for(k=0; k<(*nv); k++){
            iproc = proc[k];
            ptr_dst[map[iproc]][count[iproc]] = ((char*)v) + k * item_size;
            if (p_handle != 0) {
              gam_Loc_ptr(iproc, handle,  (subscript+k*ndim),
                          ptr_src[map[iproc]]+count[iproc]);
            } else {
              gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                          ptr_src[map[iproc]]+count[iproc]);
            }
            count[iproc]++;
        }
        
        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            if (p_handle < 0) {
              iproc = aproc[k];
            } else {
              iproc = PGRP_LIST[p_handle].inv_map_proc_list[aproc[k]];
            }
            rc=ARMCI_GetV(&desc, 1, (int)iproc);
            if(rc) ga_error("gather failed in armci",rc);
        }
        GAstat.numgat_procs += naproc;
        break;
      case SCATTER:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            iproc = proc[k];
            ptr_src[map[iproc]][count[iproc]] = ((char*)v) + k * item_size;
            if (p_handle != 0) {
              gam_Loc_ptr(iproc, handle,  (subscript+k*ndim),
                          ptr_dst[map[iproc]]+count[iproc]);
            } else {
              gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                          ptr_dst[map[iproc]]+count[iproc]);
            }
            count[iproc]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            
            if (p_handle < 0) {
              iproc = aproc[k];
            } else {
              iproc = PGRP_LIST[p_handle].inv_map_proc_list[aproc[k]];
            }
            if(GA_fence_set) fence_array[iproc]=1;

            rc=ARMCI_PutV(&desc, 1, (int)iproc);
            if(rc) ga_error("scatter failed in armci",rc);
        }
        GAstat.numsca_procs += naproc;
        break;
      case SCATTER_ACC:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            iproc = proc[k];
            ptr_src[map[iproc]][count[iproc]] = ((char*)v) + k * item_size;
            if (p_handle != 0) {
              gam_Loc_ptr(iproc, handle,  (subscript+k*ndim),
                          ptr_dst[map[iproc]]+count[iproc]);
            } else {
              gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                          ptr_dst[map[iproc]]+count[iproc]);
            }
            count[iproc]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc=0;
            
            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            if (p_handle < 0) {
              iproc = aproc[k];
            } else {
              iproc = PGRP_LIST[p_handle].inv_map_proc_list[aproc[k]];
            }
            if(GA_fence_set) fence_array[iproc]=1;
            
            if(alpha != NULL) {
                int optype=-1;
                if(type==C_DBL) optype= ARMCI_ACC_DBL;
                else if(type==C_DCPL)optype= ARMCI_ACC_DCP;
                else if(type==C_INT)optype= ARMCI_ACC_INT;
                else if(type==C_LONG)optype= ARMCI_ACC_LNG;
                else if(type==C_FLOAT)optype= ARMCI_ACC_FLT; 
                else ga_error("type not supported",type);
                rc= ARMCI_AccV(optype, alpha, &desc, 1, (int)iproc);
            }
            if(rc) ga_error("scatter_acc failed in armci",rc);
        }
        break;        
      default: ga_error("operation not supported",op);
    }

    gai_free(buf2); gai_free(buf1);
    
    ga_free(proc);
    GA_POP_NAME;
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR nga_gather_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_gather");
#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_GATHER,__FILE__,__LINE__);
#endif
  GA_PUSH_NAME("nga_gather");
  GAstat.numgat++;

  gai_gatscat(GATHER,g_a,v,subscript,nv,&GAbytes.gattot,&GAbytes.gatloc, NULL);

  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_GATHER,__FILE__,__LINE__);
#endif
}


void FATR nga_scatter_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_SCATTER,__FILE__,__LINE__);
#endif
  ga_check_handleM(g_a, "nga_scatter");
  GA_PUSH_NAME("nga_scatter");
  GAstat.numsca++;

  gai_gatscat(SCATTER,g_a,v,subscript,nv,&GAbytes.scatot,&GAbytes.scaloc, NULL);

  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_SCATTER,__FILE__,__LINE__);
#endif
}

void FATR nga_scatter_acc_(Integer *g_a, void* v, Integer subscript[],
                           Integer *nv, void *alpha)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_scatter_acc");
  GA_PUSH_NAME("nga_scatter_acc");
  GAstat.numsca++;

  gai_gatscat(SCATTER_ACC, g_a, v, subscript, nv, &GAbytes.scatot,
              &GAbytes.scaloc, alpha);

  GA_POP_NAME;
}

void FATR  ga_gather000_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
int k;
Integer *sbar = (Integer*)malloc(2*sizeof(Integer)*  (int)*nv);
     if(!sbar) ga_error("gather:malloc failed",*nv);
     for(k=0;k<*nv;k++){
          sbar[2*k] = i[k];
          sbar[2*k+1] = j[k];
     }
     nga_gather_(g_a,v,sbar,nv);
     free(sbar);
}
  


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter000_(g_a, v, i, j, nv)
     Integer *g_a, *nv, *i, *j;
     Void *v;
{
int k;
Integer *sbar = (Integer*)malloc(2*sizeof(Integer)* (int) *nv);
     if(!sbar) ga_error("scatter:malloc failed",*nv);
     for(k=0;k<*nv;k++){
          sbar[2*k] = i[k];
          sbar[2*k+1] = j[k];
     }
     nga_scatter_(g_a,v,sbar,nv);
     free(sbar);
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR  ga_gather_(Integer *g_a, void *v, Integer *i, Integer *j,
                      Integer *nv)
{
    Integer k, kk, proc, item_size;
    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    Integer handle = *g_a + GA_OFFSET;
    Integer nproc, p_handle, iproc;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp, *owner;
    char **ptr_ref;
    
    if (*nv < 1) return;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_GATHER,__FILE__,__LINE__);
#endif
    ga_check_handleM(g_a, "ga_gather");
    GA_PUSH_NAME("ga_gather");
    GAstat.numgat++;

    /* determine how many processors are associated with array */
    p_handle = GA[handle].p_handle;
    if (p_handle < 0) {
      nproc = GAnproc;
    } else {
      nproc = PGRP_LIST[p_handle].map_nproc;
    }

    /* allocate temp memory */
    buf1 = gai_malloc((int)(nproc *4  + *nv)*  (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*nproc);
    
    owner = (Integer *)buf1; 
    count = owner+ *nv;
    nelem = count + nproc;
    aproc = count + 2 * nproc;
    map =   count + 3 * nproc;
   
    /* initialize the counters and nelem */
    for(kk=0; kk<nproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp: */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, owner+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", (long)i[k], (long)j[k]);
            ga_error(err_string, *g_a);
        }
        iproc = owner[k];
        nelem[iproc]++;
    }

    naproc = 0;
    for(k=0; k<nproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    GAstat.numgat_procs += naproc;
    
    buf2 = gai_malloc((int)(2*naproc*sizeof(void **) + 2*(*nv)*sizeof(void *) +
                      5*naproc*sizeof(Integer) + naproc*sizeof(char*)));
    if(buf2 == NULL) ga_error("gai_malloc failed", naproc);
 
    ptr_src = (void ***)buf2;
    ptr_dst = (void ***)(buf2 + naproc*sizeof(void **));
    ptr_org = (void **)(buf2 + 2*naproc*sizeof(void **));
    ptr_ref = (char **)(buf2+2*naproc*sizeof(void **)+2*(*nv)*sizeof(void *));
    ilo = (Integer *)(((char*)ptr_ref) + naproc*sizeof(char*));
    ihi = ilo + naproc;
    jlo = ihi + naproc;
    jhi = jlo + naproc;
    ldp = jhi + naproc;

    for(kk=0; kk<naproc; kk++) {
        iproc = aproc[kk];
        ga_distribution_(g_a, &iproc,
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));

        /*gaShmemLocation(iproc, *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk])); */
    }
    
    item_size = GA[GA_OFFSET + *g_a].elemsize;
    GAbytes.gattot += (double)item_size**nv;
    /*This stuff is probably completely wrong. Doesn't affect performance,
     * just statistics. */
    iproc = owner[GAme];
    GAbytes.gatloc += (double)item_size * nelem[iproc];

    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        Integer this_count;
        proc = owner[k];
        this_count = count[proc]; 
        count[proc]++;
        proc = map[proc]; 
        ptr_dst[proc][this_count] = ((char*)v) + k * item_size;

        if(i[k] < ilo[proc] || i[k] > ihi[proc]  ||
           j[k] < jlo[proc] || j[k] > jhi[proc]){
          sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)><[%ld:%ld,%ld:%ld]",
                 (int)proc,i[k],j[k],ilo[proc],ihi[proc],jlo[proc], jhi[proc]);
            ga_error(err_string, *g_a);
        }
        ptr_src[proc][this_count] = ptr_ref[proc] + item_size *
            ((j[k] - jlo[proc])* ldp[proc] + i[k] - ilo[proc]);
    }
    /* source and destination pointers are ready for all processes */
    for(k=0; k<naproc; k++) {
        int rc;

        desc.bytes = (int)item_size;
        desc.src_ptr_array = ptr_src[k];
        desc.dst_ptr_array = ptr_dst[k];
        desc.ptr_array_len = (int)nelem[aproc[k]];
        if (p_handle < 0) {
          iproc = aproc[k];
        } else {
          iproc = PGRP_LIST[p_handle].inv_map_proc_list[aproc[k]]; 
        }
        rc=ARMCI_GetV(&desc, 1, (int)iproc);
        if(rc) ga_error("gather failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);
    GA_POP_NAME;
#ifdef GA_USE_VAMPIR
    vampir_end(GA_GATHER,__FILE__,__LINE__);
#endif
}
      



/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR nga_read_inc_(Integer* g_a, Integer* subscript, Integer* inc)
{
Integer *ptr, ldp[MAXDIM], proc, handle=GA_OFFSET+*g_a, p_handle;
int optype,ivalue;
long lvalue;
void *pval;

#ifdef GA_USE_VAMPIR
    vampir_begin(NGA_READ_INC,__FILE__,__LINE__);
#endif
    ga_check_handleM(g_a, "nga_read_inc");
    GA_PUSH_NAME("nga_read_inc");
    /* BJP printf("p[%d] g_a: %d subscript: %d inc: %d\n",GAme, *g_a, subscript[0], *inc); */

    if(GA[handle].type!=C_INT && GA[handle].type!=C_LONG) ga_error("type must be integer",GA[handle].type);

    GAstat.numrdi++;
    GAbytes.rditot += (double)sizeof(Integer);
    p_handle = GA[handle].p_handle;

    /* find out who owns it */
    nga_locate_(g_a, subscript, &proc);

    /* get an address of the g_a(subscript) element */
    gam_Location(proc, handle,  subscript, (char**)&ptr, ldp);
    if (p_handle != -1) {   
       proc=PGRP_LIST[GA[handle].p_handle].inv_map_proc_list[proc];
       /*printf("\n%d:proc=%d",GAme,proc);fflush(stdout);*/
    }

    if(GA[handle].type==C_INT){
       optype = ARMCI_FETCH_AND_ADD;
       pval = &ivalue;
    }else{
       optype = ARMCI_FETCH_AND_ADD_LONG;
       pval = &lvalue;
    }

    if(GAme == proc)GAbytes.rdiloc += (double)sizeof(Integer);

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

    ARMCI_Rmw(optype, (int*)pval, (int*)ptr, (int)*inc, (int)proc);

   GA_POP_NAME;
#ifdef GA_USE_VAMPIR
   vampir_end(NGA_READ_INC,__FILE__,__LINE__);
#endif

    if(GA[handle].type==C_INT)
         return (Integer) ivalue;
    else
         return (Integer) lvalue;
}




/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer  value, subscript[2];

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_READ_INC,__FILE__,__LINE__);
#endif
#ifdef GA_TRACE
       trace_stime_();
#endif

   subscript[0] =*i;
   subscript[1] =*j;
   value = nga_read_inc_(g_a, subscript, inc);

#  ifdef GA_TRACE
     trace_etime_();
     op_code = GA_OP_RDI;
     trace_genrec_(g_a, i, i, j, j, &op_code);
#  endif
#ifdef GA_USE_VAMPIR
   vampir_end(GA_READ_INC,__FILE__,__LINE__);
#endif

   return(value);
}

/*\ UTILITY FUNCTION TO CORRECT PATCH INDICES FOR PRESENCE OF
 *  SKIPS. IF NO DATA IS LEFT IN PATCH IT RETURNS FALSE.
\*/
Integer gai_correct_strided_patch(Integer ndim,
                                  Integer *lo,
                                  Integer *skip,
                                  Integer *plo,
                                  Integer *phi)
{
  Integer i, delta;
  for (i=0; i<ndim; i++) {
    delta = plo[i]-lo[i];
    if (delta%skip[i] != 0) {
      plo[i] = lo[i] + delta - delta%skip[i] + skip[i];
    }
    delta = phi[i]-lo[i];
    if (delta%skip[i] != 0) {
      phi[i] = lo[i] + delta - delta%skip[i];
    }
    if (phi[i]<plo[i]) return 0;
  }
  return 1;
}

/*\ UTILITY FUNCTION TO CALCULATE NUMBER OF ELEMENTS IN EACH STRIDE
 *  DIRECTION
\*/

int gai_ComputeCountWithSkip(Integer ndim, Integer *lo, Integer *hi,
                             Integer *skip, int *count, Integer *nstride)
{
  Integer idx;
  int i, istride = 0;
  for (i=0; i<(int)ndim; i++) {
    idx = hi[i] - lo[i];
    if (idx < 0) return 0;
    if (skip[i] > 1) {
      count[istride] = 1;
      istride++;
      count[istride] = (int)(idx/skip[i]+1);
      istride++;
    } else {
      count[istride] = (int)idx+1;
      istride++;
    }
  }
  *nstride = istride;
  return 1;
}

/*\ UTILITY FUNCTION TO CALCULATE STRIDE LENGTHS ON LOCAL AND
 *  REMOTE PROCESSORS, TAKING INTO ACCOUNT THE PRESENCE OF
 *  SKIPS
\*/
void gai_SetStrideWithSkip(Integer ndim, Integer size, Integer *ld,
                          Integer *ldrem, int *stride_rem,
                          int *stride_loc, Integer *skip)
{
  int i, nstride;
  int ts_loc[MAXDIM], ts_rem[MAXDIM];
  stride_rem[0] = stride_loc[0] = (int)size;
  ts_loc[0] = ts_rem[0] = (int)size;
  ts_loc[0] *= ld[0];
  ts_rem[0] *= ldrem[0];
  nstride = 0;
  if (skip[0] > 1) {
    stride_loc[nstride] = (int)(size*skip[0]);
    stride_rem[nstride] = (int)(size*skip[0]);
    nstride++;
    stride_loc[nstride] = (int)(size*ld[0]);
    stride_rem[nstride] = (int)(size*ldrem[0]);
  } else {
    stride_loc[nstride] = (int)(size*ld[0]);
    stride_rem[nstride] = (int)(size*ldrem[0]);
  }
  for (i=1; i<(int)ndim; i++) {
    if (skip[i] > 1) {
      nstride++;
      stride_loc[nstride] = ts_loc[i-1]*((int)skip[i]);
      stride_rem[nstride] = ts_rem[i-1]*((int)skip[i]);
    }
    nstride++;
    stride_loc[nstride] = ts_loc[i-1];
    stride_rem[nstride] = ts_rem[i-1];
    if (i<ndim-1) {
      ts_loc[i] *= ld[i];
      ts_rem[i] *= ldrem[i];
    }
  }
}

/*\ PUT AN N-DIMENSIONAL PATCH OF STRIDED DATA INTO A GLOBAL ARRAY
\*/
void FATR nga_strided_put_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   Integer *skip,
                   void    *buf,
                   Integer *ld)
{
  /* g_a:    Global Array handle
     lo[]:   Array of lower indices of patch of global array
     hi[]:   Array of upper indices of patch of global array
     skip[]: Array of skips for each dimension
     buf[]:  Local buffer that patch will be copied from
     ld[]:   ndim-1 physical dimensions of local buffer */
  Integer p, np, handle = GA_OFFSET + *g_a;
  Integer idx, size, nstride, p_handle;
  int i, proc, ndim;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_STRIDED_PUT,__FILE__,__LINE__);
#endif

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* check values of skips to make sure they are legitimate */
  for (i = 0; i<ndim; i++) {
    if (skip[i]<1) {
      ga_error("nga_strided_put: Invalid value of skip along coordinate ",i);
    }
  }

  GA_PUSH_NAME("nga_strided_put");

  /* Locate the processors containing some portion of the patch
     specified by lo and hi and return the results in _ga_map,
     GA_proclist, and np. GA_proclist contains the list of processors
     containing some portion of the patch, _ga_map contains the
     lower and upper indices of the portion of the total patch held by
     a given processor, and np contains the total number of processors
     that contain some portion of the patch. */
  if (!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

   p_handle = GA[handle].p_handle;

  /* Loop over all processors containing a portion of patch */
  gaPermuteProcList(np);
  for (idx=0; idx<np; idx++) {
    Integer ldrem[MAXDIM];
    int stride_rem[2*MAXDIM], stride_loc[2*MAXDIM], count[2*MAXDIM];
    Integer idx_buf, *plo, *phi;
    char *pbuf, *prem;

    p = (Integer)ProcListPerm[(int)idx];
    /* find visible portion of patch held by processor p and return
       the result in plo and phi. Also, get actual processor index
       corresponding to p and store the result in proc. */
    gam_GetRangeFromMap(p, ndim, &plo, &phi);
    proc = (int)GA_proclist[(int)p];

    /* Correct ranges to account for skips in original patch. If no
       data is left in patch jump to next processor in loop. */
    if (!gai_correct_strided_patch((Integer)ndim, lo, skip, plo, phi))
      continue;

    /* get pointer prem in remote buffer to location indexed by plo.
       Also get leading physical dimensions of remote buffer in memory
       in ldrem */
    gam_Location(proc, handle, plo, &prem, ldrem);

    /* get pointer in local buffer to point indexed by plo given that
       the corner of the buffer corresponds to the point indexed by lo */
    gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
    pbuf = size*idx_buf + (char*)buf;

    /* Compute number of elements in each stride region and compute the
       number of stride regions. Store the results in count and nstride */
    if (!gai_ComputeCountWithSkip(ndim, plo, phi, skip, count, &nstride))
      continue;

    /* Scale first element in count by element size. The ARMCI_PutS routine
       uses this convention to figure out memory sizes. */
    count[0] *= size;

    /* Calculate strides in memory for remote processor indexed by proc and
       local buffer */ 
    gai_SetStrideWithSkip(ndim, size, ld, ldrem, stride_rem, stride_loc,
                          skip);

    /* BJP */
    if (p_handle != -1) {
      proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
    }
#ifdef PERMUTE_PIDS
    if (GA_Proc_List) proc = GA_inv_Proc_list[proc];
#endif
    ARMCI_PutS(pbuf, stride_loc, prem, stride_rem, count, nstride-1, proc);
  }
  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_STRIDED_PUT,__FILE__,__LINE__);
#endif
}

/*\ GET AN N-DIMENSIONAL PATCH OF STRIDED DATA FROM A GLOBAL ARRAY
\*/
void FATR nga_strided_get_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   Integer *skip,
                   void    *buf,
                   Integer *ld)
{
  /* g_a:    Global Array handle
     lo[]:   Array of lower indices of patch of global array
     hi[]:   Array of upper indices of patch of global array
     skip[]: Array of skips for each dimension
     buf[]:  Local buffer that patch will be copied from
     ld[]:   ndim-1 physical dimensions of local buffer */
  Integer p, np, handle = GA_OFFSET + *g_a;
  Integer idx, size, nstride, p_handle;
  int i, proc, ndim;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_STRIDED_GET,__FILE__,__LINE__);
#endif

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;

  /* check values of skips to make sure they are legitimate */
  for (i = 0; i<ndim; i++) {
    if (skip[i]<1) {
      ga_error("nga_strided_get: Invalid value of skip along coordinate ",i);
    }
  }

  GA_PUSH_NAME("nga_strided_get");

  /* Locate the processors containing some portion of the patch
     specified by lo and hi and return the results in _ga_map,
     GA_proclist, and np. GA_proclist contains the list of processors
     containing some portion of the patch, _ga_map contains the
     lower and upper indices of the portion of the total patch held by
     a given processor, and np contains the total number of processors
     that contain some portion of the patch. */
  if (!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

  /* If default proc group is not world group, translate indices in
     GA_proclist to world group indices */
  p_handle = GA[handle].p_handle;

  /* Loop over all processors containing a portion of patch */
  gaPermuteProcList(np);
  for (idx=0; idx<np; idx++) {
    Integer ldrem[MAXDIM];
    int stride_rem[2*MAXDIM], stride_loc[2*MAXDIM], count[2*MAXDIM];
    Integer idx_buf, *plo, *phi;
    char *pbuf, *prem;

    p = (Integer)ProcListPerm[(int)idx];
    /* find visible portion of patch held by processor p and return
       the result in plo and phi. Also, get actual processor index
       corresponding to p and store the result in proc. */
    gam_GetRangeFromMap(p, ndim, &plo, &phi);
    proc = (int)GA_proclist[(int)p];

    /* Correct ranges to account for skips in original patch. If no
       data is left in patch jump to next processor in loop. */
    if (!gai_correct_strided_patch((Integer)ndim, lo, skip, plo, phi))
      continue;

    /* get pointer prem in remote buffer to location indexed by plo.
       Also get leading physical dimensions of remote buffer in memory
       in ldrem */
    gam_Location(proc, handle, plo, &prem, ldrem);

    /* get pointer in local buffer to point indexed by plo given that
       the corner of the buffer corresponds to the point indexed by lo */
    gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
    pbuf = size*idx_buf + (char*)buf;

    /* Compute number of elements in each stride region and compute the
       number of stride regions. Store the results in count and nstride */
    if (!gai_ComputeCountWithSkip(ndim, plo, phi, skip, count, &nstride))
      continue;

    /* Scale first element in count by element size. The ARMCI_PutS routine
       uses this convention to figure out memory sizes. */
    count[0] *= size;

    /* Calculate strides in memory for remote processor indexed by proc and
       local buffer */ 
    gai_SetStrideWithSkip(ndim, size, ld, ldrem, stride_rem, stride_loc,
                          skip);

    /* BJP */
    if (p_handle != -1) {
      proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
    }
#ifdef PERMUTE_PIDS
    if (GA_Proc_List) proc = GA_inv_Proc_list[proc];
#endif
    ARMCI_GetS(prem, stride_rem, pbuf, stride_loc, count, nstride-1, proc);
  }
  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_STRIDED_GET,__FILE__,__LINE__);
#endif
}

/*\ ACCUMULATE OPERATION FOR AN N-DIMENSIONAL PATCH OF STRIDED DATA OF A
 *  GLOBAL ARRAY
 *                g_a += alpha * strided_patch
\*/
void FATR nga_strided_acc_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   Integer *skip,
                   void    *buf,
                   Integer *ld,
                   void    *alpha)
{
  /* g_a:    Global Array handle
     lo[]:   Array of lower indices of patch of global array
     hi[]:   Array of upper indices of patch of global array
     skip[]: Array of skips for each dimension
     buf[]:  Local buffer that patch will be copied from
     ld[]:   ndim-1 physical dimensions of local buffer
     alpha:  muliplicative scale factor */
  Integer p, np, handle = GA_OFFSET + *g_a;
  Integer idx, size, nstride, type, p_handle;
  int i, optype=-1, proc, ndim;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_STRIDED_ACC,__FILE__,__LINE__);
#endif

  size = GA[handle].elemsize;
  ndim = GA[handle].ndim;
  type = GA[handle].type;

  /* check values of skips to make sure they are legitimate */
  for (i = 0; i<ndim; i++) {
    if (skip[i]<1) {
      ga_error("nga_strided_acc: Invalid value of skip along coordinate ",i);
    }
  }

  GA_PUSH_NAME("nga_strided_acc");

  /* Locate the processors containing some portion of the patch
     specified by lo and hi and return the results in _ga_map,
     GA_proclist, and np. GA_proclist contains the list of processors
     containing some portion of the patch, _ga_map contains the
     lower and upper indices of the portion of the total patch held by
     a given processor, and np contains the total number of processors
     that contain some portion of the patch. */
  if (!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np))
      ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

  /* If default proc group is not world group, translate indices in
     GA_proclist to world group indices */
  p_handle = GA[handle].p_handle;

  if (type == C_DBL) optype = ARMCI_ACC_DBL;
  else if (type == C_FLOAT) optype = ARMCI_ACC_FLT;
  else if (type == C_DCPL) optype = ARMCI_ACC_DCP;
  else if (type == C_INT) optype = ARMCI_ACC_INT;
  else if (type == C_LONG) optype = ARMCI_ACC_LNG;
  else ga_error("nga_strided_acc: type not supported",type);

  /* Loop over all processors containing a portion of patch */
  gaPermuteProcList(np);
  for (idx=0; idx<np; idx++) {
    Integer ldrem[MAXDIM];
    int stride_rem[2*MAXDIM], stride_loc[2*MAXDIM], count[2*MAXDIM];
    Integer idx_buf, *plo, *phi;
    char *pbuf, *prem;

    p = (Integer)ProcListPerm[(int)idx];
    /* find visible portion of patch held by processor p and return
       the result in plo and phi. Also, get actual processor index
       corresponding to p and store the result in proc. */
    gam_GetRangeFromMap(p, ndim, &plo, &phi);
    proc = (int)GA_proclist[(int)p];

    /* Correct ranges to account for skips in original patch. If no
       data is left in patch jump to next processor in loop. */
    if (!gai_correct_strided_patch((Integer)ndim, lo, skip, plo, phi))
      continue;

    /* get pointer prem in remote buffer to location indexed by plo.
       Also get leading physical dimensions of remote buffer in memory
       in ldrem */
    gam_Location(proc, handle, plo, &prem, ldrem);

    /* get pointer in local buffer to point indexed by plo given that
       the corner of the buffer corresponds to the point indexed by lo */
    gam_ComputePatchIndex(ndim, lo, plo, ld, &idx_buf);
    pbuf = size*idx_buf + (char*)buf;

    /* Compute number of elements in each stride region and compute the
       number of stride regions. Store the results in count and nstride */
    if (!gai_ComputeCountWithSkip(ndim, plo, phi, skip, count, &nstride))
      continue;

    /* Scale first element in count by element size. The ARMCI_PutS routine
       uses this convention to figure out memory sizes. */
    count[0] *= size;

    /* Calculate strides in memory for remote processor indexed by proc and
       local buffer */ 
    gai_SetStrideWithSkip(ndim, size, ld, ldrem, stride_rem, stride_loc,
                          skip);

    /* BJP */
    if (p_handle != -1) {
      proc = PGRP_LIST[p_handle].inv_map_proc_list[proc];
    }
#ifdef PERMUTE_PIDS
    if (GA_Proc_List) proc = GA_inv_Proc_list[proc];
#endif
    ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem, count,
               nstride-1, proc);
  }
  GA_POP_NAME;
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_STRIDED_GET,__FILE__,__LINE__);
#endif
}
