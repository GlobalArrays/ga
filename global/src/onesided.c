/* $Id: onesided.c,v 1.10 2001-10-01 16:57:45 d3g293 Exp $ */
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

int ProcListPerm[MAX_NPROC];            /* permuted list of processes */


/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

char *fence_array;
static int GA_fence_set=0;
Integer *_ga_map;       /* used in get/put/acc */

extern void ga_sort_scat(Integer*,Void*,Integer*,Integer*,Integer*, Integer);
extern void ga_sort_gath_(Integer*, Integer*, Integer*, Integer*);

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
    ld[_d] = _hi[_d] - _lo[_d] + 1 + 2*GA[handle].width[0];                    \
    _factor *= ld[_d];                                                         \
  }                                                                            \
  _offset += subscript[_last] * _factor;                                       \
  *(ptr_loc) = GA[handle].ptr[proc] + _offset*GA[handle].elemsize;             \
}


/*\ SYNCHRONIZE ALL THE PROCESSES
\*/
void FATR ga_sync_()
{
extern int GA_fence_set;
#ifdef CHECK_MA
Integer status;
#endif

       ARMCI_AllFence();
       ga_msg_sync_();
       if(GA_fence_set)bzero(fence_array,(int)GAnproc);
       GA_fence_set=0;
#ifdef CHECK_MA
       status = MA_verify_allocator_stuff();
#endif
}


/*\ wait until requests intiated by calling process are completed
\*/
void FATR ga_fence_()
{
    int proc;
    if(GA_fence_set<1)ga_error("ga_fence: fence not initialized",0);
    GA_fence_set--;
    for(proc=0;proc<GAnproc;proc++)if(fence_array[proc])ARMCI_Fence(proc);
    bzero(fence_array,(int)GAnproc);
}

/*\ initialize tracing of request completion
\*/
void FATR ga_init_fence_()
{
    GA_fence_set++;
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
        Integer handle, idx, elems = (bytes+sizeof(double)-1)/sizeof(double)+1; 
        if(MA_push_get(MT_DBL, elems, "GA malloc temp", &handle, &idx)){
            MA_get_pointer(handle, &ptr);
            *((Integer*)ptr)= handle;
            ptr = ((double*)ptr)+ 1;  /*needs sizeof(double)>=sizeof(Integer) */
        }else
            ptr=NULL;
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
        Integer handle= *( (Integer*) (-1 + (double*)ptr));
        if(!MA_pop_stack(handle)) ga_error("gai_free:MA_pop_stack failed",0);
    }
}

        


#define gaShmemLocation(proc, g_a, _i, _j, ptr_loc, _pld)                      \
{                                                                              \
Integer _ilo, _ihi, _jlo, _jhi, offset, proc_place, g_handle=(g_a)+GA_OFFSET;  \
Integer _lo[2], _hi[2];                                                        \
Integer _iw = GA[g_handle].width[0];                                           \
Integer _jw = GA[g_handle].width[1];                                           \
                                                                               \
      ga_ownsM(g_handle, (proc),_lo,_hi);                                      \
      _ilo = _lo[0]; _ihi=_hi[0];                                              \
      _jlo = _lo[1]; _jhi=_hi[1];                                              \
                                                                               \
      if((_i)<_ilo || (_i)>_ihi || (_j)<_jlo || (_j)>_jhi){                    \
       sprintf(err_string,"%s:p=%ld invalid i/j (%ld,%ld)><(%ld:%ld,%ld:%ld)", \
                 "gaShmemLocation", proc, (_i),(_j), _ilo, _ihi, _jlo, _jhi);  \
          ga_error(err_string, g_a );                                          \
      }                                                                        \
      offset = ((_i)-_ilo+_iw) + (_ihi-_ilo+1+2*_iw)*((_j)-_jlo+_jw);          \
                                                                               \
      /* find location of the proc in current cluster pointer array */         \
      proc_place =  proc;                                                      \
      *(ptr_loc) = GA[g_handle].ptr[proc_place] +                              \
                   offset*GAsizeofM(GA[g_handle].type);                        \
      *(_pld) = _ihi-_ilo+1+2*_iw;                                             \
}


#define gaCheckSubscriptM(subscr, lo, hi, ndim)                             \
{                                                                              \
Integer _d;                                                                    \
   for(_d=0; _d<  ndim; _d++)                                                  \
      if( subscr[_d]<  lo[_d] ||  subscr[_d]>  hi[_d]){                  \
        sprintf(err_string,"check subscript failed:%ld not in (%ld:%ld) dim=", \
                  subscr[_d],  lo[_d],  hi[_d]);                            \
          ga_error(err_string, _d);                                            \
      }\
}

/*\ Return pointer (ptr_loc) to location in memory of element with subscripts
 *  (subscript). Also return physical dimensions of array in memory in ld.
\*/
#define gam_Location(proc, g_handle,  subscript, ptr_loc, ld)                  \
{                                                                              \
Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;               \
Integer _lo[MAXDIM], _hi[MAXDIM];                                              \
                                                                               \
      ga_ownsM(g_handle, proc, _lo, _hi);                                      \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);               \
      if(_last==0) ld[0]=_hi[0]- _lo[0]+1+2*GA[g_handle].width[0];             \
      for(_d=0; _d < _last; _d++)            {                                 \
          _w = GA[g_handle].width[_d];                                         \
          _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                     \
          ld[_d] = _hi[_d] - _lo[_d]+1+2*_w;                                   \
          _factor *= ld[_d];                                                   \
      }                                                                        \
      _offset += (subscript[_last]-_lo[_last]+GA[g_handle].width[_last])       \
               * _factor;                                                      \
      *(ptr_loc) =  GA[g_handle].ptr[proc]+_offset*GA[g_handle].elemsize;      \
}


/*\ Just return pointer (ptr_loc) to location in memory of element with
 *  subscripts (subscript).
\*/
#define gam_Loc_ptr(proc, g_handle,  subscript, ptr_loc)                      \
{                                                                             \
Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;              \
Integer _lo[MAXDIM], _hi[MAXDIM];                                             \
                                                                              \
      ga_ownsM(g_handle, proc, _lo, _hi);                                     \
      gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);              \
      for(_d=0; _d < _last; _d++)            {                                \
          _w = GA[g_handle].width[_d];                                        \
          _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                    \
          _factor *= _hi[_d] - _lo[_d]+1+2*_w;                                \
      }                                                                       \
      _offset += (subscript[_last]-_lo[_last]+GA[g_handle].width[_last])      \
               * _factor;                                                     \
      *(ptr_loc) =  GA[g_handle].ptr[proc]+_offset*GA[g_handle].elemsize;     \
}

#define ga_check_regionM(g_a, ilo, ihi, jlo, jhi, string){                     \
   if (*(ilo) <= 0 || *(ihi) > GA[GA_OFFSET + *(g_a)].dims[0] ||               \
       *(jlo) <= 0 || *(jhi) > GA[GA_OFFSET + *(g_a)].dims[1] ||               \
       *(ihi) < *(ilo) ||  *(jhi) < *(jlo)){                                   \
       sprintf(err_string,"%s:req(%ld:%ld,%ld:%ld) out of range (1:%ld,1:%ld)",\
               string, *(ilo), *(ihi), *(jlo), *(jhi),                         \
               GA[GA_OFFSET + *(g_a)].dims[0], GA[GA_OFFSET + *(g_a)].dims[1]);\
       ga_error(err_string, *(g_a));                                           \
   }                                                                           \
}



#define gam_GetRangeFromMap0(p, ndim, plo, phi, proc){\
Integer   _mloc = p* (ndim *2 +1);\
          *plo  = (Integer*)_ga_map + _mloc;\
          *phi  = *plo + ndim;\
          *proc = *phi[ndim]; /* proc is immediately after hi */\
}

#define gam_GetRangeFromMap(p, ndim, plo, phi){\
Integer   _mloc = p* ndim *2;\
          *plo  = (Integer*)_ga_map + _mloc;\
          *phi  = *plo + ndim;\
}

/* Count total number of elmenents in array based on values of ndim, lo,
   and hi */
#define gam_CountElems(ndim, lo, hi, pelems){\
int _d;\
     for(_d=0,*pelems=1; _d< ndim;_d++)  *pelems *= hi[_d]-lo[_d]+1;\
}

#define gam_ComputeCount(ndim, lo, hi, count){\
int _d;\
          for(_d=0; _d< ndim;_d++) count[_d] = (int)(hi[_d]-lo[_d])+1;\
}

/* compute index of point subscripted by plo relative to point
   subscripted by lo, for a block with dimensions dims */
#define gam_ComputePatchIndex(ndim, lo, plo, dims, pidx){\
Integer _d, _factor;\
          *pidx = plo[0] -lo[0];\
          for(_d= 0,_factor=1; _d< ndim -1; _d++){\
             _factor *= dims[_d];\
             *pidx += _factor * (plo[_d+1]-lo[_d+1]);\
          }\
}

#define ga_RegionError(ndim, lo, hi, val){\
int _d, _l;\
   char *str= "cannot locate region: ";\
   sprintf(err_string, str); \
   _l = strlen(str);\
   for(_d=0; _d< ndim; _d++){ \
        sprintf(err_string+_l, "%ld:%ld ",lo[_d],hi[_d]);\
        _l=strlen(err_string);\
   }\
   ga_error(err_string, val);\
}


#define gam_setstride(ndim, size, ld, ldrem, stride_rem, stride_loc){\
int _i;\
          stride_rem[0]= stride_loc[0] = (int)size;\
          for(_i=0;_i<ndim;_i++){\
                stride_rem[_i] *=  (int)ldrem[_i];\
                stride_loc[_i] *=  (int)ld[_i];\
                if(_i<ndim-1){\
                     stride_rem[_i+1] = stride_rem[_i]; \
                     stride_loc[_i+1] = stride_loc[_i];\
                }\
          }\
}

/*\ PUT A 2-DIMENSIONAL PATCH OF DATA INTO A GLOBAL ARRAY
\*/
void FATR nga_put_(Integer *g_a, 
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
Integer  p, np, handle=GA_OFFSET + *g_a;
Integer  idx, elems, size;
int proc, ndim;

      GA_PUSH_NAME("nga_put");

      if(!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np ))
          ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

      size = GA[handle].elemsize;
      ndim = GA[handle].ndim;

      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.puttot += (double)size*elems;
      GAstat.numput++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
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

          if(GA_fence_set)fence_array[proc]=1;
#ifdef PERMUTE_PIDS
    if(GA_Proc_list){
       /* fprintf(stderr,"permuted %d %d\n",proc,GA_inv_Proc_list[proc]);*/
       proc = GA_inv_Proc_list[proc];
    }
#endif

          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.putloc += (double)size*elems;
          }
          ARMCI_PutS(pbuf, stride_loc, prem, stride_rem, count, ndim -1, proc);

      }

      GA_POP_NAME;
}



void FATR  ga_put_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_put_(g_a, lo, hi, buf, ld);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_PUT; 
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}



/*\ GET AN N-DIMENSIONAL PATCH OF DATA FROM A GLOBAL ARRAY
\*/
void FATR nga_get_(Integer *g_a,
                   Integer *lo,
                   Integer *hi,
                   void    *buf,
                   Integer *ld)
{
Integer  p, np, handle=GA_OFFSET + *g_a;
Integer  idx, elems, size;
int proc, ndim;

      GA_PUSH_NAME("nga_get");

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

      size = GA[handle].elemsize;
      ndim = GA[handle].ndim;

      /* get total size of patch */
      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.gettot += (double)size*elems;
      GAstat.numget++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
          /* Find portion of patch held by processor p and return
             the result in plo and phi. Also get actual processor
             index corresponding to p and store the result in proc. */
          gam_GetRangeFromMap(p, ndim, &plo, &phi);
          proc = (int)GA_proclist[p];

          /* get pointer prem to location indexed by plo. Also get
             leading physical dimensions in memory in ldrem */
          gam_Location(proc,handle, plo, &prem, ldrem);

          /* find the right spot in the user buffer for the point
             subscripted by plo given that the corner of the user
             buffer is subscripted by lo */
          gam_ComputePatchIndex(ndim,lo, plo, ld, &idx_buf);
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

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif
          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.getloc += (double)size*elems;
          }
          ARMCI_GetS(prem, stride_rem, pbuf, stride_loc, count, ndim -1, proc);

      }

      GA_POP_NAME;
}


void FATR  ga_get_(g_a, ilo, ihi, jlo, jhi, buf, ld)
   Integer  *g_a,  *ilo, *ihi, *jlo, *jhi,  *ld;
   Void  *buf;
{
Integer lo[2], hi[2];

#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_get_(g_a, lo, hi, buf, ld);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_GET;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
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
Integer  p, np, handle=GA_OFFSET + *g_a;
Integer  idx, elems, size, type;
int optype, proc, ndim;

      GA_PUSH_NAME("nga_acc");

      if(!nga_locate_region_(g_a, lo, hi, _ga_map, GA_proclist, &np ))
          ga_RegionError(ga_ndim_(g_a), lo, hi, *g_a);

      size = GA[handle].elemsize;
      type = GA[handle].type;
      ndim = GA[handle].ndim;

      if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
      else if(type==MT_F_REAL) optype= ARMCI_ACC_FLT;
      else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
      else if(size==sizeof(int))optype= ARMCI_ACC_INT;
      else if(size==sizeof(long))optype= ARMCI_ACC_LNG;
      else ga_error("type not supported",type);

      gam_CountElems(ndim, lo, hi, &elems);
      GAbytes.acctot += (double)size*elems;
      GAstat.numacc++;

      gaPermuteProcList(np);
      for(idx=0; idx< np; idx++){
          Integer ldrem[MAXDIM];
          int stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
          Integer idx_buf, *plo, *phi;
          char *pbuf, *prem;

          p = (Integer)ProcListPerm[idx];
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

          if(GA_fence_set)fence_array[proc]=1;

#ifdef PERMUTE_PIDS
          if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif
          if(proc == GAme){
             gam_CountElems(ndim, plo, phi, &elems);
             GAbytes.accloc += (double)size*elems;
          }

          ARMCI_AccS(optype, alpha, pbuf, stride_loc, prem, stride_rem, count, ndim-1, proc);

      }

      GA_POP_NAME;
}



void FATR  ga_acc_(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *ld;
   void *buf, *alpha;
{
Integer lo[2], hi[2];
#ifdef GA_TRACE
   trace_stime_();
#endif

   lo[0]=*ilo;
   lo[1]=*jlo;
   hi[0]=*ihi;
   hi[1]=*jhi;
   nga_acc_(g_a, lo, hi, buf, ld, alpha);

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ACC;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif
}


void nga_access_ptr(Integer* g_a, Integer lo[], Integer hi[],
                      void* ptr, Integer ld[])

{
char *lptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i;

   GA_PUSH_NAME("nga_access_ptr");
   if(!nga_locate_(g_a,lo,&ow))ga_error("locate top failed",0);
   if(ow != GAme) ga_error("cannot access top of the patch",ow);
   if(!nga_locate_(g_a,hi, &ow))ga_error("locate bottom failed",0);
   if(ow != GAme) ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }

   gam_Location(ow,handle, lo, &lptr, ld);
   *(char**)ptr = lptr; 
   GA_POP_NAME;
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


/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR nga_access_(Integer* g_a, Integer lo[], Integer hi[],
                      Integer* index, Integer ld[])
{
char     *ptr;
Integer  handle = GA_OFFSET + *g_a;
Integer  ow,i;
unsigned long    elemsize;
unsigned long    lref, lptr;

   GA_PUSH_NAME("nga_access");
   if(!nga_locate_(g_a,lo,&ow))ga_error("locate top failed",0);
   if(ow != GAme) ga_error("cannot access top of the patch",ow);
   if(!nga_locate_(g_a,hi, &ow))ga_error("locate bottom failed",0);
   if(ow != GAme) ga_error("cannot access bottom of the patch",ow);

   for (i=0; i<GA[handle].ndim; i++)
       if(lo[i]>hi[i]) {
           ga_RegionError(GA[handle].ndim, lo, hi, *g_a);
       }


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
   switch (GA[handle].type){
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
   switch (GA[handle].type){
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


/*\ PROVIDE ACCESS TO A PATCH OF A GLOBAL ARRAY
\*/
void FATR ga_access_(g_a, ilo, ihi, jlo, jhi, index, ld)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *index, *ld;
{
Integer lo[2], hi[2],ndim=ga_ndim_(g_a);

     if(ndim != 2) 
        ga_error("ga_access: 2D API cannot be used for array dimension",ndim);

     lo[0]=*ilo;
     lo[1]=*jlo;
     hi[0]=*ihi;
     hi[1]=*jhi;
     nga_access_(g_a,lo,hi,index,ld);
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
armci_giov_t desc;
register Integer k, offset;
int rc;

  if (nv < 1) return;

  GA_PUSH_NAME("ga_scatter_local");

  ga_distribution_(&g_a, &proc, &ilo, &ihi, &jlo, &jhi);

  /* get address of the first element owned by proc */
  gaShmemLocation(proc, g_a, ilo, jlo, &ptr_ref, &ldp);

  type = GA[GA_OFFSET + g_a].type;
  item_size = GAsizeofM(type);

  ptr_src = gai_malloc((int)nv*2*sizeof(void*));
  if(ptr_src==NULL)ga_error("gai_malloc failed",nv);
  else ptr_dst=ptr_src+ nv;

  for(k=0; k< nv; k++){
     if(i[k] < ilo || i[k] > ihi  || j[k] < jlo || j[k] > jhi){
       sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)>< [%ld:%ld,%ld:%ld]",
               (int)proc, i[k], j[k], ilo, ihi, jlo, jhi); 
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

  if(alpha != NULL) {
    int optype;
    if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
    else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
    else if(item_size==sizeof(int))optype= ARMCI_ACC_INT;
    else if(item_size==sizeof(long))optype= ARMCI_ACC_LNG;
    else if(type==MT_F_REAL)optype= ARMCI_ACC_FLT;  
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
    Integer pindex, phandle, item_size;
    Integer proc, type=GA[GA_OFFSET + *g_a].type;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp;
    char **ptr_ref;
    
    if (*nv < 1) return;
    
    ga_check_handleM(g_a, "ga_scatter");
    GA_PUSH_NAME("ga_scatter");
    GAstat.numsca++;
    
    if(!MA_push_get(MT_F_INT,*nv, "ga_scatter--p", &phandle, &pindex))
        ga_error("MA alloc failed ", *g_a);

    /* allocate temp memory */
    buf1 = gai_malloc((int) GAnproc *4 * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
    
    /* initialize the counters and nelem */
    for(kk=0; kk<GAnproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
            ga_error(err_string,*g_a);
        }
        nelem[INT_MB[pindex+k]]++;
    }

    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    
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
        ga_distribution_(g_a, &aproc[kk],
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));
    }
    
    /* determine limit for message size --  v,i, & j will travel together */
    item_size = GAsizeofM(type);
    GAbytes.scatot += (double)item_size**nv ;
    GAbytes.scaloc += (double)item_size* nelem[INT_MB[pindex+GAme]];

    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        Integer this_count;
        proc = INT_MB[pindex+k]; this_count = count[proc]; count[proc]++;
        proc = map[proc];
        ptr_src[proc][this_count] = ((char*)v) + k * item_size;
        if(i[k] < ilo[proc] || i[k] > ihi[proc]  ||
           j[k] < jlo[proc] || j[k] > jhi[proc]){
          sprintf(err_string,"proc=%d invalid i/j=(%ld,%ld)><[%ld:%ld,%ld:%ld]",
             (int)proc, i[k], j[k], ilo[proc], ihi[proc], jlo[proc], jhi[proc]);
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
        
        rc = ARMCI_PutV(&desc, 1, (int)aproc[k]);
        if(rc) ga_error("scatter failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);

    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}
      


/*\ SCATTER OPERATION elements of v into the global array
\*/
void FATR  ga_scatter_acc_(g_a, v, i, j, nv, alpha)
     Integer *g_a, *nv, *i, *j;
     Void *v, *alpha;
{
register Integer k;
Integer pindex, phandle, item_size;
Integer first, nelem, proc, type=GA[GA_OFFSET + *g_a].type;

  if (*nv < 1) return;

  ga_check_handleM(g_a, "ga_scatter_acc");
  GA_PUSH_NAME("ga_scatter_acc");
  GAstat.numsca++;

  if(!MA_push_get(MT_F_INT,*nv, "ga_scatter_acc--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* determine limit for message size --  v,i, & j will travel together */
  item_size = GAsizeofM(type);
  GAbytes.scatot += (double)item_size**nv ;

  /* Sort the entries by processor */
  ga_sort_scat(nv, v, i, j, INT_MB+pindex, type );

  /* go through the list again executing scatter for each processor */

  first = 0;
  do {
      proc  = INT_MB[pindex+first];
      nelem = 0;

      /* count entries for proc from "first" to last */
      for(k=first; k< *nv; k++){
        if(proc == INT_MB[pindex+k]) nelem++;
        else break;
      }

      if(proc == GAme){
             GAbytes.scaloc += (double)item_size* nelem ;
      }

      ga_scatter_acc_local(*g_a, ((char*)v)+item_size*first, i+first,
                           j+first, nelem, alpha, proc);
      first += nelem;

  }while (first< *nv);

  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);

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
Integer pindex, phandle;
extern void ga_sort_permutation();

  if (*nv < 1) return;

  if(!MA_push_get(MT_F_INT,*nv, "ga_sort_permut--p", &phandle, &pindex))
            ga_error("MA alloc failed ", *g_a);

  /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
  for(k=0; k< *nv; k++) if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
         sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
         ga_error(err_string,*g_a);
  }

  /* Sort the entries by processor */
  ga_sort_permutation(nv, index, INT_MB+pindex);
  if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
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
    Integer *proc, phandle;

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

    if(!MA_push_stack(MT_F_INT,*nv,"ga_gat-p",&phandle)) 
        ga_error("MAfailed",*g_a);
    if(!MA_get_pointer(phandle, &proc)) ga_error("MA pointer failed ", *g_a);

    ndim = GA[handle].ndim;
    type = GA[handle].type;
    item_size = GA[handle].elemsize;
    *totbytes += (double)item_size**nv;

    /* allocate temp memory */
    buf1 = gai_malloc((int) GAnproc * 4 * (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
    
    /* initialize the counters and nelem */
    for(k=0; k<GAnproc; k++) count[k] = 0; 
    for(k=0; k<GAnproc; k++) nelem[k] = 0;

    /* get the process id that the element should go and count the
     * number of elements for each process
     */
    for(k=0; k<*nv; k++) {
        if(!nga_locate_(g_a, subscript+k*ndim, proc+k)) {
            gai_print_subscript("invalid subscript",ndim, subscript+k*ndim,"\n");
            ga_error("failed -element:",k);
        }
        nelem[proc[k]]++;
    }

    /* find the active processes (with which transfer data) */
    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
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
            ptr_dst[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_src[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }
        
        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            rc=ARMCI_GetV(&desc, 1, (int)aproc[k]);
            if(rc) ga_error("gather failed in armci",rc);
        }
        break;
      case SCATTER:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            ptr_src[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_dst[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;

            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            if(GA_fence_set) fence_array[aproc[k]]=1;
            
            rc=ARMCI_PutV(&desc, 1, (int)aproc[k]);
            if(rc) ga_error("scatter failed in armci",rc);
        }
        break;
      case SCATTER_ACC:
        /* go through all the elements
         * for process 0: ptr_src[0][0, 1, ...] = v + offset0...
         *                ptr_dst[0][0, 1, ...] = subscript + offset0...
         * for process 1: ptr_src[1][...] ...
         *                ptr_dst[1][...] ...
         */
        for(k=0; k<(*nv); k++){
            ptr_src[map[proc[k]]][count[proc[k]]] = ((char*)v) + k * item_size;
            gam_Loc_ptr(proc[k], handle,  (subscript+k*ndim),
                        ptr_dst[map[proc[k]]]+count[proc[k]]);
            count[proc[k]]++;
        }

        /* source and destination pointers are ready for all processes */
        for(k=0; k<naproc; k++) {
            int rc;
            
            desc.bytes = (int)item_size;
            desc.src_ptr_array = ptr_src[k];
            desc.dst_ptr_array = ptr_dst[k];
            desc.ptr_array_len = (int)nelem[aproc[k]];
            
            if(GA_fence_set) fence_array[aproc[k]]=1;
            
            if(alpha != NULL) {
                int optype;
                if(type==MT_F_DBL) optype= ARMCI_ACC_DBL;
                else if(type==MT_F_DCPL)optype= ARMCI_ACC_DCP;
                else if(item_size==sizeof(int))optype= ARMCI_ACC_INT;
                else if(item_size==sizeof(long))optype= ARMCI_ACC_LNG;
                else if(type==MT_F_REAL)optype= ARMCI_ACC_FLT; 
                else ga_error("type not supported",type);
                rc= ARMCI_AccV(optype, alpha, &desc, 1, (int)aproc[k]);
            }
            if(rc) ga_error("scatter_acc failed in armci",rc);
        }
        break;        
      default: ga_error("operation not supported",op);
    }

    gai_free(buf2); gai_free(buf1);
    
    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}



/*\ GATHER OPERATION elements from the global array into v
\*/
void FATR nga_gather_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_gather");
  GA_PUSH_NAME("nga_gather");
  GAstat.numgat++;

  gai_gatscat(GATHER,g_a,v,subscript,nv,&GAbytes.gattot,&GAbytes.gatloc, NULL);

  GA_POP_NAME;
}


void FATR nga_scatter_(Integer *g_a, void* v, Integer subscript[], Integer *nv)
{

  if (*nv < 1) return;
  ga_check_handleM(g_a, "nga_scatter");
  GA_PUSH_NAME("nga_scatter");
  GAstat.numsca++;

  gai_gatscat(SCATTER,g_a,v,subscript,nv,&GAbytes.scatot,&GAbytes.scaloc, NULL);

  GA_POP_NAME;
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
    register Integer k;
    Integer kk;
    Integer pindex, phandle, item_size;
    Integer proc;

    Integer *aproc, naproc; /* active processes and numbers */
    Integer *map;           /* map the active processes to allocated space */
    char *buf1, *buf2;
    
    Integer *count;   /* counters for each process */
    Integer *nelem;   /* number of elements for each process */
    /* source and destination pointers for each process */
    void ***ptr_src, ***ptr_dst; 
    void **ptr_org; /* the entire pointer array */
    armci_giov_t desc;
    Integer *ilo, *ihi, *jlo, *jhi, *ldp;
    char **ptr_ref;
    
    if (*nv < 1) return;

    ga_check_handleM(g_a, "ga_gather");
    GA_PUSH_NAME("ga_gather");
    GAstat.numgat++;

    if(!MA_push_get(MT_F_INT, *nv, "ga_gather--p", &phandle, &pindex))
        ga_error("MA failed ", *g_a);

    /* allocate temp memory */
    buf1 = gai_malloc((int)GAnproc *4 *  (sizeof(Integer)));
    if(buf1 == NULL) ga_error("gai_malloc failed", 3*GAnproc);
    
    count = (Integer *)buf1;
    nelem = (Integer *)(buf1 + GAnproc * sizeof(Integer));
    aproc = (Integer *)(buf1 + 2 * GAnproc * sizeof(Integer));
    map = (Integer *)(buf1 + 3 * GAnproc * sizeof(Integer));
   
    /* initialize the counters and nelem */
    for(kk=0; kk<GAnproc; kk++) {
        count[kk] = 0; nelem[kk] = 0;
    }
    
    /* find proc that owns the (i,j) element; store it in temp: INT_MB[] */
    for(k=0; k< *nv; k++) {
        if(! ga_locate_(g_a, i+k, j+k, INT_MB+pindex+k)){
            sprintf(err_string,"invalid i/j=(%ld,%ld)", i[k], j[k]);
            ga_error(err_string, *g_a);
        }
        nelem[INT_MB[pindex+k]]++;
    }

    naproc = 0;
    for(k=0; k<GAnproc; k++) if(nelem[k] > 0) {
        aproc[naproc] = k;
        map[k] = naproc;
        naproc ++;
    }
    
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
        ga_distribution_(g_a, &aproc[kk],
                         &(ilo[kk]), &(ihi[kk]), &(jlo[kk]), &(jhi[kk]));
        
        /* get address of the first element owned by proc */
        gaShmemLocation(aproc[kk], *g_a, ilo[kk], jlo[kk], &(ptr_ref[kk]),
                        &(ldp[kk]));
    }
    
    item_size = GA[GA_OFFSET + *g_a].elemsize;
    GAbytes.gattot += (double)item_size**nv;
    GAbytes.gatloc += (double)item_size * nelem[INT_MB[pindex+GAme]];

    ptr_src[0] = ptr_org; ptr_dst[0] = ptr_org + (*nv);
    for(k=1; k<naproc; k++) {
        ptr_src[k] = ptr_src[k-1] + nelem[aproc[k-1]];
        ptr_dst[k] = ptr_dst[k-1] + nelem[aproc[k-1]];
    }
    
    for(k=0; k<(*nv); k++){
        Integer this_count;
        proc = INT_MB[pindex+k]; 
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
        
        rc=ARMCI_GetV(&desc, 1, (int)aproc[k]);
        if(rc) ga_error("gather failed in armci",rc);
    }

    gai_free(buf2);
    gai_free(buf1);

    if(! MA_pop_stack(phandle)) ga_error(" pop stack failed!",phandle);
    GA_POP_NAME;
}
      
           



/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR nga_read_inc_(Integer* g_a, Integer* subscript, Integer* inc)
{
Integer *ptr, ldp[MAXDIM], value, proc, handle=GA_OFFSET+*g_a;
int optype;

    ga_check_handleM(g_a, "nga_read_inc");
    GA_PUSH_NAME("ga_read_inc");

    if(GA[handle].type!=MT_F_INT) ga_error("type must be integer",*g_a);

    GAstat.numrdi++;
    GAbytes.rditot += (double)sizeof(Integer);

    /* find out who owns it */
    nga_locate_(g_a, subscript, &proc);

    /* get an address of the g_a(subscript) element */
    gam_Location(proc, handle,  subscript, (char**)&ptr, ldp);

#   ifdef EXT_INT
      optype = ARMCI_FETCH_AND_ADD_LONG;
#   else
      optype = ARMCI_FETCH_AND_ADD;
#   endif

    if(GAme == proc)GAbytes.rdiloc += (double)sizeof(Integer);

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) proc = GA_inv_Proc_list[proc];
#endif

    ARMCI_Rmw(optype, (int*)&value, (int*)ptr, (int)*inc, (int)proc);

   GA_POP_NAME;
   return(value);
}




/*\ READ AND INCREMENT AN ELEMENT OF A GLOBAL ARRAY
\*/
Integer FATR ga_read_inc_(g_a, i, j, inc)
        Integer *g_a, *i, *j, *inc;
{
Integer  value, subscript[2];

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

   return(value);
}

/*\ UPDATE GHOST CELLS OF GLOBAL ARRAY
\*/
void FATR ga_update_ghosts_(Integer *g_a)
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

  ga_sync_();
  GA_PUSH_NAME("ga_update_ghosts");

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
          fprintf(stderr,"\n Value of inx is %d\n\n",inx);
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
              ndim - 1, proc_rem);
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
          fprintf(stderr,"\n Value of inx is %d\n\n",inx);
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
              ndim - 1, proc_rem);
        }
      }
    }
    /* synchronize all processors and update increment array */
    ga_sync_();
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
  Integer lo_rem[MAXDIM], hi_rem[MAXDIM];
  Integer tlo_rem[MAXDIM], thi_rem[MAXDIM];
  Integer plo_rem[MAXDIM], phi_rem[MAXDIM];
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
                GAme,ipx,ipx,GA[handle].mapc[ipx]);
          }
          return FALSE;
        }
      } else {
        if (GA[handle].dims[idx]-GA[handle].mapc[ipx]+1<width[idx]) {
          if (DEBUG) {
            fprintf(stderr,"ERR2 p[%d] dims[%d] = %d  ipx = %d mapc[%d] = %d\n",
                GAme,idx,GA[handle].dims[idx],
                ipx,ipx,GA[handle].mapc[ipx]);
          }
          return FALSE;
        }
      }
      ipx++;
    }
  }

  ga_sync_();
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
    if (DEBUG) {
      fprintf(stderr,"\n");
      for (idx=0; idx<ndim; idx++) {
        fprintf(stderr,"p[%d] ipx = %d  mask[%d] = %d\n",
            GAme,ipx,idx,mask[idx]);
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
            GAme,ipx,idx,tlo_loc[idx],idx,thi_loc[idx]);
        fprintf(stderr,"p[%d] ipx = %d tlo_rem[%d] = %d thi_rem[%d] = %d\n",
            GAme,ipx,idx,tlo_rem[idx],idx,thi_rem[idx]);
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
        phi_rem[idx] = phi_loc[idx];
      } else if (mask[idx] == -1) {
        plo_loc[idx] = width[idx];
        phi_loc[idx] = 2*width[idx]-1;
        plo_rem[idx] = thi_rem[idx]-tlo_rem[idx]+width[idx]+1;
        phi_rem[idx] = thi_rem[idx]-tlo_rem[idx]+2*width[idx];
      } else if (mask[idx] == 1) {
        plo_loc[idx] = hi_loc[idx]-lo_loc[idx]+1;
        phi_loc[idx] = hi_loc[idx]-lo_loc[idx]+width[idx];
        plo_rem[idx] = 0;
        phi_rem[idx] = width[idx]-1;
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
          ndim - 1, proc_rem);
  }

  ga_sync_();
  GA_POP_NAME;
  return TRUE;
}
