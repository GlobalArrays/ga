/* $Id: copy.h,v 1.39 2002-07-17 23:07:00 d3h325 Exp $ */
#ifndef _COPY_H_
#define _COPY_H_

#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#  include <string.h>
#endif
#ifdef DECOSF
#include <c_asm.h>
#endif

#if defined(NOFORT) || defined(HITACHI)
#  define MEMCPY
#endif

#if defined(SGI) || defined(FUJITSU) || defined(HPUX) || defined(SOLARIS) || defined (DECOSF) || defined(__ia64__)
#   define PTR_ALIGN
#endif

/* macros to ensure ordering of consecutive puts or gets following puts */
#if defined(LAPI)

#   include "lapidefs.h"

#elif defined(_CRAYMPP) || defined(QUADRICS)
#ifdef CRAY
#   include <mpp/shmem.h>
#else
#   include <unistd.h>
#ifndef ptrdiff_t
#   include <malloc.h>
#endif
#   include <shmem.h>
#endif
#   ifdef ELAN
#     define FENCE_NODE(p) {shmem_quiet(); \
          if(((p)<armci_clus_first)||((p)>armci_clus_last))armci_elan_fence(p);}
#     define UPDATE_FENCE_STATE(p, op, nissued) 
#   else
      int cmpl_proc;
#     ifdef DECOSF
#       define FENCE_NODE(p) if(cmpl_proc == (p)){\
             if(((p)<armci_clus_first)||((p)>armci_clus_last))shmem_quiet();\
             else asm ("mb"); }
#     else
#       define FENCE_NODE(p) if(cmpl_proc == (p)){\
             if(((p)<armci_clus_first)||((p)>armci_clus_last))shmem_quiet(); }
#     endif
#     define UPDATE_FENCE_STATE(p, op, nissued) if((op)==PUT) cmpl_proc=(p);
#   endif
#else

#   define FENCE_NODE(p)
#   define UPDATE_FENCE_STATE(p, op, nissued)

#endif


#define THRESH 32
#define THRESH1D 512 
#define ALIGN_SIZE sizeof(double)

/********* interface to fortran 1D and 2D memory copy functions ***********/
/* dcopy2d_u_ uses explicit unrolled loops to depth 4 */
#if   defined(AIX)
#     define DCOPY2D	dcopy2d_u
#     define DCOPY1D	dcopy1d_u
#elif defined(LINUX) || defined(HPUX64) || defined(DECOSF)
#     define DCOPY2D	dcopy2d_n_
#     define DCOPY1D	dcopy1d_n_
#elif defined(CRAY)  || defined(WIN32) || defined(HITACHI)
#     define DCOPY2D    DCOPY2D_N
#     define DCOPY1D    DCOPY1D_N
#else
#     define DCOPY2D	dcopy2d_u_
#     define DCOPY1D	dcopy1d_u_
#endif
void FATR DCOPY2D(int*, int*, void*, int*, void*, int*); 
void FATR DCOPY1D(void*, void*, int*); 


/***************************** 1-Dimensional copy ************************/

#if defined(QUADRICS)
#include <elan/elan.h>
#      define armci_put(src,dst,n,proc)\
           if(((proc)<=armci_clus_last) && ((proc>= armci_clus_first))){\
              armci_copy(src,dst,n);\
           } else { shmem_putmem((dst),(src),(int)(n),(proc));}
#      define armci_get(src,dst,n,proc) \
           if(((proc)<=armci_clus_last) && ((proc>= armci_clus_first))){\
             armci_copy(src,dst,n);\
           } else { shmem_getmem((dst),(src),(int)(n),(proc));}

#elif defined(CRAY_T3E)
#      define armci_copy_disabled(src,dst,n)\
        if((n)<256 || n%sizeof(long) ) memcpy((dst),(src),(n));\
        else shmem_put((long*)(dst),(long*)(src),(int)(n)/sizeof(long),armci_me)

#      define armci_put(src,dst,n,proc) \
              shmem_int_put((int*)(dst),(int*)(src),(int)(n)/sizeof(int),(proc))

#      define armci_get(src,dst,n,proc)\
              shmem_get((int*)(dst),(int*)(src),(int)(n)/sizeof(int),(proc))

#elif  defined(HITACHI)

        extern void armcill_put(void *src, void *dst, int bytes, int proc);
        extern void armcill_get(void *src, void *dst, int bytes, int proc);

#      define armci_put(src,dst,n,proc) \
            if(((proc)<=armci_clus_last) && ((proc>= armci_clus_first))){\
               armci_copy(src,dst,n);\
            } else { armcill_put((src), (dst),(n),(proc));}

#      define armci_get(src,dst,n,proc)\
            if(((proc)<=armci_clus_last) && ((proc>= armci_clus_first))){\
               armci_copy(src,dst,n);\
            } else { armcill_get((src), (dst),(n),(proc));}

#elif  defined(FUJITSU)

#      include "fujitsu-vpp.h"
#      ifndef __sparc
#         define armci_copy(src,dst,n)  _MmCopy((char*)(dst), (char*)(src), (n))
#      endif
#      define armci_put  CopyTo
#      define armci_get  CopyFrom

#elif  defined(LAPI)

#      include <lapi.h>
       extern lapi_handle_t lapi_handle;

#      define armci_put(src,dst,n,proc)\
              if(proc==armci_me){\
                 armci_copy(src,dst,n);\
              } else {\
              if(LAPI_Put(lapi_handle, (uint)proc, (uint)n, (dst), (src),\
                 NULL, &ack_cntr.cntr, &cmpl_arr[proc].cntr))\
                  ARMCI_Error("LAPI_put failed",0); else; }

       /**** this copy is nonblocking and requires fence to complete!!! ****/
#      define armci_get(src,dst,n,proc) \
              if(proc==armci_me){\
                 armci_copy(src,dst,n);\
              } else {\
              if(LAPI_Get(lapi_handle, (uint)proc, (uint)n, (src), (dst), \
                 NULL, &get_cntr.cntr))\
                 ARMCI_Error("LAPI_Get failed",0);else;}

#else

#      define armci_get(src,dst,n,p)    armci_copy((src),(dst),(n))
#      define armci_put(src,dst,n,p)    armci_copy((src),(dst),(n))

#endif

#ifdef COPY686 
     extern void *armci_asm_memcpy(void *dst, const void *src, size_t n, int tid);
     extern void *armci_asm_memcpy_nofence(void *d,const void *s,size_t n, int id);
#    ifdef SERVER_CONTEXT
#      define armci_copy(src,dst,n) {\
        int _id= (SERVER_CONTEXT)?1:0; armci_asm_memcpy((dst), (src), (n), _id);}
#    else
#      define armci_copy(src,dst,n)  armci_asm_memcpy((dst), (src), (n), 0)
#    endif
#    define armci_copy_nofence(_s,_d,_n) armci_asm_memcpy_nofence((_d),(_s),(_n),0)
#    ifndef MEMCPY
#       define MEMCPY
#    endif
#    define MEM_FENCE armci_asm_mem_fence
     extern void armci_asm_mem_fence();
#endif
                                                 
#if  defined(MEMCPY)  && !defined(armci_copy)
#    define armci_copy(src,dst,n)  memcpy((dst), (src), (n)) 
#endif

#ifndef armci_copy
# ifdef PTR_ALIGN
#   define armci_copy(src,dst,n)     \
     do if( ((n) < THRESH1D)   || ((n)%ALIGN_SIZE) || \
            ((unsigned long)(src)%ALIGN_SIZE) ||\
            ((unsigned long)(dst)%ALIGN_SIZE)) memcpy((dst),(src),(n));\
        else{ int _bytes=(n)/sizeof(double); DCOPY1D((src),(dst),&_bytes);}\
     while (0)
# else
#   define armci_copy(src,dst,n)     \
     do if( ((n) < THRESH1D) || ((n)%ALIGN_SIZE) ) memcpy((dst), (src), (n));\
          else{ int _bytes=(n)/sizeof(double); DCOPY1D((src),(dst),&_bytes);}\
     while (0)
# endif
#endif

/****************************** 2D Copy *******************/


#ifndef MEMCPY
#   define DCopy2D(rows, cols, src_ptr, src_ld, dst_ptr, dst_ld){\
      int rrows, ldd, lds, ccols;\
          rrows = (rows);\
          lds =   (src_ld);\
          ldd =   (dst_ld);\
          ccols = (cols);\
          DCOPY2D(&rrows, &ccols, src_ptr, &lds,dst_ptr,&ldd);\
      }

#else
#   define DCopy2D(rows, cols, src_ptr, src_ld, dst_ptr, dst_ld){\
    int j, nbytes = sizeof(double)* rows;\
    char *ps=src_ptr, *pd=dst_ptr;\
      for (j = 0;  j < cols;  j++){\
          armci_copy(ps, pd, nbytes);\
          ps += sizeof(double)* src_ld;\
          pd += sizeof(double)* dst_ld;\
      }\
    }
#endif


#   define ByteCopy2D(bytes, count, src_ptr, src_stride, dst_ptr,dst_stride){\
    int _j;\
    char *ps=src_ptr, *pd=dst_ptr;\
      for (_j = 0;  _j < count;  _j++){\
          armci_copy(ps, pd, bytes);\
          ps += src_stride;\
          pd += dst_stride;\
      }\
    }

#if defined(FUJITSU) || defined(LAPI2)

#   define armci_put2D(p, bytes,count,src_ptr,src_stride,dst_ptr,dst_stride)\
           CopyPatchTo(src_ptr, src_stride, dst_ptr, dst_stride, count,bytes, p)

#   define armci_get2D(p, bytes, count, src_ptr,src_stride,dst_ptr,dst_stride)\
           CopyPatchFrom(src_ptr, src_stride, dst_ptr, dst_stride,count,bytes,p)

#elif defined(HITACHI) || defined(_ELAN_PUTGET_H)

#ifdef QUADRICS
#if 0
#   define WAIT_FOR_PUTS elan_putWaitAll(elan_base->state,200)
#   define WAIT_FOR_GETS elan_getWaitAll(elan_base->state,200)
#else
#   define WAIT_FOR_PUTS armcill_wait_put()
#   define WAIT_FOR_GETS armcill_wait_get()
#endif
#endif

    extern void armcill_put2D(int proc, int bytes, int count,
                void* src_ptr,int src_stride, void* dst_ptr,int dst_stride);
    extern void armcill_get2D(int proc, int bytes, int count,
                void* src_ptr,int src_stride, void* dst_ptr,int dst_stride);
#   define armci_put2D armcill_put2D
#   define armci_get2D armcill_get2D


#else
#   define armci_put2D(proc,bytes,count,src_ptr,src_stride,dst_ptr,dst_stride){\
    int _j;\
    char *ps=src_ptr, *pd=dst_ptr;\
      for (_j = 0;  _j < count;  _j++){\
          armci_put(ps, pd, bytes, proc);\
          ps += src_stride;\
          pd += dst_stride;\
      }\
    }


#   define armci_get2D(proc,bytes,count,src_ptr,src_stride,dst_ptr,dst_stride){\
    int _j;\
    char *ps=src_ptr, *pd=dst_ptr;\
      for (_j = 0;  _j < count;  _j++){\
          armci_get(ps, pd, bytes, proc);\
          ps += src_stride;\
          pd += dst_stride;\
      }\
    }
#endif

#ifndef MEM_FENCE
#   define MEM_FENCE() 
#endif
#ifndef armci_copy_fence
#   define armci_copy_fence armci_copy
#endif

#endif
