/* $Id: copy.h,v 1.29 2001-05-31 18:44:11 d3h325 Exp $ */
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

#ifdef NOFORT
#  define MEMCPY
#endif

#if defined(SGI) || defined(FUJITSU) || defined(HPUX) || defined(SOLARIS)
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
#elif defined(CRAY)  || defined(WIN32)
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
              shmem_put((long*)(dst),(long*)(src),(int)(n)/sizeof(long),(proc))

#      define armci_get(src,dst,n,proc)\
              shmem_get((long*)(dst),(long*)(src),(int)(n)/sizeof(long),(proc))

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
     extern void *armci_asm_memcpy(void *dst, const void *src, size_t n);
#    define armci_copy(src,dst,n)  armci_asm_memcpy((dst), (src), (n)) 
#    ifndef MEMCPY
#       define MEMCPY
#    endif
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

#endif
