#include <stdlib.h>
#ifdef WIN32
#  include <string.h>
#endif

#ifdef NOFORT
#define MEMCPY
#endif

/* macros to ensure ordering of consecutive puts or gets following puts */
#if defined(LAPI)

#   include "lapidefs.h"

#elif defined(CRAY) || defined(QUADRICS)
#ifdef CRAY
#   include <mpp/shmem.h>
#else
#   include <elan/shmem.h>
#endif
    int cmpl_proc;
#   define FENCE_NODE(p) if(cmpl_proc == (p)){\
           shmem_quiet(); cmpl_proc=-1;}
#   define UPDATE_FENCE_STATE(p, op, nissued) if((op)==PUT) cmpl_proc=(p);

#else

#   define FENCE_NODE(p)
#   define UPDATE_FENCE_STATE(p, op, nissued)

#endif


#define THRESH 32
#define THRESH1D 1000
#define ALIGN_SIZE sizeof(double)

/* dcopy2d_u_ uses explicit unrolled loops to depth 4 */
#if defined(AIX)
#    define DCOPY2D	dcopy2d_u
#elif defined(LINUX)  || defined(HPUX64)
#    define DCOPY2D	dcopy2d_n_
#elif !defined(CRAY) && !defined(WIN32)
#    define DCOPY2D	dcopy2d_u_
#else
#    define DCOPY2D	DCOPY2D_U
#endif
void FATR DCOPY2D(int*, int*, void*, int*, void*, int*); 

/***************************** 1-Dimensional copy ************************/

#if defined(QUADRICS)
#      define armci_copy(src,dst,n)           memcpy((dst),(src),(n))
#      define armci_put(src,dst,n,proc)\
              if(proc==armci_me){\
                 armci_copy(src,dst,n);\
              } else { shmem_int_put((int*)(dst),(int*)(src),(int)(n)/sizeof(int),(proc));}
#      define armci_get(src,dst,n,proc) \
              if(proc==armci_me){\
                 armci_copy(src,dst,n);\
              } else { shmem_int_get((int*)(dst),(int*)(src),(int)(n)/sizeof(int),(proc));}

#elif defined(CRAY_T3E)
#      define armci_copy(src,dst,n)\
        if((n)<128 || n%sizeof(long) ) memcpy((dst),(src),(n));\
        else shmem_put((long*)(dst),(long*)(src),(int)(n)/sizeof(long),armci_me)

#      define armci_put(src,dst,n,proc) \
              shmem_put((long*)(dst),(long*)(src),(int)(n)/sizeof(long),(proc))

#      define armci_get(src,dst,n,proc)\
              shmem_get((long*)(dst),(long*)(src),(int)(n)/sizeof(long),(proc))

#elif defined(CRAY)

#      define armci_copy(src,dst,n)           memcpy((dst),(src),(n))

#      define armci_put(src,dst,n,proc)  memcpy((dst),(src),(n))
#      define armci_get(src,dst,n,proc) memcpy((dst),(src),(n)) 

#elif  defined(FUJITSU)

#      include "fujitsu-vpp.h"
#      define armci_copy(src,dst,n)     _MmCopy((char*)(dst), (char*)(src), (n))
#      define armci_put  CopyTo
#      define armci_get  CopyFrom

#elif  defined(LAPI)

#      include <lapi.h>
       extern lapi_handle_t lapi_handle;
#      define armci_copy(src,dst,n)           memcpy((dst),(src),(n))

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

#      define armci_copy(src,dst,n)     memcpy((dst), (src), (n))
#      define armci_get(src,dst,n,p)    armci_copy((src),(dst),(n))
#      define armci_put(src,dst,n,p)    armci_copy((src),(dst),(n))

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
