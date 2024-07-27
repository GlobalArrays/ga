/* $Id: copy.h,v 1.86.2.6 2007-08-29 17:32:32 manoj Exp $ */
#ifndef _COPY_H_
#define _COPY_H_

#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif

#if 1 || defined(CRAY_T3E) || defined(CRAY_XT)
#  define MEMCPY
#endif

#ifndef EXTERN
#   define EXTERN extern
#endif
 
#ifdef NEC
#  define memcpy1 _VEC_memcpy
#  define armci_copy1(src,dst,n) _VEC_memcpy((dst),(src),(n))
   EXTERN long long _armci_vec_sync_flag;
#endif

#if defined(FUJITSU) || defined(HPUX) || defined(SOLARIS) || defined(__ia64__) || defined(__crayx1)
#   define PTR_ALIGN
#endif

#if defined(NB_NONCONT) && !defined(CRAY_SHMEM)
#error NB_NONCONT is only available on CRAY_SHMEM
#endif

#if defined(SHMEM_HANDLE_SUPPORTED) && !defined(CRAY_SHMEM)
#error SHMEM_HANDLE_SUPPORTED should not be defined on a non CRAY_SHMEM network
#endif

#if  defined(MEMCPY)  && !defined(armci_copy)
#  define armci_copy(src,dst,n)  memcpy((dst), (src), (n)) 
#endif

#ifdef NEC
#    define MEM_FENCE {mpisx_clear_cache(); _armci_vec_sync_flag=1;mpisx_syncset0_long(&_armci_vec_sync_flag);}
#endif

#if defined(NEED_MEM_SYNC)
#  ifdef AIX
#    define MEM_FENCE {int _dummy=1; _clear_lock((int *)&_dummy,0); }
#  elif defined(__ia64)
#    if defined(__GNUC__) && !defined (__INTEL_COMPILER)
#       define MEM_FENCE __asm__ __volatile__ ("mf" ::: "memory");
#    else /* Intel Compiler */ 
        extern void _armci_ia64_mb();
#       define MEM_FENCE _armci_ia64_mb();
#    endif
#  elif defined(LINUX) && defined(__GNUC__) && defined(__ppc__)
#    define MEM_FENCE \
             __asm__ __volatile__ ("isync" : : : "memory");
#  endif
#endif

#ifndef armci_copy
# ifdef PTR_ALIGN
#   define armci_copy(src,dst,n)     \
     do if( ((n) < THRESH1D)   || ((n)%ALIGN_SIZE) || \
            ((unsigned long)(src)%ALIGN_SIZE) ||\
            ((unsigned long)(dst)%ALIGN_SIZE)) memcpy((dst),(src),(n));\
        else{ int _bytes=(n)/sizeof(double); DCOPY1D((double*)(src),(double*)(dst),&_bytes);}\
     while (0)
# else
#   define armci_copy(src,dst,n)     \
     do if( ((n) < THRESH1D) || ((n)%ALIGN_SIZE) ) memcpy((dst), (src), (n));\
          else{ int _bytes=(n)/sizeof(double); DCOPY1D((double*)(src),(double*)(dst),&_bytes);}\
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

#if defined(FUJITSU)

#   define armci_put2D(p, bytes,count,src_ptr,src_stride,dst_ptr,dst_stride)\
           CopyPatchTo(src_ptr, src_stride, dst_ptr, dst_stride, count,bytes, p)

#   define armci_get2D(p, bytes, count, src_ptr,src_stride,dst_ptr,dst_stride)\
           CopyPatchFrom(src_ptr, src_stride, dst_ptr, dst_stride,count,bytes,p)

#elif defined(NB_NONCONT)

    extern void armcill_wait_put();
    extern void armcill_wait_get();
#   define WAIT_FOR_PUTS armcill_wait_put()
#   define WAIT_FOR_GETS armcill_wait_get()
 
    extern void armcill_put2D(int proc, int bytes, int count,
                void* src_ptr,int src_stride, void* dst_ptr,int dst_stride);
    extern void armcill_get2D(int proc, int bytes, int count,
                void* src_ptr,int src_stride, void* dst_ptr,int dst_stride);
#   define armci_put2D armcill_put2D
#   define armci_get2D armcill_get2D

#  if defined(CRAY_SHMEM)

#       define armcill_nb_wait(_hdl)\
               shmem_wait_nb(_hdl)
/*VT:this should be ifdef'ed based on if shmem_handle is defined or not*/
#       if defined (CRAY_XT)
#           define armcill_nb_put(_dst, _src, _sz, _proc, _hdl)\
                   shmem_putmem(_dst, _src, (size_t)_sz, _proc)
#           define armcill_nb_get(_dst, _src, _sz, _proc, _hdl)\
                   shmem_getmem(_dst, _src, (size_t)_sz, _proc)
#       else
#           define armcill_nb_put(_dst, _src, _sz, _proc, _hdl)\
                   _hdl = shmem_putmem_nb(_dst, _src, (size_t)_sz, _proc, &(_hdl))
#           define armcill_nb_get(_dst, _src, _sz, _proc, _hdl)\
                   _hdl = shmem_getmem_nb(_dst, _src, (size_t)_sz, _proc, &(_hdl))
#       endif
#   endif

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
   
/* macros to ensure ordering of consecutive puts or gets following puts */
#if defined(_CRAYMPP) || defined(__crayx1)\
   || defined(CRAY_SHMEM) 
#if defined(CRAY) || defined(CRAY_XT)
#   include <mpp/shmem.h>
#else
#   include <unistd.h>
#ifndef ptrdiff_t
#   include <malloc.h>
#endif
#   include <shmem.h>
#endif
    int cmpl_proc;
#   define FENCE_NODE(p) if(cmpl_proc == (p)){\
        if(((p)<armci_clus_first)||((p)>armci_clus_last))shmem_quiet(); }
#   define UPDATE_FENCE_STATE(p, op, nissued) if((op)==PUT) cmpl_proc=(p);
#else
#   define FENCE_NODE(p)
#   define UPDATE_FENCE_STATE(p, op, nissued)

#endif


#ifdef NEC
#  define THRESH 1
#  define THRESH1D 1
#else
#  define THRESH 32
#  define THRESH1D 512
#endif
#define ALIGN_SIZE sizeof(double)

/********* interface to C 1D and 2D memory copy functions ***********/
/* dcopy2d_u_ uses explicit unrolled loops to depth 4 */
void c_dcopy2d_n_(const int*    const restrict rows,
                  const int*    const restrict cols,
                  const double* const restrict A,
                  const int*    const restrict ald,
                        double* const restrict B,
                  const int*    const restrict bld);
void c_dcopy2d_u_(const int*    const restrict rows,
                  const int*    const restrict cols,
                  const double* const restrict A,
                  const int*    const restrict ald,
                        double* const restrict B,
                  const int*    const restrict bld);
void c_dcopy1d_n_(const double* const restrict A,
                        double* const restrict B,
                  const int*    const restrict n);
void c_dcopy1d_u_(const double* const restrict A,
                        double* const restrict B,
                  const int*    const restrict n);
void c_dcopy21_(const int*    const restrict rows,
                const int*    const restrict cols,
                const double* const restrict A,
                const int*    const restrict ald,
                      double* const restrict buf,
                      int*    const restrict cur);
void c_dcopy12_(const int*    const restrict rows,
                const int*    const restrict cols,
                      double* const restrict A,
                const int*    const restrict ald,
                const double* const restrict buf,
                      int*    const restrict cur);
void c_dcopy31_(const int*    const restrict rows,
                const int*    const restrict cols,
                const int*    const restrict plns,
                const double* const restrict A,
                const int*    const restrict aldr,
                const int*    const restrict aldc,
                      double* const restrict buf,
                      int*    const restrict cur);
void c_dcopy13_(const int*    const restrict rows,
                const int*    const restrict cols,
                const int*    const restrict plns,
                      double* const restrict A,
                const int*    const restrict aldr,
                const int*    const restrict aldc,
                const double* const restrict buf,
                      int*    const restrict cur);

#if defined(AIX)
#    define DCOPY2D c_dcopy2d_u_
#    define DCOPY1D c_dcopy1d_u_
#elif defined(LINUX) || defined(__crayx1) || defined(HPUX64) || defined(CRAY) || defined(WIN32)
#    define DCOPY2D c_dcopy2d_n_
#    define DCOPY1D c_dcopy1d_n_
#else
#    define DCOPY2D c_dcopy2d_u_
#    define DCOPY1D c_dcopy1d_u_
#endif
#define DCOPY21 c_dcopy21_
#define DCOPY12 c_dcopy12_
#define DCOPY31 c_dcopy31_
#define DCOPY13 c_dcopy13_


/***************************** 1-Dimensional copy ************************/
#if defined(CRAY_SHMEM)
#      define armci_copy_disabled(src,dst,n)\
        if((n)<256 || n%sizeof(long) ) memcpy((dst),(src),(n));\
        else {\
          shmem_put((long*)(dst),(long*)(src),(int)(n)/sizeof(long),armci_me);\
          shmem_quiet(); }

#      define armci_put(src,dst,n,proc) \
              shmem_put32((void *)(dst),(void *)(src),(int)(n)/4,(proc));\
              shmem_quiet()

#      define armci_get(src,dst,n,proc) \
              shmem_get32((void *)(dst),(void *)(src),(int)(n)/4,(proc));\
              shmem_quiet()

#elif  defined(FUJITSU)

#      include "fujitsu-vpp.h"
#      ifndef __sparc
#         define armci_copy(src,dst,n)  _MmCopy((char*)(dst), (char*)(src), (n))
#      endif
#      define armci_put  CopyTo
#      define armci_get  CopyFrom                                                

#else

#      define armci_get(src,dst,n,p)    armci_copy((src),(dst),(n))
#      define armci_put(src,dst,n,p)    armci_copy((src),(dst),(n))

#endif

#ifndef MEM_FENCE
#   define MEM_FENCE {}
#endif
#ifndef armci_copy_fence
#   define armci_copy_fence armci_copy
#endif

#endif
