#ifndef F2CBLAS_H
#define F2CBLAS_H

#ifdef CRAY_T3D
   THIS CODES NEEDS TO BE CONVERTED TO USE CHARACTER DESCRIPTORS
#  define dlacpy_ SLACPY
#  define dgemm_ SGEMM
#  define daxpy_ SAXPY
#  define dcopy_ DCOPY
#elif defined(KSR)
#  define dlacpy_ slacpy_
#  define dgemm_ sgemm_
#  define dcopy_ scopy_
#  define daxpy_ saxpy_
#endif

#endif
