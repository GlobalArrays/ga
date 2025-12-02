#define COMEX_NETWORK_MPI3 0
#define COMEX_NETWORK_MPI_MT 0
#define COMEX_NETWORK_MPI_PR 0
#define COMEX_NETWORK_MPI_PT 0
#define COMEX_NETWORK_MPI_TS 0

#define HAVE_BLAS 0

#define BLAS_CAXPY caxpy_
#define BLAS_DAXPY daxpy_
#define BLAS_SAXPY saxpy_
#define BLAS_ZAXPY zaxpy_
#define BLAS_CCOPY ccopy_
#define BLAS_DCOPY dcopy_
#define BLAS_SCOPY scopy_
#define BLAS_ZCOPY zcopy_

#define HAVE_ASSERT_H 1
#define HAVE_BZERO 1
#define HAVE_MATH_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRCHR 1
#define HAVE_STRING_H 1
#define HAVE_UNISTD_H 1



#define FUNCTION_NAME __func__

#define HAVE_PTHREAD_SETAFFINITY_NP 0
#define HAVE_SCHED_SETAFFINITY 0
#define HAVE_SYS_WEAK_ALIAS_PRAGMA 1

/* #undef NDEBUG */
/* #undef __CRAYXE */

#define ENABLE_SYSV 0

#define SIZEOF_INT 4
#define SIZEOF_LONG 8
#define SIZEOF_LONG_LONG 8
#define SIZEOF_VOIDP 8
#define BLAS_SIZE 8
/* #define BLAS_SIZE  */

#ifndef __cplusplus
/* #undef inline */
#endif
#define restrict __restrict
