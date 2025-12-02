#include "f2c_cmake.h"

#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif

#define HAVE_ASSERT_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MALLOC_H 1
#define HAVE_MATH_H 1
#define HAVE_MEMCPY 1
#define HAVE_PAUSE 1
#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRCHR 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define HAVE_WINDOWS_H 0

#define HAVE_BZERO 1
#if !HAVE_BZERO
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)
#endif



#define ENABLE_F77 1
#define NOFORT 0

#define NOUSE_MMAP

/* #undef NDEBUG */

/* #undef CYGWIN */
/* #undef DECOSF */

#define ENABLE_EISPACK 0

/* #undef ENABLE_CHECKPOINT */
#define ENABLE_PROFILING 0
/* #undef ENABLE_TRACE */
#define STATS 1
/* #undef USE_MALLOC */

#define HAVE_ARMCI_GROUP_COMM 1
#define HAVE_ARMCI_GROUP_COMM_MEMBER 0
#define HAVE_ARMCI_INITIALIZED 1

#define HAVE_SYS_WEAK_ALIAS_PRAGMA 1

/* #undef MPI3 */
/* #undef MPI_MT */
/* #undef MPI_PR */
/* #undef MPI_PT */
/* #undef MPI_TS */

#define MSG_COMMS_MPI 1
#define ENABLE_ARMCI_MEM_OPTION 1
/* #undef ENABLE_CUDA_MEM */

#define HAVE_BLAS 0
#define HAVE_LAPACK 0

#define F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS 1

/*#define F77_FUNC(name,NAME) F77_FUNC_GLOBAL(name,NAME)*/
/*#define F77_FUNC_(name,NAME) F77_FUNC_GLOBAL_(name,NAME)*/

#define F77_FUNC(name,NAME) name ## _
#define F77_FUNC_(name,NAME) name ## _

#define FXX_MODULE 
#define F77_GETARG GETARG
#define F77_GETARG_ARGS i,s
#define F77_GETARG_DECL intrinsic GETARG
#define F77_IARGC IARGC
#define F77_FLUSH flush
#define HAVE_F77_FLUSH 1

#define SIZEOF_INT 4
#define SIZEOF_DOUBLE 8
#define SIZEOF_F77_DOUBLE_PRECISION 8
#define SIZEOF_F77_REAL 4
#define SIZEOF_F77_INTEGER 8
#define SIZEOF_FLOAT 4
#define SIZEOF_LONG 8
#define SIZEOF_LONG_DOUBLE 16
#define SIZEOF_LONG_LONG 8
#define SIZEOF_SHORT 2
#define SIZEOF_VOIDP 8
#define BLAS_SIZE 8
/* #define BLAS_SIZE  */

#define LINUX
#define LINUX64

/* #undef _FILE_OFFSET_BITS */
/* #undef _LARGEFILE_SOURCE */
/* #undef _LARGE_FILES */

#ifndef __cplusplus
/* #undef inline */
#endif
#define restrict __restrict
