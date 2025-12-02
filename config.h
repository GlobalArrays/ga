/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* TODO */
/* #undef ALLOW_PIN */

/* Define to sizeof(INTEGER) used to compile BLAS */
#define BLAS_SIZE 8

/* Define to 1 on Cygnus systems */
/* #undef CYGNUS */

/* Define to 1 on Cygwin systems */
/* #undef CYGWIN */

/* TODO */
#define DATA_SERVER 1

/* ELPA is using 8-byte integers */
/* #undef ELPA_I8 */

/* Define if checkpointing is enabled */
/* #undef ENABLE_CHECKPOINT */

/* set to 1 if CUDA memory is enabled */
/* #undef ENABLE_CUDA_MEM */

/* Define to 1 if EISPACK is enabled */
#define ENABLE_EISPACK 0

/* Define to 1 if using Fortran */
#define ENABLE_F77 1

/* Define to 1 if PeIGS is enabled */
#define ENABLE_PEIGS 0

/* set to 1 if profiling is enabled */
#define ENABLE_PROFILING 0

/* Define if tracing is enabled */
/* #undef ENABLE_TRACE */

/* whether the hidden string length comes after all other args */
#define F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS 1

/* define to 1 if Fortran-callable srand48 does not conflict with system def
   */
#define F2C_SRAND48_OK 1

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Name of F77 flush routine */
#define F77_FLUSH flush

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define F77_FUNC(name,NAME) name ## _

/* As F77_FUNC, but for C identifiers containing underscores. */
#define F77_FUNC_(name,NAME) name ## _

/* Name of routine e.g. getarg, pxfgetarg */
#define F77_GETARG GETARG

/* Arguments to getarg e.g. i,s or i,s,l,ier */
#define F77_GETARG_ARGS i,s

/* Declaration of routine e.g. external GETARG */
#define F77_GETARG_DECL intrinsic GETARG

/* Name of routine e.g. iargc, ipxfargc */
#define F77_IARGC IARGC

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* CPP symbol for function name, if available */
#define FUNCTION_NAME __func__

/* Module command when using Fortran 90 compiler e.g. use f90_unix */
#define FXX_MODULE 

/* set to 1 if ARMCI has armci_group_comm function */
#define HAVE_ARMCI_GROUP_COMM 1

/* set to 1 if ARMCI has ARMCI_Group.comm member */
/* #undef HAVE_ARMCI_GROUP_COMM_MEMBER */

/* set to 1 if ARMCI has ARMCI_Initialized function */
#define HAVE_ARMCI_INITIALIZED 1

/* set to 1 if ARMCI has armci_msg_finalize function */
#define HAVE_ARMCI_MSG_FINALIZE 1

/* set to 1 if ARMCI has armci_msg_init function */
#define HAVE_ARMCI_MSG_INIT 1

/* set to 1 if ARMCI has armci_notify function */
#define HAVE_ARMCI_NOTIFY 1

/* set to 1 if ARMCI has armci_stride_info_init function */
#define HAVE_ARMCI_STRIDE_INFO_INIT 1

/* Define to 1 if you have the <assert.h> header file, 0 if you don't */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the `bcopy' function, 0 if you don't */
#define HAVE_BCOPY 1

/* Define to 1 if using external BLAS library */
#define HAVE_BLAS 0

/* Define to 1 if you have the `bzero' function, 0 if you don't */
#define HAVE_BZERO 1

/* Define to 1 if you have the `create_shared_region' function, 0 if you don't
   */
#define HAVE_CREATE_SHARED_REGION 0

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#define HAVE_DIRENT_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the `dtime' function. */
#define HAVE_DTIME 0

/* Define to 1 if you have ELPA library. */
#define HAVE_ELPA 0

/* Define to 1 if you have 2015 ELPA lib */
#define HAVE_ELPA_2015 0

/* Define to 1 if you have ELPA library with 2STAGE alg. in 2016 ELPA lib */
#define HAVE_ELPA_2016 0

/* Define to 1 if you have ELPA library with the 2017 ELPA API */
#define HAVE_ELPA_2017 0

/* Define to 1 if you have the <errno.h> header file, 0 if you don't */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the `etime' function. */
#define HAVE_ETIME 0

/* whether F77 flush routine is available */
#define HAVE_F77_FLUSH 1

/* Define to 1 if you have the <fcntl.h> header file, 0 if you don't */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <ffio.h> header file, 0 if you don't */
#define HAVE_FFIO_H 0

/* Define to 1 if you have the 'floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if you have the `fork' function. */
#define HAVE_FORK 1

/* Define to 1 if fseeko (and presumably ftello) exists and is declared. */
#define HAVE_FSEEKO 1

/* Define to 1 if you have the `ftruncate' function, 0 if you don't */
#define HAVE_FTRUNCATE 1

/* Define to 1 if you have the `gethostbyname' function, 0 if you don't */
#define HAVE_GETHOSTBYNAME 1

/* Define to 1 if you have the `gethostname' function, 0 if you don't */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have the `getpagesize' function, 0 if you don't */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `gettimeofday' function, 0 if you don't */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the 'hypot' function. */
#define HAVE_HYPOT 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if using external LAPACK library */
#define HAVE_LAPACK 0

/* Defined if plot library is available */
/* #undef HAVE_LIBPLOT */

/* Define to 1 if you have the <limits.h> header file, 0 if you don't */
#define HAVE_LIMITS_H 1

/* Define to 1 if you have the <linux/limits.h> header file, 0 if you don't */
#define HAVE_LINUX_LIMITS_H 1

/* Define to 1 if the system has the type `long double'. */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if you have the <malloc.h> header file, 0 if you don't */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the 'mallopt' function. */
#define HAVE_MALLOPT 1

/* Define to 1 if you have the <math.h> header file, 0 if you don't */
#define HAVE_MATH_H 1

/* Define to 1 if you have the `memalign' function, 0 if you don't */
#define HAVE_MEMALIGN 1

/* Define to 1 if you have the `memcpy' function, 0 if you don't */
#define HAVE_MEMCPY 1

/* Define to 1 if you have the `memset' function, 0 if you don't */
#define HAVE_MEMSET 1

/* Define to 1 if you have the <minix/config.h> header file. */
/* #undef HAVE_MINIX_CONFIG_H */

/* Define to 1 if you have a working `mmap' system call. */
#define HAVE_MMAP 1

/* Define to 1 if you have the 'modf' function. */
#define HAVE_MODF 1

/* Define to 1 if you have the `munmap' function, 0 if you don't */
#define HAVE_MUNMAP 1

/* define if the compiler implements namespaces */
/* #undef HAVE_NAMESPACES */

/* Define to 1 if you have the <ndir.h> header file, and it defines `DIR'. */
/* #undef HAVE_NDIR_H */

/* Define to 1 if you have the `pause' function, 0 if you don't */
#define HAVE_PAUSE 1

/* whether the ScaLAPACK library implements pdsyevr */
/* #undef HAVE_PDSYEVR */

/* Define to 1 if you have the `posix_memalign' function, 0 if you don't */
#define HAVE_POSIX_MEMALIGN 1

/* Define to 1 if you have the 'pow' function. */
#define HAVE_POW 1

/* define to 1 if the C compiler has a program name global varaible */
#define HAVE_PROGNAME 1

/* Define if you have POSIX threads libraries and header files. */
#define HAVE_PTHREAD 1

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define to 1 if you have the `putenv' function, 0 if you don't */
#define HAVE_PUTENV 1

/* Define to 1 if you have ScaLAPACK library. */
#define HAVE_SCALAPACK 0

/* Define to 1 if you have the `select' function, 0 if you don't */
#define HAVE_SELECT 1

/* Define to 1 if you have the `setenv' function, 0 if you don't */
#define HAVE_SETENV 1

/* Define to 1 if you have the `shared_malloc' function, 0 if you don't */
#define HAVE_SHARED_MALLOC 0

/* Define to 1 if you have the `shmalloc' function, 0 if you don't */
#define HAVE_SHMALLOC 0

/* Define to 1 if you have the `socket' function, 0 if you don't */
#define HAVE_SOCKET 1

/* Define to 1 if you have the 'sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if you have the `srand' function, 0 if you don't */
#define HAVE_SRAND 1

/* Define to 1 if you have the `srandom' function, 0 if you don't */
#define HAVE_SRANDOM 1

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stddef.h> header file, 0 if you don't */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file, 0 if you don't */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file, 0 if you don't */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file, 0 if you don't */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strchr' function, 0 if you don't */
#define HAVE_STRCHR 1

/* Define to 1 if you have the `strdup' function, 0 if you don't */
#define HAVE_STRDUP 1

/* Define to 1 if you have the `strerror' function, 0 if you don't */
#define HAVE_STRERROR 1

/* Define to 1 if you have the <strings.h> header file, 0 if you don't */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file, 0 if you don't */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strstr' function, 0 if you don't */
#define HAVE_STRSTR 1

/* Define to 1 if you have the `strtol' function, 0 if you don't */
#define HAVE_STRTOL 1

/* Define to 1 if you have the <sys/dir.h> header file, and it defines `DIR'.
   */
/* #undef HAVE_SYS_DIR_H */

/* Define to 1 if you have the <sys/file.h> header file, 0 if you don't */
#define HAVE_SYS_FILE_H 1

/* Define to 1 if you have the <sys/fsid.h> header file, 0 if you don't */
#define HAVE_SYS_FSID_H 0

/* Define to 1 if you have the <sys/fstyp.h> header file, 0 if you don't */
#define HAVE_SYS_FSTYP_H 0

/* Define to 1 if you have the <sys/ioctl.h> header file, 0 if you don't */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/iosw.h> header file, 0 if you don't */
#define HAVE_SYS_IOSW_H 0

/* Define to 1 if you have the <sys/ipc.h> header file, 0 if you don't */
#define HAVE_SYS_IPC_H 1

/* Define to 1 if you have the <sys/mman.h> header file, 0 if you don't */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/mount.h> header file, 0 if you don't */
#define HAVE_SYS_MOUNT_H 1

/* Define to 1 if you have the <sys/ndir.h> header file, and it defines `DIR'.
   */
/* #undef HAVE_SYS_NDIR_H */

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/sem.h> header file, 0 if you don't */
#define HAVE_SYS_SEM_H 1

/* Define to 1 if you have the <sys/shm.h> header file, 0 if you don't */
#define HAVE_SYS_SHM_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/statfs.h> header file, 0 if you don't */
#define HAVE_SYS_STATFS_H 1

/* Define to 1 if you have the <sys/statvfs.h> header file, 0 if you don't */
#define HAVE_SYS_STATVFS_H 1

/* Define to 1 if you have the <sys/stat.h> header file, 0 if you don't */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file, 0 if you don't */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file, 0 if you don't */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/vfs.h> header file, 0 if you don't */
#define HAVE_SYS_VFS_H 1

/* Define to 1 if you have <sys/wait.h> that is POSIX.1 compatible. */
#define HAVE_SYS_WAIT_H 1

/* Define this if weak aliases may be created with #pragma weak */
#define HAVE_SYS_WEAK_ALIAS_PRAGMA 1

/* Define to 1 if you have the <time.h> header file, 0 if you don't */
#define HAVE_TIME_H 1

/* define if sys/sem.h has union semun */
/* #undef HAVE_UNION_SEMUN */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `vfork' function. */
#define HAVE_VFORK 1

/* Define to 1 if you have the <vfork.h> header file. */
/* #undef HAVE_VFORK_H */

/* Define to 1 if you have the <wchar.h> header file. */
#define HAVE_WCHAR_H 1

/* Define to 1 if you have the <windows.h> header file, 0 if you don't */
#define HAVE_WINDOWS_H 0

/* Define to 1 if `fork' works. */
#define HAVE_WORKING_FORK 1

/* Define to 1 if `vfork' works. */
#define HAVE_WORKING_VFORK 1

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Define to 1 on IBM SP systems */
/* #undef IBM */

/* Define to 1 on 64bit IBM SP systems */
/* #undef IBM64 */

/* Define to 1 on generic Linux systems */
#define LINUX 1

/* Define to 1 on generic 64bit Linux systems */
#define LINUX64 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to 1 on OSX systems */
/* #undef MACX */

/* Define to 1 on 64bit OSX systems */
/* #undef MACX64 */

/* TODO */
/* #undef MPI3 */

/* TODO */
/* #undef MPI_MT */

/* TODO */
/* #undef MPI_PR */

/* TODO */
/* #undef MPI_PT */

/* TODO */
/* #undef MPI_SPAWN */

/* TODO */
#define MPI_TS 1

/* Use MPI for messaging */
#define MSG_COMMS_MPI 1

/* Define to 1 if assertions should be disabled. */
/* #undef NDEBUG */

/* whether to wait until the last moment to call ARMCI_Init() in TCGMSG-MPI */
#define NEED_DELAY_TCGMSG_MPI_STARTUP 0

/* Define to 1 if not using Fortran */
#define NOFORT 0

/* Set some mallopt options */
#define NOUSE_MMAP 1

/* TODO */
/* #undef OPENIB */

/* Name of package */
#define PACKAGE "ga"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://github.com/GlobalArrays/ga/issues"

/* Define to the full name of this package. */
#define PACKAGE_NAME "Global Arrays (GA)"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "Global Arrays (GA) 5.9.2"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "ga"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://hpc.pnl.gov/globalarrays/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "5.9.2"

/* TODO */
/* #undef PEND_BUFS */

/* TODO */
/* #undef POSIX_THREADS */

/* define to the name of the program name global variable */
#define PROGNAME __progname

/* TODO */
#define PTHREADS 1

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* TODO */
/* #undef REPORT_SHMMAX */

/* Define as the return type of signal handlers (`int' or `void'). */
#define RETSIGTYPE void

/* ScaLAPACK is using 8-byte integers */
/* #undef SCALAPACK_I8 */

/* Define to the type of arg 1 for `select'. */
#define SELECT_TYPE_ARG1 int

/* Define to the type of args 2, 3 and 4 for `select'. */
#define SELECT_TYPE_ARG234 (fd_set *)

/* Define to the type of arg 5 for `select'. */
#define SELECT_TYPE_ARG5 (struct timeval *)

/* TODO */
#define SERVER_THREAD 1

/* The size of `char', as computed by sizeof. */
#define SIZEOF_CHAR 1

/* The size of `double', as computed by sizeof. */
#define SIZEOF_DOUBLE 8

/* The size of 'DOUBLE PRECISION' as computed by C's sizeof. */
#define SIZEOF_F77_DOUBLE_PRECISION 8

/* The size of 'INTEGER' as computed by C's sizeof. */
#define SIZEOF_F77_INTEGER 8

/* The size of 'REAL' as computed by C's sizeof. */
#define SIZEOF_F77_REAL 4

/* The size of `float', as computed by sizeof. */
#define SIZEOF_FLOAT 4

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 8

/* The size of `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE 16

/* The size of `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `void*', as computed by sizeof. */
#define SIZEOF_VOIDP 8

/* TODO */
/* #undef SOCKETS */

/* Define to 1 on Solaris systems */
/* #undef SOLARIS */

/* Define to 1 on 64bit Solaris systems */
/* #undef SOLARIS64 */

/* This is always defined. Dunno why. */
#define STATS 1

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Defined if we want this system to use SYSV shared memory */
#define SYSV 1

/* enable test_sicm define */
/* #undef TEST_SICM */

/* specify sicm test hw */
/* #undef TEST_SICM_DEV */

/* turn on thread safety */
/* #undef THREAD_SAFE */

/* enable sicm define */
/* #undef USE_SICM */

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable general extensions on macOS.  */
#ifndef _DARWIN_C_SOURCE
# define _DARWIN_C_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable X/Open compliant socket functions that do not require linking
   with -lxnet on HP-UX 11.11.  */
#ifndef _HPUX_ALT_XOPEN_SOCKET_API
# define _HPUX_ALT_XOPEN_SOCKET_API 1
#endif
/* Identify the host operating system as Minix.
   This macro does not affect the system headers' behavior.
   A future release of Autoconf may stop defining this macro.  */
#ifndef _MINIX
/* # undef _MINIX */
#endif
/* Enable general extensions on NetBSD.
   Enable NetBSD compatibility extensions on Minix.  */
#ifndef _NETBSD_SOURCE
# define _NETBSD_SOURCE 1
#endif
/* Enable OpenBSD compatibility extensions on NetBSD.
   Oddly enough, this does nothing on OpenBSD.  */
#ifndef _OPENBSD_SOURCE
# define _OPENBSD_SOURCE 1
#endif
/* Define to 1 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_SOURCE
/* # undef _POSIX_SOURCE */
#endif
/* Define to 2 if needed for POSIX-compatible behavior.  */
#ifndef _POSIX_1_SOURCE
/* # undef _POSIX_1_SOURCE */
#endif
/* Enable POSIX-compatible threading on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-5:2014.  */
#ifndef __STDC_WANT_IEC_60559_ATTRIBS_EXT__
# define __STDC_WANT_IEC_60559_ATTRIBS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-1:2014.  */
#ifndef __STDC_WANT_IEC_60559_BFP_EXT__
# define __STDC_WANT_IEC_60559_BFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-2:2015.  */
#ifndef __STDC_WANT_IEC_60559_DFP_EXT__
# define __STDC_WANT_IEC_60559_DFP_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-4:2015.  */
#ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
# define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TS 18661-3:2015.  */
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
# define __STDC_WANT_IEC_60559_TYPES_EXT__ 1
#endif
/* Enable extensions specified by ISO/IEC TR 24731-2:2010.  */
#ifndef __STDC_WANT_LIB_EXT2__
# define __STDC_WANT_LIB_EXT2__ 1
#endif
/* Enable extensions specified by ISO/IEC 24747:2009.  */
#ifndef __STDC_WANT_MATH_SPEC_FUNCS__
# define __STDC_WANT_MATH_SPEC_FUNCS__ 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable X/Open extensions.  Define to 500 only if necessary
   to make mbstate_t available.  */
#ifndef _XOPEN_SOURCE
/* # undef _XOPEN_SOURCE */
#endif


/* TODO */
/* #undef VAPI */

/* This is always defined. Dunno why. */
#define VERIFY 1

/* Version number of package */
#define VERSION "5.9.2"

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define to 1 to make fseeko visible on some hosts (e.g. glibc 2.2). */
/* #undef _LARGEFILE_SOURCE */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* TODO */
/* #undef _REENTRANT */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to the type of a signed integer type of width exactly 64 bits if
   such a type exists and the standard includes do not define it. */
/* #undef int64_t */

/* Define to `long int' if <sys/types.h> does not define. */
/* #undef off_t */

/* Define as a signed integer type capable of holding a process identifier. */
/* #undef pid_t */

/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported only directly.  */
#define restrict __restrict__
/* Work around a bug in older versions of Sun C++, which did not
   #define __restrict__ or support _Restrict or __restrict__
   even though the corresponding Sun C compiler ended up with
   "#define restrict _Restrict" or "#define restrict __restrict__"
   in the previous line.  This workaround can be removed once
   we assume Oracle Developer Studio 12.5 (2016) or later.  */
#if defined __SUNPRO_CC && !defined __RESTRICT && !defined __restrict__
# define _Restrict
# define __restrict__
#endif

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */

/* Define as `fork' if `vfork' does not work. */
/* #undef vfork */

/* Define to empty if the keyword `volatile' does not work. Warning: valid
   code using `volatile' can become incorrect without. Disable with care. */
/* #undef volatile */
