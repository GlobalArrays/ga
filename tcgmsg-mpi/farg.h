/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/farg.h,v 1.4 1999-07-01 19:51:42 d3h325 Exp $ */

/*
  This include file defines ARGC_ and ARGV_ which are the appropriate
  global variables to get at the command argument list in the
  FORTRAN runtime environment. 
  This will need to be modified for each new machine (try using nm or 
  generating a full load map). 
*/

/* Used to be SEQUENT here but charged to _X_argv */
#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || defined(CONVEX) || \
  defined(KSR)
#define ARGC_ xargc
#define ARGV_ xargv
#endif

#if defined(ARDENT)
#define ARGC_ _UT_argc
#define ARGV_ _UT_argv
#endif

#if (defined(SGI) || defined(ULTRIX))
#define ARGC_ f77argc
#define ARGV_ f77argv
#endif

#if defined(DECOSF)
#define ARGC_ __Argc
#define ARGV_ __Argv
#endif

#if defined(AIX)
#define ARGC_ p_xargc
#define ARGV_ p_xargv
#endif

#if defined(CRAY)
#define ARGC_ _argc
#define ARGV_ _argv
#endif

/* g77/gcc fortran argc/argv interface on linux is unstable */
#if defined(LINUX)
#if ((__GNUC__ > 2) || ((__GNUC__ == 2) && (__GNUC_MINOR__ > 90)))
#   define ARGC_ f__xargc
#   define ARGV_ f__xargv
#else
#   define ARGC_ xargc
#   define ARGV_ xargv
#endif
#endif

/* PGI compilers on Linux */
#if defined(PGLINUX)
#define ARGC_ __argc_save
#define ARGV_ __argv_save
#endif


#ifdef SEQUENT
#define ARGC_ _X_argc
#define ARGV_ _X_argv
#endif

#if defined(NEXT)
#define ARGC_ _NXArgc
#define ARGV_ _NXArgv
#endif

#if defined(HPUX)
/* ARGC_ and ARGV_ are allocated and constructed in pbeginf */
#else

extern int ARGC_;
extern char **ARGV_;

#endif

/* Eample use

static void PrintCommandArgList()
{
  int i;

  for (i=0; i<ARGC_; i++)
    (void) printf("argv(%d)=%s\n", i, ARGV_[i]);
}

*/
