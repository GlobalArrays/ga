/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/farg.h,v 1.10 2001-06-28 00:06:52 edo Exp $ */

/*
  This include file defines ARGC_ and ARGV_ which are the appropriate
  global variables to get at the command argument list in the
  FORTRAN runtime environment. 
  This will need to be modified for each new machine (try using nm or 
  generating a full load map). 
*/

/* Used to be SEQUENT here but charged to _X_argv */
#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) ||  \
  defined(CONVEX) ||  defined(KSR)
#define ARGC_ xargc
#define ARGV_ xargv
#endif

#if defined(ARDENT)
#define ARGC_ _UT_argc
#define ARGV_ _UT_argv
#endif

#if (defined(SGI) || defined(ULTRIX)) && !defined(DECFORT)
#define ARGC_ f77argc
#define ARGV_ f77argv
#endif

#if defined(DECFORT)
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

#if defined(PGLINUX)
#define ARGC_ __argc_save
#define ARGV_ __argv_save
#elif defined(IFCLINUX)
#define ARGC_ xargc
#define ARGV_ xargv
#elif defined(SGILINUX)
#define ARGC_ _f90argc
#define ARGV_ _f90argv
#else
#if defined(LINUX)
/*---------------------------------------------------------------------------*\
 There are a multitutde of LINUX distributions and ALL of them differ
 to some extent.  To compile and use this software with code compiled 
 using g77 you MUST use the C compiler that was used to build g77.  
 Most distributions are shipping the egcs version of g77 and gcc as 
 well as the 2.7.2.3 version from the GNU group because of the known 
 LINUX kernel bug.  
 On Redhat 5.2 and 6.0 the egcs C compiler is called egcs so you 
    need to do a top level make with "CC=egcs" for the tools to work
    with g77 based applications.
 On Slackware 4.0 gcc is a link to the 2.7.2.3 version change the link to 
    point to gcc-egcs-1.1.2 or do a top level make with "CC=gcc-egcs-1.1.2"
 On Caldera 2.2 you only get EGCS compilers.  

 The folowing test is our best guess as to the proper environment using
 the EGCS c compilers with g77.  The major mode > 2 should use f__xarg{c|v}
 Assuming that the convention does not change in the next release.  
 For sure 2.91 or egcs 1.1.2 uses f__xarg{c|v}

 To determine which arguments are needed for the version of g77 you are using:
  1) write a hello world fortran code
  2) compile and link the code (generate a.out)
  3) nm a.out | grep xarg and then fix this file.
\*---------------------------------------------------------------------------*/

#if ((__GNUC__ > 2) || ((__GNUC__ == 2) && (__GNUC_MINOR__ > 90)))
#define ARGC_ f__xargc
#define ARGV_ f__xargv
#else
#define ARGC_ xargc
#define ARGV_ xargv
#endif
#endif
#endif /* end of ifdef PGLINUX */

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
