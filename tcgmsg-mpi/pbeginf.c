/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/pbeginf.c,v 1.7 2000-05-10 18:28:05 edo Exp $ */

#include <stdio.h>
#include "farg.h"
#include "sndrcv.h"
#define LEN 255

extern void PBEGIN_();

#if defined(HPUX) || defined(SUN) || defined(SOLARIS) ||defined(PARAGON) ||defined(FUJITSU) || defined(WIN32) ||defined(LINUX64)
#define HAS_GETARG 1
#endif

#ifdef WIN32
#define iargc_ IARGC
#define getarg_ GETARG
#include "winutil.h"
#else
#define FATR 
#endif


#if !defined(HAS_GETARG)
void FATR PBEGINF_()
/*
  Interface routine between FORTRAN and c version of pbegin.
  Relies on knowing global address of program argument list ... see farg.h
*/
{
  PBEGIN_(ARGC_, ARGV_);
}
#else
void FATR PBEGINF_()
/*
  Hewlett Packard Risc box, SparcWorks F77 2.* and Paragon compilers.
  Have to construct the argument list by calling FORTRAN.
*/
{
  extern char *strdup();

#if defined(WIN32)
    extern int FATR iargc_();
    extern void FATR getarg_(short*, char*, int, short*);
    int argc = iargc_() + 1;
#elif !defined(HPUX)
    extern int iargc_();
    extern void getarg_();
    int argc = iargc_() + 1;
#else
#   ifndef EXTNAME
#     define hpargv_ hpargv
#     define hpargc_ hpargc
#   endif
    extern int hpargv_();
    extern int hpargc_();
    int argc = hpargc_();
#endif
    int i, len, maxlen=LEN;
    char *argv[LEN], arg[LEN];

    for (i=0; i<argc; i++) {
#      if defined(HPUX)
          len = hpargv_(&i, arg, &maxlen);
#      elif defined(WIN32)
          short n=(short)i, status;
          getarg_(&n, arg, maxlen, &status);
          if(status == -1)Error("getarg failed for argument",i); 
          len = status;
#      else
          getarg_(&i, arg, maxlen);
          for(len = maxlen-2; len && (arg[len] == ' '); len--);
          len++;
#      endif

       arg[len] = '\0'; /* insert string terminator */
       /* printf("%10s, len=%d\n", arg, len);  fflush(stdout); */
       argv[i] = strdup(arg);
  }

  PBEGIN_(argc, argv);
}
#endif

void FATR PBGINF_()
/*
  Alternative entry for those senstive to FORTRAN making reference
  to 7 character external names
*/
{
  PBEGINF_();
}
