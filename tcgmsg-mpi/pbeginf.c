/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/pbeginf.c,v 1.3 1997-10-14 23:10:52 d3h325 Exp $ */

#include <stdio.h>
#include "farg.h"
#include "srftoc.h"

extern void PBEGIN_();

#if defined(HPUX) || defined(SUNF77_2) ||defined(PARAGON) ||defined(FUJITSU)
#define HAS_GETARG 1
#endif


#if !defined(HAS_GETARG)
void PBEGINF_()
/*
  Interface routine between FORTRAN and c version of pbegin.
  Relies on knowing global address of program argument list ... see farg.h
*/
{
  PBEGIN_(ARGC_, ARGV_);
}
#else
void PBEGINF_()
/*
  Hewlett Packard Risc box, SparcWorks F77 2.* and Paragon compilers.
  Have to construct the argument list by calling FORTRAN.
*/
{
  extern char *strdup();
#if defined(HAS_GETARG)
  extern int iargc_();
  extern void getarg_();
  int argc = iargc_() + 1;
#else
#  ifndef EXTNAME
#    define hpargv_ hpargv
#    define hpargc_ hpargc
#  endif
   extern int hpargv_();
   extern int hpargc_();
   int argc = hpargc_();
#endif
  int i, len, maxlen=256;
  char *argv[256], arg[256];

  for (i=0; i<argc; i++) {
#if defined(HAS_GETARG)
    getarg_(&i, arg, maxlen);
    for(len = maxlen-2; len && (arg[len] == ' '); len--);
    len++;
#else
    len = hpargv_(&i, arg, &maxlen);
#endif
    arg[len] = '\0';
    /* printf("%10s, len=%d\n", arg, len);  fflush(stdout); */
    argv[i] = strdup(arg);
  }

  PBEGIN_(argc, argv);
}
#endif

void PBGINF_()
/*
  Alternative entry for those senstive to FORTRAN making reference
  to 7 character external names
*/
{
  PBEGINF_();
}
