/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/pbeginf.c,v 1.3 1995-02-24 02:14:12 d3h325 Exp $ */

#include <stdio.h>
#include "farg.h"
#include "srftoc.h"

extern void PBEGIN_();

#if !(defined(HPUX) || defined(SUNF77_2))
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
  Hewlett Packard Risc box and new SparcWorks F77 2.* compilers.
  Have to construct the argument list by calling FORTRAN.
*/
{
  extern char *strdup();
#ifdef SUNF77_2
  extern int iargc_();
  extern void getarg_();
  int argc = iargc_() + 1;
#else
#ifndef EXTNAME
#define hpargv_ hpargv
#define hpargc_ hpargc
#endif
  extern int hpargv_();
  extern int hpargc_();
  int argc = hpargc_();
#endif
  int i, len, maxlen=256;
  char *argv[256], arg[256];

  for (i=0; i<argc; i++) {
#ifdef SUNF77_2
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
