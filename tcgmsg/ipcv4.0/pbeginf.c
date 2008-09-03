/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/pbeginf.c,v 1.16 2006-07-19 00:21:43 manoj Exp $ */

#include <stdio.h>
#include "../farg.h"
#include "srftoc.h"
#include "typesf2c.h"

extern void PBEGIN_();

#ifndef HAS_GETARG
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
  
#if defined(WIN32) || defined(__crayx1)
    int argc = IARGC() + 1;
#elif defined(HPUX)
    int argc = hpargc_();
#else
    int argc = iargc_() + 1;
#endif
    
  int i, len, maxlen=256;
  char *argv[256], arg[256];

  for (i=0; i<argc; i++) {
#if defined(HPUX64)
    Integer ii=i, lmax=maxlen;
    len = hpargv_(&ii, arg, &lmax);
#elif defined(HPUX)
    len = hpargv_(&i, arg, &maxlen);
#else
    getarg_(&i, arg, maxlen);
    for(len = maxlen-2; len && (arg[len] == ' '); len--);
    len++;
#endif
    arg[len] = '\0';
     printf("%10s, len=%d\n", arg, len);  fflush(stdout); 
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
