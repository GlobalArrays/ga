/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/pbeginf.c,v 1.18.4.1 2006-12-14 13:24:55 manoj Exp $ */

#include <stdio.h>
#include "../tcgmsg/farg.h"
#include "sndrcv.h"
#define LEN 255

extern void PBEGIN_();

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

#if defined(WIN32) || defined(__crayx1)
    int argc = IARGC() + 1;
#elif defined(HPUX)
    int argc = hpargc_();
#else
    int argc = iargc_() + 1;
#endif
    
    int i, len, maxlen=LEN;
    char *argv[LEN], arg[LEN];

    for (i=0; i<argc; i++) {
#      if defined(HPUX64) && defined(EXT_INT)
          long ii=i, lmax=LEN;
          len = hpargv_(&ii, arg, &lmax);
#      elif defined(HPUX)
          len = hpargv_(&i, arg, &maxlen);
#      elif defined(WIN32) 
          short n=(short)i, status;
          getarg_(&n, arg, maxlen, &status);
          if(status == -1)Error("getarg failed for argument",i); 
          len = status;
#      elif  defined(__crayx1)
          NTYPE n=(NTYPE)i, status,ilen;
          getarg_(&n, arg, &ilen, &status,maxlen);
          if(status )Error("getarg failed for argument",i); 
          len=(int)ilen;
#      else
          getarg_(&i, arg, maxlen);
          for(len = maxlen-2; len && (arg[len] == ' '); len--);
          len++;
#      endif

       arg[len] = '\0'; /* insert string terminator */
       /*printf("%10s, len=%d\n", arg, len);  fflush(stdout);*/ 
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
