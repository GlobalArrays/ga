/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/pbeginf.c,v 1.15 2003-09-06 01:29:19 edo Exp $ */

#include <stdio.h>
#include "farg.h"
#include "sndrcv.h"
#define LEN 255

extern void PBEGIN_();

#if defined(HPUX) || defined(SUN) || defined(SOLARIS) ||defined(PARAGON) ||defined(FUJITSU) || defined(WIN32) ||defined(LINUX64) || defined(NEC)||(defined(LINUX) & !defined(IFCV8)) || defined(HITACHI) || defined(__crayx1)
#define HAS_GETARG 1
#endif

#ifdef WIN32
#define getarg_ GETARG
extern int FATR IARGC(void);
#include <windows.h>
#include "winutil.h"
#define NTYPE short
extern void FATR getarg_( NTYPE *, char*, int, NTYPE*);
#else
#define FATR 
#endif

#if defined(__crayx1) || defined(IFCV8)
#define getarg_  pxfgetarg_
#define IARGC  ipxfargc_
#define NTYPE  int 
extern void FATR getarg_( NTYPE *, char*, NTYPE*, NTYPE*, int);
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

#if defined(WIN32) || defined(__crayx1)
    int argc = IARGC() + 1;
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
