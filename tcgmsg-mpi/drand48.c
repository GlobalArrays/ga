#include "sndrcv.h"
#include <stdio.h>
#ifdef CRAY_YMP
#include <stdlib.h>
#else
extern long random();
extern int srandom();
#endif

double FATR DRAND48_()
{
  double val=((double) random() ) * 4.6566128752458e-10;
  return val;
}

void FATR SRAND48_(seed)
  unsigned *seed;
{
  (void) srandom(*seed);
}


double ran(unsigned int flag)
{
  static unsigned long seed = 76521;

  if(flag != 0) seed = flag;

  seed = seed *1812433253 + 12345;

  return ((double) (seed & 0x7fffffff)) * 4.6566128752458e-10;
}

double drand_(flag)
    unsigned long *flag;
{
/* on YMP/J90 need to use thread safe version of rand */
#ifdef CRAY_YMP

  return ran((unsigned int)*flag);

#else
  if (*flag)
    srandom((unsigned) *flag);

  return ((double) random()) * 4.6566128752458e-10;
#endif
}

double FATR DRAND(flag)
    unsigned long *flag;
{
return (drand_(flag));
}

