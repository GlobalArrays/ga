#include "srftoc.h"

extern long random();
extern int srandom();

Double DRAND48_()
{
  return ( (Double) random() ) * 4.6566128752458e-10;
}

void SRAND48_(seed)
  unsigned *seed;
{
  (void) srandom(*seed);
}
