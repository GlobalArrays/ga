/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv5.0/drand48.c,v 1.1 1994-12-29 06:57:06 og845 Exp $ */

#include "srftoc.h"

extern long random();
extern int srandom();

double DRAND48_()
{
  return ( (double) random() ) * 4.6566128752458e-10;
}

void SRAND48_(seed)
  unsigned *seed;
{
  (void) srandom(*seed);
}
