/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/drand48.c,v 1.1.1.1 1994-03-29 06:44:46 d3g681 Exp $ */

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
