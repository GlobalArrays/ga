/*$Id: niceftn.c,v 1.2 1995-02-02 23:25:22 d3g681 Exp $*/
#include "srftoc.h"

int NICEFTN_(ival)
     int *ival;
/*
  Wrapper around nice for FORTRAN users courtesy of Rick Kendall
  ... C has the system interface
*/
{
#ifndef IPSC
  return nice(*ival);
#else
  return 0;
#endif
}
