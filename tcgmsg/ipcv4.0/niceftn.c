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
