#include "srftoc.h"
#ifndef IPSC
#include <unistd.h>
#endif

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
