#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "srftoc.h"
#include <unistd.h>

/*
  Wrapper around nice for FORTRAN users courtesy of Rick Kendall
  ... C has the system interface
*/
int NICEFTN_(int * ival)
{
  return nice(*ival);
}
