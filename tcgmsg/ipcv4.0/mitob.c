/*$Id: mitob.c,v 1.2 1995-02-02 23:25:17 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/mitob.c,v 1.2 1995-02-02 23:25:17 d3g681 Exp $ */

#include "sndrcv.h"

/*
  These routines use C's knowledge of the sizes of data types
  to generate a portable mechanism for FORTRAN to translate
  between bytes, integers and doubles. Note that we assume that
  FORTRAN integers are the same size as C longs.
*/

long MITOB_(n)
     long *n;
/*
  Return the no. of bytes that n ints=longs occupy
*/
{
  if (*n < 0)
    Error("MITOB_: negative argument",*n);

  return (long) (*n * sizeof(long));
}
