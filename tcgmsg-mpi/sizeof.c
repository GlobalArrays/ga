
#include "sndrcv.h"

/*
  These routines use C's knowledge of the sizes of data types
  to generate a portable mechanism for FORTRAN to translate
  between bytes, integers and doubles. 
*/



long MDTOB_(n)
     long *n;
/*
  Return the no. of bytes that n doubles occupy
*/
{
  if (*n < 0)
    Error("MDTOB_: negative argument",*n);

  return (long) (*n * sizeof(double));
}



long MDTOI_(n)
     long *n;
/*
  Return the minimum no. of integers which will hold n doubles.
*/
{
  if (*n < 0)
    Error("MDTOI_: negative argument",*n);

   return (long) ( (MDTOB_(n) + sizeof(long) - 1) / sizeof(long) );
}


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


long MITOD_(n)
     long *n;
/*
  Return the minimum no. of doubles in which we can store n longs
*/
{
  if (*n < 0)
    Error("MITOD_: negative argument",*n);

  return (long) ( (MITOB_(n) + sizeof(double) - 1) / sizeof(double) );
}
