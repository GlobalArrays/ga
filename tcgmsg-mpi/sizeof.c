
#include "sndrcv.h"

/*
  These routines use C's knowledge of the sizes of data types
  to generate a portable mechanism for FORTRAN to translate
  between bytes, Ints and Doubles. 
*/



Int MDTOB_(n)
     Int *n;
/*
  Return the no. of bytes that n Doubles occupy
*/
{
  if (*n < 0)
    Error("MDTOB_: negative argument",*n);

  return (Int) (*n * sizeof(Double));
}



Int MDTOI_(n)
     Int *n;
/*
  Return the minimum no. of integers which will hold n Doubles.
*/
{
  if (*n < 0)
    Error("MDTOI_: negative argument",*n);

   return (Int) ( (MDTOB_(n) + sizeof(Int) - 1) / sizeof(Int) );
}


Int MITOB_(n)
     Int *n;
/*
  Return the no. of bytes that n ints=Ints occupy
*/
{
  if (*n < 0)
    Error("MITOB_: negative argument",*n);

  return (Int) (*n * sizeof(Int));
}


Int MITOD_(n)
     Int *n;
/*
  Return the minimum no. of Doubles in which we can store n Ints
*/
{
  if (*n < 0)
    Error("MITOD_: negative argument",*n);

  return (Int) ( (MITOB_(n) + sizeof(Double) - 1) / sizeof(Double) );
}
