
#include "sndrcv.h"

/*
  These routines use C's knowledge of the sizes of data types
  to generate a portable mechanism for FORTRAN to translate
  between bytes, Integers and doubles. 
*/



Integer FATR MDTOB_(n)
     Integer *n;
/*
  Return the no. of bytes that n doubles occupy
*/
{
  if (*n < 0)
    Error("MDTOB_: negative argument",*n);

  return (Integer) (*n * sizeof(double));
}



Integer FATR MDTOI_(n)
     Integer *n;
/*
  Return the minimum no. of integers which will hold n doubles.
*/
{
  if (*n < 0)
    Error("MDTOI_: negative argument",*n);

   return (Integer) ( (MDTOB_(n) + sizeof(Integer) - 1) / sizeof(Integer) );
}


Integer FATR MITOB_(n)
     Integer *n;
/*
  Return the no. of bytes that n ints=Integers occupy
*/
{
  if (*n < 0)
    Error("MITOB_: negative argument",*n);

  return (Integer) (*n * sizeof(Integer));
}


Integer FATR MITOD_(n)
     Integer *n;
/*
  Return the minimum no. of doubles in which we can store n Integers
*/
{
  if (*n < 0)
    Error("MITOD_: negative argument",*n);

  return (Integer) ( (MITOB_(n) + sizeof(double) - 1) / sizeof(double) );
}
