/* file typesf2c.h */
#ifndef _TYPES_F2C_H_
#define _TYPES_F2C_H_

#  ifdef WIN32
#    include "winf2c.h"
#  else
#    define FATR 
#  endif

#  ifdef  EXT_INT
    typedef long   Integer;
#  else
    typedef int    Integer;
#  endif

#  ifdef  EXT_DBL
    typedef long double  DoublePrecision;
#  else
    typedef double       DoublePrecision;
#  endif

   typedef Integer logical;
   typedef Integer Logical;

#  if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
     typedef void Void;
#  else
     typedef char Void;
#  endif


   typedef struct{
        DoublePrecision real;
        DoublePrecision imag;
   }DoubleComplex;

#endif
