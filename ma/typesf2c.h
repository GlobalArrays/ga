#ifndef _TYPES_F2C_H_
#define _TYPES_F2C_H_

/*
 * $Id: typesf2c.h,v 1.3 2000-07-04 05:54:56 d3g001 Exp $
 */

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

#endif /* _TYPES_F2C_H_ */
