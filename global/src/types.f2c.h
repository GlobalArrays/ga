/* file types.f2c.h */

#ifndef _INTEGER_DEFINED_
#  define _INTEGER_DEFINED_ 1
#  ifdef  EXT_INT
    typedef long   Integer;
#  else
    typedef int    Integer;
#  endif
#endif

#ifndef _DOUBLE_DEFINED_
#  define _DOUBLE_DEFINED_ 1
#  ifdef  EXT_DBL
    typedef long double  DoublePrecision;
#  else
    typedef double       DoublePrecision;
#  endif
#endif

#ifndef _LOGICAL_DEFINED_
#  define _LOGICAL_DEFINED_ 1
   typedef Integer logical;
#endif

#ifndef _VOID_DEFINED_
#  define _VOID_DEFINED_ 1
#  if defined(__STDC__) || defined(__cplusplus)
     typedef void Void;
#  else
     typedef char Void;
#  endif
#endif


#ifndef _COMPLEX_DEFINED_
#  define _COMPLEX_DEFINED_ 1
   typedef struct{
        DoublePrecision real;
        DoublePrecision imag;
   }DoubleComplex;
#endif

