/* file types.f2c.h */

#ifndef _matypes_h
#ifdef  EXT_INT
  typedef long   Integer;
#else
  typedef int    Integer;
#endif
#endif

#ifdef  EXT_DBL
  typedef long double  DoublePrecision;
#else
  typedef double       DoublePrecision;
#endif

typedef Integer logical;

#if defined(__STDC__) || defined(__cplusplus)
   typedef void Void;
#else
   typedef char Void;
#endif


typedef struct{
        DoublePrecision real;
        DoublePrecision imag;
}DoubleComplex;

