/* file types.h */

#ifdef  EXT_INT
  typedef long   Int;
#else
  typedef int    Int;
#endif

#ifdef  EXT_DBL
  typedef long double  Double;
#else
  typedef double       Double;
#endif

#if defined(__STDC__) || defined(__cplusplus)
   typedef void Void;
#else
   typedef char Void;
#endif
