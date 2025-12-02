#ifndef TYPESF2C_H_
#define TYPESF2C_H_

#if defined(WIN32) &&!defined(__MINGW32__)
#   define FATR __stdcall
#else
#   define FATR 
#endif

typedef long Integer;
typedef float Real;
typedef double DoublePrecision;

typedef Integer logical;
typedef Integer Logical;

typedef struct {
    DoublePrecision real;
    DoublePrecision imag;
} DoubleComplex;

typedef struct {
    Real real;
    Real imag;
} SingleComplex;

typedef long intp;

#endif /* TYPESF2C_H_ */
