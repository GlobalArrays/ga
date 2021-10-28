#ifndef _COMEX_COMMON_ACC_H_
#define _COMEX_COMMON_ACC_H_

#include "comex.h"

/* needed for complex accumulate */
typedef struct {
    double real;
    double imag;
} DoubleComplex;

typedef struct {
    float real;
    float imag;
} SingleComplex;

#if SIZEOF_INT == BLAS_SIZE
#define BLAS_INT int
#elif SIZEOF_LONG == BLAS_SIZE
#define BLAS_INT long
#elif SIZEOF_LONG_LONG == BLAS_SIZE
#define BLAS_INT long long
#endif

#if HAVE_BLAS
void sscal(const BLAS_INT *n, const float  *a, const float  *x, const BLAS_INT *incx);
void dscal(const BLAS_INT *n, const double *a, const double *x, const BLAS_INT *incx);
void BLAS_SAXPY(const BLAS_INT *n,  const float *a,  const float *x,
                 const BLAS_INT *incx, float *y, const BLAS_INT *incy);
void BLAS_DAXPY(const BLAS_INT *n,  const double *a,  const double *x,
                 const BLAS_INT *incX, double *Y, const BLAS_INT *incy);
void BLAS_CAXPY(const BLAS_INT *n,  const void *a,  const void *x,
                 const BLAS_INT *incX, void *Y, const BLAS_INT *incy);
void BLAS_ZAXPY(const BLAS_INT *n,  const void *a,  const void *x,
                 const BLAS_INT *incX, void *Y, const BLAS_INT *incy);
void BLAS_SCOPY(const BLAS_INT *n,  const float *x,
                 const BLAS_INT *incx, float *y, const BLAS_INT *incy);
void BLAS_DCOPY(const BLAS_INT *n,  const double *x,
                 const BLAS_INT *incx, double *y, const BLAS_INT *incy);
void BLAS_CCOPY(const BLAS_INT *n,  const void *x,
                 const BLAS_INT *incx, void *y, const BLAS_INT *incy);
void BLAS_ZCOPY(const BLAS_INT *n,  const void *x,
                 const BLAS_INT *incx, void *y, const BLAS_INT *incy);
#endif

#define IADD_SCALE_REG(A,B,C) (A) += (B) * (C)
#define IADD_SCALE_CPL(A,B,C) \
    (A).real += ((B).real*(C).real) - ((B).imag*(C).imag);\
    (A).imag += ((B).real*(C).imag) + ((B).imag*(C).real);
#define MUL_REG(A,B,C) (A) = (B) * (C)
#define MUL_CPL(A,B,C) \
    (A).real = ((B).real*(C).real) - ((B).imag*(C).imag);\
    (A).imag = ((B).real*(C).imag) + ((B).imag*(C).real);

static inline void _scale(
        const int op,
        const int bytes,
        void * const restrict dst,
        const void * const restrict src,
        const void * const restrict scale)
{
#define SCALE_BLAS(COMEX_TYPE, C_TYPE, LETTER)                              \
    if (op == COMEX_TYPE) {                                                 \
        const BLAS_INT ONE = 1;                                             \
        const BLAS_INT N = bytes/sizeof(C_TYPE);                            \
        BLAS_##LETTER##COPY(&N, src, &ONE, dst, &ONE);                      \
        BLAS_##LETTER##AXPY(&N, scale, src, &ONE, dst, &ONE);               \
    } else
#define SCALE(WHICH, COMEX_TYPE, C_TYPE)                                    \
    if (op == COMEX_TYPE) {                                                 \
        int m;                                                              \
        const int m_lim = bytes/sizeof(C_TYPE);                             \
        C_TYPE * const restrict iterator = (C_TYPE * const restrict )dst;   \
        const C_TYPE * const restrict value = (const C_TYPE * const restrict)src;\
        const C_TYPE calc_scale = *(const C_TYPE * const restrict )scale;   \
        for (m = 0 ; m < m_lim; ++m) {                                      \
            MUL_##WHICH(iterator[m], value[m], calc_scale);                 \
        }                                                                   \
    } else
#if HAVE_BLAS
    SCALE_BLAS(COMEX_ACC_DBL, double, D)
    SCALE_BLAS(COMEX_ACC_FLT, float, S)
    SCALE(REG, COMEX_ACC_INT, int)
    SCALE(REG, COMEX_ACC_LNG, long)
    SCALE_BLAS(COMEX_ACC_DCP, DoubleComplex, Z)
    SCALE_BLAS(COMEX_ACC_CPL, SingleComplex, C)
#else
    SCALE(REG, COMEX_ACC_DBL, double)
    SCALE(REG, COMEX_ACC_FLT, float)
    SCALE(REG, COMEX_ACC_INT, int)
    SCALE(REG, COMEX_ACC_LNG, long)
    SCALE(CPL, COMEX_ACC_DCP, DoubleComplex)
    SCALE(CPL, COMEX_ACC_CPL, SingleComplex)
#endif
    {
#ifdef COMEX_ASSERT
        COMEX_ASSERT(0);
#else
        assert(0);
#endif
    }
#undef SCALE_BLAS
#undef SCALE
}

static inline void _acc(
        const int op,
        const int bytes,
        void * const restrict dst,
        const void * const restrict src,
        const void * const restrict scale)
{
#define ACC_BLAS(COMEX_TYPE, C_TYPE, LETTER)                                \
    if (op == COMEX_TYPE) {                                                 \
        const BLAS_INT ONE = 1;                                             \
        const BLAS_INT N = bytes/sizeof(C_TYPE);                            \
        BLAS_##LETTER##AXPY(&N, scale, src, &ONE, dst, &ONE);               \
    } else                                                                 
#define ACC(WHICH, COMEX_TYPE, C_TYPE)                                      \
    if (op == COMEX_TYPE) {                                                 \
        int m;                                                              \
        const int m_lim = bytes/sizeof(C_TYPE);                             \
        C_TYPE * const restrict iterator = (C_TYPE * const restrict)dst;    \
        const C_TYPE * const restrict value = (const C_TYPE * const restrict)src;\
        const C_TYPE calc_scale = *(const C_TYPE * const restrict)scale;    \
        for (m = 0 ; m < m_lim; ++m) {                                      \
            IADD_SCALE_##WHICH(iterator[m], value[m], calc_scale);          \
        }                                                                   \
    } else
#if HAVE_BLAS
    ACC_BLAS(COMEX_ACC_DBL, double, D)
    ACC_BLAS(COMEX_ACC_FLT, float, S)
    ACC(REG, COMEX_ACC_INT, int)
    ACC(REG, COMEX_ACC_LNG, long)
    ACC_BLAS(COMEX_ACC_DCP, DoubleComplex, Z)
    ACC_BLAS(COMEX_ACC_CPL, SingleComplex, C)
#else
    ACC(REG, COMEX_ACC_DBL, double)
    ACC(REG, COMEX_ACC_FLT, float)
    ACC(REG, COMEX_ACC_INT, int)
    ACC(REG, COMEX_ACC_LNG, long)
    ACC(CPL, COMEX_ACC_DCP, DoubleComplex)
    ACC(CPL, COMEX_ACC_CPL, SingleComplex)
#endif
    {
#ifdef COMEX_ASSERT
        COMEX_ASSERT(0);
#else
        assert(0);
#endif
    }
#undef ACC_BLAS
#undef ACC
}

#undef IADD_SCALE_REG
#undef IADD_SCALE_CPL
#undef MUL_REG
#undef MUL_CPL
#undef BLAS_INT

#ifdef ENABLE_DEVICE
/**
 * Skip the intense macro usage and just do this the old-fashion way
 * @param op type of operation including data type
 * @param bytes number of bytes
 * @param dst destination buffer (on device)
 * @param src source buffer
 * @param scale factor for scaling value before accumalation
 */
static inline void _acc_dev(
        const int op,
        const int bytes,
        void * const restrict dst,
        const void * const restrict src,
        const void * const restrict scale)
{
  if (op == COMEX_ACC_DBL) {
  } else if (op == COMEX_ACC_FLT) {
  } else if (op == COMEX_ACC_INT) {
  } else if (op == COMEX_ACC_LNG) {
  } else if (op == COMEX_ACC_DCP) {
  } else if (op == COMEX_ACC_CPL) {
  } else {
  }
}
#endif

#endif /* _COMEX_COMMON_ACC_H_ */
