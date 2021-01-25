#ifndef _CMX_COMMON_ACC_H_
#define _CMX_COMMON_ACC_H_

#include "cmx.h"

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
#define SCALE_BLAS(CMX_TYPE, C_TYPE, LETTER)                              \
    if (op == CMX_TYPE) {                                                 \
        const BLAS_INT ONE = 1;                                             \
        const BLAS_INT N = bytes/sizeof(C_TYPE);                            \
        BLAS_##LETTER##COPY(&N, src, &ONE, dst, &ONE);                      \
        BLAS_##LETTER##AXPY(&N, scale, src, &ONE, dst, &ONE);               \
    } else
#define SCALE(WHICH, CMX_TYPE, C_TYPE)                                    \
    if (op == CMX_TYPE) {                                                 \
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
    SCALE_BLAS(CMX_ACC_DBL, double, D)
    SCALE_BLAS(CMX_ACC_FLT, float, S)
    SCALE(REG, CMX_ACC_INT, int)
    SCALE(REG, CMX_ACC_LNG, long)
    SCALE_BLAS(CMX_ACC_DCP, DoubleComplex, Z)
    SCALE_BLAS(CMX_ACC_CPL, SingleComplex, C)
#else
    SCALE(REG, CMX_ACC_DBL, double)
    SCALE(REG, CMX_ACC_FLT, float)
    SCALE(REG, CMX_ACC_INT, int)
    SCALE(REG, CMX_ACC_LNG, long)
    SCALE(CPL, CMX_ACC_DCP, DoubleComplex)
    SCALE(CPL, CMX_ACC_CPL, SingleComplex)
#endif
    {
#ifdef CMX_ASSERT
        CMX_ASSERT(0);
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
#define ACC_BLAS(CMX_TYPE, C_TYPE, LETTER)                                \
    if (op == CMX_TYPE) {                                                 \
        const BLAS_INT ONE = 1;                                             \
        const BLAS_INT N = bytes/sizeof(C_TYPE);                            \
        BLAS_##LETTER##AXPY(&N, scale, src, &ONE, dst, &ONE);               \
    } else                                                                 
#define ACC(WHICH, CMX_TYPE, C_TYPE)                                      \
    if (op == CMX_TYPE) {                                                 \
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
    ACC_BLAS(CMX_ACC_DBL, double, D)
    ACC_BLAS(CMX_ACC_FLT, float, S)
    ACC(REG, CMX_ACC_INT, int)
    ACC(REG, CMX_ACC_LNG, long)
    ACC_BLAS(CMX_ACC_DCP, DoubleComplex, Z)
    ACC_BLAS(CMX_ACC_CPL, SingleComplex, C)
#else
    ACC(REG, CMX_ACC_DBL, double)
    ACC(REG, CMX_ACC_FLT, float)
    ACC(REG, CMX_ACC_INT, int)
    ACC(REG, CMX_ACC_LNG, long)
    ACC(CPL, CMX_ACC_DCP, DoubleComplex)
    ACC(CPL, CMX_ACC_CPL, SingleComplex)
#endif
    {
#ifdef CMX_ASSERT
        CMX_ASSERT(0);
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

#endif /* _CMX_COMMON_ACC_H_ */
