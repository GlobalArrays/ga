#ifndef _COMEX_COMMON_ACC_H_
#define _COMEX_COMMON_ACC_H_

/* needed for complex accumulate */
typedef struct {
    double real;
    double imag;
} DoubleComplex;

typedef struct {
    float real;
    float imag;
} SingleComplex;

static inline void _acc(
        int op,
        int bytes,
        void * restrict dst,
        const void * restrict src,
        void *scale)
{
#define EQ_ONE_REG(A) ((A) == 1.0)
#define EQ_ONE_CPL(A) ((A).real == 1.0 && (A).imag == 0.0)
#define IADD_REG(A,B) (A) += (B)
#define IADD_CPL(A,B) (A).real += (B).real; (A).imag += (B).imag
#define IADD_SCALE_REG(A,B,C) (A) += (B) * (C)
#define IADD_SCALE_CPL(A,B,C) \
    (A).real += ((B).real*(C).real) - ((B).imag*(C).imag);\
    (A).imag += ((B).real*(C).imag) + ((B).imag*(C).real);
#define ACC_BLAS(COMEX_TYPE, C_TYPE, FUNC)                              \
    if (op == COMEX_TYPE) {                                             \
        int ONE = 1;                                                    \
        int N = bytes/sizeof(C_TYPE);                                   \
        FUNC(&N, scale, src, &ONE, dst, &ONE);                          \
    } else
#define ACC(WHICH, COMEX_TYPE, C_TYPE)                                  \
    if (op == COMEX_TYPE) {                                             \
        int m;                                                          \
        int m_lim = bytes/sizeof(C_TYPE);                               \
        C_TYPE *iterator = (C_TYPE *)dst;                               \
        C_TYPE *value = (C_TYPE *)src;                                  \
        C_TYPE calc_scale = *(C_TYPE *)scale;                           \
        if (EQ_ONE_##WHICH(calc_scale)) {                               \
            for (m = 0 ; m < m_lim; ++m) {                              \
                IADD_##WHICH(iterator[m], value[m]);                    \
            }                                                           \
        }                                                               \
        else {                                                          \
            for (m = 0 ; m < m_lim; ++m) {                              \
                IADD_SCALE_##WHICH(iterator[m], value[m], calc_scale);  \
            }                                                           \
        }                                                               \
    } else
#if HAVE_BLAS
    ACC_BLAS(COMEX_ACC_DBL, double, BLAS_DAXPY)
    ACC_BLAS(COMEX_ACC_FLT, float, BLAS_SAXPY)
    ACC(REG, COMEX_ACC_INT, int)
    ACC(REG, COMEX_ACC_LNG, long)
    ACC_BLAS(COMEX_ACC_DCP, DoubleComplex, BLAS_ZAXPY)
    ACC_BLAS(COMEX_ACC_CPL, SingleComplex, BLAS_CAXPY)
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
#undef ACC
#undef EQ_ONE_REG
#undef EQ_ONE_CPL
#undef IADD_REG
#undef IADD_CPL
#undef IADD_SCALE_REG
#undef IADD_SCALE_CPL
}

#endif /* _COMEX_COMMON_ACC_H_ */
