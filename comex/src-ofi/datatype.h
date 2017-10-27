#ifndef DATATYPE_H_
#define DATATYPE_H_

#include "log.h"

#define COMEX_DTYPES_COUNT (COMEX_ACC_LNG - COMEX_ACC_OFF)

// comex datatype index starting from 0
#define COMEX_DTYPE_IDX(comex_dtype) (comex_dtype - COMEX_ACC_OFF - 1)

enum fi_datatype comex_to_fi_dtype[COMEX_DTYPES_COUNT] = 
{FI_INT32, FI_DOUBLE, FI_FLOAT, FI_FLOAT_COMPLEX, FI_DOUBLE_COMPLEX, FI_INT64};

#define GET_FI_DTYPE(comex_dtype) \
  (COMEX_DTYPE_IDX(comex_dtype) < COMEX_DTYPES_COUNT ? comex_to_fi_dtype[COMEX_DTYPE_IDX(comex_dtype)] : -1)

#define COMEX_DTYPE_SIZEOF(comex_dtype, datasize)                 \
  do                                                              \
  {                                                               \
      switch (comex_dtype)                                        \
      {                                                           \
          case COMEX_ACC_INT:                                     \
              datasize = sizeof(int);                             \
              break;                                              \
          case COMEX_ACC_DBL:                                     \
              datasize = sizeof(double);                          \
              break;                                              \
          case COMEX_ACC_FLT:                                     \
              datasize = sizeof(float);                           \
              break;                                              \
          case COMEX_ACC_LNG:                                     \
              datasize = sizeof(long);                            \
              break;                                              \
          case COMEX_ACC_DCP:                                     \
              datasize = sizeof(DoubleComplex);                   \
              break;                                              \
          case COMEX_ACC_CPL:                                     \
              datasize = sizeof(SingleComplex);                   \
              break;                                              \
          default:                                                \
              COMEX_OFI_LOG(WARN, "incorrect comex_datatype: %d", \
                            comex_dtype);                         \
              assert(0);                                          \
              goto fn_fail;                                       \
              break;                                              \
      }                                                           \
} while (0)

/* needed for complex accumulate */
typedef struct
{
    double real;
    double imag;
} DoubleComplex;

typedef struct
{
    float real;
    float imag;
} SingleComplex;

static inline int scale_is_1(int datatype, void* scale)
{
    if (!scale) return 1;

    switch (datatype)
    {
        case COMEX_ACC_INT:
            return *(int*)scale == 1;
        case COMEX_ACC_DBL:
            return *(double*)scale == 1.0;
        case COMEX_ACC_FLT:
            return *(float*)scale == 1.0f;
        case COMEX_ACC_LNG:
            return *(long*)scale == 1;
        case COMEX_ACC_DCP:
            return (((DoubleComplex*)scale)->real == 1.0 &&
                    ((DoubleComplex*)scale)->imag == 0.0);
        case COMEX_ACC_CPL:
            return (((SingleComplex*)scale)->real == 1.0f &&
                    ((SingleComplex*)scale)->imag == 0.0f);
        default:
            COMEX_OFI_LOG(WARN, "scale_is_1: incorrect data type: %d", datatype);
            assert(0);
            return 1;
    }
}

#endif /* DATATYPE_H_ */
