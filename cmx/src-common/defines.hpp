/* cmx defines header file */
#ifndef _CMX_COMMON_DEFINES_H
#define _CMX_COMMON_DEFINES_H

#define CMX_SUCCESS 0
#define CMX_FAILURE 1

#define CMX_SWAP 10
#define CMX_SWAP_LONG 11
#define CMX_FETCH_AND_ADD 12
#define CMX_FETCH_AND_ADD_LONG 13

#define CMX_ACC_OFF 36
#define CMX_ACC_INT (CMX_ACC_OFF + 1)
#define CMX_ACC_DBL (CMX_ACC_OFF + 2)
#define CMX_ACC_FLT (CMX_ACC_OFF + 3)
#define CMX_ACC_CPL (CMX_ACC_OFF + 4)
#define CMX_ACC_DCP (CMX_ACC_OFF + 5)
#define CMX_ACC_LNG (CMX_ACC_OFF + 6)

#define CMX_MAX_STRIDE_LEVEL 8

#define CMX_NOT_SET 0
#define CMX_INT     1
#define CMX_LONG    2
#define CMX_FLOAT   3
#define CMX_DOUBLE  4
#define CMX_COMPLEX 5
#define CMX_DCMPLX  6
#define CMX_USER    7

typedef int cmxInt;

typedef struct {
    void **loc; /**< array of local starting addresses */
    cmxInt *rem; /**< array of remote offsets */
    cmxInt count; /**< size of address arrays (src[count],dst[count]) */
    cmxInt bytes; /**< length in bytes for each src[i]/dst[i] pair */
} cmx_giov_t;

#endif /* _CMX_H */
