#ifndef __GPBASE_H__
#define __GPBASE_H__

#include "gacommon.h"
#include "typesf2c.h"

#if SIZEOF_VOIDP == SIZEOF_INT
#   define GP_POINTER_TYPE C_INT
#elif SIZEOF_VOIDP == SIZEOF_LONG
#   define GP_POINTER_TYPE C_LONG
#else
#   error sizeof(void*) is not sizeof(int) nor sizeof(long)
#endif

/* Set maximum number of Global Pointer Arrays */
#define MAX_GP_ARRAYS 1024

/* Set maximum dimension of Global Pointer ARRAYS */
#define MAX_GP_DIM 7

/* Define handle numbering offset for GP Arrays */ 
#define GP_OFFSET 1000

typedef struct{
  Integer gp_size_array;      /* Handle to Global Array holding sizes    */
  Integer gp_ptr_array;       /* Handle to Global Array holding pointers */
  Integer active;             /* Handle is currently active              */
  Integer ndim;               /* Dimension of GP                         */
  Integer dims[MAX_GP_DIM];   /* Axes dimensions of GP                   */
  Integer lo[MAX_GP_DIM];     /* Lower indices of local block            */
  Integer hi[MAX_GP_DIM];     /* Upper indices of local block            */
  Integer ld[MAX_GP_DIM-1];   /* Stride of local block                   */
} gp_array_t;

extern gp_array_t *GP;
#endif /* __GPBASE_H__ */
