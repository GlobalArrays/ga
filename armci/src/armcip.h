/* armci private header file */
#ifndef _ARMCI_P_H
#   define _ARMCI_P_H
#   include <stdlib.h> 
#   include "armci.h"

#define MAX_STRIDE_LEVEL 10
#define PUT 1
#define GET 2
extern  int armci_me, armci_nproc;

extern void armci_die(char *msg, int code);
extern int armci_op_strided(int op, void* scale, int proc,void *src_ptr, 
			int src_stride_arr[],  
		       void* dst_ptr, int dst_stride_arr[], 
                       int count[], int stride_levels);
extern int armci_copy_vector(int op, /* operation code */
                armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              );

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))
#endif
