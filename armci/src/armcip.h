/* armci private header file */
#ifndef _ARMCI_P_H
#   define _ARMCI_P_H
#   include <stdlib.h> 
#   include "armci.h"

#ifdef WIN32

#define bzero(a,len){\
 int _i;\
 for(_i=0; _i< (len); _i++)((char*)(a))[_i]=0;\
 }
#else
# include <strings.h>
#endif

#if defined (CRAY_T3E) || defined(WIN32)
#define ACC_COPY
#endif

#ifndef FATR
# ifdef WIN32
#   define FATR __stdcall
# else
#   define FATR 
# endif
#endif

/* packing algorithm for double complex numbers requires even number */
#define BUFSIZE_DBL 16384
/*#define BUFSIZE_DBL 8 */
#define BUFSIZE  (BUFSIZE_DBL * sizeof(double))

#define MAX_PROC 8096
#define MAX_STRIDE_LEVEL 10

#define PUT 1
#define GET 2

extern  int armci_me, armci_nproc;
extern  double armci_internal_buffer[BUFSIZE_DBL];

extern void armci_die(char *msg, int code);
extern int armci_op_strided(int op, void* scale, int proc,void *src_ptr, 
			int src_stride_arr[],  
		       void* dst_ptr, int dst_stride_arr[], 
                       int count[], int stride_levels, int lockit);
extern int armci_copy_vector(int op, /* operation code */
                armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              );

extern int armci_acc_vector(int op, /* operation code */
                void *scale,        /* scale factor */
                armci_giov_t darr[],/* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              );


extern int armci_pack_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int fit_level, int nb);


extern void armci_lockmem(void *pstart, void* pend, int proc);
extern void armci_unlockmem(void);

extern int armci_acc_copy_strided(int optype, void* scale, int proc,
                                  void* src_ptr, int src_stride_arr[],  
		                  void* dst_ptr, int dst_stride_arr[], 
                                  int count[], int stride_levels);

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))
#define ACC(op)  (((op)-ARMCI_ACC_INT)>=0)
#endif
