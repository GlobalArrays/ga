/* ARMCI header file */
#ifndef _ARMCI_H
#define _ARMCI_H   

typedef struct {
    void **src_ptr_array;
    void **dst_ptr_array;
    int  ptr_array_len;
    int bytes;
} armci_giov_t;

#define ARMCI_Put(src, dst, bytes, proc) \
        ARMCI_PutS((src), NULL, (dst), NULL, &(bytes), 0, proc)
#define ARMCI_Get(src, dst, bytes, proc) \
        ARMCI_GetS((src), NULL, (dst), NULL, &(bytes), 0, proc)

extern int ARMCI_Init(void);    /* initialize ARMCI */
extern int ARMCI_PutS(          /* strided put */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc	      /* remote process(or) ID */
                );

extern int ARMCI_AccS(                /* strided accumulate */
                int  optype,          /* operation */
                void *scale,          /* scale factor x += scale*y */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc	      /* remote process(or) ID */
                );



extern int ARMCI_GetS(          /* strided get */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc	      /* remote process(or) ID */
                );

extern int ARMCI_GetV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              );

extern int ARMCI_PutV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              );

extern int ARMCI_AccV( int op,       /* operation code */
                void *scale,         /* scaling factor for accumulate */
                armci_giov_t darr[], /* descriptor array */
                int len,             /* length of descriptor array */
                int proc             /* remote process(or) ID */
              );


extern int ARMCI_Malloc(void* ptr_arr[], int bytes);
extern int ARMCI_Free(void *ptr);

extern void ARMCI_Finalize();    /* terminate ARMCI */
extern void ARMCI_Error(char *msg, int code);
extern void ARMCI_Fence(int proc);
extern void ARMCI_AllFence(void);
extern int  ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc);
extern void ARMCI_Cleanup(void);
extern int ARMCI_Create_mutexes(int num);
extern int ARMCI_Destroy_mutexes(void);
extern void ARMCI_Lock(int mutex, int proc);
extern void ARMCI_Unlock(int mutex, int proc);
extern void ARMCI_Set_shm_limit(unsigned long shmemlimit);
extern int ARMCI_Uses_shm();
extern void ARMCI_Copy(void *src, void *dst, int n);

#define FAIL  -1
#define FAIL2 -2
#define FAIL3 -3
#define FAIL4 -4
#define FAIL5 -5
#define FAIL6 -6
#define FAIL7 -7
#define FAIL8 -8
#define ARMCI_FETCH_AND_ADD 88
#define ARMCI_FETCH_AND_ADD_LONG 89
#define ARMCI_ACC_OFF 100
#define ARMCI_ACC_INT (ARMCI_ACC_OFF + 1)
#define ARMCI_ACC_DBL (ARMCI_ACC_OFF + 2)
#define ARMCI_ACC_FLT (ARMCI_ACC_OFF + 3)
#define ARMCI_ACC_CPL (ARMCI_ACC_OFF + 4)
#define ARMCI_ACC_DCP (ARMCI_ACC_OFF + 5)
#define ARMCI_ACC_LNG (ARMCI_ACC_OFF + 6)

/* PVM group
 * On CrayT3E: the default group is the global group which is (char *)NULL
 *             It is the only working group.
 * On Workstations: the default group is "mp_working_group". User can set
 *                  the group name by calling the ARMCI_PVM_init (defined
 *                  in message.c) and passing the group name to the library.
 */

extern char *mp_group_name;

#endif /* _ARMCI_H */
