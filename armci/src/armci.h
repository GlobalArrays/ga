/* ARMCI header file */
#ifndef _ARMCI_H
#define _ARMCI_H   

typedef struct {
    void **src_ptr_array;
    void **dst_ptr_array;
    int  ptr_array_len;
    int bytes;
} armci_giov_t;
extern int armci_notify(int proc);
extern int armci_notify_wait(int proc,int *pval);
extern int ARMCI_Init(void);    /* initialize ARMCI */

extern int ARMCI_Put(void *src, void* dst, int bytes, int proc);
extern int ARMCI_Put_flag(void *src, void* dst,int bytes,int *f,int v,int proc);

extern int ARMCI_PutS(          /* strided put */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc	      /* remote process(or) ID */
                );


extern int ARMCI_PutS_flag(
                void *src_ptr,        /* pointer to 1st segment at source*/
                int src_stride_arr[], /* array of strides at source */
                void* dst_ptr,        /* pointer to 1st segment at destination*/
                int dst_stride_arr[], /* array of strides at destination */
                int count[],          /* number of segments at each stride 
                                         levels: count[0]=bytes*/
                int stride_levels,    /* number of stride levels */
                int *flag,            /* pointer to remote flag */
                int val,              /* value to set flag upon completion of
                                         data transfer */
                int proc              /* remote process(or) ID */
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


extern int ARMCI_Get(void *src, void* dst, int bytes, int proc);

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

extern int ARMCI_PutValueInt(int src,     /* value in a register to put     */
			     void *dst,   /* dest starting addr to put data */
			     int proc     /* remote process (or) ID         */
			     );

extern int ARMCI_PutValueLong(long src,   /* value in a register to put     */
			      void *dst,  /* dest starting addr to put data */
			      int proc    /* remote process (or) ID         */
			      );

extern int ARMCI_PutValueFloat(float src, /* value in a register to put     */
			       void *dst, /* dest starting addr to put data */
			       int proc   /* remote process (or) ID         */
			       );

extern int ARMCI_PutValueDouble(double src,/* value in a register to put     */
				void *dst, /* dest starting addr to put data */
				int proc   /* remote process (or) ID         */
				);

extern int ARMCI_GetValue(void *src, /* source starting addr to put    */
			  void *dst, /* dest starting addr to put data */
			  int proc,  /* remote process (or) ID         */
			  int bytes /* size of the data               */
			  );

extern int ARMCI_Malloc(void* ptr_arr[], int bytes);
extern int ARMCI_Free(void *ptr);
extern int ARMCI_Same_node(int proc);

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
#define ARMCI_SWAP 90
#define ARMCI_SWAP_LONG 91

#define ARMCI_ACC_OFF 100
#define ARMCI_ACC_INT (ARMCI_ACC_OFF + 1)
#define ARMCI_ACC_DBL (ARMCI_ACC_OFF + 2)
#define ARMCI_ACC_FLT (ARMCI_ACC_OFF + 3)
#define ARMCI_ACC_CPL (ARMCI_ACC_OFF + 4)
#define ARMCI_ACC_DCP (ARMCI_ACC_OFF + 5)
#define ARMCI_ACC_LNG (ARMCI_ACC_OFF + 6)

#define ARMCI_MAX_STRIDE_LEVEL 8

/************ locality information **********************************************/
typedef int armci_domain_t;
#define ARMCI_DOMAIN_SMP 0        /* SMP node domain for armci_domain_XXX calls */
extern int armci_domain_nprocs(armci_domain_t domain, int id);
extern int armci_domain_id(armci_domain_t domain, int glob_proc_id);
extern int armci_domain_glob_proc_id(armci_domain_t domain, int id, int loc_proc_id);
extern int armci_domain_my_id(armci_domain_t domain);
extern int armci_domain_count(armci_domain_t domain);



/* PVM group
 * On CrayT3E: the default group is the global group which is (char *)NULL
 *             It is the only working group.
 * On Workstations: the default group is "mp_working_group". User can set
 *                  the group name by calling the ARMCI_PVM_init (defined
 *                  in message.c) and passing the group name to the library.
 */

extern char *mp_group_name;

/*********************stuff for non-blocking API******************************/
/*\ the request structure for non-blocking api. 
\*/
typedef struct{
    int data[4];
#if defined(_AIX) 
#   if defined(__64BIT__)
    double dummy[27]; /*lapi_cntr_t is 200 bytes, using 216 just to be safe*/ 
#   else
    double dummy[20]; /*lapi_cntr_t is 148 bytes, using 166 just to be safe*/ 
#   endif
#else
    double dummy;
#endif
} armci_hdl_t;

#define armci_req_t armci_hdl_t

extern int ARMCI_NbPut(void *src, void* dst, int bytes, int proc,armci_hdl_t* nb_handle);

extern int ARMCI_NbPutS(          /* strided put */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc,	      /* remote process(or) ID */
                armci_hdl_t* nb_handle /*armci_non-blocking request handle*/
                );

extern int ARMCI_NbAccS(                /* strided accumulate */
                int  optype,          /* operation */
                void *scale,          /* scale factor x += scale*y */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc,	      /* remote process(or) ID */
                armci_hdl_t* nb_handle /*armci_non-blocking request handle*/
                );

extern int ARMCI_NbGet(void *src, void* dst, int bytes, int proc,armci_hdl_t* nb_handle);

extern int ARMCI_NbGetS(          /* strided get */
                void *src_ptr,        /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of units at each stride level count[0]=bytes */
		int stride_levels,    /* number of stride levels */
                int proc,	      /* remote process(or) ID */
                armci_hdl_t* nb_handler/*armci_non-blocking request handle*/
                );

extern int ARMCI_NbGetV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc,  /* remote process(or) ID */
                armci_hdl_t* nb_handle /*armci_non-blocking request handle*/
              );

extern int ARMCI_NbPutV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc,  /* remote process(or) ID */
                armci_hdl_t* nb_handle /*armci_non-blocking request handle*/
              );

extern int ARMCI_NbAccV( int op,       /* operation code */
                void *scale,         /* scaling factor for accumulate */
                armci_giov_t darr[], /* descriptor array */
                int len,             /* length of descriptor array */
                int proc,             /* remote process(or) ID */
                armci_hdl_t* nb_handle /*armci_non-blocking request handle*/
              );

extern int ARMCI_NbPutValueInt(int src,   /* value in a register to put     */
			       void *dst, /* dest starting addr to put data */
			       int proc,  /* remote process (or) ID         */
			       armci_hdl_t* nb_handle /*armci_non-blocking 
						       request handle       */
			       );

extern int ARMCI_NbPutValueLong(long src,  /* value in a register to put     */
				void *dst, /* dest starting addr to put data */
				int proc,  /* remote process (or) ID         */
				armci_hdl_t* nb_handle /*armci_non-blocking 
							request handle       */
				);

extern int ARMCI_NbPutValueFloat(float src,/* value in a register to put     */
				 void *dst,/* dest starting addr to put data */
				 int proc, /* remote process (or) ID         */
				 armci_hdl_t* nb_handle /*armci_non-blocking 
							 request handle      */
				 );

extern int ARMCI_NbPutValueDouble(double src,/* value in a register to put   */
				  void *dst,/* dest starting addr to put data*/
				  int proc,   /* remote process (or) ID      */
				  armci_hdl_t* nb_handle /*armci_non-blocking 
							  request handle     */
				  );

extern int ARMCI_NbGetValue(void *src, /* src starting addr   */
			    void *dst, /* dest starting addr */
			    int proc,  /* remote process (or) ID      */
			    int bytes, /* size of the data type */
			    armci_hdl_t* nb_handle /*armci_non-blocking 
						    request handle     */
			    );

extern int ARMCI_Wait(armci_hdl_t* nb_handle); /*non-blocking request handle*/

extern int ARMCI_WaitAll (void);

extern int ARMCI_WaitProc (int proc);

extern void ARMCI_SET_AGGREGATE_HANDLE(armci_hdl_t* nb_handle);

extern void ARMCI_UNSET_AGGREGATE_HANDLE(armci_hdl_t* nb_handle);

#define ARMCI_INIT_HANDLE(hdl) ((double *)((hdl)->data))[0]=0; \
  ((double *)((hdl)->data))[1]=0; 

#endif /* _ARMCI_H */
