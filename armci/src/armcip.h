/* armci private header file */
#ifndef _ARMCI_P_H

#define _ARMCI_P_H
#include <stdlib.h> 
#include "armci.h"
#include "message.h"

#ifdef SYSV
#  define CLUSTER 
#ifndef LAPI
#  define DATA_SERVER
#endif
#endif

#if defined(LAPI) || defined(CLUSTER)
#  include "request.h"
#endif


#ifdef WIN32
#  define bzero(a,len){\
     int _i;\
     for(_i=0; _i< (len); _i++)((char*)(a))[_i]=0;\
   }
#  define bcopy(a,b,len) memcpy(b,a,len)
#else
# include <strings.h>
#endif

#if defined (CRAY_T3E) || defined(FUJITSU)
#define ACC_COPY
#endif

#ifndef FATR
# ifdef WIN32
#   define FATR __stdcall
# else
#   define FATR 
# endif
#endif

#define MAX_PROC 8096
#define MAX_STRIDE_LEVEL 8

/* packing algorithm for double complex numbers requires even number */
#ifdef MSG_BUFLEN_DBL
#  define BUFSIZE_DBL (MSG_BUFLEN_DBL - sizeof(request_header_t)/sizeof(double)\
                       - 3*MAX_STRIDE_LEVEL)
#else
#  define BUFSIZE_DBL 32768
#endif

#define BUFSIZE  (BUFSIZE_DBL * sizeof(double))

/* note opcodes must be lower than ARMCI_ACC_OFF !!! */
#define PUT 1
#define GET 3
#define RMW 5
#define STRIDED 7
#define VECTOR  8
#define LOCK   20
#define UNLOCK 21

extern  int armci_me, armci_nproc;
extern  double armci_internal_buffer[BUFSIZE_DBL];

extern void armci_shmem_init();
extern void armci_die(char *msg, int code);
extern void armci_die2(char *msg, int code1, int code2);
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

extern int armci_rem_strided(int op, void* scale, int proc,void *src_ptr,
                        int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int lockit);

extern int armci_pack_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int fit_level, int nb);

extern int armci_pack_vector(int op, void *scale, 
                      armci_giov_t darr[],int len,int proc);

extern void armci_lockmem(void *pstart, void* pend, int proc);
extern void armci_unlockmem(int proc);

extern int armci_acc_copy_strided(int optype, void* scale, int proc,
                                  void* src_ptr, int src_stride_arr[],  
		                  void* dst_ptr, int dst_stride_arr[], 
                                  int count[], int stride_levels);

extern void armci_vector_to_buf(armci_giov_t darr[], int len, void* buf);
extern void armci_vector_from_buf(armci_giov_t darr[], int len, void* buf);
extern void armci_init_fence();

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))
#define ACC(op)  (((op)-ARMCI_ACC_INT)>=0)

#ifdef CLUSTER
   extern char *_armci_fence_arr;
#  define SAMECLUSNODE(p)\
     ( ((p) <= armci_clus_last) && ((p) >= armci_clus_first) )
#else
#  define SAMECLUSNODE(p) ((p)==armci_me) 
#endif


#ifdef LAPI
#  define ORDER(op,proc)\
        if( proc == armci_me || ( ACC(op) && ACC(PENDING_OPER(proc))) );\
        else  FENCE_NODE(proc)
#elif defined(CLUSTER)
#  define ORDER(op,proc)\
        if(proc != armci_me && op != GET )_armci_fence_arr[proc]=1
#else
#  define ORDER(op,proc) if(proc != armci_me) FENCE_NODE(proc) 
#endif
        
typedef struct {
    int  ptr_array_len;
    int bytes;
    void **ptr_array;
} armci_riov_t;

/*\ consider up to HOSTNAME_LEN characters in host name 
 *  we can truncate names of the SP nodes since it is not used
 *  to establish socket communication like on the networks of workstations
 *  SP node names must be distinct within first HOSTNAME_LEN characters
\*/
#ifdef LAPI
#  define HOSTNAME_TRUNCATE 
#  define HOSTNAME_LEN 12
#else
#  define HOSTNAME_LEN 64
#endif

typedef struct {
  int master;
  int nslave;
  char hostname[HOSTNAME_LEN];
} armci_clus_t;

extern armci_clus_t *armci_clus_info;
extern int armci_nclus, armci_clus_me, armci_master;
extern int armci_clus_first, armci_clus_last;
extern int armci_clus_id(int p);
extern void armci_serv_attach_req(void *info, int ilen, long size, 
                                  void* resp,int rlen);

#endif
