/* armci private header file */
#ifndef _ARMCI_P_H

#define _ARMCI_P_H
#include <stdlib.h> 
#include "armci.h"
#include "message.h"

#ifdef WIN32
#include <windows.h>
#define sleep(x) Sleep(100*(x))
#else
#include <unistd.h>
#endif

#if (defined(SYSV) || defined(WIN32)) && !defined(NO_SHM)
#define CLUSTER 

#ifdef SERVER_THREAD
#  define SERVER_NODE(c) (armci_clus_info[(c)].master);
#else
#  define SOFFSET -10000
#  define SERVER_NODE(c) ((int)(SOFFSET -armci_clus_info[(c)].master));
#endif

#endif

#if defined(LAPI) || defined(PTHREADS)
# include <pthread.h>
  typedef pthread_t thread_id_t;
# define  THREAD_ID_SELF pthread_self
#elif defined(WIN32)
# include <windows.h>
  typedef DWORD thread_id_t;
# define  THREAD_ID_SELF GetCurrentThreadId  
#else
  typedef int thread_id_t;
# define  THREAD_ID_SELF() 1  
#endif

extern thread_id_t armci_usr_tid;
#ifdef SERVER_THREAD
#  define SERVER_CONTEXT (armci_usr_tid != THREAD_ID_SELF())
#else
#  define SERVER_CONTEXT (armci_me<0)
#endif

#if defined(LAPI) || defined(CLUSTER)
#  include "request.h"
#endif

/* min amount of data in strided request to be sent in a single TCP/IP message*/
#ifdef SOCKETS
#  define TCP_PAYLOAD 512
#  define LONG_GET_THRESHOLD  TCP_PAYLOAD  
#  define LONG_GET_THRESHOLD_STRIDED LONG_GET_THRESHOLD  
#endif

#ifdef WIN32
#  define bzero(a,len){\
     int _i;\
     char *_c = (char*)(a);\
     for(_i=0; _i< (int)(len); _i++)_c[_i]=(char)0;\
   }
#  define bcopy(a,b,len) memcpy(b,a,len)
#else
# include <strings.h>
#endif

#if defined (CRAY_T3E) || defined(FUJITSU) || (defined(QUADRICS) && !defined(ELAN))
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
#define MAX_STRIDE_LEVEL ARMCI_MAX_STRIDE_LEVEL

/* msg tag ARMCI uses in collective ops */
#define ARMCI_TAG 30000

#ifndef EXTRA_MSG_BUFLEN_DBL
#   define RESERVED_BUFLEN ((sizeof(request_header_t)>>3)+3*MAX_STRIDE_LEVEL)
#else
#   define RESERVED_BUFLEN ((sizeof(request_header_t)>>3)+3*MAX_STRIDE_LEVEL +\
                           EXTRA_MSG_BUFLEN_DBL)
#endif

/* packing algorithm for double complex numbers requires even number */
#ifdef MSG_BUFLEN_DBL
#  define BUFSIZE_DBL (MSG_BUFLEN_DBL - RESERVED_BUFLEN)
#else
#  define BUFSIZE_DBL 32768
#endif

#define BUFSIZE  (BUFSIZE_DBL * sizeof(double))

/* note opcodes must be lower than ARMCI_ACC_OFF !!! */
#define PUT 1
#define GET 3
#define RMW 5
#define LOCK   20
#define UNLOCK 21
#define ACK (ARMCI_ACC_OFF -1)

/* must fit in two bits, see msginfo->format in request.h */
#define STRIDED 1
#define VECTOR  2

extern  int armci_me, armci_nproc;
extern  double armci_internal_buffer[BUFSIZE_DBL];

extern void armci_shmem_init();
extern void armci_die(char *msg, int code);
extern void armci_die2(char *msg, int code1, int code2);
extern void armci_write_strided(void *ptr, int stride_levels, 
                                int stride_arr[], int count[], char *buf);
extern void armci_read_strided(void *ptr, int stride_levels, 
                               int stride_arr[], int count[], char *buf);
extern int armci_op_strided(int op, void* scale, int proc,void *src_ptr, 
			int src_stride_arr[],  void* dst_ptr, int dst_stride_arr[], 
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

#ifdef SOCKETS
#ifdef SERVER_THREAD
  extern void armci_create_server_thread ( void* (* func)(void*) );
  extern void armci_terminate_server_thread();
#else
  extern void armci_create_server_process ( void* (* func)(void*) );
  extern void armci_wait_server_process();
  extern void RestoreSigChldDfl();
#endif
#endif

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))
#define ACC(op)  ((((int)(op))-ARMCI_ACC_INT)>=0)

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
#elif defined(CLUSTER) && !defined(QUADRICS)
#  define ORDER(op,proc)\
        if(!SAMECLUSNODE(proc) && op != GET )_armci_fence_arr[proc]=1
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
extern void armci_init_clusinfo();
extern void armci_set_mem_offset(void *ptr);
extern int _armci_terminating;
extern void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, 
                         void *dst_ptr, int bytes, int cols, int src_stride, 
                         int dst_stride, int lockit);
extern void armci_lockmem_scatter(void *ptr_array[], int len, int bytes, int p);
extern void armci_generic_rmw(int op, void *ploc, void *prem, int extra, int p);
extern unsigned long armci_max_region();
extern void armci_dispatch_strided(void *ptr, int stride_arr[], int count[],
                            int strides, int fit_level, int nb, int bufsize, 
                            void (*fun)(void*,int*,int*,int,void*), void *arg);

#if defined(SYSV) || defined(WIN32)
extern void armci_shmem_init();
extern void armci_set_shmem_limit(unsigned long shmemlimit);
#endif

#define ALIGN_PTR_LONG(type, x) if( ((long)(x)) % sizeof(long)) { long _y = (long)(x);\
              if(sizeof(long)==8){_y>>=3; _y<<=3; }\
                            else { _y>>=2; _y<<=2; }\
              _y += sizeof(long); (x) = (type*)_y; }

#define SIXTYFOUR 64
#define ALIGN64ADD(buf) (SIXTYFOUR-(((ssize_t)(buf))%SIXTYFOUR))

#endif
