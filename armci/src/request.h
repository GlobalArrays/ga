#ifndef _REQUEST_H_
#define _REQUEST_H_
#ifdef LAPI
#  include "lapidefs.h"
#elif defined(GM)
#  include "myrinet.h"
#elif defined(ELAN)
   typedef void* msg_tag_t; 
#elif defined(VIA)
#  include "via.h"
   typedef int msg_tag_t;
#else
   typedef long msg_tag_t;
#endif

#define ACK_QUIT 0
#define QUIT 33
#define ATTACH 34

typedef struct {
#if 0 
   int   to:16;               /* message recipient */
   int from:16;               /* message sender */
#else
   short int   to;            /* message recipient */
   short int from;            /* message sender */
#endif
unsigned int   operation:8;   /* operation code */
#ifdef CLIENT_BUF_BYPASS
unsigned int   format:2;      /* data format used */
unsigned int   pinned:1;      /* indicates if sender memory was pinned */
unsigned int   bypass:1;      /* indicate if bypass protocol used */
#else
unsigned int   format:4;      /* data format used */
#endif
unsigned int   bytes:20;      /* number of bytes requested */
         int   dscrlen;       /* >0 in lapi means that descriptor is included */
         int   datalen;       /* >0 in lapi means that data is included */
         msg_tag_t tag;       /* message tag for response to this request */
}request_header_t;


#ifndef MSG_BUFLEN_DBL
#  define MSG_BUFLEN_DBL 50000
#endif

#define MSG_BUFLEN  sizeof(double)*MSG_BUFLEN_DBL
extern  char* MessageRcvBuffer;
extern  char* MessageSndBuffer;

#ifdef LAPI
#  define REQ_TAG {MessageSndBuffer + sizeof(request_header_t), &buf_cntr.cntr }
#  define GET_SEND_BUFFER(_size) MessageSndBuffer;CLEAR_COUNTER(buf_cntr); SET_COUNTER(buf_cntr,1);
#  define GA_SEND_REPLY armci_lapi_send
#else
#  define REQ_TAG 32000
#  ifdef SOCKETS
#    define GA_SEND_REPLY(tag, buf, len, p) armci_sock_send(p,buf,len)
#  else
#    define GA_SEND_REPLY(tag, buf, len, p)  
#  endif
#endif

#ifndef GET_SEND_BUFFER
#  define GET_SEND_BUFFER(_len) MessageSndBuffer
#endif

#ifndef FREE_SEND_BUFFER
#define FREE_SEND_BUFFER(_ptr)  
#endif


typedef struct {
           void *buf; int count; int proc; int op; int extra; double scale[2];
} buf_arg_t;

#ifdef PIPE_BUFSIZE 
   extern void armcill_pipe_post_bufs(void *ptr, int stride_arr[], int count[],
                                      int strides, void* argvoid);
   extern void armcill_pipe_extract_data(void *ptr,int stride_arr[],int count[],
                                         int strides, void* argvoid);
   extern void armcill_pipe_send_chunk(void *data, int stride_arr[],int count[],
                                       int strides, void* argvoid);
#endif

extern void armci_send_strided(int proc, request_header_t *msginfo, char *bdata,
                         void *ptr, int strides, int stride_arr[], int count[]);

extern char *armci_rcv_data(int proc, request_header_t *msginfo);
extern void armci_rcv_strided_data_bypass(int proc, int datalen,
                                          void *ptr, int stride_levels);
extern void armci_send_strided_data_bypass(int proc, request_header_t *msginfo,
            void *loc_buf, int msg_buflen, void *loc_ptr, int *loc_stride_arr,
            void *rem_ptr, int *rem_stride_arr, int *count, int stride_levels);

extern void armci_rcv_strided_data(int proc, request_header_t* msginfo, 
                  int datalen, void *ptr, int strides,int stride_arr[],int count[]);
extern void armci_send_strided_data(int proc,  request_header_t *msginfo, 
            char *bdata, void *ptr, int strides, int stride_arr[], int count[]);
extern void armci_send_req(int proc, request_header_t* msginfo, int len);
extern void armci_server_rmw(request_header_t* msginfo,void* ptr, void* pextra);
extern int armci_rem_vector(int op, void *scale, armci_giov_t darr[],int len,
                            int proc);
extern int armci_rem_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int lockit);

extern void armci_rem_rmw(int op, int *ploc, int *prem, int extra, int proc);
extern void armci_rem_ack(int clus);
extern void armci_server(request_header_t *msginfo, char *dscr, char* buf, 
                         int buflen);
extern void armci_server_vector(request_header_t *msginfo,
                                char *dscr, char* buf, int buflen);
extern void armci_serv_attach_req(void *info, int ilen, long size,
                                  void* resp,int rlen);
extern void armci_server_lock(request_header_t *msginfo);
extern void armci_server_unlock(request_header_t *msginfo, char* dscr);
extern void armci_create_server_thread ( void* (* func)(void*) );
extern int armci_server_lock_mutex(int mutex, int proc, msg_tag_t tag);
extern void armci_send_data(request_header_t* msginfo, void *data);
extern int armci_server_unlock_mutex(int mutex, int p, int tkt, msg_tag_t* tag);
extern void armci_rcv_vector_data(int p, request_header_t* msginfo, armci_giov_t dr[], int len);

#ifndef LAPI
extern void armci_wait_for_server();
extern void armci_start_server();
extern void armci_transport_cleanup();
extern int armci_send_req_msg(int proc, void *buf, int bytes);
extern void armci_WriteToDirect(int proc, request_header_t* msginfo, void *buf);
extern char *armci_ReadFromDirect(int proc, request_header_t *msginfo, int len);
extern void armci_init_connections();
extern void *armci_server_code(void *data);
extern void armci_rcv_req(void *mesg, void *phdr, void *pdescr, 
                          void *pdata, int *buflen);
extern void armci_client_connect_to_servers();
extern void armci_data_server(void *mesg);
extern void armci_server_initial_connection();
extern void armci_call_data_server();
#endif
#ifdef SOCKETS
extern void armci_ReadStridedFromDirect(int proc, request_header_t* msginfo,
                  void *ptr, int strides, int stride_arr[], int count[]);
extern void armci_WriteStridedToDirect(int proc, request_header_t* msginfo,
                         void *ptr, int strides, int stride_arr[], int count[]);
extern void armci_serv_quit();
extern int armci_send_req_msg_strided(int proc, request_header_t *msginfo,
                          char *ptr, int strides, int stride_arr[],int count[]);
extern void armci_server_goodbye(request_header_t* msginfo);
#endif

extern void armci_server_ipc(request_header_t* msginfo, void* descr,
                             void* buffer, int buflen);

#ifdef PIPE_BUFSIZE
extern void armci_pipe_prep_receive_strided(request_header_t *msginfo,char *buf,
                       int strides, int stride_arr[], int count[], int bufsize);
extern void armci_pipe_receive_strided(request_header_t* msginfo, void *ptr,
                                int stride_arr[], int count[], int strides);
extern void armci_pipe_send_req(int proc, void *buf, int bytes);
#endif

#endif
