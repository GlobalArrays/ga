#ifndef _REQUEST_H_
#define _REQUEST_H_
#ifdef LAPI
#  include "lapidefs.h"
#else
   typedef int msg_tag_t;
#endif

typedef struct {
  short int to;         /* message recipient */
  short int from;       /* message sender */
  short int operation;  /* operation code */
  short int format;     /* data format used */
  int   bytes;          /* number of bytes requested */ 
  int   datalen;        /* >0 in lapi indicates if data is included */
  int   dscrlen;        /* >0 in lapi indicates if descriptor is included */
  msg_tag_t tag;        /* message tag for response to this request */
}request_header_t;

#define MSG_BUFLEN_DBL 12500
#define MSG_BUFLEN  sizeof(double)*MSG_BUFLEN_DBL
extern  char* MessageRcvBuffer;
extern  char* MessageSndBuffer;

#ifdef LAPI
#  define REQ_TAG {MessageSndBuffer + sizeof(request_header_t), &buf_cntr.cntr }
#  define GET_SEND_BUFFER CLEAR_COUNTER(buf_cntr); SET_COUNTER(buf_cntr,1);
#else
#  define REQ_TAG 32000
#  define GET_SEND_BUFFER
#endif

#define GA_SEND_REPLY(tag, buf, len, p) armci_sock_send(p,buf,len)

extern void armci_server_rmw(request_header_t* msginfo,void* ptr, void* pextra);
extern int armci_rem_vector(int op, void *scale, armci_giov_t darr[],int len,
                            int proc);
extern int armci_rem_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, int lockit);

extern void armci_server(request_header_t *msginfo, char *dscr, char* buf, 
                         int buflen);

extern void armci_server_vector( request_header_t *msginfo,
                          char *dscr, char* buf, int buflen);


#endif
