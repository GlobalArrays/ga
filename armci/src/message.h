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
  int   datalen;        /* >0 indicates if data is included */
  int   dscrlen;        /* >0 indicates if descriptor is included */
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
