#ifndef _VIA_H_
#define _VIA_H_ 
extern int armci_long_buf_free, armci_long_buf_taken_srv;
extern void armci_via_wait_ack();

#define PIPE_BUFSIZE  (8*4096)
#define PIPE_MIN_BUFSIZE 8192
#define PIPE_MEDIUM_BUFSIZE (2*8192)

/* we need buffer size to be 64-byte alligned */
#define EXTRA_MSG_BUFLEN_DBL  128 
#define EXTRA_MSG_BUFLEN  ((EXTRA_MSG_BUFLEN_DBL)<<3) 
#define VBUF_DLEN 64*1023
#define MSG_BUFLEN_DBL ((VBUF_DLEN)>>3)

#ifdef PIPE_BUFSIZE
#  define STRIDED_GET_BUFLEN_DBL 34*1024
#  define STRIDED_GET_BUFLEN (STRIDED_GET_BUFLEN_DBL<<3)
#  define MAX_BUFLEN (STRIDED_GET_BUFLEN+EXTRA_MSG_BUFLEN)
#else
#  define MAX_BUFLEN (MSG_BUFLEN+EXTRA_MSG_BUFLEN)
#endif

#define GET_SEND_BUFFER_(_size) MessageSndBuffer;if(!armci_long_buf_free)armci_via_wait_ack()
extern char* armci_getbuf(int size);
extern void armci_relbuf(void *buf);

#define GET_SEND_BUFFER(_size) armci_getbuf(_size)
#define FREE_SEND_BUFFER(x) armci_relbuf(x)


#endif
