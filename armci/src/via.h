#ifndef _VIA_H_
#define _VIA_H_ 
extern int armci_long_buf_free, armci_long_buf_taken_srv;
extern void armci_via_wait_ack();

/* we need buffer size to be 64-byte alligned */
#define STRIDED_GET_BUFLEN_DBL 32*1024
#define STRIDED_GET_BUFLEN (STRIDED_GET_BUFLEN_DBL<<3)
#define VBUF_DLEN 64*1023
#define MSG_BUFLEN_DBL ((VBUF_DLEN)>>3)
#define EXTRA_MSG_BUFLEN_DBL  128 
#define EXTRA_MSG_BUFLEN  ((EXTRA_MSG_BUFLEN_DBL)>>3) 
#define GET_SEND_BUFFER(_size) MessageSndBuffer;if(!armci_long_buf_free)armci_via_wait_ack()

#endif
