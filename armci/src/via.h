#ifndef _VIA_H_
#define _VIA_H_ 
extern int armci_long_buf_free, armci_long_buf_taken_srv;
extern void armci_via_wait_ack();

/* we need buffer size to be 64-byte alligned */
#define VBUF_DLEN 64*1023
#define MSG_BUFLEN_DBL ((VBUF_DLEN)>>3)
#define GET_SEND_BUFFER(_size) MessageSndBuffer;if(!armci_long_buf_free)armci_via_wait_ack()

#endif
