/***
   AUTHOR
     Jialin Ju, PNNL
   NAME
     myrinet.h
   PURPOSE
     
   NOTES
     
   HISTORY
     jju - Mar 1, 2000: Created.
     jn  - Oct, 2000: restructured memory allocation, extra optimization
                      bug fixes
***/

#ifndef MYRINET_H
#define MYRINET_H

#include "mpi.h"

#define CLIENT_BUF_BYPASS 
#ifdef __i386__
#define LONG_GET_THRESHOLD 266248
#define LONG_GET_THRESHOLD_STRIDED 30000 
#define INTERLEAVE_GET_THRESHOLD 66248
#else
#define LONG_GET_THRESHOLD 524288
#define LONG_GET_THRESHOLD_STRIDED 30000 
#define INTERLEAVE_GET_THRESHOLD 524288 
#endif

/* two ports used by ARMCI and their boards iff STATIC_PORTS defined */
#define ARMCI_GM_SERVER_RCV_PORT 5
#define ARMCI_GM_SERVER_RCV_DEV 0
#define ARMCI_GM_SERVER_SND_PORT 6
#define ARMCI_GM_SERVER_SND_DEV 0

/* message types */
#define ARMCI_GM_BLOCKING 1
#define ARMCI_GM_NONBLOCKING 2

#define ARMCI_GM_FAILED  2

typedef struct {
    void *data_ptr;         /* pointer where the data should go */
    long ack;               /* header ack */
} msg_tag_t;


extern void armci_server_send_ack(int client);
extern int armci_pin_contig(void *ptr, int bytes);
extern void armci_unpin_contig(void *ptr, int bytes);
extern void armci_serv_send_ack(int client);
extern int armci_pin_memory(void *ptr, int stride_arr[], int count[], int lev);
extern void armci_unpin_memory(void *ptr,int stride_arr[],int count[],int lev);
extern int armci_serv_send_complete();
extern void armci_server_direct_send(int p,char *src,char *dst,int len,int typ);
extern void armci_data_server(void *msg);
extern void armci_serv_send_nonblocking_complete(int max_outstanding);
extern void armci_wait_for_data_bypass();
extern int  armci_wait_pin_client(int);
extern void armci_client_send_ack(int p, int success);

#endif /* MYRINET_H */
