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

/* in GM 1.4 memory registration got so slow we cannot use 0-copy protocols
 * we are disabling it for medium messages by changing thresholds */
#if defined(GM_MAX_DEFAULT_MESSAGE_SIZE) && !defined(GM_ENABLE_PROGRESSION)
#   define GM_1_2      /* most likely we have GM <1.4 */
#endif

#define CLIENT_BUF_BYPASS 
#ifdef __i386__
# ifdef GM_1_2
#   define LONG_GET_THRESHOLD 66248
#   define LONG_GET_THRESHOLD_STRIDED 3000
# else
#   define LONG_GET_THRESHOLD 266248
#   define LONG_GET_THRESHOLD_STRIDED 30000 
# endif
#define INTERLEAVE_GET_THRESHOLD 66248
#else
#define LONG_GET_THRESHOLD 524288
#define LONG_GET_THRESHOLD_STRIDED 30000 
#define INTERLEAVE_GET_THRESHOLD 524288 
#endif

/* context for callback routine */
typedef struct {
    int tag;
    volatile int done;
} armci_gm_context_t;

#define MULTIPLE_SND_BUFS_ 
#ifdef MULTIPLE_SND_BUFS 
#define GET_SEND_BUFFER armci_gm_getbuf
#define FREE_SEND_BUFFER armci_gm_freebuf
#else
#define GET_SEND_BUFFER(x) (char*)(((armci_gm_context_t*)MessageSndBuffer)+1);
/*        armci_client_send_complete((armci_gm_context_t*)MessageSndBuffer);
*/
#define FREE_SEND_BUFFER(x) 
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

#include <mpi.h>

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
extern void armci_gm_freebuf(void *ptr);
extern char* armci_gm_getbuf(size_t size);
extern void armci_client_send_complete(armci_gm_context_t*);


#endif /* MYRINET_H */
