#include <elan/elan.h>
#include <elan3/elan3.h>
#include "armcip.h"
#include "copy.h"

#define DEBUG_ 1

static int armci_server_terminating=0;
static ELAN_MAIN_QUEUE *mq;
#define _ELAN_SLOTSIZE 320
#define MSG_DATA_LEN (_ELAN_SLOTSIZE - sizeof(request_header_t))

void armci_init_connections()
{
ELAN_QUEUE *q;
int nslots=armci_nproc, slotsize=_ELAN_SLOTSIZE;


    if ((q = elan_gallocQueue(elan_base->galloc, elan_base->allGroup)) == NULL)
        armci_die( "elan_gallocElan",0 );

    if (!(mq = elan_mainQueueInit( elan_base->state, q, nslots, slotsize)))
        armci_die("Failed to to initialise Main Queue",0);

  printf("%d: vp=%d localId=%d\n",armci_me,elan_base->state->vp,elan_base->state->localId); 
   sleep(1);
}


extern void armci_send_data_to_client(int proc, void *buf, int bytes);

void armci_call_data_server()
{
int usec_to_sleep=0;
char buf[_ELAN_SLOTSIZE];

    if(DEBUG_){
        printf("%d(server): waiting for request\n",armci_me); fflush(stdout);
    }
 
    while(1){
        elan_queueWait(mq, buf, usec_to_sleep );
        armci_data_server(buf);
    }

    if(DEBUG_) {printf("%d(server): done! closing\n",armci_me); fflush(stdout);}
}


/* server receives request */
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    request_header_t *msginfo = (request_header_t *)mesg;
    *(void **)phdr = msginfo;

    if(DEBUG_) {
       printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
                  armci_me, msginfo->operation, msginfo->dscrlen,
                  msginfo->datalen, msginfo->from); fflush(stdout);
    }

#ifdef CLIENT_BUF_BYPASS
    if(msginfo->bypass) {
        *(void**)pdata  = MessageRcvBuffer;
        *buflen = MSG_BUFLEN;
    } else
#endif
    {
        /* leave space for header ack */
        *(void**)pdata  = MessageRcvBuffer + sizeof(msginfo->tag.ack);
        *buflen = MSG_BUFLEN - sizeof(request_header_t);
    }

    if(msginfo->bytes){
        *(void **)pdescr = msginfo+1;
        if(msginfo->operation != GET){
           int payload = msginfo->dscrlen+msginfo->datalen;
           *(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);
           if( payload > MSG_DATA_LEN){
               armci_get(msginfo->tag.data_ptr, msginfo+1, payload, msginfo->from); 
           }
        }
    }else
        *(void**)pdescr = NULL;
}


/*\ server sends data to client buffer
\*/
void armci_WriteToDirect(int dst, request_header_t *msginfo, void *buffer)
{
/* none in acc */
}

char *armci_ReadFromDirect(request_header_t * msginfo, int len)
{
    char *buf = (char*) msginfo;
    return(buf);
}


/*\ send request to server thread
\*/
int armci_send_req_msg(int proc, void *vbuf, int len)
{
    char *buf = (char*)vbuf;
    request_header_t *msginfo = (request_header_t *)buf;
    int size=_ELAN_SLOTSIZE;

    /* set the message tag */
    msginfo->tag.data_ptr = (void *)(buf + sizeof(request_header_t));
    msginfo->tag.ack = 0;
    elan_queueReq(mq, proc, vbuf, size);

    return 0;
}



void armci_wait_for_server()
{
  armci_server_terminating=1;
}

void armci_transport_cleanup() {}
void armci_client_connect_to_servers(){}
void armci_server_initial_connection(){}

