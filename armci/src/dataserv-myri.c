#include "armcip.h"
#include "request.h"
#include "message.h"
#include "memlock.h"
#include "copy.h"
#include "myrinet.h"
#include <stdio.h>
#include <errno.h>

#define DEBUG_ 0

#define ACK 0
#define QUIT 33
#define ATTACH 34


/*********************************************************************
                        UTILITY FUNCTIONS                            
 *********************************************************************/

long check_flag(long *buf)
{
    return(*buf);
}

/*******************************************************************
 *                   CLIENT SIDE                                   *
 *******************************************************************/

/* client sends request to server */
void armci_send_req(int proc)
{   
    int hdrlen = sizeof(request_header_t);
    int dscrlen = ((request_header_t*)MessageSndBuffer)->dscrlen;
    int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
    int operation = ((request_header_t*)MessageSndBuffer)->operation;
    int cluster = armci_clus_id(proc);
    int stat;
    int bytes;

    if(((request_header_t*)MessageSndBuffer)->operation == GET)
        bytes = dscrlen + hdrlen;
    else
        bytes = ((request_header_t*)MessageSndBuffer)->bytes + hdrlen;
    
    if(DEBUG_) {
        fprintf(stdout,
             "%d: sending req %d (len=%d dscr=%d data=%d) to %d (nodeid %d)\n",
                armci_me, operation, bytes, dscrlen, datalen, proc,
                proc_gm->node_map[proc]);
        fflush(stdout);
    }

    stat = armci_dma_send_gm(proc, MessageSndBuffer, bytes);
    if(stat == ARMCI_GM_FAILED) armci_die("armci_send_req: write failed", stat);
}


/* client sends strided data + request to server */
void armci_send_strided(int proc, request_header_t *msginfo, char *bdata, 
                        void *ptr, int strides, int stride_arr[], int count[])
{
    int hdrlen = sizeof(request_header_t);
    int dscrlen = msginfo->dscrlen;
    int datalen = msginfo->datalen;
    int cluster = armci_clus_id(proc);
    int stat;
    int bytes;
    
    if(DEBUG_) {
        fprintf(stdout, "%d:armci_send_strided: op=%d to=%d bytes= %d \n",
                armci_me, msginfo->operation, proc, datalen);
        fflush(stdout);
    }
    
    bytes = msginfo->bytes + hdrlen;
    
    if(DEBUG_){
        fprintf(stdout,
            "%d: sending req %d to(%d,%d,%d) bytes = %d (dslen=%d dlen=%d),\n",
                armci_me, msginfo->operation, msginfo->to,
                cluster, proc, bytes, dscrlen, datalen);
        fflush(stdout);
    }
    
    /* for small contiguous blocks copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);
    
    stat = armci_dma_send_gm(proc, msginfo, bytes);
    if(stat == ARMCI_GM_FAILED)
        armci_die("armci_send_strided: write failed", stat);
}

/* client receives data from server */
char *armci_rcv_data(int proc)
{
    int cluster = armci_clus_id(proc);
    int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
    char *buf;
    int stat;
    
    if(DEBUG_) {
        fprintf(stdout, "%d:armci_rcv_data:  bytes= %d \n", armci_me, datalen);
        fflush(stdout);
    }
    
    if(datalen == 0) armci_die("armci_rcv_data: no data to receive",datalen); 
    if(datalen > (MSG_BUFLEN-sizeof(request_header_t)-sizeof(long)))
        armci_die("armci_rcv_data:data overflowing rcv buffer",datalen);

    /* buf = header ack + data(len = datalen) + tail ack */
    buf = armci_ReadFromDirect(MessageSndBuffer, datalen);

    if(DEBUG_){
        printf("%d:armci_rcv_data: got %d bytes \n",armci_me,datalen);
        fflush(stdout);
    }

    return(buf);
}

/* client receives vector data from server and unpack to the right loc */
void armci_rcv_vector_data(int proc, char *buf, armci_giov_t darr[], int len)
{
    buf = armci_rcv_data(proc);
    armci_vector_from_buf(darr, len, buf);

    /* send notice to server the message has been received */
    /* armci_NotifyDirect(proc);*/
}

/* client receives strided data from server */
void armci_rcv_strided_data(int proc, char *buf, int datalen, void *ptr,
                            int strides, int stride_arr[], int count[])
{
    char *databuf;
    int cluster = armci_clus_id(proc);
    int stat;
    
    if(DEBUG_){
        fprintf(stdout,
                "%d: armci_rcv_strided_data: expecting datalen %d from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }

    /* the buf should be MessageSndBuffer */
    if(buf != MessageSndBuffer)
        armci_die("armci_rcv_strided_data: not the right buffer", 0L);

    /* the received data starts at
     * MessageSndBuffer + sizeof(request_header_t)
     */
    
    /* for small data segments minimize number of system calls */
    databuf = armci_ReadFromDirect(MessageSndBuffer, datalen);
    
    armci_read_strided(ptr, strides, stride_arr, count, databuf);
        
    /* send notice to server the message has been received */
    /* armci_NotifyDirect(proc); */
    
    if(DEBUG_){
        fprintf(stdout, "%d: armci_rcv_strided_data: got %d bytes from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }
}

/* get ACK from server */
void armci_rem_ack(int clus)
{
    char *databuf;
    request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
    
    msginfo->dscrlen = 0;
    msginfo->from  = armci_me;
    msginfo->to    = SERVER_NODE(clus);
    msginfo->format  = msginfo->operation = ACK;
    msginfo->bytes   =0;
    msginfo->datalen =sizeof(int);
    
    if(DEBUG_) {
        fprintf(stdout, "%d client: sending ACK to %d c=%d\n",
                armci_me, msginfo->to, clus);
        fflush(stdout);
    }
    
    armci_send_req(armci_clus_info[clus].master);
    databuf = armci_rcv_data(armci_clus_info[clus].master);  /* receive ACK */
}


void armci_wait_for_server()
{
}

void armci_client_code()
{
    int i;
    int server_id;
    
    if(DEBUG_)
        printf("%d: in client after creating thread.\n",armci_me);
    
    /* make initial conection to the server, not the server in this node */
    armci_client_create_connection_gm();
    
    if(DEBUG_){
        printf("%d: client connected to all %d data servers\n",
               armci_me, armci_nclus);
        fflush(stdout);
    }
}

/*******************************************************************
 *                   SERVER SIDE                                   *
 *******************************************************************/

/* server receives request */
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    int stat;
    request_header_t *msginfo = (request_header_t *)mesg;
    
    if(DEBUG_) {
        fprintf(stdout,
                "%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from);
        fflush(stdout);
    }
    
    *(void **)phdr = msginfo;
    
    /* leave space for header ack */ 
    *(void**)pdata  = MessageRcvBuffer + sizeof(long);
    *buflen = MSG_BUFLEN - sizeof(request_header_t) - sizeof(long);
                                        /* tail acks -^ */

    /* regular checking */
    if((msginfo->to != armci_me && msginfo->to < armci_master) ||
       msginfo->to >= armci_master + armci_clus_info[armci_clus_me].nslave)
        armci_die("armci_rcv_req: invalid to", msginfo->to);

    if(msginfo->operation != GET && msginfo->bytes > *buflen)
        armci_die("armci_rcv_req: message overflowing rcv buffer",
                  msginfo->bytes);
    
    if(msginfo->dscrlen < 0)
        armci_die("armci_rcv_req: dscrlen < 0", msginfo->dscrlen);
    if(msginfo->datalen < 0)
        armci_die("armci_rcv_req: datalen < 0", msginfo->datalen);
    if(msginfo->dscrlen > msginfo->bytes)
        armci_die2("armci_rcv_req: dsclen > bytes", msginfo->dscrlen,
                   msginfo->bytes);
    
    if(msginfo->bytes) {
        *(void **)pdescr = msginfo+1;
        if(msginfo->operation != GET)
            *(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);
    }else 
        *(void**)pdescr = NULL;

    /* make sure last send is complete and the buffer is available */
    stat = armci_serv_send_complete();
    if(stat == ARMCI_GM_FAILED)
        armci_die(" last armci_send_(strided_)data: write failed", stat);
}


/* server response - send data to client */
void armci_send_data(request_header_t* msginfo, void *data)
{
    int to = msginfo->from;

    /* if the data is in the pinned buffer: MessageRcvBuffer */
    if((data > (void *)MessageRcvBuffer) &&
       (data < (void *)(MessageRcvBuffer + MSG_BUFLEN)))
        /* write the message to the client */
        armci_WriteToDirect(to, msginfo, data);
    else {
        /* copy the data to the MessageRcvBuffer */
        /* leave space for header ack */
        int i;
        char *buf = MessageRcvBuffer + sizeof(long);
        armci_copy(data, buf, msginfo->datalen);
        armci_WriteToDirect(to, msginfo, buf);
    }

    /* wait the client send back the ack */
    /* if(msginfo->operation == GET) */
    /* armci_WaitAckDirect(to); */
}



/* server sends strided data back to client */
void armci_send_strided_data(int proc,  request_header_t *msginfo,
                             char *bdata, void *ptr, int strides,
                             int stride_arr[], int count[])
{
    int to = msginfo->from;
    
    /* for small contiguous blocks copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);
    
    if(DEBUG_){
        fprintf(stdout, "%d(server): sending datalen = %d to %d first: %f\n",
                armci_me, msginfo->datalen, to, *(double *)bdata);
        fflush(stdout);
    }
    
    /* write the message to the client */
    armci_WriteToDirect(to, msginfo, bdata);

    /* wait the client send back the ack */
    /* if(msginfo->operation == GET) */
    /* armci_WaitAckDirect(to); */
}

/* server sends ACK to client */
void armci_server_ack(request_header_t* msginfo)
{
    int ack = ACK;
    
    if(DEBUG_){
        printf("%d server: sending ACK to %d\n",armci_me, msginfo->from);
        fflush(stdout);
    }

    if(msginfo->datalen != sizeof(int))
        armci_die("armci_server_ack: bad datalen=",msginfo->datalen);

    armci_send_data(msginfo, &ack);
}


/* server initializes its copy of the memory lock data structures */
static void server_alloc_memlock(void *ptr_myclus)
{
    int i;
    
    /* for protection, set pointers for processes outside local node NULL */
    memlock_table_array = calloc(armci_nproc,sizeof(void*));
    if(!memlock_table_array)
        armci_die("malloc failed for ARMCI lock array", 0);
    
    /* set pointers for processes on local cluster node
     * ptr_myclus - corresponds to the master process
     */
    for(i=0; i< armci_clus_info[armci_clus_me].nslave; i++){
        memlock_table_array[armci_master +i] = ((char*)ptr_myclus)
            + MAX_SLOTS*sizeof(memlock_t)*i;
    } 
    
    if(DEBUG_)
       fprintf(stderr, "server initialized memlock\n");
}


/* main routine for data server process in a cluster environment
 *  the process is blocked until message arrives from
 *  the clients and services the requests
 */
void armci_data_server(void *mesg)
{
    /* message */
    request_header_t *msginfo; 
    void *descr;
    void *buffer;
    int buflen;
    
    /* read header, descriptor, data, and buffer length */
    armci_rcv_req(mesg, &msginfo, &descr, &buffer, &buflen );
    
    switch(msginfo->operation){        
      case ACK:
          if(DEBUG_) {
              fprintf(stdout, "%d(server): got ACK request from %d\n",
                      armci_me, msginfo->from);
              fflush(stdout);
          } 
          armci_server_ack(msginfo);
          break;
          
      case ARMCI_FETCH_AND_ADD:
      case ARMCI_FETCH_AND_ADD_LONG:
          armci_server_rmw(msginfo,descr,buffer);
          break;
          
      case LOCK:
          armci_server_lock(msginfo); 
          break;
          
      case UNLOCK:
          armci_server_unlock(msginfo, descr);
          break;
          
      default:
          if(msginfo->format ==VECTOR)
              armci_server_vector(msginfo, descr, buffer, buflen); 
          else if(msginfo->format ==STRIDED)
              armci_server(msginfo, descr, buffer, buflen); 
          else
              armci_die2("armci_data_serv: unknown format code",
                         msginfo->format, msginfo->from);
    }
}

void *armci_server_code(void *data)
{
    if(DEBUG_)
        printf("%d: in server after creating thread.\n",armci_me);
    
    /* make initial contact with all the computing process
     * get the port id of all computing process
     */
    armci_server_initial_connection_gm();
    
    if(DEBUG_) {
        printf("%d(server): connected to all computing processes\n",armci_me);
        fflush(stdout);
    }

    armci_data_server_gm();

    armci_transport_cleanup();
    
    return(NULL);
}

void armci_start_server()
{
    /* initialize computing processes on gm */
    if(!armci_gm_proc_init())
        armci_die("computing process initialization failed on gm", 0L);
    
    if(armci_me == armci_master) {
        /* initialize the server on gm */
        if(!armci_gm_server_init())
            armci_die("server initialization failed on gm", 0L);
        armci_create_server_thread(armci_server_code);   
    }  

    armci_client_code();
}

