#include "armcip.h"
#include "request.h"
#include "message.h"
#include "memlock.h"
#include "copy.h"
#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

#define DEBUG_ 0
#define DEBUG1 1



/*\ client initialization
\*/
void armci_client_code()
{
   if(DEBUG_){
       printf("in client after fork %d(%d)\n",armci_me,getpid());
       fflush(stdout);
   }

   armci_client_connect_to_servers();
   armci_msg_barrier();

   if(DEBUG_){
      printf("%d client connected to all %d servers\n",armci_me, armci_nclus-1);
      fflush(stdout);
   }
}


/*\ client sends request to server
\*/
void armci_send_req(int proc)
{
int hdrlen = sizeof(request_header_t);
int dscrlen = ((request_header_t*)MessageSndBuffer)->dscrlen;
int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
int operation = ((request_header_t*)MessageSndBuffer)->operation;
int bytes;

    if(((request_header_t*)MessageSndBuffer)->operation == GET)
        bytes = dscrlen + hdrlen;
    else
        bytes = ((request_header_t*)MessageSndBuffer)->bytes + hdrlen;

    if(DEBUG_) {
        printf( "%d: sending req %d (len=%d dscr=%d data=%d) to %d \n",
                armci_me, operation, bytes, dscrlen, datalen, proc);
        fflush(stdout);
    }

    if(armci_send_req_msg(proc,MessageSndBuffer, bytes))
      armci_die("armci_send_req: failed",0);
}


/*\ client sends strided data + request to server
\*/
void armci_send_strided(int proc, request_header_t *msginfo, char *bdata,
                        void *ptr, int strides, int stride_arr[], int count[])
{
    int hdrlen = sizeof(request_header_t);
    int dscrlen = msginfo->dscrlen;
    int datalen = msginfo->datalen;
    int cluster = armci_clus_id(proc);
    int bytes;

    bytes = msginfo->bytes + hdrlen;

    if(DEBUG_){
       printf("%d:sending strided %d to(%d,%d,%d) bytes=%d dslen=%d dlen=%d,\n",
                armci_me, msginfo->operation, msginfo->to,
                cluster, proc, bytes, dscrlen, datalen);
        fflush(stdout);
    }

    /* for small contiguous blocks copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);

    if(armci_send_req_msg(proc,MessageSndBuffer, bytes))
      armci_die("armci_send_strided_req: failed",0);
}


/*\ client receives data from server
\*/
char *armci_rcv_data(int proc)
{
    int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
    char *buf;

    if(DEBUG_) {
        printf("%d:armci_rcv_data:  bytes= %d \n", armci_me, datalen);
        fflush(stdout);
    }

    if(datalen == 0) armci_die("armci_rcv_data: no data to receive",datalen);
    if(datalen > (MSG_BUFLEN-sizeof(request_header_t)-sizeof(long)))
        armci_die("armci_rcv_data:data overflowing rcv buffer",datalen);

    /* buf = header ack + data(len = datalen) + tail ack */
    buf = armci_ReadFromDirect((request_header_t*)MessageSndBuffer, datalen);

    if(DEBUG_){
        printf("%d:armci_rcv_data: got %d bytes \n",armci_me,datalen);
        fflush(stdout);
    }

    return(buf);
}


/*\ client receives vector data from server and unpacks to the right loc
\*/
void armci_rcv_vector_data(int proc, char *buf, armci_giov_t darr[], int len)
{
    buf = armci_rcv_data(proc);
    armci_vector_from_buf(darr, len, buf);
}


/*\ client receives strided data from server
\*/
void armci_rcv_strided_data(int proc, char *buf, int datalen, void *ptr,
                            int strides, int stride_arr[], int count[])
{
    char *databuf;

    if(DEBUG_){
        printf("%d: armci_rcv_strided_data: expecting datalen %d from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }

    /* the buf should be MessageSndBuffer */
    if(buf != MessageSndBuffer)
       armci_die("armci_rcv_strided_data: not the right buffer", 0L);

    /* for small data segments minimize number of system calls */
    databuf = armci_ReadFromDirect((request_header_t*)MessageSndBuffer,datalen);

    armci_read_strided(ptr, strides, stride_arr, count, databuf);

    if(DEBUG_){
        printf("%d: armci_rcv_strided_data: got %d bytes from %d\n",
                armci_me, datalen, proc);
        fflush(stdout);
    }
}


/*\ get ACK from server
\*/
void armci_rem_ack(int clus)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;

    GET_SEND_BUFFER;

    msginfo->dscrlen = 0;
    msginfo->from  = armci_me;
    msginfo->to    = SERVER_NODE(clus);
    msginfo->operation = ACK;
    msginfo->bytes   =0;
    msginfo->datalen =sizeof(int);

    if(DEBUG_){
        printf("%d client: sending ACK to %d c=%d\n",armci_me,msginfo->to,clus);
        fflush(stdout);
    }

    armci_send_req(armci_clus_info[clus].master);
    armci_rcv_data(armci_clus_info[clus].master);  /* receive ACK */
}




/***************************** server side *********************************/

static void armci_check_req(request_header_t *msginfo, int buflen)
{
    if((msginfo->to != armci_me && msginfo->to < armci_master) ||
       msginfo->to >= armci_master + armci_clus_info[armci_clus_me].nslave)
        armci_die("armci_rcv_req: invalid to", msginfo->to);
   
    if(msginfo->operation != GET && msginfo->bytes > buflen)
        armci_die("armci_rcv_req: message overflowing rcv buffer",
                  msginfo->bytes);
   
    if(msginfo->dscrlen < 0)
        armci_die("armci_rcv_req: dscrlen < 0", msginfo->dscrlen);
    if(msginfo->datalen < 0)
        armci_die("armci_rcv_req: datalen < 0", msginfo->datalen);
    if(msginfo->dscrlen > msginfo->bytes)
        armci_die2("armci_rcv_req: dsclen > bytes", msginfo->dscrlen,
                   msginfo->bytes);
}


/*\ server response - send data to client
\*/
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
#ifdef GM
        /* leave space for header ack */
        char *buf = MessageRcvBuffer + sizeof(long);
#else
        char *buf = MessageRcvBuffer;
#endif
        armci_copy(data, buf, msginfo->datalen);
        armci_WriteToDirect(to, msginfo, buf);
    }
}


/*\ server sends strided data back to client
\*/
void armci_send_strided_data(int proc,  request_header_t *msginfo,
                             char *bdata, void *ptr, int strides,
                             int stride_arr[], int count[])
{
    int to = msginfo->from;

    /* for small contiguous blocks copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);

    if(DEBUG_){
        printf("%d(server): sending datalen = %d to %d\n",
                armci_me, msginfo->datalen, to);
        fflush(stdout);
    }

    /* write the message to the client */
    armci_WriteToDirect(to, msginfo, bdata);

    if(DEBUG_){
        printf("%d(server): sent datalen = %d to %d\n",
                armci_me, msginfo->datalen, to);
        fflush(stdout);
    }
}


/*\ server sends ACK to client
\*/
void armci_server_ack(request_header_t* msginfo)
{
     int ack=ACK;
     if(DEBUG_){
        printf("%d server: sending ACK to %d\n",armci_me,msginfo->from);
        fflush(stdout);
     }

     if(msginfo->datalen != sizeof(int))
        armci_die("armci_server_ack: bad datalen=",msginfo->datalen);
     armci_send_data(msginfo, &ack);
}


/*  main routine for data server process in a cluster environment
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
    int from;

    /* read header, descriptor, data, and buffer length */
    armci_rcv_req(mesg, &msginfo, &descr, &buffer, &buflen );

    /* check what we got */
    armci_check_req(msginfo,buflen);
    from = msginfo->from;

    switch(msginfo->operation){
      case ACK:
          if(DEBUG_) {
              fprintf(stdout, "%d(server): got ACK request from %d\n",
                      armci_me, msginfo->from);
              fflush(stdout);
          }
          armci_server_ack(msginfo);
          break;

      case ARMCI_SWAP:
      case ARMCI_SWAP_LONG:
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


/*\ initialize connection and start server thread/processes
\*/
void armci_start_server()
{
    armci_init_connections();

    if(armci_me == armci_master) {
        armci_create_server_thread(armci_server_code);
    }

    armci_client_code();
}



void *armci_server_code(void *data)
{
    if(DEBUG_)
        printf("%d: in server after creating thread.\n",armci_me);

    /* make initial contact with all the computing process */
    armci_server_initial_connection();

    if(DEBUG_) {
        printf("%d(server): connected to all computing processes\n",armci_me);
        fflush(stdout);
    }

    armci_call_data_server();

    armci_transport_cleanup();

    return(NULL);
}
