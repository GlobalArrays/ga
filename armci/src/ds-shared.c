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
#define DEBUG1 0



/*\ client initialization
\*/
void armci_client_code()
{
   if(DEBUG_){
      printf("in client after fork %d(%d)\n",armci_me,getpid()); fflush(stdout);
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
void armci_send_req(int proc, request_header_t* msginfo, int len)
{
int hdrlen = sizeof(request_header_t);
int bytes;

    if(msginfo->operation == GET)
        bytes = msginfo->dscrlen + hdrlen;
    else
        bytes = msginfo->bytes + hdrlen;

    if(DEBUG_){printf("%d: sending req %d (len=%d dscr=%d data=%d) to %d \n",
               armci_me, msginfo->operation, bytes,msginfo->dscrlen,
               msginfo->datalen,proc); fflush(stdout);
    }

    if(bytes > len)armci_die2("armci_send_req:buffer overflow",bytes,len);

    if(armci_send_req_msg(proc,msginfo, bytes))
                      armci_die("armci_send_req:failed",0);
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
                cluster, proc, bytes, dscrlen, datalen); fflush(stdout);
    }

#ifdef SOCKETS
    /* zero-copy optimization for large requests */
    if(count[0] >  TCP_PAYLOAD){
       if(armci_send_req_msg_strided(proc, msginfo,ptr,strides,
          stride_arr, count))armci_die("armci_send_strided_req long: failed",0);
       return; /************** done **************/
    }
#endif

    /*  copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);

    if(armci_send_req_msg(proc,msginfo, bytes))
       armci_die("armci_send_strided_req: failed",0);
}


/*\ client receives data from server
\*/
char *armci_rcv_data(int proc, request_header_t* msginfo )
{
    int datalen = msginfo->datalen;
    char *buf;

    if(DEBUG_) {
        printf("%d:armci_rcv_data:  bytes= %d \n", armci_me, datalen);
        fflush(stdout);
    }

    if(datalen == 0) armci_die("armci_rcv_data: no data to receive",datalen);
    if(datalen > (MSG_BUFLEN-sizeof(request_header_t)-sizeof(long)))
        armci_die("armci_rcv_data:data overflowing rcv buffer",datalen);

    buf = armci_ReadFromDirect(proc, msginfo, datalen);

    if(DEBUG_){
        printf("%d:armci_rcv_data: got %d bytes \n",armci_me,datalen);
        fflush(stdout);
    }

    return(buf);
}


/*\ client receives vector data from server and unpacks to the right loc
\*/
void armci_rcv_vector_data(int proc, request_header_t* msginfo, armci_giov_t darr[], int len)
{
    char *buf = armci_rcv_data(proc, msginfo);
    armci_vector_from_buf(darr, len, buf);
}


/*\ client receives strided data from server
\*/
void armci_rcv_strided_data(int proc, request_header_t* msginfo, int datalen, 
                            void *ptr, int strides,int stride_arr[],int count[])
{
    char *databuf;

    if(DEBUG_){
        printf("%d: armci_rcv_strided_data: expecting datalen %d from %d\n",
                armci_me, datalen, proc); fflush(stdout);
    }

#ifdef SOCKETS
    /* zero-copy optimization for large requests */
    if(count[0] >  TCP_PAYLOAD){
       armci_ReadStridedFromDirect(proc,msginfo,ptr,strides,stride_arr, count);
       return; /*********************** done ************************/
    }
#endif

    databuf = armci_ReadFromDirect(proc,msginfo,datalen);
    armci_read_strided(ptr, strides, stride_arr, count, databuf);
}


/*\ get ACK from server
\*/
void armci_rem_ack(int clus)
{
int bufsize = sizeof(request_header_t)+sizeof(int);
request_header_t *msginfo = (request_header_t *)GET_SEND_BUFFER(bufsize);

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

    armci_send_req(armci_clus_info[clus].master, msginfo, bufsize);
    armci_rcv_data(armci_clus_info[clus].master, msginfo);  /* receive ACK */
    FREE_SEND_BUFFER(msginfo);
}



/***************************** server side *********************************/

static void armci_check_req(request_header_t *msginfo, int buflen)
{
    if((msginfo->to != armci_me && msginfo->to < armci_master) ||
       msginfo->to >= armci_master + armci_clus_info[armci_clus_me].nslave)
        armci_die("armci_rcv_req: invalid to", msginfo->to);
#if 0   
    /* should be done in recv_req */
    if(msginfo->operation != GET && msginfo->bytes > buflen)
        armci_die2("armci_rcv_req: message overflowing rcv buffer",
                  msginfo->bytes,MSG_BUFLEN);
#endif
   
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

#if defined(VIA) || defined(GM)
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
#else
        armci_WriteToDirect(to, msginfo, data);
#endif
}


/*\ server sends strided data back to client
\*/
void armci_send_strided_data(int proc,  request_header_t *msginfo,
                             char *bdata, void *ptr, int strides,
                             int stride_arr[], int count[])
{

    int to = msginfo->from;

    if(DEBUG_){ printf("%d(server): sending datalen = %d to %d\n",
                armci_me, msginfo->datalen, to); fflush(stdout); }
 
#ifdef SOCKETS
    /* zero-copy optimization for large requests */
    if(count[0] >  TCP_PAYLOAD){
       armci_WriteStridedToDirect(to,msginfo,ptr, strides, stride_arr, count);
       return; /*********************** done ************************/
    }
#endif

    /* for small contiguous blocks copy into a buffer before sending */
    armci_write_strided(ptr, strides, stride_arr, count, bdata);

    /* write the message to the client */
    armci_WriteToDirect(to, msginfo, bdata);

    if(DEBUG_){
        printf("%d(serv):sent len=%d to %d\n",armci_me,msginfo->datalen,to);
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

    if(DEBUG_){ 
       printf("%d(serv):got %d request from %d\n",armci_me,msginfo->operation, from);
       fflush(stdout);
    }

    switch(msginfo->operation){
      case ACK:
          if(DEBUG_) {
              fprintf(stdout, "%d(server): got ACK request from %d\n",
                      armci_me, msginfo->from); fflush(stdout);
          }
          armci_server_ack(msginfo);
          break;

      case ATTACH: 
          if(DEBUG_){
             printf("%d(serv):got ATTACH request from%d\n",armci_me, from);
             fflush(stdout);
          }
          armci_server_ipc(msginfo, descr, buffer, buflen);
          break;
#ifdef SOCKETS
      case QUIT:   
          if(DEBUG_){ 
             printf("%d(serv):got QUIT request from %d\n",armci_me, from);
             fflush(stdout);
          }
          armci_server_goodbye(msginfo);
          break;
#endif

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

#ifdef SERVER_THREAD

     armci_create_server_thread( armci_server_code );
#else

     armci_create_server_process( armci_server_code );

#endif

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



/*\ request to QUIT sent by client
\*/
void armci_serv_quit()
{
int bufsize = sizeof(request_header_t)+sizeof(int);
request_header_t *msginfo= (request_header_t*)GET_SEND_BUFFER(bufsize);

    if(DEBUG_){ printf("%d master: sending quit request to server\n",armci_me);
        fflush(stdout);
    }

    msginfo->dscrlen = 0;
    msginfo->from  = armci_me;
    msginfo->to    = SERVER_NODE(armci_clus_me);
    msginfo->operation = QUIT;
    if(ACK_QUIT)
       msginfo->bytes   = msginfo->datalen = sizeof(int); /* ACK */
    else
       msginfo->bytes   = msginfo->datalen = 0; /* no ACK */

    armci_send_req(armci_master, msginfo, bufsize);

    if(ACK_QUIT){
       int stat;
       stat = *(int*)armci_rcv_data(armci_master,msginfo);  /* receive ACK */
       if(stat  != QUIT)
            armci_die("armci_serv_quit: wrong response from server", stat);
       FREE_SEND_BUFFER(msginfo);
    }
}


/*\ server action triggered by request to quit
\*/
void armci_server_goodbye(request_header_t* msginfo)
{
     int ack=QUIT;
     if(DEBUG_){
        printf("%d server: terminating request by %d\n",armci_me,msginfo->from);
        fflush(stdout);
     }

     if(msginfo->datalen){
       msginfo->datalen = -msginfo->datalen;
       if(msginfo->datalen != sizeof(int))
          armci_die("armci_server_goodbye: bad datalen=",msginfo->datalen);

       armci_send_data(msginfo, &ack);
     }

     armci_transport_cleanup();

     /* Finalizing data server process w.r.t. MPI is not portable
      */
     _exit(0);
}
