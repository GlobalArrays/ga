/* $Id: dataserv.c,v 1.17 2000-08-16 00:08:10 d3h325 Exp $ */
#include "armcip.h"
#include "sockets.h"
#include "request.h"
#include "message.h"
#include "memlock.h"
#include "shmem.h"
#include "copy.h"
#include <stdio.h>
#include <errno.h>

#ifdef WIN32
#include <windows.h>
#define sleep(x) Sleep(100*(x))
#endif
 
#define ACK_QUIT 0
#define QUIT 33
#define ATTACH 34

#define DEBUG_ 0

extern int AR_ready_sigchld;
int *SRV_sock;
int *AR_port;
int *CLN_sock;

char *msg="hello from server";
static int allocate_memlock=1;


/*\ client sends request to server
\*/
void armci_send_req(int proc)
{
int hdrlen = sizeof(request_header_t);
int dscrlen = ((request_header_t*)MessageSndBuffer)->dscrlen;
int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
int cluster = armci_clus_id(proc);
int stat;
int bytes;

    if(((request_header_t*)MessageSndBuffer)->operation == GET)
      bytes = dscrlen + hdrlen;
    else
      bytes = ((request_header_t*)MessageSndBuffer)->bytes + hdrlen;

     if(DEBUG_){
        printf("%d sending req=%d to (%d,%d,%d) dsclen=%d datlen=%d bytes=%d\n",
               armci_me, ((request_header_t*)MessageSndBuffer)->operation,
               ((request_header_t*)MessageSndBuffer)->to,
               cluster,proc,dscrlen,datalen,bytes);
        fflush(stdout);
     }
    stat = armci_WriteToSocket(SRV_sock[cluster], MessageSndBuffer, bytes);
    if(stat<0)armci_die("armci_send_req:write failed",stat);
}


void armci_write_strided_sock(void *ptr, int stride_levels, int stride_arr[], 
				   int count[], int fd)
{
    int i, j, stat;
    long idx;    /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL]; 

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<=stride_levels; j++) {
            idx += bvalue[j] * stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

	    /* memcpy(buf, ((char*)ptr)+idx, count[0]); */
	    /* buf += count[0]; */
        stat = armci_WriteToSocket(fd, ((char*)ptr)+idx, count[0]);
        if(stat<0)armci_die("armci_write_strided_sock:write failed",stat);
    }
}



void armci_read_strided_sock(void *ptr, int stride_levels, int stride_arr[], 
				   int count[], int fd)
{
    int i, j, stat;
    long idx;    /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL]; 

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<=stride_levels; j++) {
            idx += bvalue[j] * stride_arr[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

	/* memcpy(buf, ((char*)ptr)+idx, count[0]); */
	/* buf += count[0]; */
        stat = armci_ReadFromSocket(fd, ((char*)ptr)+idx, count[0]);
        if(stat<0)armci_die("armci_read_strided_sock:read failed",stat);
    }
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
int stat;
int bytes;

    if(DEBUG_){
      printf("%d:armci_send_strided: op=%d to=%d bytes= %d \n",armci_me,
					  msginfo->operation,proc,datalen);
      fflush(stdout);
    }

    bytes = msginfo->bytes + hdrlen;
    if(DEBUG_){
        printf("%d sending str req=%d to(%d,%d,%d) dslen=%d dlen=%d bytes=%d\n",
               armci_me, msginfo->operation, msginfo->to,
               cluster,proc,dscrlen,datalen,bytes);
        fflush(stdout);
    }

    if(count[0] <  TCP_PAYLOAD){

      /* for small contiguous blocks copy into a buffer before sending */
      armci_write_strided(ptr, strides, stride_arr, count, bdata);

      stat = armci_WriteToSocket(SRV_sock[cluster], msginfo, bytes);
      if(stat<0)armci_die("armci_send_strided:write failed",stat);

    }else{

      /* we write header + data descriptor */
      bytes = hdrlen + dscrlen;
      stat = armci_WriteToSocket(SRV_sock[cluster], msginfo, bytes);
      if(stat<0)armci_die("armci_send_strided:write failed",stat);

      /* for larger blocks write directly to socket thus avoiding memcopy */
      armci_write_strided_sock(ptr, strides, stride_arr,count,SRV_sock[cluster]);
    }
}



/*\ client receives vector data from server to buffer and unpacks it 
\*/
void armci_rcv_vector_data(int proc, char *buf, armci_giov_t darr[], int len)
{
    buf = armci_rcv_data(proc);
    armci_vector_from_buf(darr, len, buf);
}



/*\ server receives request
\*/
void armci_rcv_req(int p, void *phdr, void *pdescr, void *pdata, int *buflen )
{
request_header_t *msginfo = (request_header_t*)MessageRcvBuffer;
int hdrlen = sizeof(request_header_t);
int stat;
int bytes;

    stat =armci_ReadFromSocket(CLN_sock[p],MessageRcvBuffer,hdrlen);
    if(stat<0){
	fflush(stdout); 
	armci_die("armci_rcv_req: failed to receive header ",p);
    }

    if(DEBUG_){
      printf("%d(server):got %d req from %d len=(%d,%d,%d)\n",
            armci_me,msginfo->operation,p,msginfo->bytes,
            msginfo->dscrlen, msginfo->datalen );
      fflush(stdout);
    }

    *buflen = MSG_BUFLEN - hdrlen;
    *(void**)phdr = msginfo;

    if(msginfo->from != p)armci_die2("armci_rcv_req: from !=p",msginfo->from,p);
    if( (msginfo->to != armci_me && msginfo->to < armci_master ) ||
         msginfo->to >= armci_master + armci_clus_info[armci_clus_me].nslave)
         armci_die("armci_rcv_req: invalid to",msginfo->to); 

    if(msginfo->operation != GET && msginfo->bytes > MSG_BUFLEN)
       armci_die("armci_rcv_req:message overflowing rcv buffer",msginfo->bytes);

    if(msginfo->dscrlen<0)armci_die("armci_rcv_req:dscrlen<0",msginfo->dscrlen);
    if(msginfo->datalen<0)armci_die("armci_rcv_req:datalen<0",msginfo->datalen);
    if(msginfo->dscrlen > msginfo->bytes)
       armci_die2("armci_rcv_req:dsclen>bytes",msginfo->dscrlen,msginfo->bytes);

    if (msginfo->operation == GET)
      bytes = msginfo->dscrlen; 
    else
      bytes = msginfo->bytes;

    if(msginfo->bytes){
       stat = armci_ReadFromSocket(CLN_sock[p],msginfo+1,bytes);
       if(stat<0)armci_die("armci_rcv_req: read of data failed",stat);
       *(void**)pdescr = msginfo+1;
       *(void**)pdata  = msginfo->dscrlen + (char*)(msginfo+1); 
       *buflen -= msginfo->dscrlen; 

       if (msginfo->operation != GET)
           if(msginfo->datalen)*buflen -= msginfo->datalen;

    }else {

       *(void**)pdata  = msginfo+1;
       *(void**)pdescr = NULL;
    }
    
    if(msginfo->datalen>0 && msginfo->operation != GET){

       if(msginfo->datalen > MSG_BUFLEN -hdrlen -msginfo->dscrlen)
          armci_die2("armci_rcv_req:data overflowing buffer",
                      msginfo->dscrlen,msginfo->datalen);

       *buflen -= msginfo->datalen;

    }
/*        if (msginfo->operation == GET){
        printf("%d received GET datalen=%d\n",armci_me,
                                 msginfo->datalen);
        fflush(stdout);
       }
*/
}



/*\ client receives data from server
\*/
char* armci_rcv_data(int proc)
{
int cluster = armci_clus_id(proc);
int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
int stat;

    if(DEBUG_){
      printf("%d:armci_rcv_data:  bytes= %d \n",armci_me,datalen);
      fflush(stdout);
    }
    if(datalen == 0) armci_die("armci_rcv_data: no data to receive",datalen); 
    if(datalen > MSG_BUFLEN)
       armci_die("armci_rcv_data:data overflowing rcv buffer",datalen);

    stat =armci_ReadFromSocket(SRV_sock[cluster],MessageSndBuffer,datalen);
    if(stat<0)armci_die("armci_rcv_data: read failed",stat);
    if(DEBUG_){
      printf("%d:armci_rcv_data: got %d bytes \n",armci_me,datalen);
      fflush(stdout);
    }
    return MessageSndBuffer;
}


/*\ client receives strided data from server
\*/
void armci_rcv_strided_data(int proc, char *buf, int datalen, 
                        void *ptr, int strides, int stride_arr[], int count[])
{
int cluster = armci_clus_id(proc);
int stat;

    if(DEBUG_){
      printf("%d:armci_rcv_strided_data:  from %d \n",armci_me,proc);
      fflush(stdout);
    }

   if(count[0] < TCP_PAYLOAD && strides>1 ){

     /* for small data segments minimize number of system calls */
     stat =armci_ReadFromSocket(SRV_sock[cluster],buf,datalen);
     if(stat<0)armci_die("armci_rcv_data: read failed",stat);
	
     armci_read_strided(ptr, strides, stride_arr, count, buf);

   }else{
	

     /* for larger blocks read directly from socket thus avoiding memcopy */
     armci_read_strided_sock(ptr, strides, stride_arr, count, SRV_sock[cluster]);
   }

    if(DEBUG_){
      printf("%d:armci_rcv_data: got %d bytes \n",armci_me,datalen);
      fflush(stdout);
    }
}




/*\ server response - send data to client
\*/
void armci_send_data(request_header_t* msginfo, void *data)
{
int to = msginfo->from;
int stat;

    stat = armci_WriteToSocket(CLN_sock[to], data, msginfo->datalen);
    if(stat<0)armci_die("armci_send_data:write failed",stat);
}



/*\ server sends strided data back to client 
\*/
void armci_send_strided_data(int proc,  request_header_t *msginfo, char *bdata, 
                        void *ptr, int strides, int stride_arr[], int count[])
{
int to = msginfo->from;
int datalen = msginfo->datalen;

   if(count[0] < TCP_PAYLOAD && strides>1 ){

       /* minimize number of system calls */
       int stat;

       /* for small contiguous blocks copy into a buffer before sending */
       armci_write_strided(ptr, strides, stride_arr, count, bdata);

       stat = armci_WriteToSocket(CLN_sock[to], bdata, datalen);
       if(stat<0)armci_die("armci_send_strided_data:write failed",stat);

   }else{

     /* for larger blocks write directly to socket thus avoiding memcopy */
     armci_write_strided_sock(ptr, strides, stride_arr, count, CLN_sock[to]);
   }

}



/*\ server writes data to socket associated with process "to"
\*/
void armci_sock_send(int to, void* data, int len)
{
int stat;

    stat = armci_WriteToSocket(CLN_sock[to], data, len);
    if(stat<0)armci_die("armci_sock_send:write failed",stat);
}


/*\ control message to the server, e.g.: ATTACH to shmem, return ptr etc.
\*/
void armci_serv_attach_req(void *info, int ilen, long size, void* resp,int rlen)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
char *buf;

    msginfo->from  = armci_me;
    msginfo->to    = SERVER_NODE(armci_clus_me);
    msginfo->dscrlen   = ilen;
    msginfo->datalen = sizeof(long)+sizeof(rlen);
    msginfo->operation =  ATTACH;
    msginfo->bytes = msginfo->dscrlen+ msginfo->datalen;

    armci_copy(info, msginfo +1, ilen);
    buf = MessageSndBuffer + ilen + sizeof(request_header_t); 
    *((long*)buf) =size;
    *(int*)(buf+ sizeof(long)) =rlen;
    armci_send_req(armci_master);
    if(rlen){
      msginfo->datalen = rlen;
      armci_rcv_data(armci_master);  /* receive response */
      armci_copy(MessageSndBuffer, resp, rlen);
      if(DEBUG_){
         printf("%d:client attaching got ptr %d bytes \n",armci_me,rlen);
         fflush(stdout);
      }
    }
}



/*\ server initializes its copy of the memory lock data structures
\*/
static void server_alloc_memlock(void *ptr_myclus)
{
int i;

    /* for protection, set pointers for processes outside local node NULL */
    memlock_table_array = calloc(armci_nproc,sizeof(void*));
    if(!memlock_table_array) armci_die("malloc failed for ARMCI lock array",0);

    /* set pointers for processes on local cluster node
     * ptr_myclus - corresponds to the master process
     */
    for(i=0; i< armci_clus_info[armci_clus_me].nslave; i++){
        memlock_table_array[armci_master +i] = ((char*)ptr_myclus)
                + MAX_SLOTS*sizeof(memlock_t)*i;
    } 

    if(DEBUG_)
       fprintf(stderr,"server initialized memlock\n");
}
            
    

/*\ server actions triggered by client request to ATTACH
\*/
void armci_server_ipc(request_header_t* msginfo, void* descr, 
                      void* buffer, int buflen)
{
   double *ptr;
   long *idlist = (long*)descr;
   long size = *(long*)buffer;
   int rlen = *(int*)(sizeof(long)+(char*)buffer);

   if(size<0) armci_die("armci_server_ipc: size<0",(int)size);
   ptr=(double*)Attach_Shared_Region(idlist+1,size,idlist[0]);
   if(!ptr)armci_die("armci_server_ipc: failed to attach",0);

   /* provide data server with access to the memory lock data structures */
   if(allocate_memlock){
      allocate_memlock = 0;
      server_alloc_memlock(ptr);
   }

   /* compute offset if we are are really allocating new memory */
   if(size>0)armci_set_mem_offset(ptr);
   
   if(msginfo->datalen != sizeof(long)+sizeof(int))
      armci_die("armci_server_ipc: bad msginfo->datalen ",msginfo->datalen);

   if(rlen==sizeof(ptr)){
     msginfo->datalen = rlen;
     armci_send_data(msginfo, &ptr);
   }else armci_die("armci_server_ipc: bad rlen",rlen);
}


/*\ close all open sockets, called before terminating/aborting
\*/
void armci_transport_cleanup()
{
     if(SERVER_CONTEXT) 
        armci_ShutdownAll(CLN_sock,armci_nproc); /*server */
     else
        armci_ShutdownAll(SRV_sock,armci_nclus); /*client */
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

     armci_ShutdownAll(CLN_sock,armci_nproc);

     /* Finalizing data server process w.r.t. MPI is not portable 
      * some IBM implementations of MPI could hang in MPI_Finalize
      */ 
#ifdef MPICH_NAME___ 
        MPI_Finalize();
#endif

     _exit(0);
}


/*\ request to QUIT sent by client
\*/
void armci_serv_quit()
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
int stat;

     if(DEBUG_){
        printf("%d master: sending quit request to server\n",armci_me);
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
   
    armci_send_req(armci_master);

    if(ACK_QUIT){
       armci_rcv_data(armci_master);  /* receive ACK */
       stat = * (int*)MessageSndBuffer;
       if(stat  != QUIT)
            armci_die("armci_serv_quit: wrong response from server", stat);
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


/*\ get ACK from server
\*/
void armci_rem_ack(int clus)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;

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



/*\ main routine for data server process in a cluster environment
 *  the process is blocked (in select) until message arrives from
 *  the clients and services the requests
\*/
void armci_data_server()
{
int nready;
int *readylist;
int up=1;

    readylist = (int*)calloc(sizeof(int),armci_nproc);
    if(!readylist)armci_die("armci_data_server:could not allocate readylist",0);

    if(DEBUG_){
      printf("%d server waiting for request\n",armci_me); fflush(stdout);
      sleep(1);
    }

    /* server main loop; wait for and service requests until QUIT requested */
    for(;;){ 
      int i, p;
      nready = armci_WaitSock(CLN_sock, armci_nproc, readylist);

      for(i = 0; i < armci_nproc; i++){
          void *descr, *buffer;
          request_header_t *msginfo; 
          int buflen;

          p = (up) ? i : armci_nproc -1 -i;
          if(!readylist[p])continue; 
          
          /* read header, descriptor, data, and buffer length */
          armci_rcv_req(p, &msginfo, &descr, &buffer, &buflen );

          switch(msginfo->operation){

          case ACK:    if(DEBUG_){
                          printf("%d(server):got ACK request from %d\n",
                          armci_me, p);
                          fflush(stdout);
                       } 
                       armci_server_ack(msginfo);
                       break;

          case ATTACH: if(DEBUG_){
                          printf("%d(server):got ATTACH request from %d\n",
                          armci_me, p);
                          fflush(stdout);
                       }
                       armci_server_ipc(msginfo, descr, buffer, buflen); 
                       /*if(armci_me == -1001)msg[300]=0;  debug */
                       break;

          case QUIT:   if(DEBUG_){
                          printf("%d(server):got QUIT request from %d\n",
                          armci_me, p);
                          fflush(stdout);
                       } 
                       if(nready>1)
                         printf("WARNING:quit request when queue not empty%d\n",
                                 nready); fflush(stdout);
                       free(readylist);
                       armci_server_goodbye(msginfo);
                       break;

          case ARMCI_SWAP:
          case ARMCI_SWAP_LONG:
          case ARMCI_FETCH_AND_ADD:
          case ARMCI_FETCH_AND_ADD_LONG:
                       armci_server_rmw(msginfo,descr,buffer);
                       break;
             
          case LOCK:   armci_server_lock(msginfo); 
                       break;

          case UNLOCK: armci_server_unlock(msginfo, descr);
                       break;

          default:     if(msginfo->format ==VECTOR)
                          armci_server_vector(msginfo, descr, buffer, buflen); 
                       else if(msginfo->format ==STRIDED)
                          armci_server(msginfo, descr, buffer, buflen); 
                       else
                          armci_die2("armci_data_serv: unknown format code",
                                      msginfo->format, msginfo->from);
          }            
          nready--;
          if(nready==0) break; /* all sockets read */
      }

      /* fairness attempt: each time process the list in a different direction*/
      up = 1- up; /* switch directions for the next round */

      if(nready)
        armci_die("armci_dataserv:readylist not consistent with nready",nready);
    } 
}
   

/*\ Create Sockets for clients and servers 
\*/
void armci_create_connections()
{
  int p,master = armci_clus_info[armci_clus_me].master;

  /* sockets for communication with data server */
  SRV_sock = (int*) malloc(sizeof(int)*armci_nclus);
  if(!SRV_sock)armci_die("ARMCI cannot allocate SRV_sock",armci_nclus);

  /* array that will be used to exchange port info */
  AR_port = (int*) calloc(armci_nproc * armci_nclus, sizeof(int));
  if(!AR_port)armci_die("ARMCI cannot allocate AR_port",armci_nproc*armci_nclus);

  /* create sockets for communication with each user process */ 
  if(master==armci_me){
     CLN_sock = (int*) malloc(sizeof(int)*armci_nproc);
     if(!CLN_sock)armci_die("ARMCI cannot allocate CLN_sock",armci_nproc);

     for(p=0; p< armci_nproc; p++){
       int off_port = armci_clus_me*armci_nproc; 
#      ifdef SERVER_THREAD
         if(p >=armci_clus_first && p <= armci_clus_last) CLN_sock[p]=-1;
         else
#      endif
         armci_CreateSocketAndBind(CLN_sock + p, AR_port + p +off_port);
     }
  }
}



void armci_wait_for_server()
{
  if(armci_me == armci_master){
#ifndef SERVER_THREAD
     RestoreSigChldDfl();
     armci_serv_quit();
     armci_wait_server_process();
#endif
  }
}



void armci_client_code()
{
  int stat,c, nall;
  char str[100];
#ifndef SERVER_THREAD
  int p;
#endif

  if(DEBUG_){
     bzero(str,99);
     printf("in client after fork %d\n",armci_me);
     fflush(stdout);
  }

#ifndef SERVER_THREAD
  /* master has to close all sockets -- they are used by server PROCESS */ 
  if(armci_master==armci_me)for(p=0; p< armci_nproc; p++){
     close(CLN_sock[p]);
  } 
#endif

  /* exchange port numbers with processes in all cluster nodes
   * save number of messages by using global sum -only masters contribute
   */

  nall = armci_nclus*armci_nproc;
  armci_msg_igop(AR_port,nall,"+");
  
  /*using port number create socket & connect to data server in each clus node*/
  for(c=0; c< armci_nclus; c++){
      
      int off_port = c*armci_nproc; 

#ifdef SERVER_THREAD
      /*no intra node socket connection with server thread*/
      if(c == armci_clus_me) SRV_sock[c]=-1; 
      else
#endif
       SRV_sock[c] = armci_CreateSocketAndConnect(armci_clus_info[c].hostname,
                                                  AR_port[off_port + armci_me]);
      if(DEBUG_ && SRV_sock[c]!=-1){
         printf("%d: client connected to %s:%d\n",armci_me,
             armci_clus_info[c].hostname, AR_port[off_port + armci_me]);
         fflush(stdout);
      }

  }

  if(DEBUG_){
     printf("%d client connected to all %d\n",armci_me, armci_nclus);
  
     for(c=0; c< armci_nclus; c++)if(SRV_sock[c]!=-1){
        stat =armci_ReadFromSocket(SRV_sock[c],str, sizeof(msg)+1);
        if(stat<0)armci_die("read failed",stat);
        printf("in client %d message was=%s from%d\n",armci_me,str,c); 
        fflush(stdout);
     }
  }

  /* we do not need the port numbers anymore */
  free(AR_port);
}


void * armci_server_code(void *data)
{
     int stat, p;

     if(DEBUG_){
        printf("in server after fork %d\n",armci_me);
        fflush(stdout);
     }

     /* establish connections with compute processes */
#ifdef SERVER_THREAD

     if(armci_clus_first>0)
        armci_AcceptSockAll(CLN_sock, armci_clus_first);
     if(armci_clus_last< armci_nproc-1)
        armci_AcceptSockAll(CLN_sock + armci_clus_last+1, 
                            armci_nproc-armci_clus_last-1);
#else
     armci_AcceptSockAll(CLN_sock, armci_nproc);
#endif

     if(DEBUG_){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);

       sleep(1);
       for(p=0; p<armci_nproc; p++)if(CLN_sock[p]!=-1){
         stat = armci_WriteToSocket(CLN_sock[p], msg, sizeof(msg)+1);
         if(stat<0)armci_die("write failed",stat);
       }
       sleep(5);
     }

#ifndef SERVER_THREAD
     /* we do not need the port numbers anymore */
     free(AR_port);
#endif

     armci_data_server();
     return(NULL);

}


void armci_start_server()
{
    
   /* create socket connections accross the cluster */
   armci_create_connections();

   if(armci_me == armci_master){

#ifdef SERVER_THREAD

     /* skip sockets associated with processes on the current node */
     if(armci_clus_first>0)
        armci_ListenSockAll(CLN_sock, armci_clus_first); 

     if(armci_clus_last< armci_nproc-1)
        armci_ListenSockAll(CLN_sock + armci_clus_last+1, 
                            armci_nproc-armci_clus_last-1);

     armci_create_server_thread( armci_server_code );

#else

     armci_ListenSockAll(CLN_sock, armci_nproc);
     armci_create_server_process( armci_server_code );

#endif

   }  
     
   armci_client_code();
}
