/* $Id: dataserv.c,v 1.6 1999-10-14 00:18:50 d3h325 Exp $ */
#include "armcip.h"
#include "sockets.h"
#include "request.h"
#include "message.h"
#include "memlock.h"
#include "copy.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>


#define ACK 0
#define DEBUG_ 0
#define QUIT 33
#define ATTACH 34
#define SOFFSET -1000

extern int AR_ready_sigchld;
int *AR_sock;
int *AR_port;

int init_port, init_socket;
pid_t server_pid= (pid_t)0;
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
               armci_me,
               ((request_header_t*)MessageSndBuffer)->operation,
               ((request_header_t*)MessageSndBuffer)->to,
               cluster,proc,dscrlen,datalen,bytes);
        fflush(stdout);
     }
    stat = armci_WriteToSocket(AR_sock[cluster], MessageSndBuffer, 
                 bytes);
    if(stat<0)armci_die("armci_send_req:write failed",stat);
}


/*\ server receives request
\*/
void armci_rcv_req(int p, void *phdr, void *pdescr, void *pdata, int *buflen )
{
request_header_t *msginfo = (request_header_t*)MessageRcvBuffer;
int hdrlen = sizeof(request_header_t);
int stat;
int bytes;

    stat =armci_ReadFromSocket(AR_sock[p],MessageRcvBuffer,hdrlen);
    if(stat<0)armci_die("armci_rcv_req: failed to receive header ",stat);

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

    if(msginfo->bytes > MSG_BUFLEN)
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
       stat = armci_ReadFromSocket(AR_sock[p],msginfo+1,bytes);
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
void armci_rcv_data(int proc)
{
int cluster = armci_clus_id(proc);
int datalen = ((request_header_t*)MessageSndBuffer)->datalen;
int stat;

    if(datalen == 0) armci_die("armci_rcv_data: no data to receive",datalen); 
    if(datalen > MSG_BUFLEN)
       armci_die("armci_rcv_data:data overflowing rcv buffer",datalen);

    stat =armci_ReadFromSocket(AR_sock[cluster],MessageSndBuffer,datalen);
    if(stat<0)armci_die("armci_rcv_data: read failed",stat);
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

    stat = armci_WriteToSocket(AR_sock[to], data, msginfo->datalen);
    if(stat<0)armci_die("armci_send_data:write failed",stat);
}


/*\ write data to socket associated with process "to"
\*/
void armci_sock_send(int to, void* data, int len)
{
int stat;

    stat = armci_WriteToSocket(AR_sock[to], data, len);
    if(stat<0)armci_die("armci_sock_send:write failed",stat);
}


/*\ control message to the server, e.g.: ATTACH to shmem, return ptr etc.
\*/
void armci_serv_attach_req(void *info, int ilen, long size, void* resp,int rlen)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
char *buf;

    msginfo->from  = armci_me;
    msginfo->to    = SOFFSET - armci_master; /*server id derived from master*/
    msginfo->dscrlen   = ilen;
    msginfo->datalen = sizeof(long)+sizeof(rlen);
    msginfo->operation = msginfo->format  =  ATTACH;
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

   /* provide data server with access to the memory lock data structures */
   if(allocate_memlock){
      allocate_memlock = 0;
      server_alloc_memlock(ptr);
   }
   
   if(msginfo->datalen != sizeof(long)+sizeof(int))
      armci_die("armci_server_ipc: bad msginfo->datalen ",msginfo->datalen);

   if(rlen==sizeof(ptr)){
     msginfo->datalen = rlen;
     armci_send_data(msginfo, &ptr);
   }else armci_die("armci_server_ipc: bad rlen",rlen);
}


/*\ close all open sockets, called before terminating/aborting
\*/
void armci_CleanupSockets()
{
     if(armci_me < 0) 
        armci_ShutdownAll(AR_sock,armci_nproc); /*server */
     else
        armci_ShutdownAll(AR_sock,armci_nclus); /*client */
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

     armci_CleanupSockets();
/*     sleep(1);*/

#ifdef MPI
        MPI_Finalize();
#endif

     exit(0);
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
    msginfo->to    = SOFFSET - armci_master; /* server id derived from master*/
    msginfo->format  = msginfo->operation = QUIT;
    if(ACK)
       msginfo->bytes   = msginfo->datalen = sizeof(int); /* ACK */
    else
       msginfo->bytes   = msginfo->datalen = 0; /* no ACK */
   
    armci_send_req(armci_master);

    if(ACK){
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
    msginfo->to    = SOFFSET -armci_clus_info[clus].master; 
    msginfo->format  = msginfo->operation = ACK;
    msginfo->bytes   =0;
    msginfo->datalen =sizeof(int);

    armci_send_req(armci_clus_info[clus].master);
    armci_rcv_data(armci_clus_info[clus].master);  /* receive ACK */
}



/*\ main routine for data server process in cluster environment
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

    /* server main loop; wait for and service requests until QUIT requested */
    for(;;){ 
      int i, p;
      nready = armci_WaitSock(AR_sock, armci_nproc, readylist);

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
   


void armci_create_connections()
{
  int p,master = armci_clus_info[armci_clus_me].master;

  AR_sock = (int*) malloc(sizeof(int)*armci_nproc);
  if(!AR_sock)armci_die("ARMCI cannot allocate AR_sock",armci_nproc);
  AR_port = (int*) calloc(armci_nproc * armci_nclus, sizeof(int));
  if(!AR_port)armci_die("ARMCI cannot allocate ARport",armci_nproc*armci_nclus);

  /* create sockets for communication with each user process */ 
  if(master==armci_me)for(p=0; p< armci_nproc; p++){
      int off_port = armci_clus_me*armci_nproc; 
      armci_CreateSocketAndBind(AR_sock + p, AR_port + p +off_port);
/*      printf("%d:port %d\n",armci_me,AR_port[p+off_port]); fflush(stdout);*/
  }
}


void armci_wait_for_server()
{
  if(armci_me == armci_master && server_pid ){
     int stat;
     pid_t rc;
     RestoreSigChldDfl();
     armci_serv_quit();
     rc = wait (&stat);
     if (rc != server_pid){
         perror("ARMCI master: wait for child process (server) failed:");
     }
     server_pid = (pid_t)0;

  }
}



void armci_client_code()
{
  int stat,p, c, master = armci_clus_info[armci_clus_me].master;
  char str[100];


  if(DEBUG_){
     bzero(str,99);
     printf("in client after fork %d\n",armci_me);
  }

  /* master has to close all sockets -- they are used by server */ 
  if(master==armci_me)for(p=0; p< armci_nproc; p++){
     close(AR_sock[p]);
  } 

  /* exchange port numbers with processes in all cluster nodes
   * save number of messages by using global sum -only masters contribute
   */

  armci_msg_igop((long*)AR_port,armci_nclus*armci_nproc,"+",0);
  
  /*using port number create socket & connect to data server in each clus node*/
  for(c=0; c< armci_nclus; c++){
      
      int off_port = c*armci_nproc; 

      if(DEBUG_){
         printf("%d: client connecting to %s:%d\n",armci_me,
             armci_clus_info[c].hostname, AR_port[off_port + armci_me]);
         fflush(stdout);
      }
      AR_sock[c] = armci_CreateSocketAndConnect(armci_clus_info[c].hostname,
                                                AR_port[off_port + armci_me]);
  }

  if(DEBUG_){
     printf("%d client connected to all %d\n",armci_me, armci_nclus);
  
     for(c=0; c< armci_nclus; c++){
        stat =armci_ReadFromSocket(AR_sock[c],str, sizeof(msg)+1);
        if(stat<0)armci_die("read failed",stat);
        printf("in client %d message was=%s from%d\n",armci_me,str,c); 
        fflush(stdout);
     }
  }
     
}


void armci_server_code()
{
     int stat, p;

     armci_me = SOFFSET -armci_me; /* server id derived from parent id */

     if(DEBUG_)
        printf("in server after fork %d\n",armci_me);

     /* establish connections with compute processes */
    
     armci_ListenAndAcceptAll(AR_sock, armci_nproc);

     if(DEBUG_){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);

       for(p=0; p<armci_nproc; p++){
         stat = armci_WriteToSocket(AR_sock[p], msg, sizeof(msg)+1);
         if(stat<0)armci_die("write failed",stat);
       }
     }

     armci_data_server();
}



void armci_start_server()
{
   pid_t pid;
    
   armci_create_connections();

   if(armci_me == armci_master){
     if ( (pid = fork() ) < 0)
        armci_die("fork failed", (int)pid);

     else if(pid == 0) {

        armci_server_code();

     }else {

        server_pid = pid;
        armci_client_code();

     }

   } else 
     armci_client_code();

}

