/* $Id: dataserv.c,v 1.20 2001-05-25 22:09:19 d3h325 Exp $ */
#include "armcip.h"
#include "sockets.h"
#include "request.h"
#include "copy.h"
#include <stdio.h>
#include <errno.h>

#define DEBUG_ 0

extern int AR_ready_sigchld;
int *SRV_sock;
int *AR_port;
int *CLN_sock;

char *msg="hello from server";
static int *readylist=(int*)0;

/*\ client sends request message to server
\*/
int armci_send_req_msg(int proc, void *buf, int bytes)
{
int cluster = armci_clus_id(proc);
    if(armci_WriteToSocket(SRV_sock[cluster], buf, bytes) <0) return 1;
    else return 0;
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
int armci_send_req_msg_strided(int proc, request_header_t *msginfo,char *ptr,
                               int strides, int stride_arr[], int count[])
{
int cluster = armci_clus_id(proc);
int stat, bytes;

    if(DEBUG_){
      printf("%d:armci_send_req_msg_strided: op=%d to=%d bytes= %d \n",armci_me,
				  msginfo->operation,proc,msginfo->datalen);
      fflush(stdout);
    }

    /* we write header + data descriptor */
    bytes = sizeof(request_header_t) + msginfo->dscrlen;
    stat = armci_WriteToSocket(SRV_sock[cluster], msginfo, bytes);
    if(stat<0)armci_die("armci_send_strided:write failed",stat);

    /* for larger blocks write directly to socket thus avoiding memcopy */
    armci_write_strided_sock(ptr, strides,stride_arr,count,SRV_sock[cluster]);

    return 0;
}


char *armci_ReadFromDirect(int proc, request_header_t * msginfo, int len)
{
int cluster=armci_clus_id(proc);
int stat;

    if(DEBUG_){
      printf("%d:armci_ReadFromDirect:  from %d \n",armci_me,proc);
      fflush(stdout);
    }

    stat =armci_ReadFromSocket(SRV_sock[cluster],msginfo+1,len);
    if(stat<0)armci_die("armci_rcv_data: read failed",stat);
    return(char*)(msginfo+1);
}


/*\ client receives strided data from server
\*/
void armci_ReadStridedFromDirect(int proc, request_header_t* msginfo, void *ptr,
                                 int strides, int stride_arr[], int count[])
{
int cluster=armci_clus_id(proc);

    if(DEBUG_){
      printf("%d:armci_ReadStridedFromDirect:  from %d \n",armci_me,proc);
      fflush(stdout);
    }
    armci_read_strided_sock(ptr, strides, stride_arr, count, SRV_sock[cluster]);
}


/*********************************** server side ***************************/

/*\ server receives request
\*/
void armci_rcv_req(void *mesg, void *phdr, void *pdescr,void *pdata,int *buflen)
{
request_header_t *msginfo = (request_header_t*)MessageRcvBuffer;
int hdrlen = sizeof(request_header_t);
int stat, p = *(int*)mesg;
int bytes;

    stat =armci_ReadFromSocket(CLN_sock[p],MessageRcvBuffer,hdrlen);
    if(stat<0) armci_die("armci_rcv_req: failed to receive header ",p);

    if(DEBUG_){ printf("%d(server):got %d req from %d len=(%d,%d,%d)\n",
            armci_me,msginfo->operation,p,msginfo->bytes,
            msginfo->dscrlen, msginfo->datalen ); fflush(stdout);
    }

    *buflen = MSG_BUFLEN - hdrlen;
    *(void**)phdr = msginfo;

    if (msginfo->operation == GET)
      bytes = msginfo->dscrlen; 
    else{
      bytes = msginfo->bytes;
      if(bytes >*buflen)armci_die2("armci_rcv_req: message overflowing rcv buf",
                                    msginfo->bytes,*buflen);
    }

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
}


/*\ send data back to client
\*/
void armci_WriteToDirect(int to, request_header_t* msginfo, void *data)
{
int stat = armci_WriteToSocket(CLN_sock[to], data, msginfo->datalen);
    if(stat<0)armci_die("armci_WriteToDirect:write failed",stat);
}


/*\ server sends strided data back to client 
\*/
void armci_WriteStridedToDirect(int proc, request_header_t* msginfo,
                         void *ptr, int strides, int stride_arr[], int count[])
{
    if(DEBUG_){ printf("%d:armci_WriteStridedToDirect:from %d\n",armci_me,proc);
      fflush(stdout);
    }
    armci_write_strided_sock(ptr, strides, stride_arr, count, CLN_sock[proc]);
}


/*\ server writes data to socket associated with process "to"
\*/
void armci_sock_send(int to, void* data, int len)
{
int stat = armci_WriteToSocket(CLN_sock[to], data, len);
    if(stat<0)armci_die("armci_sock_send:write failed",stat);
}


/*\ close all open sockets, called before terminating/aborting
\*/
void armci_transport_cleanup()
{
     if(SERVER_CONTEXT){ 
         if(readylist)free(readylist);
         armci_ShutdownAll(CLN_sock,armci_nproc); /*server */
     }else
         armci_ShutdownAll(SRV_sock,armci_nclus); /*client */
}


/*\ main routine for data server process in a cluster environment
 *  the process is blocked (in select) until message arrives from
 *  the clients and services the requests
\*/
void armci_call_data_server()
{
int nready;
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

          p = (up) ? i : armci_nproc -1 -i;
          if(!readylist[p])continue; 
          
          armci_data_server(&p);

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
void armci_init_connections()
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

#ifdef SERVER_THREAD
     /* skip sockets associated with processes on the current node */
     if(armci_clus_first>0)
        armci_ListenSockAll(CLN_sock, armci_clus_first);

     if(armci_clus_last< armci_nproc-1)
        armci_ListenSockAll(CLN_sock + armci_clus_last+1,
                            armci_nproc-armci_clus_last-1);
#else
     armci_ListenSockAll(CLN_sock, armci_nproc);
#endif

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


void armci_client_connect_to_servers()
{
  int stat,c, nall;
  char str[100];
#ifndef SERVER_THREAD
  int p;

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
     bzero(str,99);
  
     for(c=0; c< armci_nclus; c++)if(SRV_sock[c]!=-1){
        stat =armci_ReadFromSocket(SRV_sock[c],str, sizeof(msg)+1);
        if(stat<0)armci_die("read failed",stat);
        printf("in client %d message was=%s from%d\n",armci_me,str,c); 
        fflush(stdout);
     }
  }

  free(AR_port); /* we do not need the port numbers anymore */
}


/*\ establish connections with compute processes
\*/
void armci_server_initial_connection()
{

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
       int stat, p;
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
}
