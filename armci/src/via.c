#include <stdio.h>
#include <strings.h>
#include <assert.h>
#include "armcip.h" 

#define VIPL095  
#include <vipl.h>

#define DEBUG_ 0
#define DEBUG0 0
#define DEBUG1 0
#define PAUSE_ON_ERROR__ 

#define VIADEV_NAME "/dev/clanvi0"
#define ADDR_LEN 6
#define SIXTYFOUR 64
#define NONE -1
#define GET_DATA_PTR(buf) (sizeof(request_header_t) + (char*)buf)

typedef struct {
  VIP_NIC_HANDLE handle;
  VIP_NIC_ATTRIBUTES attr;
  VIP_PROTECTION_HANDLE ptag;
  VIP_CQ_HANDLE scq;
  VIP_CQ_HANDLE rcq;
  int maxtransfersize;
} nic_t;

typedef double discrim_t;
#define MAKE_DISCRIMINATOR(base, id) (discrim_t)(1100.0 * base + id + 3) 

static nic_t nic_arr[2];
static nic_t *SRV_nic= nic_arr;
static nic_t *CLN_nic= nic_arr+1;


struct cb_args {
    pthread_cond_t *cond;
    pthread_mutex_t *mutex;
};


typedef struct {
   VIP_NET_ADDRESS *rem;
   VIP_NET_ADDRESS *loc;
   char st_remote[40];
   char st_local[40];
   VIP_VI_HANDLE vi;
} armci_connect_t;


typedef struct {
  VIP_DESCRIPTOR dscr;
  char buf[VBUF_DLEN];
}vbuf_t;


typedef struct {
  VIP_DESCRIPTOR snd_dscr;
  VIP_DESCRIPTOR rcv_dscr;
  char buf[MSG_BUFLEN];
}vbuf_long_t;


static vbuf_t *serv_buf_arr;
static vbuf_long_t *client_buf, *serv_buf;
static VIP_MEM_HANDLE serv_memhandle, client_memhandle;
static armci_connect_t *SRV_con;
static armci_connect_t *CLN_con;
static int *AR_discrim;

char *MessageSndBuffer;
char *MessageRcvBuffer;
int armci_long_buf_free=1;
int armci_long_buf_taken_srv=NONE;
static int armci_server_terminating=0;
static int armci_ack_proc=NONE;

extern void armci_send_data_to_client(int proc, void *buf, int bytes);

void armci_wait_for_server()
{
  armci_server_terminating=1;
}

void armci_transport_cleanup()
{
}


/*\ server receives request
\*/
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    int stat;
    vbuf_t *vbuf = (vbuf_t*)mesg;
    request_header_t *msginfo = (request_header_t *)vbuf->buf;
    *(void **)phdr = msginfo;

    if(DEBUG0) {
        printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from);
        fflush(stdout);
    }

    /* we leave room for msginfo on the client side */
    *buflen = MSG_BUFLEN - sizeof(request_header_t);

    if(msginfo->bytes) {
        *(void **)pdescr = msginfo+1;
        if(msginfo->operation == GET)
            *(void **)pdata = MessageRcvBuffer;
        else
            *(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);
    }else {
        *(void**)pdescr = NULL;
        *(void **)pdata = MessageRcvBuffer;
    }  
}



static void armci_check_status(int debug, VIP_RETURN rc, char *msg)
{
#define BLEN 80

     if(rc != VIP_SUCCESS){

        char buf[BLEN];
        
        if(armci_server_terminating){
           /* server got interrupted when clients terminate connections */
           sleep(1);
           _exit(0);
        }
          
        fprintf(stderr,"%d in check FAILURE %s\n",armci_me,msg);
        assert(strlen(msg)<BLEN-20);
        sprintf(buf,"ARMCI(via):failure: %s code ",msg);
#       ifdef  PAUSE_ON_ERROR
          printf("%d(%d): Error from VIPL: %s - pausing\n",
                 armci_me, getpid(), msg);
          fflush(stdout);
          pause();
#       endif
        armci_die(buf,(int)rc);

     }else if(debug){

         printf("%d:ARMCI(via): %s successful\n",armci_me,msg); 
         fflush(stdout);

     }
}


/*\ establish h/w connection to nic
\*/
void armci_init_nic(nic_t *nic, int scq_entries, int rcq_entries)
{
VIP_RETURN rc;
void armci_err_callback(void *context, VIP_ERROR_DESCRIPTOR * d);

    bzero(nic,sizeof(nic_t));

    rc = VipOpenNic(VIADEV_NAME, &nic->handle);
    armci_check_status(DEBUG0, rc,"open nic");
    
    rc = VipQueryNic(nic->handle, &nic->attr);
    armci_check_status(DEBUG0, rc,"query nic");
    
    rc = VipCreatePtag(nic->handle, &nic->ptag);
    armci_check_status(DEBUG0, rc,"create ptag");

    if(scq_entries){
       rc = VipCreateCQ(nic->handle, scq_entries, &nic->scq);
       armci_check_status(DEBUG0, rc,"create send completion queue");
    }

    if(rcq_entries){
       rc = VipCreateCQ(nic->handle, rcq_entries, &nic->rcq);
       armci_check_status(DEBUG0, rc,"create receive completion queue");
    }


    VipErrorCallback(nic->handle, 0, armci_err_callback);
}


/*\ create vi instance to nic, no connection
\*/
VIP_VI_HANDLE armci_create_vi(nic_t *nic)
{
VIP_RETURN rc;
VIP_VI_HANDLE vi;
VIP_VI_ATTRIBUTES vattr;

    bzero(&vattr,sizeof(VIP_VI_ATTRIBUTES));

    vattr.EnableRdmaWrite  = VIP_TRUE; 
    vattr.ReliabilityLevel = VIP_SERVICE_RELIABLE_DELIVERY;
    vattr.MaxTransferSize  = nic->attr.MaxTransferSize;

    rc= VipCreateVi(nic->handle, &vattr, nic->scq, nic->rcq, &vi);

    armci_check_status(DEBUG0, rc,"create vi");

    return(vi);
}
    

void armci_make_netaddr(VIP_NET_ADDRESS *pnaddr, char* hostname, discrim_t dm)
{
VIP_RETURN rc;
char *p = (char*)pnaddr;

    rc = VipNSGetHostByName(SRV_nic->handle,hostname,pnaddr,0);
    armci_check_status(DEBUG0, rc,"get host name address");

    pnaddr->HostAddressLen = SRV_nic->attr.NicAddressLen;  
    p = pnaddr->HostAddress + SRV_nic->attr.NicAddressLen;
    pnaddr->DiscriminatorLen = sizeof(discrim_t);
    *(discrim_t*)p = dm;
}


void armci_create_connections()
{
VIP_RETURN rc;
int c,s;
VIP_NET_ADDRESS *pnetaddr;
int *AR_base;
    
    /* get base for connection descriptor - we use process id */
    AR_base = (int*)malloc(armci_nproc * sizeof(int));
    if(!AR_base)armci_die("malloc failed for AR_base",0);
    bzero(AR_base,armci_nproc * sizeof(int));
    AR_base[armci_me]=(int)getpid();
    armci_msg_igop(AR_base,armci_nproc,"+"); /*exchange it globally */

    /* initialize nic connection for talking to servers */
    armci_init_nic(SRV_nic,0,0);

    /* for pier network address we need name service */
    rc = VipNSInit(SRV_nic->handle, NULL);
    armci_check_status(DEBUG0, rc,"init name service");

    /* allocate and initialize connection structs */
    SRV_con=(armci_connect_t*)malloc(sizeof(armci_connect_t)*armci_nclus);
    if(!SRV_con)armci_die("cannot allocate SRV_con",armci_nclus);

    for(s=0; s< armci_nclus; s++)if(armci_clus_me != s){
       char *ptr;
       discrim_t dm;
       int cluster = s;
       int master  = armci_clus_info[cluster].master;
       armci_connect_t *con = SRV_con + s;

       con->loc = (void*)con->st_local;
       con->rem = (void*)con->st_remote;
       con->vi  = armci_create_vi(SRV_nic);

       dm = MAKE_DISCRIMINATOR(AR_base[master], armci_me);
       if(DEBUG_)printf("%d:discriminator(%d)=%lf\n",armci_me,master,dm);
       armci_make_netaddr(con->loc, armci_clus_info[armci_clus_me].hostname,dm);
       armci_make_netaddr(con->rem, armci_clus_info[cluster].hostname, dm);
       
    }
    if(DEBUG_) printf("%d: connections ready for client\n",armci_me);

    /* ............ masters also set up connections for clients ......... */

    if(armci_me == armci_master){

       int clients = armci_nproc - armci_clus_info[armci_clus_me].nslave;

       /* master initializes nic connection for talking to clients */
       armci_init_nic(CLN_nic,0,clients);

       /* allocate and initialize connection structs */
       CLN_con=(armci_connect_t*)malloc(sizeof(armci_connect_t)*armci_nproc);
       if(!CLN_con)armci_die("cannot allocate SRV_con",armci_nproc);

       for(c=0; c< armci_nproc; c++)if(!SAMECLUSNODE(c)){
          char *ptr;
          discrim_t dm;
          int cluster  = armci_clus_id(c);
          armci_connect_t *con = CLN_con + c;

          con->loc = (void*)con->st_local;
          con->rem = (void*)con->st_remote;
          con->vi  = armci_create_vi(CLN_nic);

          dm = MAKE_DISCRIMINATOR(AR_base[armci_me], c);
          if(DEBUG_)printf("%d(s):discriminator(%d)=%lf\n",armci_me,c,dm);

          armci_make_netaddr(con->loc, armci_clus_info[armci_clus_me].hostname, dm);
          armci_make_netaddr(con->rem, armci_clus_info[cluster].hostname, dm);
   
       }
       if(DEBUG_) printf("%d: connections ready for server\n",armci_me);
    }

    if(DEBUG_) printf("%d: all connections ready \n",armci_me);
    /* cleanup we do not need that anymore */
    free(AR_base); 
    rc = VipNSShutdown(SRV_nic->handle);
    armci_check_status(DEBUG0, rc,"shut down name service");
}


static void armci_init_vbuf(VIP_DESCRIPTOR *d, char* buf, int len, VIP_MEM_HANDLE mhandle)
{
    d->CS.Control  = VIP_CONTROL_OP_SENDRECV;
    d->CS.Length   = (unsigned)len;
    d->CS.SegCount = 1;
    d->CS.Reserved = 0;
    d->CS.Status   = 0; 
    d->DS[0].Local.Data.Address = buf;
    d->DS[0].Local.Handle = mhandle;
    d->DS[0].Local.Length = (unsigned)len;
}


static void armci_call_data_server()
{
VIP_RETURN rc;
VIP_VI_HANDLE vi;
VIP_BOOLEAN rcv;
VIP_DESCRIPTOR *pdscr;
vbuf_t *vbuf;
int c;

     for(;;){
        
       /* wait for a request message to arrive */

       rc = VipCQWait(CLN_nic->rcq,VIP_INFINITE, &vi, &rcv); 
       armci_check_status(DEBUG0, rc,"server out of CQ wait");
       if(rcv==VIP_FALSE)armci_die("server got null handle for vbuf",0);

       /* dequeue the completed descriptor */
       rc = VipRecvDone(vi, &pdscr); 
       if(!pdscr) armci_die("server got null dscr ptr from VipRecvDone",0);
       armci_check_status(DEBUG0, rc,"server out of VipRecvDone");

       vbuf = (vbuf_t*) pdscr;

       /* look at the request to see where it came from */
       armci_ack_proc= c= ((request_header_t*)vbuf->buf)->from; 

       if(DEBUG0){
         printf("%d(s): got REQUEST from %d\n",armci_me, c); fflush(stdout);
       }

       /* we should post it this vbuf again even though data it contains 
        * will be processedi in armci_data_server() later
        * this is safe since the corresponding client cannot send us 
        * new request before receiving response/ack
        */
       armci_init_vbuf(pdscr, vbuf->buf, VBUF_DLEN, serv_memhandle);
       rc = VipPostRecv((CLN_con+c)->vi,pdscr, serv_memhandle);
       armci_check_status(DEBUG0, rc,"server Repost recv vbuf");

       armci_data_server(vbuf);

       /* flow control: send ack for this request no response was sent */
       if(armci_ack_proc != NONE){
          armci_send_data_to_client(armci_ack_proc,serv_buf->buf,0);
       }
    }
}


void * armci_server_code(void *data)
{
int c, ib;
VIP_RETURN rc;
VIP_MEM_ATTRIBUTES mattr;
int mod, bytes;
char *tmp;
int clients = armci_nproc - armci_clus_info[armci_clus_me].nslave;

     if(DEBUG1){
        printf("in server after fork %d (%d)\n",armci_me,getpid());
        fflush(stdout);
     }

     /* allocate memory for the recv buffers-must be alligned on 64byte bnd */
     bytes = clients*sizeof(vbuf_t) + sizeof(vbuf_long_t);
     tmp = malloc(bytes + SIXTYFOUR);
     if(!tmp) armci_die("failed to malloc recv vbufs",bytes);

     /* setup buffer pointers */
     mod = ((ssize_t)tmp)%SIXTYFOUR;
     serv_buf_arr = (vbuf_t*)(tmp+SIXTYFOUR-mod);
     serv_buf = (vbuf_long_t*)(serv_buf_arr+clients); /* buffer for response */
     MessageRcvBuffer = serv_buf->buf;

     /* setup memory attributes for the region */
     mattr.Ptag = CLN_nic->ptag;
     mattr.EnableRdmaWrite = VIP_FALSE;
     mattr.EnableRdmaRead  = VIP_FALSE;
     
     /* lock it */
     rc = VipRegisterMem(CLN_nic->handle,serv_buf_arr,bytes,
                         &mattr,&serv_memhandle);
     armci_check_status(DEBUG0, rc,"server register recv vbuf");
     if(!serv_memhandle)armci_die("server got null handle for vbuf",0);
     /* setup descriptors and post nonblocking receives */
     for(c = ib= 0; c < armci_nproc; c++) if(!SAMECLUSNODE(c)){
        vbuf_t *vbuf = serv_buf_arr+ib;
        armci_init_vbuf(&vbuf->dscr, vbuf->buf, VBUF_DLEN, serv_memhandle);
        rc = VipPostRecv((CLN_con+c)->vi,&vbuf->dscr, serv_memhandle);
        armci_check_status(DEBUG_, rc,"server post recv vbuf");
        ib++;
     }

     /* establish connections with compute processes/clients */
     for(c = 0; c < armci_nproc; c++)if(!SAMECLUSNODE(c)){
         VIP_VI_ATTRIBUTES rattrs;
         VIP_CONN_HANDLE con_hndl;
         armci_connect_t *con = CLN_con + c;
         rc = VipConnectWait(CLN_nic->handle,con->loc,VIP_INFINITE,con->rem,
                             &rattrs,&con_hndl);   
         armci_check_status(DEBUG0, rc,"server connect wait");
         
         rc = VipConnectAccept(con_hndl, con->vi);
         armci_check_status(DEBUG1, rc,"server connect wait");
     }

     if(DEBUG1){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);
     }

     armci_call_data_server();
     return(NULL);
}



void armci_client_code()
{
VIP_MEM_ATTRIBUTES mattr;
VIP_RETURN rc;
int s, mod, bytes;
char *tmp;

   if(DEBUG1){
       printf("in client after fork %d(%d)\n",armci_me,getpid());
       fflush(stdout);
   }

   /* allocate memory for the msg buffers-must be alligned on 64byte bnd */
   bytes = sizeof(vbuf_long_t);
   tmp = malloc(bytes + SIXTYFOUR);
   if(!tmp) armci_die("failed to malloc recv vbufs",bytes);

   /* setup buffer pointers */
   mod = ((ssize_t)tmp)%SIXTYFOUR;
   client_buf = (vbuf_long_t*)(tmp+SIXTYFOUR-mod);
   MessageSndBuffer = client_buf->buf;

   /* setup memory attributes for the region */
   mattr.Ptag = SRV_nic->ptag;
   mattr.EnableRdmaWrite = VIP_FALSE;
   mattr.EnableRdmaRead  = VIP_FALSE;
     
   /* lock it */
   rc = VipRegisterMem(SRV_nic->handle,client_buf,bytes,
                       &mattr,&client_memhandle);
   armci_check_status(DEBUG0, rc,"client register snd vbuf");
   if(!client_memhandle)armci_die("client got null handle for vbuf",0);


   /* connect to data server on each cluster node*/
   for(s=0; s< armci_nclus; s++)if(armci_clus_me != s){
      armci_connect_t *con = SRV_con + s;
      VIP_VI_ATTRIBUTES rattrs;

again:
      rc = VipConnectRequest(con->vi,con->loc, con->rem, VIP_INFINITE, &rattrs);
      if (rc == VIP_NO_MATCH) {
            usleep(10000);
            goto again;
      }
      armci_check_status(DEBUG1, rc,"client connect request");
   }

   armci_msg_barrier();

   if(DEBUG1){
      printf("%d client connected to all %d servers\n",armci_me, armci_nclus-1);
      fflush(stdout);
   }

   sleep(1);
}



void armci_start_server()
{
   /* create via connections accross the cluster */
   armci_create_connections();

   if(armci_me == armci_master){

      armci_create_server_thread( armci_server_code );
   }
   
   armci_client_code();

}


/******************* this code implements armci data transfers ***************/

static void armci_client_post_buf(int srv)
{
VIP_RETURN rc;
char *dataptr = GET_DATA_PTR(client_buf->buf); 
     
     armci_init_vbuf(&client_buf->rcv_dscr,dataptr,MSG_BUFLEN,client_memhandle);

     rc = VipPostRecv((SRV_con+srv)->vi,&client_buf->rcv_dscr,client_memhandle);
     armci_check_status(DEBUG0, rc,"client prepost vbuf");
     armci_long_buf_taken_srv = srv;
     armci_long_buf_free = 0;
}


void armci_via_wait_ack()
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;

     if(armci_long_buf_free) armci_die("armci_via_wait_ack: nothing posted",0);

     armci_long_buf_free =1; /* mark up the buffer as free */

     rc = VipRecvWait((SRV_con+armci_long_buf_taken_srv)->vi, VIP_INFINITE, &pdscr);
     armci_check_status(DEBUG0, rc,"client wait_ack out of recv wait");

     if(DEBUG0){
       printf("%d client got ack for req\n",armci_me);fflush(stdout);
     }
}


/*\ client sends request to server
\*/
int armci_send_req_msg(int proc, void *buf, int bytes)
{
VIP_RETURN rc;
int cluster  = armci_clus_id(proc);
VIP_DESCRIPTOR *cmpl_dscr;

    armci_client_post_buf(cluster); /* ack/response */

    armci_init_vbuf(&client_buf->snd_dscr, client_buf->buf, bytes, client_memhandle);
    rc = VipPostSend((SRV_con+cluster)->vi, &client_buf->snd_dscr, client_memhandle);
    armci_check_status(DEBUG0, rc,"client sent data to server");
    
    /********** should be moved to code that gets the buffer  **********/
    rc = VipSendWait((SRV_con+cluster)->vi, VIP_INFINITE, &cmpl_dscr);
    armci_check_status(DEBUG0, rc,"client wait for send to complete");
    if(cmpl_dscr !=&client_buf->snd_dscr)
       armci_die("different descriptor completed",0);

    if(DEBUG0){ printf("%d:client sent %dbytes to server\n",armci_me,bytes);
                fflush(stdout);
    }

    return 0;
}


void armci_send_data_to_client(int proc, void *buf, int bytes)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *cmpl_dscr;

    armci_init_vbuf(&serv_buf->snd_dscr, buf, bytes, serv_memhandle);
    rc = VipPostSend((CLN_con+proc)->vi, &serv_buf->snd_dscr, serv_memhandle);
    if(bytes)
       armci_check_status(DEBUG0, rc,"server sent data to client");
    else
       armci_check_status(DEBUG0, rc,"server sent ack to client");

    /********** should be moved to where we get buffer  **********/
    rc = VipSendWait((CLN_con+proc)->vi, VIP_INFINITE, &cmpl_dscr);
    armci_check_status(DEBUG0, rc,"server wait for send to complete");
    if(cmpl_dscr !=&serv_buf->snd_dscr)
       armci_die("-different descriptor completed",0);

    if(DEBUG0){ printf("%d:SERVER sent %dbytes to %d\n",armci_me,bytes,proc);
                fflush(stdout);
    }
}



/*\ server sends data to client in response to request
\*/
void armci_WriteToDirect(int proc, request_header_t* msginfo, void *buf)
{
     armci_send_data_to_client(proc, buf, (int)msginfo->datalen); 
     armci_ack_proc=NONE;
}


char *armci_ReadFromDirect(request_header_t *msginfo, int len)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;
int cluster = armci_clus_id(msginfo->to);
char *dataptr = GET_DATA_PTR(client_buf->buf); 

    rc = VipRecvWait((SRV_con+ cluster)->vi, VIP_INFINITE, &pdscr);
    armci_check_status(DEBUG0, rc,"client getting data from server");
    armci_long_buf_free =1;

    return dataptr;
}



void armci_flow_ack(int proc)
{
     armci_ack_proc=proc;
}


/*********** this code was adopted from the Giganet SDK examples *************/

/*
   function:    armci_err_callback
   arguments:   context - INVALID_HANDLE or an event to set
   d - async error descriptor
   description: set an event and then call generic callback printing
   function.
 */

static char *armci_code_tab[] =
{
    "Error posting descriptor", "Connection lost",
    "Receive on empty queue", "VI over-run",
    "RDMA write protection error", "RDMA Write data error",
    "RDMA write abort", "*invalid* - RDMA read",
    "Protection error on completion", "RDMA transport error",
    "Catastrophic error"
};
static void armci_ErrorCallbackFunction(void *ctx, VIP_ERROR_DESCRIPTOR * d)
{
    char buf[256], *p = buf;

    switch (d->ResourceCode) {
    case VIP_RESOURCE_NIC:
        sprintf(p, "callback on NIC handle %p", d->NicHandle);
        break;
    case VIP_RESOURCE_VI:
        sprintf(p, "callback on VI handle %p", d->ViHandle);
        break;
    case VIP_RESOURCE_CQ:
        sprintf(p, "callback on CQ handle %p", d->CQHandle);
        break;
    case VIP_RESOURCE_DESCRIPTOR:
        sprintf(p, "callback on descriptor %p", d->DescriptorPtr);
        break;
    }
    p += strlen(p);
    sprintf(p, ": %s", armci_code_tab[d->ErrorCode]);

    armci_die(buf,0);
}


void armci_err_callback(void *context, VIP_ERROR_DESCRIPTOR * d)
{
    struct cb_args *args = context;

    if (args != 0) {
        pthread_mutex_lock(args->mutex);
        pthread_cond_signal(args->cond);
        pthread_mutex_unlock(args->mutex);
    }
    armci_ErrorCallbackFunction(0, d);
}

