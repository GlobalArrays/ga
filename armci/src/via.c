/* $Id: via.c,v 1.22 2002-01-08 21:56:50 vinod Exp $ */
#include <stdio.h>
#include <strings.h>
#include <assert.h>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "armcip.h" 
#include "copy.h" 
#include "via.h" 

#define VIPL095  
#include <vipl.h>

#define DEBUG_ 0
#define DEBUG0 0
#define DEBUG1 0
#define DEBUG2 0
#define PAUSE_ON_ERROR__ 
#ifndef VIP_ERROR_NOT_SUPPORTED
#   define VIP_ERROR_NOT_SUPPORTED -33333
#endif

/* Giganet/Emulex cLAN is the default */
#ifndef VIADEV_NAME
#   define VIADEV_NAME "/dev/clanvi0"
#endif

/* SHORTNAME is use to bridge the hostname as returned by gethostbyname() and
   used by VipNSGetHostByName - on Giganet at least it must be w/o domainname */
#define SHORTNAME

#define ADDR_LEN 6
#define FOURTY 40
#define NONE -1
#define GET_DATA_PTR(buf) (sizeof(request_header_t) + (char*)buf)
#define CLIENT_STAMP 101
#define SERV_STAMP 99 
static char * client_tail;
static char * serv_tail;

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

int _s=-1, _c=-1;
int armci_via_server_ready=0;
int armci_via_client_ready=0;

struct cb_args {
    pthread_cond_t *cond;
    pthread_mutex_t *mutex;
};


typedef struct {
   VIP_NET_ADDRESS *rem;
   VIP_NET_ADDRESS *loc;
   char st_remote[FOURTY];
   char st_local[FOURTY];
   VIP_VI_HANDLE vi;
} armci_connect_t;

typedef struct {
   VIP_MEM_HANDLE *prem_handle; /*address on rem server to store memory handle*/
   VIP_MEM_HANDLE handle;
}ack_t;

typedef struct {
   char st_host[FOURTY];
}armci_hostaddr_t;

typedef struct {
  VIP_DESCRIPTOR dscr;
  char buf[VBUF_DLEN];
}vbuf_t;

typedef struct {
  VIP_DESCRIPTOR snd_dscr;
  VIP_DESCRIPTOR rcv_dscr;
  char buf[VBUF_DLEN];
}vbuf_ext_t;


typedef struct {
  char *buf;
  unsigned char srv;
  char snd;
  char rcv;
}reqbuf_t;

typedef struct {
  VIP_DESCRIPTOR snd_dscr;
  VIP_DESCRIPTOR rcv_dscr;
  char buf[MAX_BUFLEN];
}vbuf_long_t;

#define MAX_BUFS 4
reqbuf_t client_buf_pool[MAX_BUFS];

static vbuf_t *serv_buf_arr, *spare_serv_buf;
static vbuf_long_t *client_buf, *serv_buf;
static VIP_MEM_HANDLE serv_memhandle, client_memhandle;
static armci_connect_t *SRV_con;
static armci_connect_t *CLN_con;
static VIP_MEM_HANDLE *CLN_handle;
static ack_t *SRV_ack;
static VIP_MEM_HANDLE *pinned_handle;
#define MAX_DESCR 16
typedef struct { 
        int avail; VIP_VI_HANDLE vi; VIP_DESCRIPTOR *descr; 
} descr_pool_t;

static descr_pool_t serv_descr_pool = {MAX_DESCR, (VIP_DESCRIPTOR *)0}; 
static descr_pool_t client_descr_pool = {MAX_DESCR, (VIP_DESCRIPTOR *)0}; 

char *MessageSndBuffer;
char *MessageRcvBuffer;
int armci_long_buf_free=1;
int armci_long_buf_taken_srv=NONE;
static int armci_server_terminating=0;
static int armci_ack_proc=NONE;

extern void armci_send_data_to_client(int proc, void *buf, int bytes);
extern void armci_via_wait_ack();
static void armci_serv_clear_sends();
extern void ensure_one_outstanding_op_per_node(char *);
#define SERVER_SEND_ACK(p) armci_send_data_to_client((p),serv_buf->buf,0)
#define CLIENT_WAIT_ACK armci_die("CLIENT_WAIT_ACK: not supported",0)

#define BUF_TO_RDESCR(buf) ((VIP_DESCRIPTOR *)(((armci_via_field_t *)((char *)(buf) - sizeof(armci_via_field_t)))->r))
#define BUF_TO_SDESCR(buf) ((VIP_DESCRIPTOR *)(((armci_via_field_t *)((char *)(buf) - sizeof(armci_via_field_t)))->s)) 


void armci_wait_for_server()
{
  armci_server_terminating=1;
}

void armci_transport_cleanup()
{
  if(SERVER_CONTEXT) if(*serv_tail != SERV_STAMP){
        printf("%d: server_stamp %d %d\n",armci_me,*serv_tail, SERV_STAMP);
        armci_die("ARMCI Internal Error: end-of-buffer overwritten",0); 
  }
  if(!SERVER_CONTEXT) if(*client_tail != CLIENT_STAMP){
        printf("%d: client_stamp %d %d\n",armci_me,*client_tail, CLIENT_STAMP);
        //armci_die("ARMCI Internal Error: end-of-buffer overwritten",0); 
  }
}


/*\ server receives request
\*/
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    vbuf_t *vbuf = (vbuf_t*)mesg;
    request_header_t *msginfo = (request_header_t *)vbuf->buf;
    *(void **)phdr = msginfo;

    if(DEBUG0) {
        printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from); fflush(stdout);
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
          *(void**)pdata = MessageRcvBuffer;
    }  
}


static char *via_err_msg(VIP_RETURN rc)
{
  switch (rc) {
     case VIP_SUCCESS: return ("VIP_SUCCESS");break;
     case VIP_NOT_DONE: return ("VIP_NOT_DONE");break;
     case VIP_INVALID_PARAMETER: return ("VIP_INVALID_PARAMETER");break;
     case VIP_ERROR_RESOURCE: return ("VIP_ERROR_RESOURCE");break;
     case VIP_TIMEOUT: return ("VIP_TIMEOUT");break;
     case VIP_REJECT: return ("VIP_REJECT");break;
     case VIP_INVALID_RELIABILITY_LEVEL: return ("VIP_INVALID_RELIABILITY_LEVEL");break;
     case VIP_INVALID_MTU: return ("VIP_INVALID_MTU");break;
     case VIP_INVALID_QOS: return ("VIP_INVALID_QOS");break;
     case VIP_INVALID_PTAG: return ("VIP_INVALID_PTAG");break;
     case VIP_INVALID_RDMAREAD: return ("VIP_INVALID_RDMAREAD");break;
     case VIP_DESCRIPTOR_ERROR: return ("VIP_DESCRIPTOR_ERROR");break;
     case VIP_INVALID_STATE: return ("VIP_INVALID_STATE");break;
     case VIP_ERROR_NAMESERVICE: return ("VIP_ERROR_NAMESERVICE");break;
     case VIP_NO_MATCH: return ("VIP_NO_MATCH");break;
     case VIP_NOT_REACHABLE: return ("VIP_NOT_REACHABLE");break;
     case VIP_ERROR_NOT_SUPPORTED: return ("VIP_ERROR_NOT_SUPPORTED");break;
     default: return ("");
  }
}
 


static void armci_check_status(int debug, VIP_RETURN rc, char *msg)
{
#define BLEN 100 

     if(rc != VIP_SUCCESS){

        char buf[BLEN];
        
        if(armci_server_terminating){
           /* server got interrupted when clients terminate connections */
           sleep(1);
           _exit(0);
        }
          
        fprintf(stderr,"%d in check FAILURE %s\n",armci_me,msg);
        assert(strlen(msg)<BLEN-20);
        sprintf(buf,"ARMCI(via):failure:%s:%s code %d %d ",via_err_msg(rc),msg,_s,_c);
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
    
#define SHORTNAME  
void armci_make_netaddr_(VIP_NET_ADDRESS *pnaddr, char* hostname, discrim_t dm)
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

void armci_make_netaddr(VIP_NET_ADDRESS *pnaddr, char* hostname)
{
VIP_RETURN rc;
#ifdef SHORTNAME
    char *found=strchr(hostname,'.'); 
    if(found)*found='\0';  /* truncate hostname to cut off domainname */
#endif

    rc = VipNSGetHostByName(SRV_nic->handle,hostname,pnaddr,0);
    armci_check_status(DEBUG0, rc,"get host name address");

#ifdef SHORTNAME
    if(found)*found='.'; /* restores the original full name */
#endif

    pnaddr->HostAddressLen = SRV_nic->attr.NicAddressLen;
    pnaddr->DiscriminatorLen = sizeof(discrim_t);
}


void armci_netaddr_dm(VIP_NET_ADDRESS *pnaddr, discrim_t dm)
{
char *p = (char*)(  pnaddr->HostAddress + SRV_nic->attr.NicAddressLen);
    *(discrim_t*)p = dm;
}

static VIP_DESCRIPTOR* armci_malloc_descriptors()
{
char *tmp;
int mod;
     tmp = malloc(sizeof(VIP_DESCRIPTOR)*MAX_DESCR + SIXTYFOUR);
     if(!tmp) armci_die("failed to malloc descriptors bufs",MAX_DESCR);
     mod = ((ssize_t)tmp)%SIXTYFOUR;
     return (VIP_DESCRIPTOR*)(tmp+SIXTYFOUR-mod);
}

/*\ allocate receive buffers for data server thread
\*/
void armci_server_alloc_bufs()
{
VIP_RETURN rc;
VIP_MEM_ATTRIBUTES mattr;
int mod, bytes, total, extra =sizeof(VIP_DESCRIPTOR)*MAX_DESCR+SIXTYFOUR;
int mhsize = armci_nproc*sizeof(VIP_MEM_HANDLE); /* ack */
char *tmp, *tmp0;
int clients = armci_nproc - armci_clus_info[armci_clus_me].nslave;

     /* allocate memory for the recv buffers-must be alligned on 64byte bnd */
     /* note we add extra one to repost it for the client we are received req */
     bytes = (clients+1)*sizeof(vbuf_t) + sizeof(vbuf_long_t) + extra+ mhsize;
     total = bytes + SIXTYFOUR;
     tmp0=tmp = malloc(total);
     if(!tmp) armci_die("failed to malloc server vbufs",total);

     /* stamp the last byte */
     serv_tail= tmp + bytes+SIXTYFOUR-1;
     *serv_tail=SERV_STAMP;

     /* allocate memory for client memory handle to support put response 
        in dynamic memory registration protocols */
     CLN_handle = (VIP_MEM_HANDLE*)tmp;
     memset(CLN_handle,0,mhsize); /* set it to zero */
     tmp += mhsize;

     /* setup descriptor memory */
     mod = ((ssize_t)tmp)%SIXTYFOUR;
     serv_descr_pool.descr= (VIP_DESCRIPTOR*)(tmp+SIXTYFOUR-mod);
     tmp += extra;

     /* setup buffer pointers */
     mod = ((ssize_t)tmp)%SIXTYFOUR;
     serv_buf_arr = (vbuf_t*)(tmp+SIXTYFOUR-mod);
     spare_serv_buf = serv_buf_arr+clients; /* spare buffer is at the end */
     serv_buf = (vbuf_long_t*)(serv_buf_arr+clients+1); /* buffer for response*/
     MessageRcvBuffer = serv_buf->buf;

     /* setup memory attributes for the region */
     mattr.Ptag = CLN_nic->ptag;
     mattr.EnableRdmaWrite = VIP_TRUE;
     mattr.EnableRdmaRead  = VIP_FALSE;

     /* lock it */
     rc = VipRegisterMem(CLN_nic->handle,tmp0, total, &mattr,&serv_memhandle);
     armci_check_status(DEBUG0, rc,"server register recv vbuf");

     if(!serv_memhandle)armci_die("server got null handle for vbuf",0);

     if(DEBUG1){
        printf("%d(s):registered mem %p %dbytes mhandle=%d mharr starts%p\n",
               armci_me, tmp0, total, serv_memhandle,CLN_handle);fflush(stdout);
     }
}

static void armci_print_attr(nic_t *nic)
{
      printf("Max Vi=%ld\n",nic->attr.MaxVI);
      printf("Max Register Bytes=%ld\n",nic->attr.MaxRegisterBytes);
      printf("Max Register Regions=%ld\n",nic->attr.MaxRegisterRegions);
      printf("Max Register Block Bytes=%ld\n",nic->attr.MaxRegisterBlockBytes);
      printf("Max Descriptors PerQueue=%ld\n",nic->attr.MaxDescriptorsPerQueue);
      printf("Max Segments Per Desc=%ld\n",nic->attr.MaxSegmentsPerDesc);
      printf("Max Transfer Size=%ld\n",nic->attr.MaxTransferSize);
      printf("Native MTU=%ld\n",nic->attr.NativeMTU);
      printf("Max Ptags=%ld\n",nic->attr.MaxPtags);
      printf("Max CQ=%ld\n",nic->attr.MaxCQ);
      printf("Max CQ Entries=%ld\n",nic->attr.MaxCQEntries);
      fflush(stdout);
}



static void armci_set_serv_mh()
{
int s, ratio = sizeof(ack_t)/sizeof(int);

    /* exchange address of ack/memhandle flag on servers */
    SRV_ack = (ack_t*)calloc(armci_nclus,sizeof(ack_t));
    if(!SRV_ack)armci_die("buffer alloc failed",ratio);

    /* first collect addrresses on all masters */
    if(armci_me == armci_master){
      SRV_ack[armci_clus_me].prem_handle=CLN_handle;
      SRV_ack[armci_clus_me].handle =serv_memhandle;
      armci_msg_gop_scope(SCOPE_MASTERS,SRV_ack,ratio*armci_nclus,"+",ARMCI_INT);
    }

    /* now master broadcasts the addresses within its node */
    armci_msg_bcast_scope(SCOPE_NODE,SRV_ack,armci_nclus*sizeof(ack_t),
                                                         armci_master);
    
    /* now save address corresponding to my id on each server */
    for(s=0; s< armci_nclus; s++){ 
        SRV_ack[s].prem_handle += armci_me;
        /*printf("%d: my addr on %d = %p\n",armci_me,s,SRV_ack[s].prem_handle);
        fflush(stdout); */
    }
}

 
/* initialize connection data structures - called by main thread */
void armci_init_connections()
{
VIP_RETURN rc;
int c,s, *AR_base;
armci_hostaddr_t *host_addr;
    
    /* get base for connection descriptor - we use process id */
    AR_base = (int*)malloc(armci_nproc * sizeof(int));
    if(!AR_base)armci_die("malloc failed for AR_base",0);
    bzero(AR_base,armci_nproc * sizeof(int));
    AR_base[armci_me]=(int)getpid();
    armci_msg_igop(AR_base,armci_nproc,"+"); /*exchange it globally */

    host_addr = (armci_hostaddr_t*)malloc(armci_nclus*sizeof(armci_hostaddr_t));
    if(!host_addr)armci_die("malloc failed for host_addr",0);
    bzero(host_addr,armci_nclus*sizeof(armci_hostaddr_t));

    /* initialize nic connection for talking to servers */
    armci_init_nic(SRV_nic,0,0);
    if(DEBUG2 && (armci_me==0))armci_print_attr(SRV_nic);

    /* for peer network address we need name service */
    rc = VipNSInit(SRV_nic->handle, NULL);
    armci_check_status(DEBUG0, rc,"init name service");

    /* allocate and initialize connection structs */
    SRV_con=(armci_connect_t*)malloc(sizeof(armci_connect_t)*armci_nclus);
    if(!SRV_con)armci_die("cannot allocate SRV_con",armci_nclus);

    for(s=0; s< armci_nclus; s++){
          VIP_NET_ADDRESS *addr = (VIP_NET_ADDRESS *)(host_addr+s)->st_host;
          armci_make_netaddr(addr, armci_clus_info[s].hostname);
    }

    for(s=0; s< armci_nclus; s++)if(armci_clus_me != s){
       discrim_t dm;
       int cluster = s;
       int master  = armci_clus_info[cluster].master;
       armci_connect_t *con = SRV_con + s;

       dm = MAKE_DISCRIMINATOR(AR_base[master], armci_me);
       if(DEBUG_)printf("%d:discriminator(%d)=%f\n",armci_me,master,dm);

       con->loc = (void*)con->st_local;
       con->rem = (void*)con->st_remote;
       con->vi  = armci_create_vi(SRV_nic);

#if 0
       armci_make_netaddr(con->loc, armci_clus_info[armci_clus_me].hostname);
       armci_make_netaddr(con->rem, armci_clus_info[cluster].hostname);
#else
       armci_copy((host_addr+armci_clus_me)->st_host, con->st_local, FOURTY);
       armci_copy((host_addr+cluster)->st_host, con->st_remote, FOURTY);
#endif
       armci_netaddr_dm(con->loc,dm);
       armci_netaddr_dm(con->rem,dm);
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
          discrim_t dm;
          int cluster  = armci_clus_id(c);
          armci_connect_t *con = CLN_con + c;

          con->loc = (void*)con->st_local;
          con->rem = (void*)con->st_remote;
          con->vi  = armci_create_vi(CLN_nic);

          dm = MAKE_DISCRIMINATOR(AR_base[armci_me], c);
          if(DEBUG_)printf("%d(s):discriminator(%d)=%f\n",armci_me,c,dm);
#if 0
          armci_make_netaddr(con->loc, armci_clus_info[armci_clus_me].hostname);
          armci_make_netaddr(con->rem, armci_clus_info[cluster].hostname);
#else
          armci_copy((host_addr+armci_clus_me)->st_host, con->st_local, FOURTY);
          armci_copy((host_addr+cluster)->st_host, con->st_remote, FOURTY);
#endif
          armci_netaddr_dm(con->loc,dm);
          armci_netaddr_dm(con->rem,dm);
   
       }

       if(DEBUG_) printf("%d: connections ready for server\n",armci_me);

       armci_server_alloc_bufs(); /* get receive buffers for server thread */

    }

    /* cleanup we do not need that anymore */
    free(host_addr); 
    free(AR_base); 
    rc = VipNSShutdown(SRV_nic->handle);
    armci_check_status(DEBUG0, rc,"shut down name service");

    armci_set_serv_mh();

    if(DEBUG_) printf("%d: all connections ready \n",armci_me);
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

#define ARMCI_BUF_FROM_DESCR(d) d->DS[0].Local.Data.Address

void armci_call_data_server()
{
VIP_RETURN rc;
VIP_VI_HANDLE vi;
VIP_BOOLEAN rcv;
VIP_DESCRIPTOR *pdscr;
vbuf_t *vbuf;
int c, need_ack;
request_header_t *msginfo;

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
       msginfo = (request_header_t*)vbuf->buf;
       armci_ack_proc= c= msginfo->from; 

       if(DEBUG1){
         printf("%d(s):got REQUEST %d from %d\n",armci_me,msginfo->operation,c);         fflush(stdout);
       }

       /* we replace (repost) the current buffer with the spare one 
        * so that the client has one available ASAP for next request */ 
       armci_init_vbuf(&spare_serv_buf->dscr,spare_serv_buf->buf, VBUF_DLEN, 
                       serv_memhandle);
       rc = VipPostRecv((CLN_con+c)->vi,&spare_serv_buf->dscr,serv_memhandle);
       armci_check_status(DEBUG0, rc,"server repost recv vbuf");
       spare_serv_buf = vbuf; /* save the current buf as spare */
       
       if((msginfo->operation == PUT) || ACC(msginfo->operation)){
           /* for operations that do not send data back we can send ACK now */
           SERVER_SEND_ACK(armci_ack_proc);
           need_ack=0;
       }else need_ack=1;

       armci_data_server(vbuf);

       if ((msginfo->operation==GET)&&(PIPE_MIN_BUFSIZE<msginfo->datalen))
          armci_serv_clear_sends(); /* for now we complete all pending sends */
      
       /* flow control: send ack for this request since no response was sent */
       if(need_ack &&(armci_ack_proc != NONE)) SERVER_SEND_ACK(armci_ack_proc);

       if(DEBUG0){
         printf("%d(s): REQUEST from %d processed\n",armci_me,c);fflush(stdout);
       }
    }
}


#ifdef PEER_CONNECTION
static void via_connect_peer(armci_connect_t *con_arr, int num, int serv)
{
int n;
armci_connect_t *con;
VIP_RETURN rc;

     /* post connection requests */
     for(n = 0; n < num; n++){

         if(serv){ if(SAMECLUSNODE(n))    continue;} 
         else    { if(n == armci_clus_me) continue;}

         con = con_arr + n;

         rc = VipConnectPeerRequest(con->vi,con->loc, con->rem, VIP_INFINITE);
         armci_check_status(DEBUG0, rc,"peer connect request");
     }

     /* wait for all connections to be established */
     for(n = 0; n < num; n++){
         VIP_VI_ATTRIBUTES rattrs;

         if(serv){ if(SAMECLUSNODE(n))    continue;} 
         else    { if(n == armci_clus_me) continue;}

         con = con_arr + n;
         if(serv)_c=n;
         else _s=n;

         rc = VipConnectPeerWait(con->vi, &rattrs);
         armci_check_status(DEBUG1, rc," connect wait");
     }
}
#else

static void via_connect_server()
{
int c,start,i;
VIP_RETURN rc;

     /* start from master task on next node */
     c = (armci_clus_me+1)%armci_nclus;
     start = armci_clus_info[c].master;

     for(i = 0; i < (armci_nproc-armci_clus_info[armci_clus_me].nslave); i++){
         VIP_VI_ATTRIBUTES rattrs;
         VIP_CONN_HANDLE con_hndl;
         armci_connect_t *con;

         c = (start+i)%armci_nproc; /* wrap up */
         con = CLN_con + c;
         rc = VipConnectWait(CLN_nic->handle,con->loc,VIP_INFINITE,con->rem,
                             &rattrs,&con_hndl);
         armci_check_status(DEBUG0, rc,"server connect wait");

         rc = VipConnectAccept(con_hndl, con->vi);
         armci_check_status(DEBUG1, rc,"server connect wait");
     }
}

static void via_connect_client()
{
int s,i,start;
VIP_RETURN rc;

   /* start from from server on my_node -1 */
   start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;

   for(i=0; i< armci_nclus-1; i++){
      armci_connect_t *con;
      VIP_VI_ATTRIBUTES rattrs;
  
      s = (start -i)%armci_nclus;
      if(s<0) s+=armci_nclus;
      con = SRV_con + s;

again:
      rc = VipConnectRequest(con->vi,con->loc, con->rem, VIP_INFINITE, &rattrs);
      if (rc == VIP_NO_MATCH) {
            usleep(10);
            goto again;
      }
      armci_check_status(DEBUG1, rc,"client connect request");
   }
}
#endif



void armci_server_initial_connection()
{
int c, ib;
VIP_RETURN rc;

     if(DEBUG1){ printf("in server after fork %d (%d)\n",armci_me,getpid());
        fflush(stdout);
     }
 
     /* setup descriptors and post nonblocking receives */
     for(c = ib= 0; c < armci_nproc; c++) if(!SAMECLUSNODE(c)){
        vbuf_t *vbuf = serv_buf_arr+ib;
        armci_init_vbuf(&vbuf->dscr, vbuf->buf, VBUF_DLEN, serv_memhandle);
        rc = VipPostRecv((CLN_con+c)->vi,&vbuf->dscr, serv_memhandle);
        armci_check_status(DEBUG_, rc,"server post recv vbuf");
        ib++;
     }

     armci_via_server_ready=1;

     /* establish connections with compute processes/clients */
#ifdef PEER_CONNECTION
     via_connect_peer(CLN_con, armci_nproc, 1);
#else
     via_connect_server();
#endif

     if(DEBUG1){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);
     }
}


/*\ allocates pinned memory for the client
\*/
char * armci_via_client_mem_alloc(int size)
{
VIP_MEM_ATTRIBUTES mattr;
VIP_RETURN rc;
int mod,  total, extra = MAX_DESCR*sizeof(VIP_DESCRIPTOR)+SIXTYFOUR;
char *tmp,*tmp0;    

   /*we use the size passed by the armci_init_bufs routine instead of bytes
   */
   total = size + extra+ SIXTYFOUR;
   tmp0  = tmp = malloc(total);
   if(!tmp) armci_die("failed to malloc client bufs",total);
   /* stamp the last byte */
   client_tail= tmp + extra+ size +SIXTYFOUR-1;
   *client_tail=CLIENT_STAMP;
 
   /* we also have a place to store memhandle for zero-copy get */
   pinned_handle =(VIP_MEM_HANDLE *) (tmp + extra+ size +SIXTYFOUR-16);
 
   /* setup descriptor memory */
   mod = ((ssize_t)tmp)%SIXTYFOUR;
   client_descr_pool.descr= (VIP_DESCRIPTOR*)(tmp+SIXTYFOUR-mod);
   tmp += extra; 
   mattr.Ptag = SRV_nic->ptag;
   mattr.EnableRdmaWrite = VIP_FALSE;
   mattr.EnableRdmaRead  = VIP_FALSE;
   
   /* lock allocated memory */
   rc = VipRegisterMem(SRV_nic->handle, tmp0, total, &mattr,&client_memhandle);
   armci_check_status(DEBUG0, rc,"client register snd vbuf");
   if(!client_memhandle)armci_die("client got null handle for vbuf",0);

   if(DEBUG2){
        printf("%d: registered client memory %p %dsize tmp=%p memhandle=%d\n",
                armci_me,tmp0, total, tmp,client_memhandle);
        fflush(stdout);
   
   }

   return(tmp);
}

void armci_client_connect_to_servers()
{
   /* initialize buffer managment module */
   _armci_buf_init();

   /* connect to data server on each cluster node*/
#ifdef PEER_CONNECTION
   via_connect_peer(SRV_con, armci_nclus, 0);
#else
   via_connect_client();
#endif

}

void armci_via_complete_buf(armci_via_field_t *field,int snd,int rcv,int to){
VIP_RETURN rc;
VIP_DESCRIPTOR *cmpl_dscr,* snd_dscr,* rcv_dscr;
    if(snd){
        snd_dscr=(VIP_DESCRIPTOR *)(field->s);
	do{rc = VipSendDone((SRV_con+armci_clus_id(to))->vi, &cmpl_dscr);}
        while(rc==VIP_NOT_DONE);
       armci_check_status(DEBUG0, rc,"complete_buf: wait for send to complete");
       if(snd_dscr != cmpl_dscr)
       armci_die("armci_via_complete_buf: wrong dscr completed",0);
    }
    if(rcv){
	rcv_dscr=(VIP_DESCRIPTOR *)(field->r);
        do{rc = VipRecvDone((SRV_con+armci_clus_id(to))->vi, &cmpl_dscr);}
        while(rc==VIP_NOT_DONE);
        if(rcv_dscr != cmpl_dscr){
           printf("%d(c): DIFFERENT recv DESCRIPTOR %p %p buf= \n",
                  armci_me,cmpl_dscr ,&rcv_dscr);
           fflush(stdout);
           armci_die("armci_via_complete_buf: wrong rcv dscr completed",0);
        } 
    }
}    
static void armci_complete_buf_index(int b)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *cmpl_dscr;
vbuf_ext_t *evbuf = (vbuf_ext_t*)client_buf_pool[b].buf;

    if(DEBUG1){
       printf("%d(c): completing %d snd=%d rcv=%d  buf=%p\n",armci_me,b,
              client_buf_pool[b].snd,client_buf_pool[b].rcv,evbuf->buf);
       fflush(stdout);
    }

    if(client_buf_pool[b].snd){
#if 0
       rc = VipSendWait((SRV_con+client_buf_pool[b].srv)->vi, VIP_INFINITE,
                         &cmpl_dscr);
#else
       do{rc = VipSendDone((SRV_con+client_buf_pool[b].srv)->vi, &cmpl_dscr);}
       while(rc==VIP_NOT_DONE);
#endif
       armci_check_status(DEBUG0, rc,"complete_buf: wait for send to complete");
       if(&evbuf->snd_dscr != cmpl_dscr)
       armci_die("armci_complete_buf: wrong dscr completed",b);
       client_buf_pool[b].snd = 0;
    }
    if(client_buf_pool[b].rcv){
#if 0
       rc = VipRecvWait((SRV_con+client_buf_pool[b].srv)->vi, VIP_INFINITE,
                            &cmpl_dscr);
#else
       do{rc = VipRecvDone((SRV_con+client_buf_pool[b].srv)->vi, &cmpl_dscr);}
       while(rc==VIP_NOT_DONE);
#endif
       armci_check_status(DEBUG0, rc,"complete_buf: wait receive to complete");
       if(&evbuf->rcv_dscr != cmpl_dscr){
           printf("%d(c): DIFFERENT recv DESCRIPTOR %p %p buf=%p \n",
                  armci_me,cmpl_dscr ,&evbuf->rcv_dscr, &client_buf->rcv_dscr);
        
           fflush(stdout);
           armci_die("armci_complete_buf: wrong rcv dscr completed",b);
       }
       client_buf_pool[b].rcv = 0;
    }
}


char* armci_getbuf(int size)
{
static int avail=0;

      if(size >= VBUF_DLEN){
         int i;
         /* clear all the buffers */
         for(i=0;i<MAX_BUFS;i++)if(client_buf_pool[i].snd)
             armci_complete_buf_index(i);
         avail =0;
      }else{
         avail +=1;
         avail %= MAX_BUFS;
         if(client_buf_pool[avail].snd) armci_complete_buf_index(avail);
      }
      if(DEBUG1){
         printf("%d(c) GETBUF size=%d %d %p\n",armci_me,size,avail,
               client_buf_pool[avail].buf); fflush(stdout);
      }
#if 0
      if(size > 15000){printf("%d: size=%d %d %p %p\n",armci_me,
                       size,avail,client_buf_pool[avail].buf,
                       ((vbuf_ext_t*)client_buf_pool[avail].buf)->buf);
                      fflush(stdout);
      } 
#endif
      return(((vbuf_ext_t*)client_buf_pool[avail].buf)->buf);
     
}

#define BUF_TO_EVBUF(buf) (vbuf_ext_t*)(((char*)buf) - 2*sizeof(VIP_DESCRIPTOR))

static int armci_buf2index(void *buf)
{
char *evbuf = ((char*)buf) - 2*sizeof(VIP_DESCRIPTOR);
int i, found=0;
    for(i=0; i<MAX_BUFS; i++)if(client_buf_pool[i].buf == evbuf){found=1;break;}
    if(!found){
        printf("%d(c) not found %p %p\n",armci_me,client_buf_pool[0].buf,evbuf);
        armci_die("armci_buf2index: not found",0);
    }
    return(i);
}

void armci_relbuf(void *buf)
{
int index = armci_buf2index(buf);
    armci_complete_buf_index(index);
}


/******************* this code implements armci data transfers ***************/

static void armci_client_post_buf(int srv,void* buf)
{
VIP_RETURN rc;
char *dataptr = GET_DATA_PTR(buf); 
VIP_DESCRIPTOR *rcv_dscr;
     rcv_dscr = BUF_TO_RDESCR(buf);     
     armci_init_vbuf(rcv_dscr,dataptr,MSG_BUFLEN,client_memhandle);
     rc = VipPostRecv((SRV_con+srv)->vi,rcv_dscr,client_memhandle);
     armci_check_status(DEBUG0, rc,"client prepost vbuf");
#if 0
     armci_long_buf_taken_srv = srv;
     armci_long_buf_free = 0;
#endif
}


static void  armci_dequeue_send_descr(VIP_VI_HANDLE vi)
{

VIP_DESCRIPTOR *cmpl_dscr;
VIP_RETURN   rc;
#if 0
    rc = VipSendWait(vi, VIP_INFINITE, &cmpl_dscr);
#endif
    do{rc = VipSendDone(vi, &cmpl_dscr);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"WAIT for send to complete");
}

/* ........ */


/*\ wait until all outstanding sends completed 
\*/
static void armci_clear_sends(descr_pool_t *dp )
{
int i, outstanding;

  outstanding = MAX_DESCR-dp->avail;
  if( !outstanding) return;
  if( outstanding<0) armci_die("armci_serv_clear_sends: error",outstanding);

  if(DEBUG1){
     printf("%d CLEARING DESCRIPTORS %d %p\n",armci_me,outstanding,dp);fflush(stdout);
  }

  for(i=0; i<outstanding; i++){
    VIP_DESCRIPTOR *pdscr;
    VIP_RETURN rc;

    if(DEBUG0){ printf("%d CLEARING %d %p\n",armci_me,i, dp->descr+i);                          fflush(stdout);
    }

#if 0
    rc = VipSendWait(dp->vi, VIP_INFINITE, &pdscr);
    do{ rc = VipSendDone(dp->vi, &pdscr); }while(rc==VIP_NOT_DONE);
#endif

    rc = VipSendDone(dp->vi, &pdscr); 
    if(rc==VIP_NOT_DONE) rc = VipSendWait(dp->vi, VIP_INFINITE, &pdscr); 
    armci_check_status(DEBUG0, rc,"wait & clear send to complete");

    /* make sure the right descriptor in send work queue completed */
    if(pdscr != (dp->descr+i))
        armci_die("armci_clear_sends:different descr completed",i); 

    dp->avail++;
  } 
}

static void armci_serv_clear_sends()
{
  armci_clear_sends(&serv_descr_pool);
}

void armci_client_clear_sends()
{
  armci_clear_sends(&client_descr_pool); 
}

/*\ prepost buffers for receiving data from server (pipeline)
\*/
void armcill_pipe_post_bufs(void *ptr, int stride_arr[], int count[], 
                            int strides, void* argvoid)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdesc;
int bytes,i, extra;
buf_arg_t *arg = (buf_arg_t*)argvoid;
descr_pool_t *dp;
VIP_MEM_HANDLE *pmhandle;

     if(arg->op ==GET){
        int srv= armci_clus_id(arg->proc);
        dp = &client_descr_pool;
        dp->vi= (SRV_con+srv)->vi; 
        pmhandle = &client_memhandle; 
     }else{
        dp = &serv_descr_pool;
        dp->vi= (CLN_con+arg->proc)->vi;
        pmhandle = &serv_memhandle; 
     }

     if(!dp->avail)
         armci_die("armci_pipe_post_bufs: all descriptors used",MAX_DESCR);

     pdesc = dp->descr+MAX_DESCR-dp->avail;

     for(i=0, bytes=1; i<=strides; i++)bytes*=count[i];

     /* allign receive buffer on 64-byte boundary */
     extra = ALIGN64ADD(arg->buf);
     arg->buf+=extra;

     if(DEBUG2){
        printf("%d: posting pipe receive %d (%p,%p) %d vi=%p\n",
               armci_me, arg->proc,pdesc,arg->buf,bytes,dp->vi);
        fflush(stdout);
     }

     armci_init_vbuf(pdesc, arg->buf, bytes, *pmhandle);

     rc = VipPostRecv(dp->vi, pdesc, *pmhandle);
     armci_check_status(DEBUG0, rc,"pipe prepost vbuf");

     dp->avail--;
     arg->buf+=bytes;
     /*if(DEBUG1)if(arg->buf > client_tail)
               armci_die("overwriting end of buf",arg->buf-client_tail);*/
}


void armcill_pipe_extract_data(void *ptr, int stride_arr[], int count[], 
                               int strides, void* argvoid)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;
VIP_MEM_HANDLE *pmhandle;
void *buf;
buf_arg_t *arg = (buf_arg_t*)argvoid;
descr_pool_t *dp;

    if(arg->op ==GET){
        int srv= armci_clus_id(arg->proc);
        dp = &client_descr_pool;
        dp->vi= (SRV_con+srv)->vi;
        if(!arg->count) { 
                          armci_dequeue_send_descr(dp->vi);
        }
        pmhandle = &client_memhandle;
    }else{
        dp = &serv_descr_pool;
        dp->vi= (CLN_con+arg->proc)->vi;
        pmhandle = &serv_memhandle;
        if(!arg->count){
            SERVER_SEND_ACK(arg->proc); /* notify client we ready for data */
            armci_ack_proc=NONE; /* prevent sending another ACK by server */
        }
    }

    if(DEBUG0){ printf("%d:extracting pipe received data from %d %d vi=%p\n",
               armci_me,arg->proc,arg->count,dp->vi); fflush(stdout);
    }

#if 0
    rc = VipRecvWait(dp->vi, VIP_INFINITE, &pdscr);
#else
    do{ rc = VipRecvDone(dp->vi, &pdscr); }while(rc==VIP_NOT_DONE);
#endif
    armci_check_status(DEBUG0, rc,"pipe extract getting data");
    if(pdscr != (dp->descr+arg->count)){
        printf("%d:extracting pipe %d:wrong descr %p %p\n",armci_me, arg->count,
                pdscr,dp->descr+arg->count); fflush(stdout);
        armci_die("armci_pipe_extract_data:wrong descr completed",arg->count);
    }

    /* get the ptr to data buf corresponding to completed descriptor */
    
    buf = ARMCI_BUF_FROM_DESCR(pdscr);
  
    /* copy data to the user buffer identified by ptr */
    armci_read_strided(ptr, strides, stride_arr, count, buf);
    if(DEBUG1 ){ printf("%d(c): extracting: data copy to user buf success (%p,%p) first=%f\n",armci_me,
                pdscr,buf,((double*)buf)[0]); fflush(stdout);
    }

    arg->count++;
    dp->avail++;
    if(DEBUG1){
      if(*client_tail != CLIENT_STAMP){
         printf("%d:stamp corrupted %d tail=%p cur buf %p,%p %p %p %p %d %d\n",
                armci_me, *client_tail, client_tail, buf,
                client_buf_pool[0].buf, client_buf_pool[1].buf,
                client_buf_pool[2].buf, client_buf_pool[3].buf,
                sizeof(vbuf_ext_t),sizeof(vbuf_long_t));fflush(stdout);    
         armci_die("end of buf",*client_tail);
      }
    }
}
    


void armcill_pipe_send_chunk(void *data, int stride_arr[], int count[],
                             int strides, void* argvoid)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdesc;
VIP_MEM_HANDLE *pmhandle;
int bytes,i,extra;
buf_arg_t *arg = (buf_arg_t*)argvoid;
descr_pool_t *dp;

     if(arg->op ==GET){
        dp = &serv_descr_pool;
        dp->vi = (CLN_con+arg->proc)->vi;
        pmhandle = &serv_memhandle;
        if(!arg->count)armci_ack_proc=NONE; /* prevent sending ACK by server */
     }else{
        int srv= armci_clus_id(arg->proc);
        dp = &client_descr_pool;
        dp->vi= (SRV_con+srv)->vi;
        pmhandle = &client_memhandle;
        if(!arg->count){
            armci_clear_sends(dp); /* make sure all previous sends complete*/
            CLIENT_WAIT_ACK; /* wait for server to post bufs */
        }
     }

     if(DEBUG1){ printf("%d(s):SENDING pipe data %d to %d %p (%p,%p) vi=%p\n",
                 armci_me, arg->count, arg->proc, dp,&client_descr_pool,
                 &serv_descr_pool,dp->vi); fflush(stdout);
     }

     if(!dp->avail)
         armci_die("armci_pipe_send_chunk: all descriptors used",MAX_DESCR);

     /* allign send buffer on 64-byte boundary */
     extra = ALIGN64ADD(arg->buf);
     arg->buf+=extra;

     /* copy data to buffer */
     armci_write_strided(data, strides, stride_arr, count, arg->buf);

     /* setup descriptor */
     for(i=0, bytes=1;i<=strides;i++)bytes*=count[i];
     pdesc = dp->descr+MAX_DESCR-dp->avail;
     armci_init_vbuf(pdesc, arg->buf, bytes, *pmhandle);

     rc = VipPostSend(dp->vi, pdesc, *pmhandle);
     armci_check_status(DEBUG0, rc,"pipe data sent");

     if(DEBUG1){ printf("%d(s): out of send %d bytes=%d first=%f\n",armci_me,
               arg->count,bytes,((double*)arg->buf)[0]); fflush(stdout);
     }

     arg->buf += bytes+extra;
     arg->count++;
     dp->avail--;
}


#if 0
void armci_via_wait_ack()
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;

     if(armci_long_buf_free) armci_die("armci_via_wait_ack: nothing posted",0);

     armci_long_buf_free =1; /* mark up the buffer as free */

     /* make sure that msg associated with buffer completed */
     armci_dequeue_send_descr((SRV_con+armci_long_buf_taken_srv)->vi);

     /* wait for ack */
     rc=VipRecvWait((SRV_con+armci_long_buf_taken_srv)->vi,VIP_INFINITE,&pdscr);
     armci_check_status(DEBUG0, rc,"client wait_ack out of recv wait");

     if(DEBUG1){printf("%d(c) client got ack for req dscr=%p\n",armci_me,pdscr);
                fflush(stdout);
     }
}
#endif



/*\ client sends request for pipelined get to server
\*/
void armci_pipe_send_req(int proc, void *buf, int bytes)
{
VIP_RETURN rc;
int cluster = armci_clus_id(proc);
VIP_DESCRIPTOR *snd_dscr;

    snd_dscr = BUF_TO_SDESCR((char *)buf); 

    /* flow control for remote server - we cannot send if send/recv descriptor
       is pending for that node -- we will complete these descriptors
       we need to do that for small pipeline gets that still fit in a small buf
     */
     _armci_buf_ensure_one_outstanding_op_per_node(buf,cluster);

    armci_init_vbuf(snd_dscr, buf, bytes, client_memhandle);
    rc = VipPostSend((SRV_con+cluster)->vi, snd_dscr, client_memhandle);
    armci_check_status(DEBUG0, rc,"client sent data to server");

    if(DEBUG1){ printf("%d:client sent PIPE GET req %dbytes to server vi=%p\n",
                       armci_me,bytes,(SRV_con+cluster)->vi); fflush(stdout);
    }
}


/*\ client sends request to server
\*/
int armci_send_req_msg(int proc, void *buf, int bytes)
{
VIP_RETURN rc;
int cluster = armci_clus_id(proc);
VIP_DESCRIPTOR *snd_dscr;

    snd_dscr = BUF_TO_SDESCR((char *)buf);  

    /* flow control for remote server - we cannot send if send/recv descriptor
       is pending for that node -- we will complete these descriptors */
     _armci_buf_ensure_one_outstanding_op_per_node(buf,cluster);  

    armci_client_post_buf(cluster,buf); /* post descriptor for ack/response */
    armci_init_vbuf(snd_dscr, buf, bytes, client_memhandle);
    rc = VipPostSend((SRV_con+cluster)->vi, snd_dscr, client_memhandle);
    armci_check_status(DEBUG0, rc,"client sent data to server");

    if(DEBUG1){ printf("%d:client sent REQ %dbytes to server vi=%p\n",
                       armci_me,bytes,(SRV_con+cluster)->vi); fflush(stdout);
    }
    return 0;
}


/*\ client reads response from server
\*/
char *armci_ReadFromDirect(int proc, request_header_t *msginfo, int len)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;
int i, mybufid=0, cluster = armci_clus_id(proc);
vbuf_ext_t* evbuf=BUF_TO_EVBUF(msginfo);
char *dataptr = GET_DATA_PTR(evbuf->buf);

    if(DEBUG1){ printf("%d(c):read direct %d vi=%p\n",armci_me,
                (int)msginfo->datalen,(SRV_con+cluster)->vi); fflush(stdout);
    }
    for(i=0; i< MAX_BUFS; i++){
        vbuf_ext_t* cur = (vbuf_ext_t*)client_buf_pool[i].buf;
        if(cur == evbuf){ mybufid = i; break;}
    }
    
    do{rc = VipSendDone((SRV_con+cluster)->vi, &pdscr);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"WAIT for send to complete");

#if 0
    armci_dequeue_send_descr((SRV_con+cluster)->vi);

    rc = VipRecvWait((SRV_con+ cluster)->vi, VIP_INFINITE, &pdscr);
#endif

    do{rc = VipRecvDone((SRV_con+cluster)->vi, &pdscr);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"client getting data from server");
    if(pdscr != &evbuf->rcv_dscr){
       printf("%d(c): reading data client-DIFFERENT DESCRIPTOR %p %p buf=%p\n",
              armci_me,pdscr ,&evbuf->rcv_dscr, msginfo);fflush(stdout); 
       armci_die("reading data client-different descriptor completed",0);
    }
    
    client_buf_pool[mybufid].rcv =0;
    client_buf_pool[mybufid].snd =0;

#if 0
    armci_long_buf_free =1;
#endif

    return dataptr;
}

void armci_rcv_strided_data_bypass(int proc, request_header_t* msginfo,
                                   void *ptr, int stride_levels)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *pdscr;
vbuf_ext_t* evbuf=BUF_TO_EVBUF(msginfo);
int mybufid=-1,i,cluster = armci_clus_id(proc);

#if 0
    do{rc = VipSendDone((SRV_con+cluster)->vi, &pdscr);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"WAIT for send msg req to complete");
#endif
    for(i=0; i< MAX_BUFS; i++){
        vbuf_ext_t* cur = (vbuf_ext_t*)client_buf_pool[i].buf;
        if(cur == evbuf){ mybufid = i; break;}
    }

    if(mybufid<0)armci_die("rcv_strided_data_bypass:did not find buf",0);
    if(!client_buf_pool[mybufid].rcv)armci_die("rcv_strided_data_bypass: cv",0);

    if(DEBUG2){
       printf("%d:rcv_strided_data_bypass wait for ack%d\n",armci_me,mybufid);
       fflush(stdout);
    }
    do{rc = VipRecvDone((SRV_con+cluster)->vi, &pdscr);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"client getting ACK data from server");
    if(pdscr != &evbuf->rcv_dscr)
       armci_die("rcv_strided_data_bypass:different descriptor completed",0);

    client_buf_pool[mybufid].rcv =0;
    client_buf_pool[mybufid].snd =0;

    if(DEBUG2){
       printf("%d(c):rcv_strided_data_bypass buf %d\n",armci_me,mybufid);
       fflush(stdout);
    }
}
    

void armci_send_data_to_client(int proc, void *buf, int bytes)
{
VIP_RETURN rc;
VIP_DESCRIPTOR *cmpl_dscr;

    armci_init_vbuf(&serv_buf->snd_dscr, buf, bytes, serv_memhandle);
    rc = VipPostSend((CLN_con+proc)->vi, &serv_buf->snd_dscr, serv_memhandle);
    armci_check_status(DEBUG0, rc,"server send");
    if(bytes)
       armci_check_status(DEBUG0, rc,"server sent data to client");
    else
       armci_check_status(DEBUG0, rc,"server sent ack to client");

    /********** should be moved to where we get buffer  **********/
    rc = VipSendWait((CLN_con+proc)->vi, VIP_INFINITE, &cmpl_dscr);
    armci_check_status(DEBUG0, rc,"server wait for send to complete");
    if(cmpl_dscr !=&serv_buf->snd_dscr)
       armci_die("sending data to client-different descriptor completed",bytes);

    if(DEBUG1){ printf("%d:SERVER sent %dbytes to %d vi=%p\n",armci_me,bytes,
                proc,(CLN_con+proc)->vi); fflush(stdout);
    }
}



/*\ server sends data to client in response to request
\*/
void armci_WriteToDirect(int proc, request_header_t* msginfo, void *buf)
{
    if(DEBUG0){ printf("%d(s):write to direct sent %d to %d\n",armci_me,(int)msginfo->datalen,proc);
                fflush(stdout);
    }
     armci_send_data_to_client(proc, buf, (int)msginfo->datalen); 
     armci_ack_proc=NONE;
}


static void armci_iput(VIP_VI_HANDLE vi, VIP_DESCRIPTOR *d, 
                       void *src, VIP_MEM_HANDLE smhandle,
                       void *dst, VIP_MEM_HANDLE dmhandle, int len)
{
VIP_RETURN rc;

    memset(d,0,sizeof(VIP_DESCRIPTOR));
    d->CS.Control  = VIP_CONTROL_OP_RDMAWRITE;
    d->CS.Length   = (unsigned)len;
    d->CS.SegCount = 2; /* address segment and data segment */
    d->CS.Reserved = 0;
    d->CS.Status   = 0;
    d->DS[0].Remote.Data.Address = dst;
    d->DS[0].Remote.Handle = dmhandle;
    d->DS[1].Local.Data.Address = src;
    d->DS[1].Local.Handle = smhandle;
    d->DS[1].Local.Length = (unsigned)len;

    if(SERVER_CONTEXT){
      rc = VipPostSend(vi, d, serv_memhandle);
      armci_check_status(DEBUG0, rc,"server put");
    }else{
      rc = VipPostSend(vi, d, client_memhandle);
      armci_check_status(DEBUG0, rc,"client put");
    }
}


void armci_client_send_ack(int proc, int n)
{
int srv = armci_clus_id(proc);
descr_pool_t *dp = &client_descr_pool;
VIP_DESCRIPTOR *pcmpl, *pdesc = dp->descr+MAX_DESCR-dp->avail;
VIP_RETURN rc;
void *ptr_ack=pinned_handle;

    armci_iput((SRV_con+srv)->vi, pdesc, ptr_ack, client_memhandle,
               (SRV_ack+srv)->prem_handle, (SRV_ack+srv)->handle,
               sizeof(VIP_MEM_HANDLE));

    if(DEBUG2){
       printf("%d(c): sent memhandle/ack %d to server at (%p,%d)\n",armci_me,
             *pinned_handle, (SRV_ack+srv)->prem_handle, (SRV_ack+srv)->handle);
       fflush(stdout);
    }

    /* this is for req message */
    do{rc = VipSendDone((SRV_con+srv)->vi, &pcmpl);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"WAIT for send msg req to complete");

    /* this is for put */
    do{rc = VipSendDone((SRV_con+srv)->vi, &pcmpl);}while(rc==VIP_NOT_DONE);
    armci_check_status(DEBUG0, rc,"WAIT for put ack to complete");
    if(pcmpl != pdesc)armci_die("put ack: wrong descr completed ",srv);
}


VIP_MEM_HANDLE _armci_getval(VIP_MEM_HANDLE *p) { return *p; }
void armcill_server_wait_ack(int proc, int n)
{
volatile VIP_MEM_HANDLE ack;
     if(DEBUG2){printf("%d(s):waiting for ack at%p\n",armci_me,CLN_handle+proc);
                fflush(stdout);
     }
     do{ ack = _armci_getval(CLN_handle+proc); }while (!ack); 
     if(DEBUG2){printf("%d(s): got memhandle %d from %d\n",armci_me,ack,proc);
        fflush(stdout);
     }
}

VIP_MEM_HANDLE serv_pin_memhandle;
int armci_pin_contig(void *ptr, int len)
{
VIP_MEM_ATTRIBUTES mattr;
VIP_RETURN rc;

     mattr.EnableRdmaWrite = VIP_TRUE;
     mattr.EnableRdmaRead  = VIP_FALSE;
     if(SERVER_CONTEXT){
        mattr.Ptag = CLN_nic->ptag;
        rc = VipRegisterMem(CLN_nic->handle,ptr,len,&mattr,&serv_pin_memhandle);
     }else{
        mattr.Ptag = SRV_nic->ptag;
        rc = VipRegisterMem(SRV_nic->handle,ptr,len,&mattr,pinned_handle);
        if(DEBUG2){
           printf("%d:pinned %d bytes handle %d\n",armci_me,len,*pinned_handle);
           fflush(stdout);
        }
     }
     armci_check_status(DEBUG0, rc,"pinning memory");
     return 1;
}

void armci_unpin_contig(void *ptr, int len)
{
VIP_RETURN rc;

     if(SERVER_CONTEXT){
        rc = VipDeregisterMem(CLN_nic->handle,ptr,serv_pin_memhandle);
     }else{
        rc = VipDeregisterMem(SRV_nic->handle,ptr,*pinned_handle);
     }
     armci_check_status(DEBUG0, rc,"unpinning memory");
}

int armci_pin_memory(void *ptr, int stride_arr[], int count[], int strides)
{
    if(strides ==0)return armci_pin_contig(ptr,count[0]);
    printf("%d: cannot pin stridedd",strides);
    return 0;
}

void armci_unpin_memory(void *ptr, int stride_arr[], int count[], int strides)
{
    if(strides ==0) armci_unpin_contig(ptr,count[0]);
    else armci_die("strided was not pinned",strides);
}


void armcill_server_put(int proc, void* s, void *d, int len)
{
descr_pool_t   *dp = &serv_descr_pool;
VIP_DESCRIPTOR *pdesc;
char *src=(char*)s;
char *dst=(char*)d;
    if(dp->avail != MAX_DESCR)
       armci_die("armci_server_put: expected",MAX_DESCR);
    dp->vi = (CLN_con+proc)->vi;
    while(len){
      int bytes = VBUF_DLEN>len? len: VBUF_DLEN;
      if(!dp->avail)armci_serv_clear_sends();
      pdesc = dp->descr+MAX_DESCR-dp->avail;
      armci_iput((CLN_con+proc)->vi, pdesc, src, serv_pin_memhandle,
                  dst, CLN_handle[proc], bytes);
      len  -= bytes;
      src  += bytes;
      dst  += bytes;
      dp->avail--; 
    }
    armci_serv_clear_sends();  
    CLN_handle[proc] =0; /*clear is for next round */
//  SERVER_SEND_ACK(proc); /* server code does not expect ack in GET*/
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
