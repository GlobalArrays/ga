/********************************************************************* 
  Initial version of ARMCI Port for the Infiniband VAPI
  Contiguous sends and noncontiguous sends need a LOT of optimization
  most of the structures are very similar to those in VIA code.
 *********************************************************************/
#include <stdio.h>
#include <strings.h>
#include <assert.h>
#include <unistd.h>

#include "armcip.h"
#include "copy.h"
/*our incude*/
#include "armci-vapi.h"
#define DEBUG_INIT 0
#define DEBUG_SERVER 0
#define DEBUG_CLN 0
/* The device name is "InfiniHost0" */
#  define VAPIDEV_NAME "InfiniHost0"
#  define INVAL_HNDL 0xFFFFFFFF
typedef struct {
   VAPI_qp_hndl_t qp;
   VAPI_qp_prop_t qp_prop;         /*mostly for getting scq num*/
  VAPI_qp_num_t sqpnum;             /*we need to exchng qp nums,arr for that*/
   VAPI_qp_num_t *rqpnum;          /*we need rqp nums,arr for that*/
   IB_lid_t lid;
} armci_connect_t;
armci_connect_t *CLN_con,*SRV_con;
VAPI_hca_id_t   hca_id= VAPIDEV_NAME;


/*\
 * datastrucure for infinihost NIC
\*/
typedef struct {
  VAPI_hca_hndl_t handle;           /*IB nic handle*/
  VAPI_hca_id_t   hca_id;
  VAPI_hca_vendor_t vendor;
  VAPI_hca_cap_t attr;              /*IB nic attributes*/
  VAPI_pd_hndl_t ptag;              /*protection tag*/
  VAPI_hca_port_t hca_port;         /*mostly for getting lid*/
  VAPI_cq_hndl_t scq;               /*send completion queue*/
  VAPI_cq_hndl_t rcq;               /*recv completion queue*/
  IB_lid_t *lid_arr;                /*we need to exchange lids, arr for that*/
  VAPI_qp_num_t rqpnum;            /*we need to exchng qp nums,arr for that*/
  EVAPI_compl_handler_hndl_t rcq_eventh;
  int maxtransfersize; 
} vapi_nic_t;

typedef struct {
  armci_vapi_memhndl_t *prem_handle; /*address server to store memory handle*/ 
  armci_vapi_memhndl_t handle;
}ack_t;

armci_vapi_memhndl_t *CLN_handle;
armci_vapi_memhndl_t serv_memhandle, client_memhandle;
armci_vapi_memhndl_t *handle_array;
armci_vapi_memhndl_t *pinned_handle;

static vapi_nic_t nic_arr[3];
static vapi_nic_t *SRV_nic= nic_arr;
static vapi_nic_t *CLN_nic= nic_arr+1;
static int armci_server_terminating;

#define NONE -1
static int armci_ack_proc=NONE;

static int armci_vapi_server_ready;
static int armci_vapi_server_stage1=0;
static int armci_vapi_client_stage1=0;
static int armci_vapi_server_stage2=0;
static int armci_vapi_client_ready;
int _s=-1,_c=-1;
static int server_can_poll=0;
static int armci_vapi_max_inline_size=-1;
#define CLIENT_STAMP 101
#define SERV_STAMP 99 

static char * client_tail;
static char * serv_tail;
static ack_t *SRV_ack;
typedef struct {
	  VAPI_rr_desc_t dscr;
	  VAPI_sg_lst_entry_t    sg_entry;
	  char buf[VBUF_DLEN];
}vapibuf_t;


typedef struct {
          VAPI_sr_desc_t         snd_dscr;
          VAPI_sg_lst_entry_t    ssg_entry;
          VAPI_rr_desc_t         rcv_dscr;
          VAPI_sg_lst_entry_t    rsg_entry;
          char buf[VBUF_DLEN];
}vapibuf_ext_t;

static vapibuf_t **serv_buf_arr, *spare_serv_buf,*spare_serv_bufptr;
static vapibuf_ext_t *serv_buf;

#define MAX_DESCR 16
typedef struct { 
    int avail; VAPI_qp_hndl_t qp; VAPI_rr_desc_t *descr;
} descr_pool_t;

static int* _gtmparr;

char *MessageRcvBuffer;

extern void armci_util_wait_int(volatile int *,int,int);
void armci_send_data_to_client(int proc, void *buf,int bytes,void *dbuf);
void armci_server_register_region(void *,long,ARMCI_MEMHDL_T *);
static descr_pool_t serv_descr_pool = {MAX_DESCR, 0, (VAPI_rr_desc_t *)0};
static descr_pool_t client_descr_pool = {MAX_DESCR,0,(VAPI_rr_desc_t *)0};

#define GET_DATA_PTR(buf) (sizeof(request_header_t) + (char*)buf)

#define BUF_TO_SDESCR(buf) ((VAPI_sr_desc_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->sdscr))

#define BUF_TO_RDESCR(buf) ((VAPI_rr_desc_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->rdscr))

#define BUF_TO_SSGLST(buf) ((VAPI_sg_lst_entry_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->ssg_entry))

#define BUF_TO_RSGLST(buf) ((VAPI_sg_lst_entry_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->rsg_entry))

#define BUF_TO_EVBUF(buf) (vapibuf_ext_t*)(((char*)buf) - (sizeof(VAPI_sr_desc_t)+sizeof(VAPI_rr_desc_t)+2*sizeof(VAPI_sg_lst_entry_t)))

#define SERVER_SEND_ACK(p) {*((long *)serv_buf->buf)=ARMCI_VAPI_COMPLETE;armci_send_data_to_client((p),serv_buf->buf,sizeof(long),msginfo->tag.ack_ptr);}


/*\ descriptors will have unique ID's for the wait on descriptor routine to 
 * complete a descriptor and know where it came from
\*/

#define MAX_PENDING 8
#define DSCRID_NBDSCR 10000
#define DSCRID_NBDSCR_END (10000+MAX_PENDING)

#define DSCRID_SERVSEND 20000
#define DSCRID_SERVSEND_END 20000+10000


#define NUMOFBUFFERS 20
#define DSCRID_FROMBUFS 1
#define DSCRID_FROMBUFS_END DSCRID_FROMBUFS+NUMOFBUFFERS
static int mark_buf_send_complete[NUMOFBUFFERS];


static sdescr_t armci_vapi_nbsdscr_array[MAX_PENDING];

/*\
 *   function to get the next available context for non-blocking RDMA. We limit
 *   the number of uncompleted non-blocking RDMA sends to 8.
\*/

void armci_check_status(int,VAPI_ret_t,char*);
void armci_client_send_complete(VAPI_sr_desc_t *snd_dscr, char *from)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_wc_desc_t pdscr1;
VAPI_wc_desc_t *pdscr=&pdscr1;

    if(DEBUG_CLN){
       printf("\n%d:client send complete called from %s id=%ld\n",armci_me,
               from,snd_dscr->id);fflush(stdout);
    }
    do{
       while(rc == VAPI_CQ_EMPTY)  
         rc = VAPI_poll_cq(SRV_nic->handle, SRV_nic->scq, pdscr);
       armci_check_status(DEBUG_CLN,rc,"client_send_complete wait for send");
       rc =VAPI_CQ_EMPTY;

       if(DEBUG_CLN){
         printf("\n%d:client_send_complete completed %ld",armci_me, pdscr->id);
         fflush(stdout);
       }

       if(pdscr->id!=snd_dscr->id){
         if(pdscr->id >=DSCRID_FROMBUFS && pdscr->id < DSCRID_FROMBUFS_END)
           mark_buf_send_complete[pdscr->id-DSCRID_FROMBUFS]=1;
         else if(pdscr->id >=DSCRID_NBDSCR && pdscr->id < DSCRID_NBDSCR_END)
           armci_vapi_nbsdscr_array[pdscr->id-DSCRID_NBDSCR].tag=0;
         else armci_die("client send complete got weird id",pdscr->id);
       }
    }while(pdscr->id!=snd_dscr->id);
    if(pdscr->id >=DSCRID_NBDSCR && pdscr->id < DSCRID_NBDSCR_END)
       armci_vapi_nbsdscr_array[pdscr->id-DSCRID_NBDSCR].tag=0;


}

sdescr_t *armci_vapi_get_next_descr(int nbtag)
{
static int avail=-1;
sdescr_t *retdscr;
    if(avail==-1){
       int i;
       for(i=0;i<MAX_PENDING;i++){
         armci_vapi_nbsdscr_array[i].tag=0;
         bzero(&armci_vapi_nbsdscr_array[i].descr,sizeof(VAPI_sr_desc_t));
         armci_vapi_nbsdscr_array[i].descr.id = DSCRID_NBDSCR + i; 
       }
       avail=0;
    }
    if(armci_vapi_nbsdscr_array[avail].tag!=0){
       armci_client_send_complete(&armci_vapi_nbsdscr_array[avail].descr,"armci_vapi_get_next_descr");
    }
    armci_vapi_nbsdscr_array[avail].tag=nbtag;
    retdscr= (armci_vapi_nbsdscr_array+avail);
    memset(&retdscr->descr,0,sizeof(VAPI_sr_desc_t));
    retdscr->descr.id = DSCRID_NBDSCR + avail;
    avail = (avail+1)%MAX_PENDING;
    return(retdscr);
}




void armci_wait_for_server()
{
	  armci_server_terminating=1;
}


void armci_check_status(int debug,VAPI_ret_t rc,char *msg)
{
    if(rc != VAPI_OK){
       char buf[100];
       if(armci_server_terminating){
         /* server got interrupted when clients terminate connections */
         sleep(1);
         _exit(0);
       }
       fprintf(stderr,"%d in check FAILURE %s\n",armci_me,msg);
       assert(strlen(msg)<100-20);
       sprintf(buf,"ARMCI(vapi):failure:%d:%s code %d %d ",rc,msg,
               _s,_c);
#     ifdef  PAUSE_ON_ERROR
       printf("%d(%d): Error from VIPL: %s - pausing\n",
              armci_me, getpid(), msg);
       fflush(stdout);
       pause();
#     endif
       armci_die(buf,(int)rc);
    }else if(debug){
       printf("%d:ARMCI(vapi): %s successful\n",armci_me,msg);
       fflush(stdout);
    }
}




/*\
 * create QP == create VI in via
\*/
static void armci_create_qp(vapi_nic_t *nic,VAPI_qp_hndl_t *qp,VAPI_qp_prop_t *qp_prop)
{
call_result_t rc;
VAPI_qp_init_attr_t initattr;

    bzero(&initattr, sizeof(VAPI_qp_init_attr_t));
    *qp=INVAL_HNDL;

    initattr.cap.max_oust_wr_rq = DEFAULT_MAX_WQE;
    initattr.cap.max_oust_wr_sq = DEFAULT_MAX_WQE;
    initattr.cap.max_sg_size_rq = DEFAULT_MAX_SG_LIST;
    initattr.cap.max_sg_size_sq = DEFAULT_MAX_SG_LIST;
    initattr.pd_hndl            = nic->ptag;
    initattr.rdd_hndl           = 0;
    initattr.rq_cq_hndl         = nic->rcq;
    initattr.sq_cq_hndl         = nic->scq;
    initattr.rq_sig_type        = VAPI_SIGNAL_REQ_WR;
    initattr.sq_sig_type        = VAPI_SIGNAL_REQ_WR;
    initattr.ts_type            = IB_TS_RC;

    rc = VAPI_create_qp(nic->handle, &initattr, qp, qp_prop);
    if(!armci_vapi_max_inline_size){
       armci_vapi_max_inline_size = qp_prop->cap.max_inline_data_sq;
       if(DEBUG_CLN)
       printf("\n%d:max inline size= %d\n",armci_me,armci_vapi_max_inline_size);
    }

    armci_check_status(DEBUG_INIT, rc,"create qp");

}




/*\
 * doesnt have to be static void, but we should make this a coding rule.
 * functions used only inside that file should always be static
\*/
static void armci_init_nic(vapi_nic_t *nic, int scq_entries, int rcq_entries)
{
VAPI_ret_t rc;
VAPI_cqe_num_t num;

    bzero(nic,sizeof(vapi_nic_t));
    /*hca_id = VAPIDEV_NAME;*/
    nic->lid_arr    = (IB_lid_t *)calloc(armci_nproc,sizeof(IB_lid_t));
    if(!nic->lid_arr)
       armci_die("malloc for nic_t arrays in vapi.c failed",0);

    /*first open nic*/ 
    rc = VAPI_open_hca(hca_id, &nic->handle);
    /*armci_check_status(DEBUG_INIT, rc,"open nic");*/

    rc = EVAPI_get_hca_hndl(hca_id, &nic->handle);
    armci_check_status(DEBUG_INIT, rc,"get handle");

    nic->maxtransfersize = MAX_RDMA_SIZE;

    /*now, query for properties, why?*/
    rc = VAPI_query_hca_cap(nic->handle, &nic->vendor, &nic->attr);
    armci_check_status(DEBUG_INIT, rc,"query nic");

    /*query nic port basically for lid, lid in IB is required*/
    VAPI_query_hca_port_prop(nic->handle,(IB_port_t)DEFAULT_PORT,
                             &(nic->hca_port));
    armci_check_status(DEBUG_INIT, rc,"query for lid");

    /*save the lid for doing a global exchange later */
    nic->lid_arr[armci_me] = nic->hca_port.lid;

    /*allocate tag (protection domain) */
    rc = VAPI_alloc_pd(nic->handle, &nic->ptag);
    armci_check_status(DEBUG_INIT, rc,"create protection domain");

    /*properties of scq and rcq required for the cq number, this also needs 
     * to be globally exchanged
     */
    nic->scq = INVAL_HNDL;
    nic->rcq = INVAL_HNDL;
    /*do the actual queue creation */
    if(scq_entries){
       rc = VAPI_create_cq(nic->handle,DEFAULT_MAX_CQ_SIZE, 
                           &nic->scq,&num);
       armci_check_status(DEBUG_INIT, rc,"create send completion queue");
    }
    if(rcq_entries){
       /*rc = VAPI_create_cq(nic->handle,(VAPI_cqe_num_t)rcq_entries, 
                           &nic->rcq,&num);*/
       rc = VAPI_create_cq(nic->handle,DEFAULT_MAX_CQ_SIZE, 
                           &nic->rcq,&num);
       armci_check_status(DEBUG_INIT, rc,"create recv completion queue");
    }
    /*VAPIErrorCallback(nic->handle, 0, armci_err_callback);*/
}

void armci_server_alloc_bufs()
{
VAPI_ret_t rc;
VAPI_mrw_t mr_in,mr_out;
int mod, bytes, total, extra =sizeof(VAPI_rr_desc_t)*MAX_DESCR+SIXTYFOUR;
int mhsize = armci_nproc*sizeof(armci_vapi_memhndl_t); /* ack */
char *tmp, *tmp0; 
int clients = armci_nproc,i,j=0;

    /* allocate memory for the recv buffers-must be alligned on 64byte bnd */
    /* note we add extra one to repost it for the client we are received req */ 
    bytes =(clients+1)*sizeof(vapibuf_t)+sizeof(vapibuf_ext_t) + extra+ mhsize;
    total = bytes + SIXTYFOUR;
    if(total%4096!=0) 
       total = total - (total%4096) + 4096;
    tmp0=tmp = VMALLOC(total);
    if(!tmp) armci_die("failed to malloc server vapibufs",total);
    /* stamp the last byte */
    serv_tail= tmp + bytes+SIXTYFOUR-1;
    *serv_tail=SERV_STAMP;
    /* allocate memory for client memory handle to support put response 
     *         in dynamic memory registration protocols */
    CLN_handle = (armci_vapi_memhndl_t*)tmp;
    memset(CLN_handle,0,mhsize); /* set it to zero */
    tmp += mhsize;
    /* setup descriptor memory */
    mod = ((ssize_t)tmp)%SIXTYFOUR;
    serv_descr_pool.descr= (VAPI_rr_desc_t*)(tmp+SIXTYFOUR-mod);
    tmp += extra;
    /* setup buffer pointers */
    mod = ((ssize_t)tmp)%SIXTYFOUR;
    serv_buf_arr = (vapibuf_t **)malloc(sizeof(vapibuf_t*)*armci_nproc);
    for(i=0;i<armci_nproc;i++){
       serv_buf_arr[i] = (vapibuf_t*)(tmp+SIXTYFOUR-mod) + j++;
    }
    i=0;
    while(serv_buf_arr[i]==NULL)i++;
    spare_serv_buf = serv_buf_arr[i]+clients; /* spare buffer is at the end */
    spare_serv_bufptr = spare_serv_buf;    /* save the pointer for later */
    serv_buf =(vapibuf_ext_t*)(serv_buf_arr[i]+clients+1);
    MessageRcvBuffer = serv_buf->buf;
    /* setup memory attributes for the region */
    /*mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_ATOM | VAPI_EN_REMOTE_WRITE |VAPI_EN_REMOTE_READ;*/
    mr_in.acl =  VAPI_EN_LOCAL_WRITE | VAPI_EN_REMOTE_WRITE | VAPI_EN_REMOTE_READ;
    mr_in.l_key = 0;
    mr_in.pd_hndl = CLN_nic->ptag;
    mr_in.r_key = 0;
    mr_in.size = total;
    mr_in.start = (VAPI_virt_addr_t)(MT_virt_addr_t)tmp0;
    mr_in.type = VAPI_MR;

    if(DEBUG_SERVER){
      printf("\n%d(s):registering mem %p %dbytes ptag=%ld handle=%d\n",
             armci_me, tmp0,total,CLN_nic->ptag,CLN_nic->handle);fflush(stdout);
    }

    rc = VAPI_register_mr(CLN_nic->handle,&mr_in,&(serv_memhandle.memhndl),&mr_out);
    armci_check_status(DEBUG_INIT, rc,"server register recv vbuf");
    
    serv_memhandle.lkey = mr_out.l_key;
    serv_memhandle.rkey = mr_out.r_key;
     
    /* exchange address of ack/memhandle flag on servers */


    if(!serv_memhandle.memhndl)armci_die("server got null handle for vbuf",0);
    if(DEBUG_SERVER){
       printf("%d(s):registered mem %p %dbytes mhandle=%d mharr starts%p\n",
              armci_me, tmp0, total, serv_memhandle.memhndl,CLN_handle);
       fflush(stdout); 
    }
}


void armci_set_serv_mh()
{
int s, ratio = sizeof(ack_t)/sizeof(int);
    /* first collect addrresses on all masters */
    if(armci_me == armci_master){
       SRV_ack[armci_clus_me].prem_handle=CLN_handle;
       SRV_ack[armci_clus_me].handle =serv_memhandle;
       armci_msg_gop_scope(SCOPE_MASTERS,SRV_ack,ratio*armci_nclus,"+",
                           ARMCI_INT);
    }
    /* next master broadcasts the addresses within its node */
    armci_msg_bcast_scope(SCOPE_NODE,SRV_ack,armci_nclus*sizeof(ack_t),
                          armci_master);

    /* Finally save address corresponding to my id on each server */
    for(s=0; s< armci_nclus; s++){
       SRV_ack[s].prem_handle += armci_me;
       /*printf("%d: my addr on %d = %p\n",armci_me,s,SRV_ack[s].prem_handle);
         fflush(stdout); */
    }

}


/*\
 * init_connections, client_connect_to_servers -- client code
 * server_initial_connection, call_data_server -- server code 
\*/ 
void armci_init_connections()
{
int c,s;
int sz;
int *tmparr;
    
    /* initialize nic connection for qp numbers and lid's */
    armci_init_nic(SRV_nic,1,1);
    bzero(mark_buf_send_complete,sizeof(int)*NUMOFBUFFERS);
    _gtmparr = (int *)calloc(armci_nproc,sizeof(int)); 

    /*qp_numbers and lids need to be exchanged globally*/
    tmparr = (int *)calloc(armci_nproc,sizeof(int));
    tmparr[armci_me] = SRV_nic->lid_arr[armci_me];
    sz = armci_nproc;
    armci_msg_gop_scope(SCOPE_ALL,tmparr,sz,"+",ARMCI_INT);
    for(c=0;c<armci_nproc;c++){
       SRV_nic->lid_arr[c]=tmparr[c];
       tmparr[c]=0;
    }

    /*SRV_con is for client to connect to servers */
    SRV_con=(armci_connect_t *)malloc(sizeof(armci_connect_t)*armci_nclus);
    if(!SRV_con)armci_die("cannot allocate SRV_con",armci_nclus);
    bzero(SRV_con,sizeof(armci_connect_t)*armci_nclus);

    CLN_con=(armci_connect_t*)malloc(sizeof(armci_connect_t)*armci_nproc);
    if(!CLN_con)armci_die("cannot allocate SRV_con",armci_nproc);
    bzero(CLN_con,sizeof(armci_connect_t)*armci_nproc);

    /*every client creates a qp with every server other than the one on itself*/
    sz = armci_nproc*(sizeof(VAPI_qp_num_t)/sizeof(int));
    armci_vapi_max_inline_size = 0;
    for(s=0; s< armci_nclus; s++){
       armci_connect_t *con = SRV_con + s;
       con->rqpnum = (VAPI_qp_num_t *)malloc(sizeof(VAPI_qp_num_t)*armci_nproc);
       bzero(con->rqpnum,sizeof(VAPI_qp_num_t)*armci_nproc);
       /*if(armci_clus_me != s)*/
       {
         armci_create_qp(SRV_nic,&con->qp,&con->qp_prop);
         con->sqpnum  = con->qp_prop.qp_num;
         con->rqpnum[armci_me]  = con->qp_prop.qp_num;
         con->lid     = SRV_nic->lid_arr[s];
       }
       armci_msg_gop_scope(SCOPE_ALL,con->rqpnum,sz,"+",ARMCI_INT);
    }

    if(DEBUG_CLN) printf("%d: connections ready for client\n",armci_me);

    /* ............ masters also set up connections for clients ......... */
    SRV_ack = (ack_t*)calloc(armci_nclus,sizeof(ack_t));
    if(!SRV_ack)armci_die("buffer alloc failed",armci_nclus*sizeof(ack_t));

    handle_array = (armci_vapi_memhndl_t *)calloc(sizeof(armci_vapi_memhndl_t),armci_nproc);
    if(!handle_array)armci_die("handle_array malloc failed",0);

}

static void vapi_connect_client()
{
int i,start,sz=0,c;
call_result_t rc;
VAPI_qp_attr_t         qp_attr;
VAPI_qp_cap_t          qp_cap;
VAPI_qp_attr_mask_t    qp_attr_mask;
    if(armci_me==armci_master)
       armci_util_wait_int(&armci_vapi_server_stage1,1,10000);
    armci_msg_barrier();
    sz = armci_nproc;
    if(armci_me==armci_master){
       armci_msg_gop_scope(SCOPE_MASTERS,_gtmparr,sz,"+",ARMCI_INT);
       for(c=0;c<armci_nproc;c++){
         CLN_nic->lid_arr[c]=_gtmparr[c];
         _gtmparr[c]=0;
       }
       if(DEBUG_CLN){
         printf("\n%d(svc): mylid = %d",armci_me,CLN_nic->lid_arr[armci_me]);
         fflush(stdout);
       }
    }

    armci_vapi_client_stage1 = 1;

    /* allocate and initialize connection structs */
    sz = armci_nproc*(sizeof(VAPI_qp_num_t)/sizeof(int));

    if(armci_me==armci_master)
       armci_util_wait_int(&armci_vapi_server_stage2,1,10000);
    armci_msg_barrier();
    for(c=0; c< armci_nproc; c++){
       armci_connect_t *con = CLN_con + c;
       if(armci_me!=armci_master){
         con->rqpnum=(VAPI_qp_num_t *)malloc(sizeof(VAPI_qp_num_t)*armci_nproc);
         bzero(con->rqpnum,sizeof(VAPI_qp_num_t)*armci_nproc);
       }
       armci_msg_gop_scope(SCOPE_ALL,con->rqpnum,sz,"+",ARMCI_INT);
    }

    /*armci_set_serv_mh();*/

    if(DEBUG_CLN)printf("%d: all connections ready \n",armci_me);

    /* Modifying  QP to INIT */
    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state = VAPI_INIT;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.pkey_ix  = DEFAULT_PKEY_IX;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.port     = DEFAULT_PORT;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PORT);
    qp_attr.remote_atomic_flags = VAPI_EN_REM_WRITE | VAPI_EN_REM_READ;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_REMOTE_ATOMIC_FLAGS);

    /* start from from server on my_node -1 */
    start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;
    for(i=0; i< armci_nclus; i++){
       armci_connect_t *con;
       con = SRV_con + i;
       rc = VAPI_modify_qp(SRV_nic->handle,(con->qp),&qp_attr, &qp_attr_mask, &qp_cap);
       armci_check_status(DEBUG_INIT, rc,"client connect requesti RST->INIT");
    }

    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state = VAPI_RTR;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.qp_ous_rd_atom = 2;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_OUS_RD_ATOM);
    qp_attr.av.sl            = 0;
    qp_attr.av.grh_flag      = FALSE;
    qp_attr.av.static_rate   = 0; /* 1x */
    qp_attr.av.src_path_bits = 0;
    qp_attr.path_mtu         = MTU1024;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PATH_MTU);
    qp_attr.rq_psn           = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RQ_PSN);
    qp_attr.pkey_ix = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.min_rnr_timer = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_MIN_RNR_TIMER);

    start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;
    for(i=0; i< armci_nclus; i++){
       armci_connect_t *con;
       armci_connect_t *conS;
       con = SRV_con + i;
       conS = CLN_con + armci_me;
       qp_attr.dest_qp_num=conS->rqpnum[armci_clus_info[i].master];
       QP_ATTR_MASK_SET(qp_attr_mask, QP_ATTR_DEST_QP_NUM);
       qp_attr.av.dlid = SRV_nic->lid_arr[armci_clus_info[i].master];
       QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_AV);
       rc = VAPI_modify_qp(SRV_nic->handle,(con->qp),&qp_attr, &qp_attr_mask, 
                           &qp_cap);
       armci_check_status(DEBUG_INIT, rc,"client connect request INIT->RTR");
    }

    /*to to to RTS, other side must be in RTR*/
    armci_msg_barrier();

    armci_vapi_client_ready=1; 

    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state   = VAPI_RTS;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.sq_psn   = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_SQ_PSN);
    qp_attr.timeout   = 0x20;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_TIMEOUT);
    qp_attr.retry_count   = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RETRY_COUNT);
    qp_attr.rnr_retry     = 3;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RNR_RETRY);
    qp_attr.ous_dst_rd_atom  = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_OUS_DST_RD_ATOM);

    start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;
    for(i=0; i< armci_nclus; i++){
       armci_connect_t *con;
       con = SRV_con + i;
       rc=VAPI_modify_qp(SRV_nic->handle,(con->qp),&qp_attr,&qp_attr_mask,
                         &qp_cap);
       armci_check_status(DEBUG_CLN, rc,"client connect request RTR->RTS");
    }

}

void armci_client_connect_to_servers()
{
    /* initialize buffer managment module */
    extern void armci_util_wait_int(volatile int *,int,int);
    _armci_buf_init();

    vapi_connect_client();
    if(armci_me==armci_master)
      armci_util_wait_int(&armci_vapi_server_ready,1,10000);
    armci_msg_barrier();
    if(DEBUG_CLN && armci_me==armci_master){
       printf("\n%d:server_ready=%d\n",armci_me,armci_vapi_server_ready);
       fflush(stdout);
    }
}


void armci_init_vapibuf_recv(VAPI_rr_desc_t *rd,VAPI_sg_lst_entry_t *sg_entry, char* buf, int len, armci_vapi_memhndl_t *mhandle)
{
     memset(rd,0,sizeof(VAPI_rr_desc_t));
     rd->opcode = VAPI_RECEIVE;
     rd->comp_type = VAPI_SIGNALED;
     rd->sg_lst_len = 1;
     rd->sg_lst_p  = sg_entry;
     rd->id = 0;

     sg_entry->lkey = mhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)buf;
     sg_entry->len = len;
}


void armci_init_vapibuf_send(VAPI_sr_desc_t *sd,VAPI_sg_lst_entry_t *sg_entry, char* buf, int len, armci_vapi_memhndl_t *mhandle)
{
     sd->opcode = VAPI_SEND;
     sd->comp_type = VAPI_SIGNALED;
     sd->sg_lst_len = 1;
     sd->sg_lst_p  = sg_entry;
     /*sd->id = 0;*/
     sd->remote_qkey=0;
     sd->set_se = TRUE;
     sd->fence = FALSE;

     sg_entry->lkey = mhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)buf;
     sg_entry->len = len;
}



static void armci_init_vbuf_srdma(VAPI_sr_desc_t *sd, VAPI_sg_lst_entry_t *sg_entry, char* lbuf, char *rbuf, int len, armci_vapi_memhndl_t *lhandle,armci_vapi_memhndl_t *rhandle)
{
     sd->opcode = VAPI_RDMA_WRITE;
     sd->comp_type = VAPI_SIGNALED;
     sd->sg_lst_len = 1;
     sd->sg_lst_p  = sg_entry;
     /*sd->id = 0;*/
     sd->remote_qkey=0;
     if(rhandle)sd->r_key = rhandle->rkey;
     sd->remote_addr = (VAPI_virt_addr_t) (MT_virt_addr_t) rbuf;
     sd->set_se = TRUE;
     sd->fence = FALSE;

     if(lhandle)sg_entry->lkey = lhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)lbuf;
     sg_entry->len = len;
}



static void armci_init_vbuf_rrdma(VAPI_sr_desc_t *sd, VAPI_sg_lst_entry_t *sg_entry, char* lbuf, char *rbuf, int len, armci_vapi_memhndl_t *lhandle,armci_vapi_memhndl_t *rhandle)
{
     sd->opcode = VAPI_RDMA_READ;
     sd->comp_type = VAPI_SIGNALED;
     sd->sg_lst_len = 1;
     sd->sg_lst_p  = sg_entry;
     /*sd->id = 0;*/
     sd->remote_qkey=0;
     if(rhandle)sd->r_key = rhandle->rkey;
     sd->remote_addr = (VAPI_virt_addr_t) (MT_virt_addr_t) rbuf;
     sd->set_se = TRUE;
     sd->fence = FALSE;

     if(lhandle)sg_entry->lkey = lhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)lbuf;
     sg_entry->len = len;
}

static void vapi_signal_comp_handler(VAPI_hca_hndl_t hca_hndl,
                                     VAPI_cq_hndl_t cq_hndl,void* sem_p)
{
  /*printf("%d:in comp handler",armci_me);fflush(stdout);
  MOSAL_sem_rel((MOSAL_semaphore_t*)sem_p);
  printf("%d:in comp handler - semaphore released",armci_me);fflush(stdout);
  */
}



void armci_server_initial_connection()
{
int c, ib;
VAPI_ret_t rc;
VAPI_qp_attr_t         qp_attr;
VAPI_qp_cap_t          qp_cap;
VAPI_qp_attr_mask_t    qp_attr_mask;
char *enval;

    if(DEBUG_SERVER){ 
       printf("in server after fork %d (%d)\n",armci_me,getpid());
       fflush(stdout);
    }

    armci_init_nic(CLN_nic,1,1);

    /*MOSAL_sem_init(&(res->rq_sem),0);*/

    _gtmparr[armci_me] = CLN_nic->lid_arr[armci_me];
    armci_vapi_server_stage1 = 1;
    armci_util_wait_int(&armci_vapi_client_stage1,1,10000);

    for(c=0; c< armci_nproc; c++){
       armci_connect_t *con = CLN_con + c;
       con->rqpnum = (VAPI_qp_num_t *)malloc(sizeof(VAPI_qp_num_t)*armci_nproc);
       bzero(con->rqpnum,sizeof(VAPI_qp_num_t)*armci_nproc);
       armci_create_qp(CLN_nic,&con->qp,&con->qp_prop);
       con->sqpnum  = con->qp_prop.qp_num;
       con->lid      = CLN_nic->lid_arr[c];
       con->rqpnum[armci_me]  = con->qp_prop.qp_num;
    }

    armci_vapi_server_stage2 = 1;

    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state = VAPI_INIT;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.pkey_ix  = DEFAULT_PKEY_IX;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.port     = DEFAULT_PORT;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PORT);
    qp_attr.remote_atomic_flags = VAPI_EN_REM_WRITE | VAPI_EN_REM_READ;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_REMOTE_ATOMIC_FLAGS);
    for(c=0; c< armci_nproc; c++){
       armci_connect_t *con = CLN_con + c;
       rc = VAPI_modify_qp(CLN_nic->handle,(con->qp),&qp_attr, &qp_attr_mask,
                           &qp_cap);
       armci_check_status(DEBUG_INIT, rc,"master connect request RST->INIT");
    }
    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state = VAPI_RTR;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.qp_ous_rd_atom = 2;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_OUS_RD_ATOM);
    qp_attr.av.sl            = 0;
    qp_attr.av.grh_flag      = FALSE;
    qp_attr.av.static_rate   = 0;                           /* 1x */
    qp_attr.av.src_path_bits = 0;
    qp_attr.path_mtu      = MTU1024;                        /*MTU*/
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PATH_MTU);
    qp_attr.rq_psn           = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RQ_PSN);
    qp_attr.pkey_ix = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.min_rnr_timer = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_MIN_RNR_TIMER);

    for(c=0; c< armci_nproc; c++){
       armci_connect_t *con = CLN_con + c;
       armci_connect_t *conC = SRV_con + armci_clus_me;
       qp_attr.dest_qp_num=conC->rqpnum[c];
       QP_ATTR_MASK_SET(qp_attr_mask, QP_ATTR_DEST_QP_NUM);
       qp_attr.av.dlid          = SRV_nic->lid_arr[c];
       QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_AV); 
       if(DEBUG_SERVER){
         printf("\n%d(s):connecting to %d rqp = %d dlid=%d\n",armci_me,c,
                  conC->rqpnum[c],qp_attr.av.dlid);fflush(stdout);
       }
       rc = VAPI_modify_qp(CLN_nic->handle,(con->qp),&qp_attr, &qp_attr_mask,
                           &qp_cap);
       armci_check_status(DEBUG_SERVER, rc,"master connect request INIT->RTR");
    }

    armci_util_wait_int(&armci_vapi_client_ready,1,10000);

    QP_ATTR_MASK_CLR_ALL(qp_attr_mask);
    qp_attr.qp_state   = VAPI_RTS;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_QP_STATE);
    qp_attr.sq_psn   = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_SQ_PSN);
    qp_attr.timeout   = 0x20;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_TIMEOUT);
    qp_attr.retry_count   = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RETRY_COUNT);
    qp_attr.rnr_retry     = 3;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RNR_RETRY);
    qp_attr.ous_dst_rd_atom  = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_OUS_DST_RD_ATOM);
    for(c=0; c< armci_nproc; c++){
       armci_connect_t *con = CLN_con + c;
       rc = VAPI_modify_qp(CLN_nic->handle,(con->qp),&qp_attr,&qp_attr_mask,
                           &qp_cap);
       armci_check_status(DEBUG_SERVER, rc,"master connect request RTR->RTS");
    }

    if(DEBUG_SERVER)
       printf("%d:server thread done with connections\n",armci_me);

    armci_server_alloc_bufs();/* create receive buffers for server thread */

 
    /* setup descriptors and post nonblocking receives */
    for(c = ib= 0; c < armci_nproc; c++) {
       vapibuf_t *vbuf = serv_buf_arr[c];
       armci_init_vapibuf_recv(&vbuf->dscr, &vbuf->sg_entry,vbuf->buf, 
                               VBUF_DLEN, &serv_memhandle);
       /* we use index of the buffer to identify the buffer, this index is
        * returned with a call to VAPI_poll_cq inside the VAPI_wc_desc_t */
       vbuf->dscr.id = c;
       if(DEBUG_SERVER){
         printf("\n%d(s):posted rr with lkey=%d",armci_me,vbuf->sg_entry.lkey);
         fflush(stdout);
       }
       rc = VAPI_post_rr(CLN_nic->handle,(CLN_con+c)->qp,&(vbuf->dscr));
       armci_check_status(DEBUG_SERVER, rc,"server post recv vbuf");
       
    }

    rc = EVAPI_set_comp_eventh(CLN_nic->handle,CLN_nic->rcq,
                               EVAPI_POLL_CQ_UNBLOCK_HANDLER,NULL,
                               &(CLN_nic->rcq_eventh));
    armci_check_status(DEBUG_SERVER, rc,"EVAPI_set_comp_eventh"); 

    armci_vapi_server_ready=1;
    /* check if we can poll in the server thread */
    enval = getenv("ARMCI_SERVER_CAN_POLL");
    if(enval != NULL){
       if((enval[0] != 'N') && (enval[0]!='n')) server_can_poll=1;
    } 
    else{
      if(armci_clus_info[armci_clus_me].nslave < armci_getnumcpus()) 
        server_can_poll=1;
    }


    /* establish connections with compute processes/clients */
    /*vapi_connect_server();*/

    if(DEBUG_SERVER){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);
    }
}


void armci_call_data_server()
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
vapibuf_t *vbuf,*vbufs;
request_header_t *msginfo;
int c,need_ack;

    for(;;){
       VAPI_wc_desc_t *pdscr=NULL;
       VAPI_wc_desc_t pdscr1;
       pdscr = &pdscr1;
       rc=VAPI_CQ_EMPTY;

       /*we just snoop to see if we have something */ 
       rc = VAPI_poll_cq(CLN_nic->handle, CLN_nic->rcq, pdscr);

       if(server_can_poll){
         while(rc == VAPI_CQ_EMPTY){
           rc = VAPI_poll_cq(CLN_nic->handle, CLN_nic->rcq, pdscr);
           if(armci_server_terminating){
             /* server got interrupted when clients terminate connections */
             sleep(1);
             _exit(0);
           }
         }
       }
       else {
         while(rc == VAPI_CQ_EMPTY){
           rc = EVAPI_poll_cq_block(CLN_nic->handle, CLN_nic->rcq, 0, pdscr);
           if(armci_server_terminating){
             /* server got interrupted when clients terminate connections */
             sleep(1);
             _exit(0);
           }
         }
       }

       armci_check_status(DEBUG_SERVER, rc,"server poll/block");
       /*we can figure out which buffer we got data info from the wc_desc_t id
        * this can tell us from which process we go the buffer, as well */

       vbuf = serv_buf_arr[pdscr->id];

       msginfo = (request_header_t*)vbuf->buf;
       armci_ack_proc = c = msginfo->from;

       vbufs = serv_buf_arr[pdscr->id] = spare_serv_buf;
       armci_init_vapibuf_recv(&vbufs->dscr, &vbufs->sg_entry,vbufs->buf, 
                               VBUF_DLEN, &serv_memhandle);
       vbufs->dscr.id = c;
       rc = VAPI_post_rr(CLN_nic->handle,(CLN_con+c)->qp,&(vbufs->dscr));
       armci_check_status(DEBUG_SERVER, rc,"server post recv vbuf");

       spare_serv_buf = vbuf; 

       if(DEBUG_SERVER){
         printf("%d(s):Came out of poll id=%d\n",armci_me,pdscr->id);
         fflush(stdout);
       }
       if(msginfo->operation == REGISTER){
          armci_server_register_region(*((void **)(msginfo+1)),
                           *((long *)((char *)(msginfo+1)+sizeof(void *))),
                           (ARMCI_MEMHDL_T *)(msginfo->tag.data_ptr));
          *(long *)(msginfo->tag.ack_ptr) = ARMCI_VAPI_COMPLETE;
          continue;
       }
       if((msginfo->operation == PUT) || ACC(msginfo->operation)){
         /* for operations that do not send data back we can send ACK now */
         SERVER_SEND_ACK(armci_ack_proc);
         need_ack=0;
       }else need_ack=1;

       armci_data_server(vbuf);

       if(DEBUG_SERVER){
         printf("%d(s):Done processed request\n",armci_me);
         fflush(stdout);
       }

      /*if ((msginfo->operation==GET)&&(PIPE_MIN_BUFSIZE<msginfo->datalen))
         armci_serv_clear_sends(); 
       if((msginfo->operation==GET && msginfo->bypass) && need_ack &&(armci_ack_proc != NONE)) SERVER_SEND_ACK(armci_ack_proc);*/
    }
}

char * armci_vapi_client_mem_alloc(int size)
{
VAPI_ret_t rc;
VAPI_mrw_t mr_in,mr_out;
int mod,  total;
int extra = MAX_DESCR*sizeof(VAPI_rr_desc_t)+SIXTYFOUR;
char *tmp,*tmp0;

    /*we use the size passed by the armci_init_bufs routine instead of bytes*/
    
    total = size + extra + 2*SIXTYFOUR;
    
    if(total%4096!=0)  
       total = total - (total%4096) + 4096;
    tmp0  = tmp = VMALLOC(total);
    if(ALIGN64ADD(tmp0))tmp0+=ALIGN64ADD(tmp0);
    if(!tmp) armci_die("failed to malloc client bufs",total);
    /* stamp the last byte */
    client_tail= tmp + extra+ size +2*SIXTYFOUR-1;
    *client_tail=CLIENT_STAMP;

    /* we also have a place to store memhandle for zero-copy get */
    pinned_handle =(armci_vapi_memhndl_t *) (tmp + extra+ size +SIXTYFOUR-16);

    mod = ((ssize_t)tmp)%SIXTYFOUR;
    client_descr_pool.descr= (VAPI_rr_desc_t*)(tmp+SIXTYFOUR-mod);
    tmp += extra;
    /*mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_ATOM | VAPI_EN_REMOTE_WRITE |VAPI_EN_REMOTE_READ;*/
    mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_WRITE | VAPI_EN_REMOTE_READ;
    mr_in.l_key = 0;
    mr_in.pd_hndl = SRV_nic->ptag;
    mr_in.r_key = 0;
    mr_in.size = total;
    mr_in.start = (VAPI_virt_addr_t)(MT_virt_addr_t)tmp0;
    mr_in.type = VAPI_MR;


    rc = VAPI_register_mr(SRV_nic->handle,&mr_in,&(client_memhandle.memhndl),
                          &mr_out);
    armci_check_status(DEBUG_INIT, rc,"client register snd vbuf");
    /*printf("\n%d(c):my lkey=%d",armci_me,mr_out.l_key);fflush(stdout);*/
    if(!client_memhandle.memhndl)armci_die("client got null handle for vbuf",0);

    client_memhandle.lkey = mr_out.l_key;
    client_memhandle.rkey = mr_out.r_key;
    handle_array[armci_me].lkey = client_memhandle.lkey;
    handle_array[armci_me].rkey = client_memhandle.rkey;
    handle_array[armci_me].memhndl = client_memhandle.memhndl;

    /* lock allocated memory */
    /*mattr.EnableRdmaWrite = VIP_FALSE;
    rc = VipRegisterMem(SRV_nic->handle, tmp0, total, &mattr,&client_memhandle);
    armci_check_status(DEBUG_INIT, rc,"client register snd vbuf");
    if(!client_memhandle)armci_die("client got null handle for vbuf",0); */
    if(DEBUG_INIT){
       printf("%d: registered client memory %p %dsize tmp=%p memhandle=%d\n",
               armci_me,tmp0, total, tmp,client_memhandle);
       fflush(stdout);

    }
    /*now that we have the handle array, we get every body elses RDMA handle*/
    total = (sizeof(armci_vapi_memhndl_t)*armci_nproc)/sizeof(int);
    armci_msg_gop_scope(SCOPE_ALL,handle_array,total,"+",ARMCI_INT);

    return(tmp);

}


void armci_transport_cleanup()
{

}



void armci_vapi_complete_buf(armci_vapi_field_t *field,int snd,int rcv,int to,
                             int op)
{
VAPI_sr_desc_t *snd_dscr;

BUF_INFO_T *info;
    info = (BUF_INFO_T *)((char *)field-sizeof(BUF_INFO_T));
    if(info->tag && op==GET)return;
    if(snd){
       snd_dscr=&(field->sdscr);
       if(mark_buf_send_complete[snd_dscr->id-1])
         mark_buf_send_complete[snd_dscr->id-1]=0;
       else
         armci_client_send_complete(snd_dscr,"armci_vapi_complete_buf");
    }
   
    if(rcv){
       int *last;
       long *flag;
       int loop = 0;
       request_header_t *msginfo = (request_header_t *)(field+1);
       flag = (long *)&msginfo->tag.ack;

       if(op==PUT || ACC(op)){
         while(armci_util_long_getval(flag) != ARMCI_VAPI_COMPLETE) {
         }
         *flag = 0L;
       }
       else{
         last = (int *)((char *)msginfo+msginfo->datalen-sizeof(int));
         while(armci_util_int_getval(last) == ARMCI_VAPI_COMPLETE &&
               armci_util_long_getval(flag)  != ARMCI_VAPI_COMPLETE){
           loop++;
           loop %=100000;
           if(loop==0){
             cpu_yield();
             if(DEBUG_CLN);{
               printf("%d: client last(%p)=%d flag(%p)=%ld off=%d\n",
                      armci_me,last,*last,flag,*flag,msginfo->datalen);
               fflush(stdout);
             }
           }
         }
       }
    }
}

static inline void armci_vapi_post_send(int isclient,int con_offset,
                                        VAPI_sr_desc_t *snd_dscr,char *from)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
vapi_nic_t *nic;
armci_connect_t *con;

    if(!isclient){
       nic = CLN_nic;
       con = CLN_con+con_offset;
    }
    else{
       nic = SRV_nic;
       con = SRV_con+con_offset;
    }
    if(snd_dscr->sg_lst_p->len>armci_vapi_max_inline_size)
       rc = VAPI_post_sr(nic->handle,con->qp,snd_dscr);
    else
       rc = EVAPI_post_inline_sr(nic->handle,con->qp,snd_dscr);
     
    armci_check_status(DEBUG_INIT, rc, from);
}


int armci_send_req_msg(int proc, void *buf, int bytes)
{
int cluster = armci_clus_id(proc);
request_header_t *msginfo = (request_header_t *)buf;
VAPI_sr_desc_t *snd_dscr;
VAPI_sg_lst_entry_t *ssg_lst; 

    snd_dscr = BUF_TO_SDESCR((char *)buf);
    ssg_lst  = BUF_TO_SSGLST((char *)buf);

    _armci_buf_ensure_one_outstanding_op_per_node(buf,cluster);  
    msginfo->tag.ack = 0;

    if(msginfo->operation == PUT || ACC(msginfo->operation))
       msginfo->tag.data_ptr = (void *)&msginfo->tag.ack;
    else {
       if(msginfo->operation == GET && !msginfo->bypass && msginfo->dscrlen >= (msginfo->datalen-sizeof(int)))
         msginfo->tag.data_ptr = (char *)(msginfo+1)+msginfo->dscrlen;
       else
         msginfo->tag.data_ptr = GET_DATA_PTR(buf);
    }
    msginfo->tag.ack_ptr = &(msginfo->tag.ack);

    armci_init_vapibuf_send(snd_dscr, ssg_lst,buf, 
                            bytes, &client_memhandle);

    armci_vapi_post_send(1,cluster,snd_dscr,"send_req_msg:post_send");

    if(DEBUG_CLN){
       printf("%d:client sent REQ=%d %d bytes serv=%d qp=%d remqp=%d id =%d lkey=%d\n",
               armci_me,msginfo->operation,bytes,cluster,
               (SRV_con+cluster)->qp,snd_dscr->remote_qp,snd_dscr->id,ssg_lst->lkey);
       fflush(stdout);
    }
    return(0);
}

void armci_client_direct_send(int p,void *src_buf, void *dst_buf, int len,void** contextptr,int nbtag,ARMCI_MEMHDL_T *lochdl,ARMCI_MEMHDL_T *remhdl)
{
sdescr_t *dirdscr;
int clus = armci_clus_id(p);

    /*ID for the desr that comes from get_next_descr is already set*/
    dirdscr = armci_vapi_get_next_descr(nbtag);
    if(nbtag)*contextptr = dirdscr;

    armci_init_vbuf_srdma(&dirdscr->descr,&dirdscr->sg_entry,src_buf,dst_buf,
                          len,lochdl,remhdl);

    armci_vapi_post_send(1,clus,&(dirdscr->descr),
                         "client_direct_send:post_send");

    if(!nbtag)
       armci_client_send_complete(&(dirdscr->descr),"armci_client_direct_send");
}

/*\ RDMA get 
\*/ 
void armci_client_direct_get(int p, void *src_buf, void *dst_buf, int len,
                             void** cptr,int nbtag,ARMCI_MEMHDL_T *lochdl,
                             ARMCI_MEMHDL_T *remhdl)
{      
VAPI_ret_t rc=VAPI_CQ_EMPTY;
sdescr_t *dirdscr;
int clus = armci_clus_id(p);

    /*ID for the desr that comes from get_next_descr is already set*/
    dirdscr = armci_vapi_get_next_descr(nbtag);
    if(nbtag)*cptr = dirdscr;

    if(DEBUG_CLN){
      printf("\n%d: in direct get lkey=%d rkey=%d\n",armci_me,lochdl->lkey,
               remhdl->rkey);fflush(stdout);
    }

    armci_init_vbuf_rrdma(&dirdscr->descr,&dirdscr->sg_entry,dst_buf,src_buf,
                          len,lochdl,remhdl);

    rc = VAPI_post_sr(SRV_nic->handle,(SRV_con+clus)->qp,&(dirdscr->descr));
    armci_check_status(DEBUG_CLN, rc,"client_get_direct, get");
    

    if(!nbtag)
       armci_client_send_complete(&(dirdscr->descr),"armci_client_direct_get");
}



char *armci_ReadFromDirect(int proc, request_header_t *msginfo, int len)
{
int cluster = armci_clus_id(proc);
vapibuf_ext_t* evbuf=BUF_TO_EVBUF(msginfo);
char *dataptr = GET_DATA_PTR(evbuf->buf);
extern void armci_util_wait_int(volatile int *,int,int);

    if(DEBUG_CLN){ printf("%d(c):read direct %d qp=%p\n",armci_me,
                len,&(SRV_con+cluster)->qp); fflush(stdout);
    }
   
    if(mark_buf_send_complete[evbuf->snd_dscr.id-1])
       mark_buf_send_complete[evbuf->snd_dscr.id-1]=0;
    else
       armci_client_send_complete(&(evbuf->snd_dscr),"armci_ReadFromDirect"); 
   
    if(!msginfo->bypass){
       long *flag;
       int *last;
       int loop = 0;
       flag = &(msginfo->tag.ack);
       if(msginfo->operation==GET){
         last = (int *)(dataptr+len-sizeof(int));
         if(msginfo->dscrlen >= (len-sizeof(int))){
           last = (int *)(dataptr+len+msginfo->dscrlen-sizeof(int));
           dataptr+=msginfo->dscrlen;
         }
         if(DEBUG_CLN){
           printf("\n%d: flagval=%d at ptr=%p ack=%ld dist=%d\n",armci_me,*last,
                   last,*flag,len);fflush(stdout);
                   
         }
         while(armci_util_int_getval(last) == ARMCI_VAPI_COMPLETE &&
               armci_util_long_getval(flag)  != ARMCI_VAPI_COMPLETE){
           loop++;
           loop %=100000;
           if(loop==0){
             cpu_yield();
             if(DEBUG_CLN){
               printf("%d: client last(%p)=%d flag(%p)=%ld off=%d\n",
                      armci_me,last,*last,flag,*flag,msginfo->datalen);
               fflush(stdout);
             }
           }
         }
         *flag = 0L;
       }
       else if(msginfo->operation == REGISTER){
         while(armci_util_long_getval(flag)  != ARMCI_VAPI_COMPLETE){
           loop++;
           loop %=100000;
           if(loop==0){
             cpu_yield();
             if(DEBUG_CLN){
               printf("%d: client last(%p)=%d flag(%p)=%ld off=%d\n",
                      armci_me,last,*last,flag,*flag,msginfo->datalen);
               fflush(stdout);
             }
           }
         }
       }
       else{
         int *flg = (int *)(dataptr+len);
         while(armci_util_int_getval(flg) != ARMCI_VAPI_COMPLETE){
           loop++;
           loop %=100000;
           if(loop==0){
             cpu_yield();
             if(DEBUG_CLN){
               printf("%d: client waiting (%p)=%d off=%d\n",
                      armci_me,flg,*flg,len);
               fflush(stdout);
             }
           }
         }
       }
    }
    return dataptr;
}

void armci_send_data_to_client(int proc, void *buf, int bytes,void *dbuf)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_sr_desc_t *sdscr;
VAPI_wc_desc_t *pdscr=NULL; 
VAPI_wc_desc_t pdscr1;

    pdscr = &pdscr1;
    sdscr=&serv_buf->snd_dscr;

    if(DEBUG_SERVER){
       printf("\n%d(s):sending data to client %d at %p flag = %p bytes=%d\n",
               armci_me,
               proc,dbuf,(char *)dbuf+bytes-sizeof(int),bytes);fflush(stdout);
    }

    memset(sdscr,0,sizeof(VAPI_sr_desc_t));
    armci_init_vbuf_srdma(sdscr,&serv_buf->ssg_entry,buf,dbuf,bytes,
                          &serv_memhandle,(handle_array+proc));

    if(DEBUG_SERVER){
       printf("\n%d(s):handle_array[%d]=%p dbuf=%p flag=%p bytes=%d\n",armci_me,
              proc,&handle_array[proc],(char *)dbuf,
              (char *)dbuf+bytes-sizeof(int),bytes);
       fflush(stdout);
    }

    serv_buf->snd_dscr.id = proc+armci_nproc;
    rc = VAPI_post_sr(CLN_nic->handle,(CLN_con+proc)->qp,&serv_buf->snd_dscr);
    armci_check_status(DEBUG_SERVER, rc,"server post send to client");

    rc=VAPI_CQ_EMPTY;
    while(rc == VAPI_CQ_EMPTY)
       rc = VAPI_poll_cq(CLN_nic->handle,CLN_nic->scq,pdscr);
    armci_check_status(DEBUG_SERVER, rc,"server wait post send to client");
    if(pdscr->id != proc+armci_nproc)
       armci_die2("server send data to client wrong dscr completed",pdscr->id,
                  proc+armci_nproc);

}


void armci_WriteToDirect(int proc, request_header_t* msginfo, void *buf)
{
int bytes;
int *last;
    bytes = (int)msginfo->datalen;
    if(DEBUG_SERVER){
      printf("%d(s):write to direct sent %d to %d at %p\n",armci_me,
             bytes,proc,(char *)msginfo->tag.data_ptr);
      fflush(stdout);
    }
    if(msginfo->operation!=GET){
       *(int *)((char *)buf+bytes)=ARMCI_VAPI_COMPLETE;
       bytes+=sizeof(int);
    }
    armci_send_data_to_client(proc,buf,bytes,msginfo->tag.data_ptr);
    /*if(msginfo->dscrlen >= (bytes-sizeof(int)))
       last = (int*)(((char*)(buf)) + (msginfo->dscrlen+bytes - sizeof(int)));
    else*/
       last = (int*)(((char*)(buf)) + (bytes - sizeof(int)));

    if(msginfo->operation==GET && *last == ARMCI_VAPI_COMPLETE)
       SERVER_SEND_ACK(msginfo->from);
    armci_ack_proc=NONE;
}


void armci_rcv_req(void *mesg,void *phdr,void *pdescr,void *pdata,int *buflen)
{
vapibuf_t *vbuf = (vapibuf_t*)mesg;
request_header_t *msginfo = (request_header_t *)vbuf->buf;
*(void **)phdr = msginfo;

    if(DEBUG_SERVER){
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


void armci_server_register_region(void *ptr,long bytes, ARMCI_MEMHDL_T *memhdl)
{
VAPI_ret_t rc;
VAPI_mrw_t mr_in,mr_out;
    bzero(memhdl,sizeof(ARMCI_MEMHDL_T));
    mr_in.acl =  VAPI_EN_LOCAL_WRITE | VAPI_EN_REMOTE_WRITE | VAPI_EN_REMOTE_READ;
    mr_in.l_key = 0;
    mr_in.pd_hndl = CLN_nic->ptag;
    mr_in.r_key = 0;
    mr_in.size = bytes;
    mr_in.start = (VAPI_virt_addr_t)(MT_virt_addr_t)ptr;
    mr_in.type = VAPI_MR;
    rc = VAPI_register_mr(CLN_nic->handle, &mr_in,&(serv_memhandle.memhndl), &mr_out);
    armci_check_status(DEBUG_INIT, rc,"server register region");
    memhdl->lkey = mr_out.l_key;
    memhdl->rkey = mr_out.r_key;
    if(DEBUG_SERVER){
       printf("\n%d(s):registered lkey=%d rkey=%d ptr=%p end=%p\n",armci_me,
               memhdl->lkey,memhdl->rkey,ptr,(char *)ptr+bytes);fflush(stdout);
    }
}



int armci_pin_contig_hndl(void *ptr, int bytes, ARMCI_MEMHDL_T *memhdl)
{
VAPI_ret_t rc;
VAPI_mrw_t mr_in,mr_out;
    
    mr_in.acl =  VAPI_EN_LOCAL_WRITE | VAPI_EN_REMOTE_WRITE | VAPI_EN_REMOTE_READ;
    mr_in.l_key = 0;
    mr_in.pd_hndl = SRV_nic->ptag;
    mr_in.r_key = 0;
    mr_in.size = bytes;
    mr_in.start = (VAPI_virt_addr_t)(MT_virt_addr_t)ptr;
    mr_in.type = VAPI_MR;
    rc = VAPI_register_mr(SRV_nic->handle, &mr_in,&(memhdl->memhndl), &mr_out);
    armci_check_status(DEBUG_INIT, rc,"client register region");
    memhdl->lkey = mr_out.l_key;
    memhdl->rkey = mr_out.r_key;
    if(DEBUG_CLN){
       printf("\n%d:registered lkey=%d rkey=%d ptr=%p end=%p\n",armci_me,
               memhdl->lkey,memhdl->rkey,ptr,(char *)ptr+bytes);fflush(stdout);
    }
    return 1;
}   

void armci_server_direct_send(int dst, char *src_buf, char *dst_buf, int len,
                              VAPI_lkey_t *lkey,VAPI_rkey_t *rkey)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_sr_desc_t *sdscr;
VAPI_wc_desc_t *pdscr=NULL;
VAPI_wc_desc_t pdscr1;

    pdscr = &pdscr1;
    sdscr=&serv_buf->snd_dscr;

    if(DEBUG_SERVER){
       printf("\n%d(s):sending dir data to client %d at %p bytes=%d last=%p\n",
                armci_me,dst,dst_buf,len,(dst_buf+len-4));fflush(stdout);
               
    }

    memset(sdscr,0,sizeof(VAPI_sr_desc_t));
    armci_init_vbuf_srdma(sdscr,&serv_buf->ssg_entry,src_buf,dst_buf,len,NULL,NULL);
    sdscr->r_key = *rkey;
    serv_buf->ssg_entry.lkey = *lkey;
                          
    serv_buf->snd_dscr.id = dst+armci_nproc;
    rc = VAPI_post_sr(CLN_nic->handle,(CLN_con+dst)->qp,&serv_buf->snd_dscr);
    armci_check_status(DEBUG_SERVER, rc,"server post sent dir data to client");

    while(rc == VAPI_CQ_EMPTY)
       rc = VAPI_poll_cq(CLN_nic->handle,CLN_nic->scq,pdscr);
    armci_check_status(DEBUG_SERVER, rc,"server poll sent dir data to client");

}



void armci_send_contig_bypass(int proc, request_header_t *msginfo,
                              void *src_ptr, void *rem_ptr, int bytes)
{
int *last;
VAPI_lkey_t *lkey=NULL;
VAPI_rkey_t *rkey;    
int dscrlen = msginfo->dscrlen;

    last = (int*)(((char*)(src_ptr)) + (bytes - sizeof(int)));
    if(!msginfo->pinned)armci_die("armci_send_contig_bypass: not pinned",proc);

    rkey = (VAPI_rkey_t *)((char *)(msginfo+1)+dscrlen-(sizeof(VAPI_rkey_t)+sizeof(VAPI_lkey_t)));

    if(DEBUG_SERVER){
       printf("%d(server): sending data bypass to %d (%p,%p) %d %d\n", armci_me,
               msginfo->from,src_ptr, rem_ptr,*lkey,*rkey);
       fflush(stdout);
    }
    armci_server_direct_send(msginfo->from,src_ptr,rem_ptr,bytes,lkey,rkey);

    if(*last == ARMCI_VAPI_COMPLETE){
       SERVER_SEND_ACK(msginfo->from);
    }
}

#if 0
void armci_send_strided_data_bypass(int proc, request_header_t *msginfo,
                                    void *loc_buf, int msg_buflen,
                                    void *loc_ptr, int *loc_stride_arr,
                                    void *rem_ptr, int *rem_stride_arr,
                                    int *count, int stride_levels)
{
    int i, j;
    long loc_idx, rem_idx;
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL];

    int to = msginfo->from;
    char *buf;
    int buflen;
    int msg_threshold;

    if(stride_levels==0 && msginfo->pinned){
      armci_send_contig_bypass(proc,msginfo,loc_ptr,rem_ptr,count[0]);
      return;
    }
    else {
      printf("\n %d ***SOMETHING TERRIBLY WRONG IN send_strided_data_bypass***",
              armci_me);fflush(stdout);
    }
}
#endif

void armci_rcv_strided_data_bypass_both(int proc, request_header_t *msginfo,
                                       void *ptr, int *count, int stride_levels)
{
int datalen = msginfo->datalen;
int *last;
long *ack;
int loop=0;

    if(DEBUG_CLN){ printf("%d:rcv_strided_data_both bypass from %d\n",
                armci_me,  proc); fflush(stdout);
    }

    last = (int*)(((char*)(ptr)) + (count[0] -sizeof(int)));
    ack  = (long *)&msginfo->tag;
    while(armci_util_int_getval(last) == ARMCI_VAPI_COMPLETE &&
          armci_util_long_getval(ack)  != ARMCI_VAPI_COMPLETE){
          loop++;
          loop %=1000000;
          if(loop==0){cpu_yield();
            if(DEBUG_CLN){
               printf("%d: client last(%p)=%d ack(%p)=%ld off=%d\n",
                      armci_me,last,*last,ack,*ack,(char*)last - (char*)ptr);
               fflush(stdout);
            }
          }
    }

    if(DEBUG_CLN){printf("%d:rcv_strided_data bypass both: %d bytes from %d\n",
                          armci_me, datalen, proc); fflush(stdout);
    }
}


int armci_pin_memory(void *ptr, int stride_arr[], int count[], int strides)
{
    printf("\n%d:armci_pin_memory not implemented",armci_me);fflush(stdout);
    return 0;
}    


void armci_client_send_ack(int proc, int n)
{
    printf("\n%d:client_send_ack not implemented",armci_me);fflush(stdout);
}


void armci_rcv_strided_data_bypass(int proc, request_header_t* msginfo,
                                   void *ptr, int stride_levels)
{
    printf("\n%d:armci_rcv_strided_data_bypass not implemented",armci_me);
    fflush(stdout);
}


void armci_unpin_memory(void *ptr, int stride_arr[], int count[], int strides)
{
    printf("\n%d:armci_unpin_memory not implemented",armci_me);fflush(stdout);
}


int armcill_server_wait_ack(int proc, int n)
{ 
    printf("\n%d:armcill_server_wait_ack not implemented",armci_me);
    fflush(stdout);
    return(0);
}


void armcill_server_put(int proc, void* s, void *d, int len)
{
    printf("\n%d:armcill_server_put not implemented",armci_me);fflush(stdout);
}
