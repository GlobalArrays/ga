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
/*vapi includes*/
#include "/usr/mellanox/include/vapi.h"
#include "/usr/mellanox/include/evapi.h"
#include "/usr/mellanox/include/ib_defs.h"
#include "/usr/mellanox/include/vapi_common.h"
#include "/usr/mellanox/include/mtl_common.h"
/*our incude*/
#include "armci-vapi.h"
#define DEBUG_INIT 0
#define DEBUG_SERVER 0
#define DEBUG_CLN 0
#define DIRTMP_BUF_LEN 4096
static char *dirtmp_buf;
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
  int maxtransfersize; 
} vapi_nic_t;

typedef struct {
  VAPI_lkey_t lkey;
  VAPI_rkey_t rkey;
  VAPI_mr_hndl_t memhndl;
}armci_vapi_memhndl_t;

typedef struct {
  armci_vapi_memhndl_t *prem_handle; /*address server to store memory handle*/ 
  armci_vapi_memhndl_t handle;
}ack_t;

armci_vapi_memhndl_t *CLN_handle;
armci_vapi_memhndl_t serv_memhandle, client_memhandle;
armci_vapi_memhndl_t *handle_array;
armci_vapi_memhndl_t *pinned_handle;

static vapi_nic_t nic_arr[2];
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
    int avail; VAPI_qp_hndl_t vi; VAPI_rr_desc_t *descr;
} descr_pool_t;

static int* _gtmparr;

char *MessageRcvBuffer;

extern void armci_util_wait_int(volatile int *,int,int);
void armci_send_data_to_client(int proc, void *buf,int bytes,void *dbuf);

static descr_pool_t serv_descr_pool = {MAX_DESCR, 0, (VAPI_rr_desc_t *)0};
static descr_pool_t client_descr_pool = {MAX_DESCR,0,(VAPI_rr_desc_t *)0};

#define GET_DATA_PTR(buf) (sizeof(request_header_t) + (char*)buf)

#define BUF_TO_SDESCR(buf) ((VAPI_sr_desc_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->sdscr))

#define BUF_TO_RDESCR(buf) ((VAPI_rr_desc_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->rdscr))

#define BUF_TO_SSGLST(buf) ((VAPI_sg_lst_entry_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->ssg_entry))

#define BUF_TO_RSGLST(buf) ((VAPI_sg_lst_entry_t *)(&((armci_vapi_field_t *)((char *)(buf) - sizeof(armci_vapi_field_t)))->rsg_entry))

#define BUF_TO_EVBUF(buf) (vapibuf_ext_t*)(((char*)buf) - (sizeof(VAPI_sr_desc_t)+sizeof(VAPI_rr_desc_t)+2*sizeof(VAPI_sg_lst_entry_t)))

#define SERVER_SEND_ACK(p) armci_send_data_to_client((p),serv_buf->buf,0,msginfo->tag)



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

    /*attributes for qpprop*/ 
    /*qp_prop->EnableRdmaWrite  = VIP_TRUE;*/
    /*qp_prop.MaxTransferSize  = nic->attr.MaxTransferSize;*/

    rc = VAPI_create_qp(nic->handle, &initattr, qp, qp_prop);

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
int clients = armci_nproc - armci_clus_info[armci_clus_me].nslave,i,j=0;

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
       if(SAMECLUSNODE(i))serv_buf_arr[i] = NULL;
       else serv_buf_arr[i] = (vapibuf_t*)(tmp+SIXTYFOUR-mod) + j++;
    }
    i=0;
    while(serv_buf_arr[i]==NULL)i++;
    spare_serv_buf = serv_buf_arr[i]+clients; /* spare buffer is at the end */
    spare_serv_bufptr = spare_serv_buf;    /* save the pointer for later */
    serv_buf =(vapibuf_ext_t*)(serv_buf_arr[i]+clients+1);
    MessageRcvBuffer = serv_buf->buf;
    /* setup memory attributes for the region */
    /*mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_ATOM | VAPI_EN_REMOTE_WRITE |VAPI_EN_REMOTE_READ;*/
    mr_in.acl =  VAPI_EN_LOCAL_WRITE | VAPI_EN_REMOTE_WRITE;
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
VAPI_qp_attr_t         qp_attr;
VAPI_qp_cap_t          qp_cap;
VAPI_qp_attr_mask_t    qp_attr_mask;
VAPI_ret_t rc;
int *tmparr;
    
    /* initialize nic connection for qp numbers and lid's */
    armci_init_nic(SRV_nic,1,1);

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
    for(s=0; s< armci_nclus; s++){
       armci_connect_t *con = SRV_con + s;
       con->rqpnum = (VAPI_qp_num_t *)malloc(sizeof(VAPI_qp_num_t)*armci_nproc);
       bzero(con->rqpnum,sizeof(VAPI_qp_num_t)*armci_nproc);
       if(armci_clus_me != s){
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
int s,i,start,sz=0,c;
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
    for(i=0; i< armci_nclus; i++)if(i!=armci_clus_me){
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
    qp_attr.av.static_rate   = 1; /* 1x */
    qp_attr.av.src_path_bits = 0;
    qp_attr.path_mtu         = MTU1024;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PATH_MTU);
    qp_attr.rq_psn           = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RQ_PSN);
    qp_attr.pkey_ix = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.min_rnr_timer = 5;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_MIN_RNR_TIMER);

    start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;
    for(i=0; i< armci_nclus; i++)if(i!=armci_clus_me){
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
    qp_attr.timeout   = 5;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_TIMEOUT);
    qp_attr.retry_count   = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RETRY_COUNT);
    qp_attr.rnr_retry     = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RNR_RETRY);
    qp_attr.ous_dst_rd_atom  = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_OUS_DST_RD_ATOM);

    start = (armci_clus_me==0)? armci_nclus-1 : armci_clus_me-1;
    for(i=0; i< armci_nclus; i++)if(i!=armci_clus_me){
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
     memset(sd,0,sizeof(VAPI_sr_desc_t));
     sd->opcode = VAPI_SEND;
     sd->comp_type = VAPI_SIGNALED;
     sd->sg_lst_len = 1;
     sd->sg_lst_p  = sg_entry;
     sd->id = 0;
     sd->remote_qkey=0;
     sd->set_se = FALSE;
     sd->fence = FALSE;

     sg_entry->lkey = mhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)buf;
     sg_entry->len = len;
}


static void armci_init_vbuf_srdma(VAPI_sr_desc_t *sd, VAPI_sg_lst_entry_t *sg_entry, char* lbuf, char *rbuf, int len, armci_vapi_memhndl_t *lhandle,armci_vapi_memhndl_t *rhandle)
{
     memset(sd,0,sizeof(VAPI_sr_desc_t));
     sd->opcode = VAPI_RDMA_WRITE;
     sd->comp_type = VAPI_SIGNALED;
     sd->sg_lst_len = 1;
     sd->sg_lst_p  = sg_entry;
     sd->id = 0;
     sd->remote_qkey=0;
     sd->r_key = rhandle->rkey;
     sd->remote_addr = (VAPI_virt_addr_t) (MT_virt_addr_t) rbuf;
     sd->set_se = FALSE;
     sd->fence = FALSE;

     sg_entry->lkey = lhandle->lkey;
     sg_entry->addr = (VAPI_virt_addr_t)(MT_virt_addr_t)lbuf;
     sg_entry->len = len;
}


void armci_server_initial_connection()
{
int c, ib;
int clients = armci_nproc - armci_clus_info[armci_clus_me].nslave;
VAPI_ret_t rc;
VAPI_qp_attr_t         qp_attr;
VAPI_qp_cap_t          qp_cap;
VAPI_qp_attr_mask_t    qp_attr_mask;

    if(DEBUG_SERVER){ 
       printf("in server after fork %d (%d)\n",armci_me,getpid());
       fflush(stdout);
    }

    armci_init_nic(CLN_nic,2,2);
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
    for(c=0; c< armci_nproc; c++)if(!SAMECLUSNODE(c)){
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
    qp_attr.av.static_rate   = 1;                           /* 1x */
    qp_attr.av.src_path_bits = 0;
    qp_attr.path_mtu      = MTU1024;                        /*MTU*/
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PATH_MTU);
    qp_attr.rq_psn           = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RQ_PSN);
    qp_attr.pkey_ix = 0;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_PKEY_IX);
    qp_attr.min_rnr_timer = 5;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_MIN_RNR_TIMER);

    for(c=0; c< armci_nproc; c++)if(!SAMECLUSNODE(c)){
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
    qp_attr.timeout   = 5;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_TIMEOUT);
    qp_attr.retry_count   = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RETRY_COUNT);
    qp_attr.rnr_retry     = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_RNR_RETRY);
    qp_attr.ous_dst_rd_atom  = 1;
    QP_ATTR_MASK_SET(qp_attr_mask,QP_ATTR_OUS_DST_RD_ATOM);
    for(c=0; c< armci_nproc; c++)if(!SAMECLUSNODE(c)){
       armci_connect_t *con = CLN_con + c;
       rc = VAPI_modify_qp(CLN_nic->handle,(con->qp),&qp_attr,&qp_attr_mask,
                           &qp_cap);
       armci_check_status(DEBUG_SERVER, rc,"master connect request RTR->RTS");
    }

    if(DEBUG_SERVER)
       printf("%d:server thread done with connections\n",armci_me);

    armci_server_alloc_bufs();/* create receive buffers for server thread */

 
    /* setup descriptors and post nonblocking receives */
    for(c = ib= 0; c < armci_nproc; c++) if(!SAMECLUSNODE(c)){
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

    armci_vapi_server_ready=1;

    /* establish connections with compute processes/clients */
    /*vapi_connect_server();*/

    if(DEBUG_SERVER){
       printf("%d: server connected to all clients\n",armci_me); fflush(stdout);
    }
}


void armci_call_data_server()
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_wc_desc_t *pdscr=NULL;
VAPI_wc_desc_t pdscr1;
vapibuf_t *vbuf,*vbufs;
request_header_t *msginfo;
int c,need_ack;

    for(;;){
       pdscr = &pdscr1;
       rc=VAPI_CQ_EMPTY;
       while(rc == VAPI_CQ_EMPTY){
         rc = VAPI_poll_cq(CLN_nic->handle, CLN_nic->rcq, &pdscr1);
         if(armci_server_terminating){
           /* server got interrupted when clients terminate connections */
           sleep(1);
           _exit(0);
         }
       }
       armci_check_status(DEBUG_SERVER, rc,"server poll recv got something");
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
         printf("%d:Came out of poll \n",armci_me);
         fflush(stdout);
       }
       if((msginfo->operation == PUT) || ACC(msginfo->operation)){
         /* for operations that do not send data back we can send ACK now */
         SERVER_SEND_ACK(armci_ack_proc);
         need_ack=0;
       }else need_ack=1;

       armci_data_server(vbuf);

       if(DEBUG_SERVER){
         printf("%d:Done processed request\n",armci_me);
         fflush(stdout);
       }

      /*if ((msginfo->operation==GET)&&(PIPE_MIN_BUFSIZE<msginfo->datalen))
         armci_serv_clear_sends(); */
         /* for now we complete all pending sends */

       /* flow control: send ack for this request since no response was sent */
       /* msginfo->operation===GET&&.... added because VAPI get has been
          modified to use RDMA, it is not necessary to send an ack for get as
          the flag used to indicate completion of get is an ack in itself.
          But,the pinning protocols still posts a rcv_dscr for ack.
       */
       if((msginfo->operation==GET && msginfo->bypass) && need_ack &&(armci_ack_proc != NONE)) SERVER_SEND_ACK(armci_ack_proc);
       bzero(&pdscr1,sizeof(VAPI_wc_desc_t));
    }

}

char * armci_vapi_client_mem_alloc(int size)
{
VAPI_ret_t rc;
VAPI_mrw_t mr_in,mr_out;
int mod,  total;
int extra = MAX_DESCR*sizeof(VAPI_rr_desc_t)+SIXTYFOUR+DIRTMP_BUF_LEN;
char *tmp,*tmp0;

    /*we use the size passed by the armci_init_bufs routine instead of bytes*/
    
    total = size + extra + 2*SIXTYFOUR;
    
    if(total%4096!=0)  
       total = total - (total%4096) + 4096;
    tmp0  = tmp = VMALLOC(total);

    if(!tmp) armci_die("failed to malloc client bufs",total);
    /* stamp the last byte */
    client_tail= tmp + extra+ size +2*SIXTYFOUR-1;
    *client_tail=CLIENT_STAMP;

    dirtmp_buf = tmp0+ size + extra + 2*SIXTYFOUR - DIRTMP_BUF_LEN; 

    /* we also have a place to store memhandle for zero-copy get */
    pinned_handle =(armci_vapi_memhndl_t *) (tmp + extra+ size +SIXTYFOUR-16);

    mod = ((ssize_t)tmp)%SIXTYFOUR;
    client_descr_pool.descr= (VAPI_rr_desc_t*)(tmp+SIXTYFOUR-mod);
    tmp += extra;
    /*mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_ATOM | VAPI_EN_REMOTE_WRITE |VAPI_EN_REMOTE_READ;*/
    mr_in.acl =  VAPI_EN_LOCAL_WRITE|VAPI_EN_REMOTE_WRITE;
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
    /*
    if(SERVER_CONTEXT) if(*serv_tail != SERV_STAMP){
       printf("%d: server_stamp %d %d\n",armci_me,*serv_tail, SERV_STAMP);
       armci_die("ARMCI Internal Error: end-of-buffer overwritten",0);
    }
    if(!SERVER_CONTEXT) if(*client_tail != CLIENT_STAMP){
       printf("%d: client_stamp %d %d\n",armci_me,*client_tail, CLIENT_STAMP);
       armci_die("ARMCI Internal Error: end-of-buffer overwritten",0); 
    }
    */
}



void armci_vapi_complete_buf(armci_vapi_field_t *field,int snd,int rcv,int to,
                             int op)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_sr_desc_t *snd_dscr;
VAPI_rr_desc_t *rcv_dscr;
VAPI_wc_desc_t pdscr1;
VAPI_wc_desc_t *pdscr=&pdscr1;
VAPI_wc_desc_t pdscr0;
VAPI_wc_desc_t *pdscr2=&pdscr0;

BUF_INFO_T *info;
    info = (BUF_INFO_T *)((char *)field-sizeof(BUF_INFO_T));
    if(info->tag && op==GET)return;
    if(snd){
       snd_dscr=&(field->sdscr);
       while(rc == VAPI_CQ_EMPTY)  
         rc = VAPI_poll_cq(SRV_nic->handle, SRV_nic->scq, pdscr);
       armci_check_status(DEBUG_CLN,rc,"complete_buf: wait for send complete");
       if(pdscr->id!=armci_me){
         printf("%d(c): DIFFERENT recv DESCRIPTOR %d %d buf= \n",
                armci_me,pdscr->id ,armci_me);
         fflush(stdout);
         armci_die("armci_via_complete_buf: wrong dscr completed",0);
       }
    }
   
    if(rcv){
       request_header_t *msginfo = (request_header_t *)(field+1);
       if(op==PUT || ACC(op))
         armci_util_wait_int((int *)&msginfo->tag,1,10000);
       else
         armci_util_wait_int((int *)((char *)msginfo+msginfo->datalen),1,10000);
       /*
       if(op==PUT || ACC(op))
       while ((int)msginfo->tag!=1);
       else
       while (*(int *)((char *)msginfo+msginfo->datalen)!=1);
       */

    }
}


static void armci_client_post_buf(int srv, void *buf)
{
VAPI_ret_t rc;
char *dataptr = GET_DATA_PTR(buf);
VAPI_rr_desc_t *rcv_dscr;
VAPI_sg_lst_entry_t *rsg_lst; 
    rcv_dscr = BUF_TO_RDESCR((char *)buf);     
    rsg_lst  = BUF_TO_RSGLST((char *)buf);
    armci_init_vapibuf_recv(rcv_dscr,rsg_lst,dataptr,MSG_BUFLEN,
                            &client_memhandle);
    rcv_dscr->id = armci_nproc+armci_me;
    rc = VAPI_post_rr(SRV_nic->handle,(SRV_con+srv)->qp,rcv_dscr);
    armci_check_status(DEBUG_CLN, rc,"client prepost vbuf");
}

int armci_send_req_msg(int proc, void *buf, int bytes)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
int cluster = armci_clus_id(proc);
request_header_t *msginfo = (request_header_t *)buf;
VAPI_sr_desc_t *snd_dscr;
VAPI_sg_lst_entry_t *ssg_lst; 
VAPI_wc_desc_t pdscr1;
VAPI_wc_desc_t *pdscr=&pdscr1;
    snd_dscr = BUF_TO_SDESCR((char *)buf);
    ssg_lst  = BUF_TO_SSGLST((char *)buf);

    /* flow control for remote server - we cannot send if send/recv descriptor
     * is pending for that node -- we will complete these descriptors */
    _armci_buf_ensure_one_outstanding_op_per_node(buf,cluster);  

    if(msginfo->operation==GET && !msginfo->bypass){
       msginfo->tag = (char *)(msginfo+1)+msginfo->dscrlen;
       *(int *)((char *)(msginfo+1)+msginfo->datalen+msginfo->dscrlen)=0;
       if(DEBUG_CLN){
         printf("\n%d:op=get dst %d and tag=%p flagset=%p at dist=%d\n",
                armci_me,proc,msginfo->tag,
                ((char *)(msginfo+1)+msginfo->datalen+msginfo->dscrlen),
                msginfo->datalen);
         fflush(stdout);
       }        
    }
    else{
       if(msginfo->operation==PUT || ACC(msginfo->operation))
         msginfo->tag = (void *)&msginfo->tag;
       else 
         msginfo->tag = GET_DATA_PTR(buf);
       /*armci_client_post_buf(cluster,buf); */
    }

    armci_init_vapibuf_send(snd_dscr, ssg_lst,buf, 
                            bytes, &client_memhandle);
    snd_dscr->id = armci_me;
    /*snd_dscr->remote_qp =  (CLN_con + armci_me)->rqpnum[cluster];*/
    rc = VAPI_post_sr(SRV_nic->handle,(SRV_con+cluster)->qp,snd_dscr);
    armci_check_status(DEBUG_INIT, rc,"client post send");

    if(DEBUG_CLN){
       printf("%d:client sent REQ=%d %d bytes serv=%d qp=%d remqp=%d lkey=%d\n",
               armci_me,msginfo->operation,bytes,cluster,
               (SRV_con+cluster)->qp,snd_dscr->remote_qp,ssg_lst->lkey);
       fflush(stdout);
    }
    /*
    while(rc == VAPI_CQ_EMPTY)
       rc = VAPI_poll_cq(SRV_nic->handle, SRV_nic->scq, pdscr);
    armci_check_status(DEBUG_INIT, rc,"RFD client send complete"); 
    return(0);
    */
}


char *armci_ReadFromDirect(int proc, request_header_t *msginfo, int len)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_sr_desc_t *sdscr;
VAPI_rr_desc_t *rdscr;
VAPI_wc_desc_t pdscr1;
VAPI_wc_desc_t *pdscr=&pdscr1;
int cluster = armci_clus_id(proc);
vapibuf_ext_t* evbuf=BUF_TO_EVBUF(msginfo);
char *dataptr = GET_DATA_PTR(evbuf->buf);
extern void armci_util_wait_int(volatile int *,int,int);

    if(DEBUG_CLN){ printf("%d(c):read direct %d qp=%p\n",armci_me,
                len,&(SRV_con+cluster)->qp); fflush(stdout);
    }
    
    while(rc == VAPI_CQ_EMPTY)
    rc = VAPI_poll_cq(SRV_nic->handle, SRV_nic->scq, pdscr);
    armci_check_status(DEBUG_INIT, rc,"RFD client send complete"); 
   
#if 0
    if(msginfo->operation!=GET)
    {
       rc=VAPI_CQ_EMPTY;
       while(rc == VAPI_CQ_EMPTY)
         rc = VAPI_poll_cq(SRV_nic->handle, SRV_nic->rcq, pdscr);
       armci_check_status(DEBUG_INIT, rc,"server post recv vbuf"); 
       if(pdscr->id!=armci_me+armci_nproc){
         printf("%d(c):ReadFrom-WRONG RECV DSCR got %p instead of %p buf=%p\n",
                armci_me,pdscr ,&evbuf->rcv_dscr, msginfo);fflush(stdout);
         armci_die("reading data client-different descriptor completed",0);
       }
    }
#endif
 
    if(msginfo->tag && !msginfo->bypass){
       if(msginfo->operation==GET){
         volatile int *flg = (volatile int *)(dataptr+len+msginfo->dscrlen);
         if(DEBUG_CLN){
           printf("\n%d:B flagval=%d at ptr=%p dist=%d\n",armci_me,*flg,flg,len);
           fflush(stdout);
         }
         armci_util_wait_int(flg,1,10000);
         return (dataptr+msginfo->dscrlen);
       }
       else{
         volatile int *flg = (volatile int *)(dataptr+len);
         if(DEBUG_CLN){
           printf("\n%d: flagval=%d at ptr=%p dist=%d\n",armci_me,*flg,flg,len);
           fflush(stdout);
         }
         armci_util_wait_int(flg,1,10000);
         return (dataptr+msginfo->dscrlen);
       }
    }
    return dataptr;
}

void armci_send_data_to_client(int proc, void *buf, int bytes,void *dbuf)
{
VAPI_ret_t rc=VAPI_CQ_EMPTY;
VAPI_sr_desc_t *sdscr;
VAPI_rr_desc_t *rdscr;
VAPI_wc_desc_t *pdscr=NULL; 
VAPI_wc_desc_t pdscr1;

    pdscr = &pdscr1;
    sdscr=&serv_buf->snd_dscr;
    memset(sdscr,0,sizeof(VAPI_sr_desc_t));
    if(DEBUG_SERVER){
       printf("\n%d(s):sending data to client %d at %p bytes=%d\n",armci_me,
               proc,dbuf,bytes);fflush(stdout);
    }

    *(int *)((char *)buf+bytes)=1;
    bytes+=sizeof(int);
    armci_init_vbuf_srdma(sdscr,&serv_buf->ssg_entry,buf,dbuf,bytes,
                          &serv_memhandle,(handle_array+proc));
    if(DEBUG_SERVER){
       printf("\n%d(s):handle_array[%d]=%p dbuf=%p flag=%p bytes=%d\n",armci_me,
              proc,&handle_array[proc],(char *)dbuf,
              (char *)dbuf+bytes-sizeof(int),bytes);
       fflush(stdout);
    }
    /*armci_init_vapibuf_send(&serv_buf->snd_dscr, &serv_buf->ssg_entry,buf,8,
                            &serv_memhandle);
    */
    serv_buf->snd_dscr.id = proc+armci_nproc;
    rc = VAPI_post_sr(CLN_nic->handle,(CLN_con+proc)->qp,&serv_buf->snd_dscr);
    if(bytes)
       armci_check_status(DEBUG_SERVER, rc,"server sent data to client");
    else
       armci_check_status(DEBUG_SERVER, rc,"server sent ack to client");

    rc=VAPI_CQ_EMPTY;
    while(rc == VAPI_CQ_EMPTY)
       rc = VAPI_poll_cq(CLN_nic->handle,CLN_nic->scq,pdscr);
    armci_check_status(DEBUG_SERVER, rc,"server sent data to client complete");
    if(pdscr->id != proc+armci_nproc)
       armci_die2("server send data to client wrong dscr completed",pdscr->id,
                  proc+armci_nproc);
}

void armci_WriteToDirect(int proc, request_header_t* msginfo, void *buf)
{
int bytes;
    bytes = (int)msginfo->datalen;
    if(DEBUG_SERVER){
      printf("%d(s):write to direct sent %d to %d at %p\n",armci_me,
             bytes,proc,(char *)msginfo->tag);
      fflush(stdout);
    }
    armci_send_data_to_client(proc,buf,bytes,msginfo->tag);
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
}


void armcill_server_put(int proc, void* s, void *d, int len)
{
    printf("\n%d:armcill_server_put not implemented",armci_me);fflush(stdout);
}
