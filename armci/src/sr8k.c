/* $Id: sr8k.c,v 1.5 2002-03-13 17:09:35 vinod Exp $
 *
 * Hitachi SR8000 specific code 
 *
 * *** WE NEED TO OPTIMIZE armcill_put/get AND armcill_put2D/get2D ******* 
 * *** latency by using TCW and combuf_kick_tcw_fast()
 * *** bandwidth by overlapping memory copy with RDMA nonblocking communication
 *
 * Optimisations performed:
 * [BPE, Hitachi, 01/11/01]
 * (0) Increase the value of MSG_BUFLEN_DBL in request.h
 * (1) Reuse tcws for the put operation
 * (2) Pipeline contiguous put with memory copy
 * (3) Pipeline contiguous get with memory copy
 * (4) Use combuf_stride_send() for strided put
 * LGET and LPUT are tunable parameters for the pipelining thresholds.
 */

#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <strings.h>
#include <stdlib.h>
#include "armcip.h"
#include "copy.h"
#include "locks.h"
#include "shmem.h"

/* data structure used to store ptr/size/authorization info for
   every shared memory (combuf) area on every other smp node
   -- we only need it for one process on each smp node - "master"
*/
typedef struct{
    int pauth;   /* segment descriptor for send/put operation*/
    int gauth;   /* segment descriptor for get operation*/
    long size;   /* segment size */
    char* ptr;   /* address of the segment */
    int tcwd;    /* transmission control word for put operation */
}reg_auth_t;

char *directbuffer;
rcv_field_info_t *client_rcv_field_info;
int *client_ready_flag;
int _sr8k_vector_req=0;
int *client_pending_op_count, *server_pending_op_count;
int *server_smallbuf_pending_count,*client_smallbuf_pending_count;
char *serverpendingopflag;
int clientpendingopbufdesc;
int *_sr8k_smallbuf_insync;
int *_sr8k_largebuf_insync;
double *armci_clientmap_common_buffer;
char *client_ack_buf;
client_auth_t *client_auths;
int *server_ready_flag; 
static long reg_idlist[MAX_REGIONS]; /* keys for rdma/shmem segments on this node */
static long creg_idlist[MAX_REGIONS];
static int  reg_num=0;          /* number of rdma/shmem segments on this node */
static int  creg_num=0;
static reg_auth_t *rem_reg_list;/* info about rdma segments on every node */
static reg_auth_t **rem_reg;    /* info about rdma segments on every node */
static long *reg_id;            /* buffer to exchange info about new segments */
static int  *proc_reg;          /* count of shmem/rdma segments on every node */
extern ndes_t _armci_group;     /* processor group -- set in clusterinfo */
double *armci_internal_buffer;  /* work buffer for accumulate -- in RDMA area */
static char *bufstart, *bufend; /* address range for local rdma area */
static int bufdesc,cbufdesc;             /* descriptor for local rdma buffer */

int current_clientbuf_tcwd;
static int descs[2];            /* descriptors for get pipeline buffers */
static char *getbuf[2];         /* the get pipeline buffers themselves */
char *client_buf_array[SMALL_BUFS_PER_PROCESS];
static char *client_buf_flags[SMALL_BUFS_PER_PROCESS];
static int client_buf_desc_array[SMALL_BUFS_PER_PROCESS];
int sr8k_server_ready=0;
#define DEBUG0 0
#define DEBUG2 0
#define DEBUG1 0

#define ALIGNUP(x) (4*((3+x)/4))

#define BUF_TO_TCWD(bu) ((int *)((int *)bu-1))


/**TCW manipulation routines****************************************/

      /**** Routines for PUT ****/

static char *tcw_flag;

int armci_rdma_make_tcw(void *src, Cb_size_t off, int bytes, int desc, char *flag)
{
int rc;
Cb_msg msg;
int tcwd;
     memset(&msg, 0, sizeof(msg));
     msg.data.addr = src;
     msg.data.size = (Cb_size_t)bytes;
     if ( (rc=combuf_make_tcw(&msg, desc, off, flag, &tcwd)) != COMBUF_SUCCESS)
         armci_die("combuf_make_tcw failed",rc);
     if(DEBUG0){printf("%d:put dsc=%d off=%d giving tcw=%d\n",armci_me,desc,off,tcwd); fflush(stdout);}
     return tcwd;
}

void armci_rdma_modify_tcw(int tcwd, char *src, Cb_size_t off, int bytes)
{
int rc;
int modfunc = COMBUF_MOD_ADDRESS + COMBUF_MOD_SIZE + COMBUF_MOD_OFFSET;
Cb_mod_info modinfo;
     modinfo.send_addr  = src;
     modinfo.send_size  = (Cb_size_t)bytes;
     modinfo.rcv_offset = off;
     if ( (rc=combuf_modify_tcw(tcwd, modfunc, &modinfo)) != COMBUF_SUCCESS)
         armci_die("combuf_modify_tcw failed",rc);
}
void armci_rdma_modify_tcw_partial(int tcwd,int bytes){
int rc;
int modfunc =COMBUF_MOD_SIZE;
Cb_mod_info modinfo;
     modinfo.send_size  = (Cb_size_t)bytes;
	if(DEBUG0){printf("%d:\n armci_rdma_modify_tcw_partial called with  tcwd=%d and size=%d \n",armci_me,tcwd,bytes);fflush(stdout);}
     if ( (rc=combuf_modify_tcw(tcwd, modfunc, &modinfo)) != COMBUF_SUCCESS)
         armci_die("combuf_modify_tcw partial failed",rc);
}

void armci_rdma_kick_tcw_put(int tcwd)
{
    int rc ;
    if ( (rc=combuf_kick_tcw(tcwd, COMBUF_SEND_NOBLOCK)) != COMBUF_SUCCESS)
        armci_die("combuf_kick_tcw failed",rc);
}
void armci_rdma_kick_tcw_block(int tcwd)
{
    int rc ;
    if ( (rc=combuf_kick_tcw(tcwd,0)) != COMBUF_SUCCESS)
        armci_die("combuf_kick_tcw failed",rc);
}

void armci_rdma_put_wait(int tcwd, char *flag)
{
	if(DEBUG0){printf("%d:\n put wait called with flag address %p and tcwd=%d\n",armci_me,flag,tcwd);fflush(stdout);}
    while(!combuf_check_sendflag(flag)); /* Wait */
}

      /**** Routines for GET ****/
#if 0
static int armci_rdma_make_gtcw(Cb_size_t soff, int src_dsc, 
                                Cb_size_t doff, int dst_dsc, int bytes)
{
int rc;
static Cb_msg msg;
int gtcw;

    if(DEBUG0){
     printf("%d:get s=%d d=%d bytes=%d\n",armci_me,soff,doff,bytes);fflush(stdout);}

    memset(&msg, 0, sizeof(msg));
    msg.data.addr = (void*)soff;  /* RDMA doc says to stuff offset here */
    msg.data.size = (Cb_size_t)bytes;
    rc = combuf_make_gtcw(src_dsc, &msg, dst_dsc, doff, NOOPTION, NOATOMIC, &gtcw);
    if(rc != COMBUF_SUCCESS) armci_die("combuf_make_gtcw failed",rc);
    return gtcw;
}

static void armci_rdma_kick_tcw_get(int tcwd)
{
    int rc ;
    if ( (rc=combuf_kick_tcw(tcwd, COMBUF_GET_NOBLOCK)) != COMBUF_SUCCESS)
        armci_die("combuf_kick_tcw failed",rc);
}


      /**** Other routines ****/

static void armci_rdma_free_tcw(int tcwd)
{
    int rc;
    if ((rc=combuf_free_tcw(tcwd)) != COMBUF_SUCCESS)
        armci_die("combuf_free_tcw failed",rc);
}
#endif

static void armci_rdma_get_wait(int desc)
{
    int rc;
    unsigned int recd;
    if ((rc=combuf_spin_wait(desc,-1,&recd)) != COMBUF_SUCCESS)
	                          armci_die("combuf_spin_wait failed",rc);
}


/*******************************************************************/

/*\ intialization of data structures 
 *  called by armci_register_shmem in 1st ARMCI_Malloc call in  ARMCI_Init
\*/ 
/*new temp array for client_rcv_fields should not be created here*/
/*code from client_init has to be moved here instead*/

void armci_init_sr8k()
{
     int rc;
     Cb_object_t oid,coid;
     int i, bytes = armci_nclus*MAX_REGIONS*sizeof(reg_auth_t);
	int client_bytes = armci_nproc*MAX_REGIONS*sizeof(reg_auth_t);
     long key;
     Cb_opt_region options;
	double *client_buffers;
     rem_reg = (reg_auth_t**)malloc(armci_nclus*sizeof(void*));
     if(!rem_reg)armci_die("rem_reg malloc failed",0);

     rem_reg_list = (reg_auth_t*)malloc(bytes);
     if(!rem_reg_list)armci_die("rem_reg_list malloc failed",bytes);
     bzero(rem_reg_list,bytes);

     reg_id = (long*) malloc(3*armci_nclus*sizeof(long));
     if(!reg_id)armci_die("rem_id: malloc failed",0);

     proc_reg = (int*) malloc(armci_nclus*sizeof(int));
     if(!proc_reg)armci_die("proc_reg: malloc failed",0);
     bzero(proc_reg,armci_nclus*sizeof(int));
     
     
     /* setup pointers for each smp cluster node */
     for(i = 0; i<armci_nclus; i++)rem_reg[i] = rem_reg_list + MAX_REGIONS*i;

     /* allocate internal RDMA work buffer */
     /*printf("BUFSIZE = %d, LPUT = %d, LGET = %d\n",BUFSIZE,LPUT,LGET);*/
     if (2*LPUT > BUFSIZE) armci_die("LPUT must be <= BUFSIZE/2",0);

     /* We need space for the work buffer (BUFSIZE) and two receive
      * fields (LGET) along with a flag for each of them (FLAGSIZE) */

     bytes = ROUND_UP_PAGE(BUFSIZE+FLAGSIZE+2*(LGET+FLAGSIZE));
     key   = BUF_KEY + armci_me-armci_master;
     if((rc=combuf_object_get(key, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &oid))
                       != COMBUF_SUCCESS) armci_die("combufget-1 buf failed",rc);
     if(combuf_map(oid, 0, (Cb_size_t)bytes, 0, (char**)&armci_internal_buffer)
                       != COMBUF_SUCCESS) armci_die("combuf map for buf failed",0);
     /* store the range of addresses taken by local buffer in rdma memory */
     bufstart = (char*) armci_internal_buffer;
     bufend   = (char*) armci_internal_buffer + BUFSIZE;
     tcw_flag = bufend; /* We allocated extra bytes here for the flag */

     /* Create a field for the work buffer */
     if(combuf_create_field(oid, bufstart, BUFSIZE, FIELD_NUM,
                            NOFLAG, NOOPTION, &bufdesc) != COMBUF_SUCCESS)
                                          armci_die("combufget field failed",0);
     /* Create two further fields for the get pipeline */
     
     getbuf[0] = bufend+FLAGSIZE;
     getbuf[1] = bufend+FLAGSIZE+LGET+FLAGSIZE;

     memset(&options,0, sizeof(options));
     options.flag.addr = getbuf[0]+LGET;   /* set up the spin_wait flag */
     options.flag.size = FLAGSIZE;

     if( (rc=combuf_create_field(oid, getbuf[0], LGET, CLIENT_GETBUF_FIELDNUM,
                            &options, COMBUF_CHECK_FLAG, &descs[0])) != COMBUF_SUCCESS)
                                          armci_die("combufget field 2 failed",rc);
     options.flag.addr = getbuf[1]+LGET;   /* set up the spin_wait flag */
     options.flag.size = FLAGSIZE;
     if( (rc=combuf_create_field(oid, getbuf[1], LGET, CLIENT_GETBUF_FIELDNUM+1,
                            &options, COMBUF_CHECK_FLAG, &descs[1])) != COMBUF_SUCCESS)
                                          armci_die("combufget field 3 failed",rc);
	/*we now create client buffers and fields for them*/
	bytes = ROUND_UP_PAGE(SMALL_MSG_SIZE*SMALL_BUFS_PER_PROCESS +FLAGSIZE*SMALL_BUFS_PER_PROCESS);
	key = CLIENT_SMALLBUF_KEY+armci_me;	

	if((rc=combuf_object_get(key, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &coid))
			!= COMBUF_SUCCESS) armci_die("combufget buf failed",rc);
	if((rc=combuf_map(coid, 0, (Cb_size_t)bytes, 0, (char**)&client_buffers))
                       != COMBUF_SUCCESS) armci_die("combuf map for buf failed",rc);
	for(i=0;i<SMALL_BUFS_PER_PROCESS;i++){
		int fn=CLIENT_SMALLBUF_FIELDNUM;
		client_buf_array[i] = (char*)client_buffers + i*SMALL_MSG_SIZE + i*FLAGSIZE;

		if(DEBUG0)printf("\n:%d client buf pointer=%p\n",armci_me,client_buf_array[i]);
		client_buf_flags[i] = (char*)client_buffers + (i+1)*SMALL_MSG_SIZE + i*FLAGSIZE;
		options.flag.addr = client_buf_array[i]+SMALL_MSG_SIZE;   
     	options.flag.size = FLAGSIZE;
     	if((rc=combuf_create_field(coid, client_buf_array[i], SMALL_MSG_SIZE, fn+i,
                           /*&options*/0,0/*COMBUF_CHECK_FLAG*/, &client_buf_desc_array[i])) != COMBUF_SUCCESS)
			armci_die("combuf create field for buf failed",rc);
		
	}	
     if(DEBUG0){printf("%d:armci_init_sr8k initialization done\n",armci_me);
                fflush(stdout);}
}
     

/*\ registers new rdma area - called in every call to ARMCI_Malloc
\*/
void armci_register_shmem(void *my_ptr, long size, long *idlist, long off,void *sptr)
{
     int i=0,dst,found=0,cfound=0;
     long id = idlist[2];
     long reg_size=0;

     if(DEBUG0){printf("%d: registering id=%ld size=%ld\n",armci_me,id,size);
                fflush(stdout);}
     //if(!reg_num) armci_init_sr8k();

     /* find if we allocated a new region, and if yes how much memory */
     if(size){
        if(reg_num>MAX_REGIONS)armci_die("error: reg_num corrupted",reg_num);
        for(i=0; i<reg_num; i++) if(reg_idlist[i]==id){
            found = 1;
            break;
        }

        if(!found){ 
          /* record new region id */
          reg_idlist[reg_num] = id; reg_num++; 
          reg_size = armci_shmem_reg_size(i,id);
          fflush(stdout);
        }
     }

     if(DEBUG0){
        printf("%d: regist id=%ld found=%d size=%ld reg_num=%d\n",\
			armci_me,id,found,reg_size,reg_num);
        fflush(stdout);
	}

     bzero(reg_id,3*armci_nclus*sizeof(long));

     /* store id and ptr into array of longs: sizeof(long) must be >=sizeof(void*)*/
     if(armci_me==armci_master){
        if(!found && size){
           reg_id[3*armci_clus_me]=id;
#if 0
           reg_id[3*armci_clus_me+1]=(long)armci_shmem_reg_ptr(i);
#else
           reg_id[3*armci_clus_me+1]=(long)((char *)sptr-off);
#endif
           reg_id[3*armci_clus_me+2]=reg_size;
        }
        /* master processes exchange region data */
        armci_msg_gop_scope(SCOPE_MASTERS, reg_id, 3*armci_nclus,"+",ARMCI_LONG); 
     }

     /* make the data available within each cluster smp node */
     armci_msg_clus_brdcst(reg_id,3*armci_nclus*sizeof(long));
     
	dst = armci_clus_me; 
     for(i = 0; i < armci_nclus; i++){
         void *ptr;
         long len;
         Cb_node_rt     remote;
         int rc, auth, tcwd;

         dst ++; dst %= armci_nclus;  /* select smp node */

         /* unpack region info for that smp node */
         id = reg_id[3*dst];
         ptr= (void*)reg_id[3*dst+1];
         len= reg_id[3*dst+2];
         if(!id) continue;   /* nothing to register - likely we did it before */

         /* aquire authorization to send/put */
         bzero( &remote, sizeof(remote) );
         remote.type = CB_NODE_RELATIVE;
         remote.ndes = _armci_group;
         remote.node = dst;
         if(DEBUG1){
            printf("%d:%d: %d registering sendright %d key=%ld %p %d\n",
                   armci_me,armci_clus_me,i,armci_clus_info[dst].master,id,ptr,len);
            fflush(stdout);
         }

         rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                                    id, FIELD_NUM, -1, &auth);
         if(rc != COMBUF_SUCCESS){ 
            printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
            armci_die("combuf_get_sendright:",rc);
         }

	 /* Make a generic tcw for put communication to this region.
          * This is modified later as needed.  Should reduce latency
          * [BPE] */
	 tcwd = armci_rdma_make_tcw( bufstart, 0, LPUT, auth, tcw_flag);

         rem_reg[dst][proc_reg[dst]].pauth = auth;      
         rem_reg[dst][proc_reg[dst]].size  = len;      
         rem_reg[dst][proc_reg[dst]].ptr   = ptr;      
         rem_reg[dst][proc_reg[dst]].tcwd  = tcwd;

         /* aquire authorization to do get */
         bzero( &remote, sizeof(remote) );
         remote.type = CB_NODE_RELATIVE;
         remote.ndes = _armci_group;
         remote.node = dst;

         if(DEBUG0){printf("%d:register target %d\n",armci_me,dst); fflush(stdout);}
         rc = combuf_target( (Cb_node *)&remote, sizeof(remote), id, 0, -1, &auth);
         if(rc != COMBUF_SUCCESS) armci_die("combuf_target:",rc);
         rem_reg[dst][proc_reg[dst]].gauth = auth;      

         proc_reg[dst] ++; 
     }
     if(DEBUG0){printf("%d:registered id=%ld\n",armci_me,id); fflush(stdout);}
}

/*\ find shmem region corresponding to dst address
\*/

int armci_find_shmem( int node, char *ptr, Cb_size_t *off)
{
    char *ps,*pe;
    int found, i;

    found = 0;
    for(i=0; i< proc_reg[node]; i++){
       ps = rem_reg[node][i].ptr;
       pe = rem_reg[node][i].size + ps;
       if((ptr>=ps) && (ptr<pe)){
	 found=1;
	 *off = (Cb_size_t)(ptr-ps);
	 break;
       }
    }
    return(found ? i : -1);
}

/*\ basic put operation to combuf desc field at specified offset
\*/ 
static unsigned int armci_rdma_put(void *src, Cb_size_t off, int bytes, int desc)
{
int rc;
static Cb_msg msg;
unsigned int ev;
     memset(&msg, 0, sizeof(msg));
     msg.data.addr = src;
     msg.data.size = (Cb_size_t)bytes;
     rc = combuf_send(&msg, desc, off, 0, &ev);
     if (rc != COMBUF_SUCCESS) armci_die("combuf_send failed",rc);
     if(DEBUG0){printf("%d:put dsc=%d off=%d\n",armci_me,desc,off); fflush(stdout);}
     return ev;
}

/*\ basic get operation from combuf desc field at specified offset
\*/
static unsigned int armci_rdma_get(Cb_size_t soff, int src_dsc, 
                                   Cb_size_t doff, int dst_dsc, int bytes)
{
int rc;
static Cb_msg msg;
unsigned int ev;

    if(DEBUG0){
     printf("%d:get s=%d d=%d bytes=%d\n",armci_me,soff,doff,bytes);fflush(stdout);}

    memset(&msg, 0, sizeof(msg));
    msg.data.addr = (void*)soff;  /* RDMA doc says to stuff offset here */
    msg.data.size = (Cb_size_t)bytes;
    rc = combuf_get(src_dsc, &msg, dst_dsc, doff, NOOPTION, NOATOMIC, &ev);
    if(rc != COMBUF_SUCCESS) armci_die("combuf_get failed",rc);
    return ev;
}

/*\ loop to wait for ack's
\*/
#define DEBUGACK 0
int ackcount;
void _sr8k_wait_for_ack(int node){
	int rc;
	unsigned int recvd;
	if(!_sr8k_largebuf_insync[node])
		while(server_pending_op_count[node]!=client_pending_op_count[node]){
			rc=combuf_spin_wait(clientpendingopbufdesc,10,&recvd);
			if(DEBUGACK)printf("\n%d:wait_largebuf_ack expected=%d actual=%d\n",armci_me,client_pending_op_count[node],server_pending_op_count[node]);   
		}
	_sr8k_largebuf_insync[node]=1;
	if(!_sr8k_smallbuf_insync[node])
    	while(server_smallbuf_pending_count[node]!=client_smallbuf_pending_count[node]){
           rc=combuf_spin_wait(clientpendingopbufdesc,10,&recvd); 
			if(DEBUGACK)printf("\n%d:wait_smallbuf_ack actual=%d expected=%d\n",armci_me,server_smallbuf_pending_count[node],client_smallbuf_pending_count[node]);   
          }

	_sr8k_smallbuf_insync[node]=1;
	if(DEBUGACK)printf("\n%d:OUT of buf ack\n",armci_me);
}

void _sr8k_wait_smallbuf_ack(int node){
	int rc;
	unsigned int recvd;
	if(_sr8k_smallbuf_insync[node])return;
	while(server_smallbuf_pending_count[node]!=client_smallbuf_pending_count[node]){
    		rc=combuf_spin_wait(clientpendingopbufdesc,10,&recvd);
			if(DEBUGACK)printf("\n%d:wait_smallbuf_ack actual=%d expected=%d\n",armci_me,server_smallbuf_pending_count[node],client_smallbuf_pending_count[node]);   
	}
	_sr8k_smallbuf_insync[node]=1;
	if(DEBUGACK)printf("\n%d:OUT of smallbuf ack\n",armci_me);
}
void _sr8k_wait_largebuf_ack(int node){
	int rc;
	unsigned int recvd;
	if(_sr8k_largebuf_insync[node])return;
	if(DEBUGACK)printf("\n%d:IN largebuf ack\n",armci_me);
	while(server_pending_op_count[node]!=client_pending_op_count[node]){
		rc=combuf_spin_wait(clientpendingopbufdesc,10,&recvd);
		if(DEBUGACK)printf("\n%d:wait_largebuf_ack expected=%d actual=%d\n",armci_me,client_pending_op_count[node],server_pending_op_count[node]);   
	}
	_sr8k_largebuf_insync[node]=1;	
	if(DEBUGACK)printf("\n%d:OUT of largebuf ack.....\n",armci_me);
}

/*\ basic get operation from combuf desc field at specified offset
 *  (non-blocking)
\*/
static unsigned int armci_rdma_get_nbl(Cb_size_t soff, int src_dsc, 
                                       Cb_size_t doff, int dst_dsc, int bytes)
{
int rc;
static Cb_msg msg;
unsigned int ev;

    if(DEBUG0){
     printf("%d:get s=%d d=%d bytes=%d\n",armci_me,soff,doff,bytes);fflush(stdout);}

    memset(&msg, 0, sizeof(msg));
    msg.data.addr = (void*)soff;  /* RDMA doc says to stuff offset here */
    msg.data.size = (Cb_size_t)bytes;
    rc = combuf_get(src_dsc, &msg, dst_dsc, doff, COMBUF_GET_NOBLOCK, NOATOMIC, &ev);
    if(rc != COMBUF_SUCCESS) armci_die("combuf_get failed",rc);
    return ev;
}

/*\ Strided put operation to combuf desc field at specified offset
\*/ 
static unsigned int armci_rdma_stride_put(void *src, int bytes, int count,
                    int src_stride, Cb_size_t off, int dst_stride, int desc)
{
int rc;
static Cb_stride_msg msg;
unsigned int ev;
      memset(&msg, 0, sizeof(msg));
      msg.data.addr = src;
      msg.data.elem_size = bytes;
      msg.data.elem_num = count;
      msg.data.stridesize = src_stride;
      
      rc = combuf_stride_send(&msg, desc, off, dst_stride, 0, &ev);
      if (rc != COMBUF_SUCCESS) armci_die("combuf_stride_send failed",rc);
      return(ev);
}

/*\  contiguous put  dst(proc) = src
\*/
void armcill_put(void *src, void *dst, int bytes, int proc)
{
Cb_size_t off;
int found =0, i, node = armci_clus_id(proc);
int master = armci_clus_info[node].master;
char *ptr=(char*)dst;
int desc, tcwd;
	_sr8k_wait_for_ack(node);	
    if(DEBUG0){
       printf("%d:put s=%p d=%p p=%d\n",armci_me,src,dst,proc); fflush(stdout);}

    /* Find shmem region corresponding to dst address */
    if ( (i = armci_find_shmem(node, ptr, &off)) >= 0 ) {
        desc = rem_reg[node][i].pauth;
	tcwd = rem_reg[node][i].tcwd;
    }
    else 
        armci_die("armcill_put: bad dst address for p=",proc);

    if((src >= (void*)bufstart) && (src <(void*)bufend)){

       /* no need to copy or pipeline - data is in the rdma buffer */
        (void)armci_rdma_put(src, off, bytes, desc);

    } else {

      /* If bytes > LPUT/3 then pipeline sends with memory copies
       * else this just reverts to a single copy and send      */

      char *intbuf=(char*)armci_internal_buffer; 
      int len, bufpos=0;
      int dlen=LPUT;

      /* Intermediate size messages benefit from pipelining in
       * smaller chunks */
      if (bytes<3*LPUT/2 && bytes>=LPUT/2) dlen=ALIGNUP(LPUT/2);
      if (bytes<LPUT/2   && bytes>=LPUT/3) dlen=ALIGNUP(LPUT/3);

      /* Do the smallest copy first; we can hide the others */
      len = 1+(bytes-1)%dlen;

      for(i = 0; i< bytes;){

	/* send (mostly) in dlen size chunks.  dlen <= BUFSIZE/2
         * so that data are not overwritten before they are sent */

	armci_copy((char*)src, intbuf+bufpos, len);
	if (i>0) {
	  armci_rdma_put_wait(tcwd, tcw_flag);
	}
	armci_rdma_modify_tcw(tcwd, intbuf+bufpos, off, len);
	armci_rdma_kick_tcw_put(tcwd);

        i   +=len;
        off +=len;
        src = len +(char*)src;
	bufpos = (bufpos <= BUFSIZE-2*LPUT) ? bufpos+LPUT : 0;
	len = dlen; /* We know that the rest of the message
                     * is a multiple of this size */
      }
      armci_rdma_put_wait(tcwd,tcw_flag); /* wait for the last send */
    }
}


/*\  contiguous get  src = dst(proc)
\*/
void armcill_get(void *src, void *dst, int bytes, int proc)
{
Cb_size_t off, buf_off;
int found =0, i, node = armci_clus_id(proc);
int master = armci_clus_info[node].master;
char *ptr=(char*)src;
int desc;
	 _sr8k_wait_for_ack(node);
    if(DEBUG0){
       printf("%d:get s=%p d=%p p=%d\n",armci_me,src,dst,proc); fflush(stdout);}

    /* find shmem region corresponding to dst address */
    if ( (i = armci_find_shmem(node, ptr, &off)) >= 0 )
        desc = rem_reg[node][i].gauth;
    else 
        armci_die("armcill_get: bad dst address for p=",proc);

    if((dst >= (void*)bufstart) && (dst <(void*)bufend)){

       /* no need to copy - this is our rdma buffer */
        buf_off = (Cb_size_t)(((char*)dst) - bufstart);
        (void)armci_rdma_get(off, desc, buf_off, bufdesc, bytes);

    } else {

        /* If bytes > LGET then pipeline gets with memory copies
         * else this just reverts to a single get and copy */

        /*
         * There doesn't seem to be an equivalent for get of
         * combuf_modify_tcw(), so we can't make the tcw reuse
         * optimisation for latency.  This has an adverse impact on
         * the efficiency of the pipelining.
         *
         * By using separate combuf fields for the get buffers we can
         * use spin_wait which is by far the fastest receive
         * confirmation method.  We can also initiate more than one
         * get at a time which helps to hide a little of the latency
         * for the pipelined transfers.
         */

	char *dstold;
	int len, lenold;
	int new=0, old=1;

        for(i = 0; i< bytes;){ 

          len = ((bytes -i)<LGET)? (bytes -i): LGET;

	  /* Get to current buffer: combuf_get() seems to be fastest way */
          armci_rdma_get_nbl(off, desc, 0, descs[new], len);

	  /* Wait, then copy from the old buffer */
	  if (i>0) {
	    armci_rdma_get_wait(descs[old]);
	    armci_copy(getbuf[old], dstold, lenold);
	  }

	  /* Save current values for later copy operation */
	  lenold = len;
	  dstold = (char*)dst;

          /* Update to new values */
          i   +=len;
          off +=len;
          dst = len +(char*)dst;

	  /* Switch to alternate descriptors and buffers */
	  old = new; new = 1-new;
	}
	armci_rdma_get_wait(descs[old]);
	armci_copy(getbuf[old], dstold, lenold);
    }
}


/*\ strided put
\*/
void armcill_put2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j,node = armci_clus_id(proc);
char *ps=src_ptr, *pd=dst_ptr;
  _sr8k_wait_for_ack(node);
  if (2*bytes > BUFSIZE) {

    /* 
     * No more than one block will fit into the buffer, so it is
     * convenient to send each individually making use of armcill_put()
     * for pipelining.  Alternatively we could just do a number of
     * shorter strided sends, and we could even pipeline them with the
     * packing, I guess.
     * Exercise for the reader: implement this. [BPE]
     */

    for (_j = 0;  _j < count;  _j++){
      armcill_put(ps, pd, bytes, proc);
      ps += src_stride;
      pd += dst_stride;
    }    

  } else {

    /* We choose to send blocks in buffer-loads */

    Cb_size_t off;
    int node = armci_clus_id(proc);
    int desc, stride, nocopy;
    char *source;
    int i, ict, ct, dct=BUFSIZE/bytes;  /* dct is the number of elements
                                         * we can fit into the buffer */

    /* Find shmem region corresponding to dst address */
    if ( (i = armci_find_shmem(node, pd, &off)) >= 0 )
        desc = rem_reg[node][i].pauth;
    else 
        armci_die("armcill_put_2D: bad dst address for p=",proc);

    nocopy = ((ps >= bufstart) && (ps <bufend));

    ict = 0;
    while(ict<count) {
      ct = (ict+dct <= count) ? dct : count-ict;
      
      /* Copy data to send buffer if necessary */
      if(nocopy) {
	/* Data are already in rdma buffer */
	stride = src_stride;
	source = ps;
      } else {
	/* Pack into rdma buffer */
	char *ps_tmp = ps;
	char *pd_tmp = (char*)armci_internal_buffer; 
	for (_j = 0;  _j < ct;  _j++){
	  armci_copy(ps_tmp, pd_tmp, bytes);
	  ps_tmp += src_stride;
	  pd_tmp += bytes;
	}
	stride = bytes; /* Since it's now packed */
	source = (char*)armci_internal_buffer;
      }
      
      /* Send the data */
      (void)armci_rdma_stride_put(source, bytes, ct, stride, off, dst_stride, desc);

      ps  += ct*src_stride;
      off += ct*dst_stride;
      ict += ct;
    }
  }
}



/*\ strided get: source is at proc, destination on calling process 
\*/
void armcill_get2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j;
char *ps=src_ptr, *pd=dst_ptr;
    if(DEBUG0){
       printf("%d:get s=%p d=%p p=%d\n",armci_me,src_ptr,dst_ptr,proc); fflush(stdout);}
      for (_j = 0;  _j < count;  _j++){
          armcill_get(ps, pd, bytes, proc);
          ps += src_stride;
          pd += dst_stride;
      }
    if(DEBUG0){
       printf("%d:get s=%p d=%p p=%d\n",armci_me,src_ptr,dst_ptr,proc); fflush(stdout);}
}


typedef struct {
       Cb_size_t off;
       int desc;
} sr8k_mutex_t;

static sr8k_mutex_t *_mutex_array;
sr8k_mutex_t *_server_mutex_array;

void server_set_mutex_array(char *tmp){
int rc,size,auth,i,dst,nproc=armci_clus_info[armci_clus_me].nslave;
Cb_object_t oid;
Cb_node_rt remote;
char *temp;
	size = ROUND_UP_PAGE(sizeof(sr8k_mutex_t)*armci_nclus);
	if((rc=combuf_object_get(MUTEX_ARRAY_KEY, (Cb_size_t)size, 0, &oid))
    		!= COMBUF_SUCCESS) armci_die("creat combufget fail",rc);
    if((rc=combuf_map(oid, 0, (Cb_size_t)size, COMBUF_COMMON_USE, &temp))
			!= COMBUF_SUCCESS) armci_die("combuf map failed",rc);

	_server_mutex_array = (sr8k_mutex_t*)temp;
	dst = armci_clus_me;
	for(i=0;i<armci_nclus;i++){	
    	bzero( &remote, sizeof(remote));
    	remote.type = CB_NODE_RELATIVE;
    	remote.ndes = _armci_group;
		dst++;dst%=armci_nclus;
    	remote.node = dst;
		rc = combuf_target( (Cb_node *)&remote, sizeof(remote), 1961, 0, -1, &auth);
		if(rc != COMBUF_SUCCESS)armci_die("combuf_target:",rc);
    	_mutex_array[dst].off  = _server_mutex_array[armci_clus_me].off; 
    	_mutex_array[dst].desc = auth;
	}
}
/*\ allocate the specified number of mutexes on the current SMP node
\*/
void armcill_allocate_locks(int num)
{
int bytes = num*sizeof(int);
char *ptr;
void *myptr;
Cb_object_t oid;
int nproc = armci_clus_info[armci_clus_me].nslave;
int rc,i,size,auth,dst;
long idlist[SHMIDLEN];
Cb_node_rt remote;

	if(armci_me==armci_master){
		myptr=Create_Shared_Region(idlist+1, bytes, idlist);	
	}
	armci_msg_clus_brdcst(idlist, SHMIDLEN*sizeof(long));
	if(armci_me != armci_master){
		myptr=Attach_Shared_Region(idlist+1, bytes, idlist[0]);
	}
	
	if(armci_me==armci_master){			
		size = ROUND_UP_PAGE(sizeof(sr8k_mutex_t)*armci_nclus);
		if(combuf_object_get(MUTEX_ARRAY_KEY, (Cb_size_t)size,\
			 	COMBUF_OBJECT_CREATE, &oid)!= COMBUF_SUCCESS)
			armci_die("attaching combufget fail",0);
     	if(combuf_map(oid, 0, (Cb_size_t)size, COMBUF_COMMON_USE, &ptr)
                       != COMBUF_SUCCESS) armci_die("combuf map failed",0);
		_server_mutex_array = (sr8k_mutex_t*)ptr;
   	}
	_mutex_array = (sr8k_mutex_t *)malloc(sizeof(sr8k_mutex_t)*armci_nclus);
    if(!_mutex_array)
         armci_die("armcill_allocate_locks: malloc failed",armci_nclus);
	/*get authorization to do local swap*/
	dst = armci_clus_me;
	for(i=0;i<armci_nclus;i++){	
    	bzero( &remote, sizeof(remote));
    	remote.type = CB_NODE_RELATIVE;
    	remote.ndes = _armci_group;
		dst++;dst%=armci_nclus;
    	remote.node = dst;
		rc = combuf_target( (Cb_node *)&remote, sizeof(remote), 1961, 0, -1, &auth);
		if(rc != COMBUF_SUCCESS)armci_die("combuf_target:",rc);
    	_mutex_array[dst].off  = idlist[0];
    	_mutex_array[dst].desc = auth;
		if(armci_me==armci_master){
			_server_mutex_array[dst].off  = idlist[0];
			_server_mutex_array[dst].desc = 0;
		}
    	if(DEBUG0){
			printf("%d:allocate %d locks %p \n",armci_me,num,myptr);fflush(stdout);
		}
    	if(_mutex_array[0].off >armci_shmem_reg_size(0,idlist[3])) 
			armci_die("armcill_allocate_locks:offset error",_mutex_array[i].off);
	}
	armci_msg_clus_brdcst(idlist, SHMIDLEN*sizeof(long));
}

/*\ lock specified mutex on node where process proc is running
\*/
void armcill_lock(int mutex, int proc)
{
#if 1
int desc,node = armci_clus_id(proc);
Cb_size_t off;
	//if(node!=armci_clus_me)armci_die("only local locks are implementer",node);
    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
    if(DEBUG0){
		printf("%d: lock %d on %d off=%d desc=%d\n",armci_me,mutex,proc, off,desc);
		fflush(stdout);
	}
    while(combuf_swap(desc,off,1));
    /*while(combuf_cswap(desc,off,1,0,-1));*/ /* wait 2ms for condition to be met*/ 
#endif
}


/*\ unlock specified mutex on node where process proc is running
\*/
void armcill_unlock(int mutex, int proc)
{
#if 1
int desc,node = armci_clus_id(proc);
Cb_size_t off;
	//if(node!=armci_clus_me)armci_die("only local locks are implementer",node);
    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
    combuf_swap(desc,off,0);
    if(DEBUG0){
      printf("%d: unlock %d on %d off=%d desc=%d\n",armci_me,mutex,proc, off,desc);fflush(stdout);}
#endif
}


char *client_direct_buffer;
char **client_directbuf_flag;

void armci_init_connections(){
int size,rc;
Cb_object_t moid;
Cb_opt_region options;
int mykey = CLIENT_SERV_COMMON_CB_KEY;
	armci_init_sr8k();
	size = ROUND_UP_PAGE(sizeof(rcv_field_info_t)*(armci_nproc+armci_nclus)\
			+(armci_nproc+1)*sizeof(int));
	if(armci_me==armci_master){
		if(rc=(combuf_object_get(mykey, (Cb_size_t)size,COMBUF_OBJECT_CREATE,\
				   &moid))!= COMBUF_SUCCESS) 
			armci_die("armci_init_connection-1 combufget buf failed",rc);
    	if((rc=combuf_map(moid, 0, (Cb_size_t)size, COMBUF_COMMON_USE,\
				    (char**)&armci_clientmap_common_buffer))!= COMBUF_SUCCESS)
				armci_die("combufmap init_connection-1 failed",rc);
	}
	armci_msg_barrier();
	if(armci_me!=armci_master){
		if((rc=combuf_object_get(mykey, (Cb_size_t)size,0,&moid))!= COMBUF_SUCCESS) 
			armci_die("armci_init_connection-3 combufget buf failed",rc);
    	if((rc=combuf_map(moid, 0, (Cb_size_t)size, COMBUF_COMMON_USE,\
			(char**)&armci_clientmap_common_buffer))!= COMBUF_SUCCESS) 
			armci_die("combuf map int amrci_init_connection-2 for buf failed",rc);

	}
	server_ready_flag = (int *)armci_clientmap_common_buffer;
	client_ready_flag = (int *)armci_clientmap_common_buffer+1;
 	serv_rcv_field_info = (rcv_field_info_t *)((int *)armci_clientmap_common_buffer+armci_nproc+1);
	client_rcv_field_info = serv_rcv_field_info + armci_nclus;	
	_sr8k_smallbuf_insync=(int*)calloc(sizeof(int),armci_nclus);	
	if(!_sr8k_smallbuf_insync)armci_die("malloc for _sr8k_smallbuf_insync failed",0);
	_sr8k_largebuf_insync=(int*)calloc(sizeof(int),armci_nclus);	
	if(!_sr8k_largebuf_insync)armci_die("malloc for _sr8k_largebuf_insync failed",0);
	_armci_buf_init(); /*initialize the request buffers(not used for small messages)*/
}


char **client_send_buffer;
int *client_send_bufdesc;
int clientgetdirectbufdesc;
char *clientgetdirectbufflag;


char *_sr8k_armci_buf_init(int size){
int rc,i,bufextsize,origsize;
Cb_object_t moid;
char *returnbuf;
Cb_opt_region options;
	client_directbuf_flag = (char **)malloc(sizeof(char *)*numofbuffers);
    client_send_buffer = (char **)malloc(sizeof(char *)*numofbuffers);
	client_send_bufdesc = (int *)malloc(sizeof(int)*numofbuffers);
	client_pending_op_count=(int *)malloc(sizeof(int)*armci_nproc);
	client_smallbuf_pending_count=(int *)malloc(sizeof(int)*armci_nproc);
    if(!client_directbuf_flag || !client_send_buffer || ! client_send_bufdesc
        || !client_pending_op_count || !client_smallbuf_pending_count)
        armci_die("_sr8k_armci_buf_init: malloc failed",0);

	origsize=size;
	size=ROUND_UP_PAGE(size+(numofbuffers+1)*FLAGSIZE+2*armci_nproc*sizeof(int));
	if(rc=(combuf_object_get(CLIENT_DIRECTBUF_KEY+armci_me, (Cb_size_t)size,COMBUF_OBJECT_CREATE,&moid))
                       != COMBUF_SUCCESS) armci_die("_sr8k_armci_buf_init combufget buf failed",rc);
    if((rc=combuf_map(moid, 0, (Cb_size_t)size,COMBUF_COMMON_USE,(char**)&client_direct_buffer))
                       != COMBUF_SUCCESS) armci_die("combuf map int amrci_init_connection-1 for buf failed",rc);

     if(!client_direct_buffer)  armci_die("_sr8k_armci_buf_init: map failed",0);

	/*create a field to communicate pending send/acc counts*/
	server_pending_op_count=(int *)(client_direct_buffer+origsize+(numofbuffers+1)*FLAGSIZE);
	server_smallbuf_pending_count=server_pending_op_count+armci_nproc;
	memset(&options, 0, sizeof(options));
    options.flag.addr = client_direct_buffer+origsize+numofbuffers*FLAGSIZE;
    serverpendingopflag = options.flag.addr;
    options.flag.size = FLAGSIZE;
    if((rc=combuf_create_field(moid,(client_direct_buffer+origsize+(numofbuffers+1)*FLAGSIZE),(Cb_size_t)(2*armci_nproc*sizeof(int)),
                    CLIENT_PENDING_OP_FIELDNUM,&options,COMBUF_CHECK_FLAG,&clientpendingopbufdesc)) != COMBUF_SUCCESS)
        armci_die("combuf create field for pending op count",rc);
	for(i=0;i<armci_nproc;i++){
		client_pending_op_count[i]=0;server_pending_op_count[i]=0;
		client_smallbuf_pending_count[i]=0;server_smallbuf_pending_count[i]=0;
	}


	/*create one large field out of the whole client buffer to be used when doing a get*/
	memset(&options, 0, sizeof(options));
	options.flag.addr = client_direct_buffer+origsize;
	clientgetdirectbufflag = options.flag.addr;
	options.flag.size = FLAGSIZE;
	if((rc=combuf_create_field(moid,client_direct_buffer,(Cb_size_t)(MSG_BUFLEN_SMALL*numofbuffers),
                    CLIENT_GET_DIRECTBUF_FIELDNUM,&options,COMBUF_CHECK_FLAG,&clientgetdirectbufdesc)) != COMBUF_SUCCESS)
        armci_die("combuf create field for ackbuf failed",rc);

     memset(&options, 0, sizeof(options));
	returnbuf = client_direct_buffer + ALIGN64ADD(client_direct_buffer);
	bufextsize = (origsize-64)/numofbuffers;
	for(i=0;i<numofbuffers;i++){
		memset(&options, 0, sizeof(options));
     	options.flag.addr =  returnbuf+origsize-64+i*FLAGSIZE; 
		/*NOTusing buf_pad_t as flag for allignment issues*/
		client_directbuf_flag[i]= options.flag.addr;
     	options.flag.size = FLAGSIZE;
  		if((rc=combuf_create_field(moid,(returnbuf+i*bufextsize+sizeof(BUFID_PAD_T)+sizeof(BUF_EXTRA_FIELD_T)),(Cb_size_t)(MSG_BUFLEN_SMALL),
                    CLIENT_DIRECTBUF_FIELDNUM+armci_me+i,/*&options*/0,0/*COMBUF_CHECK_FLAG*/,&client_send_bufdesc[i])) != COMBUF_SUCCESS)
		armci_die("_sr8k_armci_buf_init combuf_create_fieldfailed",rc);
		*((int *)(returnbuf+i*bufextsize+sizeof(BUFID_PAD_T)))=-1;
		client_send_buffer[i] = (returnbuf+i*bufextsize+sizeof(BUFID_PAD_T)+sizeof(BUF_EXTRA_FIELD_T));
		if(DEBUG0)printf("\n%d: client_send_buffer[%d]=%p extra=%d client_direct buf=%p\n",armci_me,i,client_send_buffer[i],ALIGN64ADD(client_direct_buffer),client_direct_buffer);
	}		
	directbuffer = returnbuf;
	return(client_direct_buffer);
}

void _sr8k_armci_buf_release(void *buf){
int i;
int count;
	/*we need to reset buffers after every get*/
	if(DEBUG0)printf("\n%d: in _sr8k_armci_buf_release \n",armci_me);
	if((char *)buf == client_buf_array[0]){ 
	  if(((request_header_t *)buf)->operation==GET && ((request_header_t *)buf)->format==STRIDED){
		count= ((request_header_t *)buf)->bytes+sizeof(request_header_t);
		count=count/MSG_BUFLEN_SMALL+(count%MSG_BUFLEN_SMALL)?1:0;
		
		for(i=0;i<count;i++){
			*((int *)(client_send_buffer[i]-sizeof(BUF_EXTRA_FIELD_T))) = -1;
			*((BUFID_PAD_T *)(client_send_buffer[i]-sizeof(BUF_EXTRA_FIELD_T)-sizeof(BUFID_PAD_T))) = 0;
		}
	   _armci_buf_release(client_send_buffer[0]);
	  } 
	  else {
	  	
	  }
	}
	else 
		_armci_buf_release(buf);

}
char *_sr8k_get_client_small_buf(int *tcwd,int dst){
	_sr8k_wait_smallbuf_ack(dst);
	*tcwd=client_auths[dst].small_buf_tcwd[0];
	client_smallbuf_pending_count[dst]++;
	_sr8k_smallbuf_insync[dst]=0;	
	return(client_buf_array[0]);
}

int _sr8k_using_smallbuf;
char *_sr8k_armci_getbuf_ptr;
long _sr8k_armci_getbuf_ofs;

char *_sr8k_armci_buf_get(int size, int op,int proc){
char *tmp;	
int b_ind,dst,rc;
unsigned int recvd;
#if defined(PIPE_BUFSIZE___)
	if(!_sr8k_vector_req)
	if(op==GET && size>2*PIPE_MIN_BUFSIZE){
		tmp = _armci_buf_get(size,op,proc);
        _sr8k_armci_getbuf_ptr=tmp;
        _sr8k_armci_getbuf_ofs=(tmp-client_direct_buffer);
    	tmp = _sr8k_get_client_small_buf(&current_clientbuf_tcwd,armci_clus_id(proc));
  		return(tmp);
    }
#endif
	if(size<SMALL_MSG_SIZE){
		if(proc<0)proc=-proc+SOFFSET;  
		_sr8k_using_smallbuf=1;
		tmp = _sr8k_get_client_small_buf(&current_clientbuf_tcwd,armci_clus_id(proc));  
	}
	else{
		tmp = _armci_buf_get(size,op,proc);
          b_ind = _sr8k_buffer_index;
		_sr8k_using_smallbuf=0;	
          if(proc<0)proc=-proc+SOFFSET;
          current_clientbuf_tcwd=client_auths[armci_clus_id(proc)].large_buf_tcwd[b_ind];
          *((int *)(tmp-sizeof(BUF_EXTRA_FIELD_T))) = current_clientbuf_tcwd;
	}	
     _sr8k_armci_getbuf_ptr=tmp;
     _sr8k_armci_getbuf_ofs=(tmp-client_direct_buffer);
	if(DEBUG0){printf("\n%d: got buf %p  _sr8k_buffer_ind=%d size=%d op=%d\n",armci_me,tmp,_sr8k_buffer_index,size,op);fflush(stdout);}
     return(tmp);
}

int _sr8k_buffer_index,_sr8k_pipeget_req=0;
int buf_index=0;
int previousbuf=-1;

int armci_send_req_msg(int proc, void *buf, int bytes){
int dst,tcwd,i,rc;
unsigned int recvd;
request_header_t *msginfo=(request_header_t *)buf;
	dst = armci_clus_id(proc);	
	if(DEBUG0)printf("\n%d in send_req bytes=%d op=%d ackcount=%d to %d\n",armci_me,bytes,msginfo->operation,client_pending_op_count[dst],dst);
	if(msginfo->operation==ACK){
		client_smallbuf_pending_count[dst]--;
		for(i=0;i<armci_nclus;i++){
			_sr8k_wait_for_ack(i);
		}
		return(0);
	}
	tcwd=current_clientbuf_tcwd;
	buf_index = _sr8k_buffer_index;	
	if(!_sr8k_pipeget_req && msginfo->operation==GET){
		_sr8k_wait_largebuf_ack(dst);
		if(!_sr8k_using_smallbuf)
			_sr8k_wait_smallbuf_ack(dst);
	}
	if((msginfo->operation==PUT || ACC(msginfo->operation))&&bytes>SMALL_MSG_SIZE){
		if(previousbuf!=-1){
			_armci_buf_ensure_one_outstanding_op_per_proc(buf,proc);
		}
		previousbuf=_sr8k_buffer_index;
		client_pending_op_count[dst]++;
		_sr8k_largebuf_insync[dst]=0;
		msginfo->tag.ack=client_pending_op_count[dst];
	}
	if(bytes>SMALL_MSG_SIZE||(_sr8k_pipeget_req)){
#if 0
		while((rc=combuf_cswap(client_auths[armci_clus_id(proc)].get_auth,((SMALL_MSG_SIZE*SMALL_MSG_NUM)+buf_index*(LARGE_MSG_SIZE/4)),10+armci_me,0,0))){
			printf("\n%d in send_req waiting for swap rc=%d\n",armci_me,rc);
			armci_util_spin(10000+armci_me*armci_me*100,NULL);	
		}
#endif
        while(combuf_swap(client_auths[armci_clus_id(proc)].get_auth,((SMALL_MSG_SIZE*SMALL_MSG_NUM)+buf_index*(LARGE_MSG_SIZE/4)),1));

	}
	armci_rdma_modify_tcw_partial(tcwd,bytes);
	armci_rdma_kick_tcw_block(tcwd);
	return(0);
}

int _sr8k_armci_wait_some =20;
double _sr8k_armci_fake_work=99.0;

long check_flag(long*buf){
	return(*buf);
}

void armci_wait_long_flag_updated(long *buf, int val)
{
    long res;
    long spin =0;

    res = check_flag(buf);
    while(res != (long)val){
       for(spin=0; spin<_sr8k_armci_wait_some; spin++)_sr8k_armci_fake_work+=0.001;
       res = check_flag(buf);
    }
    _sr8k_armci_fake_work =99.0;
}

void armci_wait_long_flag_updated_clear(long *buf, int val)
{
    armci_wait_long_flag_updated(buf,val);
    *buf = 0L;
}


char * armci_ReadFromDirect(int proc,request_header_t *msginfo, int datalen){
long *ack;
int rc;unsigned int recvd;
	if(DEBUG2){printf("\n%d:armci_ReadFromDirect op=%d bytes=%d\n",armci_me,msginfo->operation,(msginfo->bytes+sizeof(request_header_t)));fflush(stdout);}
	if(msginfo->operation==ACK)return((char*)msginfo);
	if((msginfo->bytes+sizeof(request_header_t))<SMALL_MSG_SIZE){
		ack=(long*)((char *)(msginfo+1)+msginfo->dscrlen+msginfo->datalen);
		rc=combuf_block_wait(client_buf_desc_array[0],-1,&recvd);
		if(rc!=COMBUF_SUCCESS)armci_die("wait failed",rc);
		armci_wait_long_flag_updated_clear(ack,1);
		msginfo->datalen=datalen;
		return(client_buf_array[0]+(sizeof(request_header_t)+msginfo->dscrlen));
	}
	else{
		rc=combuf_block_wait(client_send_bufdesc[_sr8k_buffer_index],-1,&recvd);
		if(rc!=COMBUF_SUCCESS)armci_die("armci_ReadFromDirect:wait fail",rc);
		ack=(long*)((char *)(msginfo+1)+msginfo->dscrlen+msginfo->datalen);
		armci_wait_long_flag_updated_clear(ack,1);	
		return((char *)(msginfo+1)+msginfo->dscrlen);
	}
}
extern char* serverbuffer;
extern int _sr8k_server_buffer_index;
void  armci_WriteToDirect(int proc,request_header_t *msginfo, void* data){
	char *buf;
	int auth,rc;
	long *ack;
	Cb_node_rt remote;
     int dst=armci_clus_id(proc);
	static Cb_msg msg;unsigned int ev;
     remote.type = CB_NODE_RELATIVE;
     remote.ndes = _armci_group;
     remote.node = dst;
	if(DEBUG2){printf("\n%d: in armci_WriteToDirect-1 sending bytes=%d to %d op=%d\n",armci_me,msginfo->datalen,dst,msginfo->operation);fflush(stdout);}
	if(msginfo->bytes+sizeof(request_header_t)<SMALL_MSG_SIZE){
		if(msginfo->operation==ATTACH || msginfo->operation==ARMCI_SWAP\
		  || msginfo->operation==ARMCI_SWAP_LONG || msginfo->operation==ARMCI_FETCH_AND_ADD\
	       || msginfo->operation==ARMCI_FETCH_AND_ADD_LONG/* || msginfo->operation==LOCK\
		  || msginfo->operation==UNLOCK*/){
			armci_copy(data,(msginfo+1),msginfo->datalen);
			data=(msginfo+1);
		}
		ack=(long*)((char *)(data)+msginfo->datalen);
		*ack=1L;
		auth = server_auths[proc].put_auths[0];
		memset(&msg, 0, sizeof(msg));
		msg.data.addr = (char *)data;
  		msg.data.size = msginfo->datalen+sizeof(long);
		rc = combuf_send(&msg,auth,(sizeof(request_header_t)+msginfo->dscrlen), 0, &ev);
  		if (rc != COMBUF_SUCCESS) armci_die("armci_WriteToDirect combuf_send failed",rc);
	}
	else{
		if(DEBUG2){printf("\n%d: in armci_WriteToDirect-1 sending to %d nobuf\n",armci_me,dst);fflush(stdout);}
		auth = server_auths[proc].lbuf_put_auths[_sr8k_server_buffer_index-SMALL_MSG_NUM];
		ack=(long*)((char *)(data)+msginfo->datalen);
		*ack=1L;
		memset(&msg, 0, sizeof(msg));
          msg.data.addr = (char *)data;
          msg.data.size = msginfo->datalen+sizeof(long);
          rc = combuf_send(&msg,auth,(sizeof(request_header_t)+msginfo->dscrlen), 0, &ev);
          if (rc != COMBUF_SUCCESS) armci_die("armci_WriteToDirect combuf_send failed",rc);
	}
}
/*Sequence of steps
 * 1.Masters wait for the data server on their node to finish creating fields
 * 2.Masters Do a broadcast of the server's recv_field_info_t among master processes
 * 3.Everybody does a local broadcast of the obtained structure
 * 3a.Repeat 2 and 3 for client's recv_field_info_t
 * 4.Everybody get authorizations for 'put' and 'get' from all the data servers
 *   Note that every client gets authorizations to only those small buffers that it uses
 *   and both the large buffers available.
 * 5.Each client has to create tcw's for all the server buffers that correspond to it
*/ 
void armci_client_connect_to_servers(){
int i,j,dst,field_num,auth,rc;
	client_auths = (client_auth_t *)malloc(sizeof(client_auth_t)*armci_nclus);
	if(!client_auths)armci_die("armci_client_connect_to_servers malloc failed",0);
	if(armci_me==armci_master){
		/*step 1 */
		if(DEBUG0){printf("\n%d:about to wait for server_ready_flag\n",armci_me);fflush(stdout);}
		while(*server_ready_flag!=1)*client_ready_flag=124*432;	
		if(DEBUG0){printf("\n%d:done wait for server_ready_flag\n",armci_me);fflush(stdout);}
		/*step 2 */
		armci_msg_gop_scope(SCOPE_MASTERS,(long *)serv_rcv_field_info,3*armci_nclus,"+",ARMCI_LONG);
	}
	/*step 3*/
     armci_msg_clus_brdcst(serv_rcv_field_info,3*armci_nclus*sizeof(long));
	sr8k_server_ready = 1;
	/*step 3a*/
	/*yet to be implemented*/
	*client_ready_flag = 1;
	/*step 4*/
	dst = armci_clus_me;
	for(i=0;i<armci_nclus;i++){
		Cb_node_rt remote; 
		dst++;dst%=armci_nclus;
		remote.type = CB_NODE_RELATIVE;
         	remote.ndes = _armci_group;
        	remote.node = dst;
/*Every process gets authorizations for two large fields, these two large 
 *fields are common
 *to all the processes. In addition to that, each process gets authorizations to a few small
 *fields that are to be used exclusively for that process. Field numbers of these fields are
 *always in sequence and fixed. This reduces the size of the serv_rcv_field_info struct needed.
*/
		/*first go get_sendright for small buffers, these small buffers are exclusive for each process*/
		field_num=(SERV_FIELD_NUM+armci_me*SMALL_BUFS_PER_PROCESS);
		for(j=0;j<SMALL_BUFS_PER_PROCESS;j++,field_num++){
			unsigned int ev;
			static Cb_msg msg;
			rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                                    serv_rcv_field_info[dst].cb_key, field_num+j, -1, &auth);
			if(rc != COMBUF_SUCCESS){
            		printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
            		armci_die("armci_client_connect_to_servers combuf_get_sendright:",rc);
         		}
			client_auths[dst].small_buf_auth[j] = auth;
			/*step 5*/
			client_auths[dst].small_buf_tcwd[j] = armci_rdma_make_tcw(client_buf_array[j],0,SMALL_MSG_SIZE,auth,client_buf_flags[j]);
			if(DEBUG0){printf("\n%d:in client_connect,got smallbuf tcwd=%d for dst=%d index=%d\n",armci_me,client_auths[dst].small_buf_tcwd[j],dst,j);fflush(stdout);}
		}
		/*now get_sendright for large buffers. There large buffers are common to all procs*/
		field_num = SERV_FIELD_NUM_FOR_LARGE_BUF;	
		for(j=0;j<numofbuffers;j++){
			rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                              serv_rcv_field_info[dst].cb_key, field_num+j, -1, &auth);
          	if(rc != COMBUF_SUCCESS){
               	printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
               	armci_die("armci_client_connect_to_servers combuf_get_sendright:",rc);
          	}

			client_auths[dst].large_buf_auth[j] = auth;
			client_auths[dst].large_buf_tcwd[j] = armci_rdma_make_tcw(client_send_buffer[j],0,MSG_BUFLEN_SMALL,auth,client_directbuf_flag[j]);
		}

		/*Get authorization for doing a get from all the servers*/

		rc = combuf_target( (Cb_node *)&remote, sizeof(remote), SERVER_GENBUF_KEY, 0, -1, &auth);
         	if(rc != COMBUF_SUCCESS) armci_die("combuf_target:",rc);
		client_auths[dst].get_auth = auth;	
		rc = combuf_target( (Cb_node *)&remote, sizeof(remote), SERVER_LARGE_GETBUF_KEY, 0,-1,&auth);
		if(rc != COMBUF_SUCCESS) armci_die("combuf_target:",rc);
		client_auths[dst].server_large_getbuf_auth=auth;
	if(DEBUG0){printf("\n%d:recvd authorizations from server %d \n",armci_me,dst);fflush(stdout);}	
	}	
}


/*code for get pipeline*/

void armci_server_direct_send(int proc, char *srcbuf,int dstoffset, int bytes_ack){
/*	armci_rdma_modify_tcw(tcwd,srcbuf,(Cb_size_t)dstoffset,bytes_ack);
	armci_rdma_kick_tcw_put(tcwd);
*/
	char *buf;
	int auth,rc;
	unsigned int complet;
	Cb_node_rt remote;
     int dst=armci_clus_id(proc);
	static Cb_msg msg;unsigned int ev;
     remote.type = CB_NODE_RELATIVE;
     remote.ndes = _armci_group;
     remote.node = dst;
	if(DEBUG2){printf("\n%d: in armci_server_direct_send sending bytes=%d dstoffset=%d to=%d\n",armci_me,bytes_ack,dstoffset,dst);fflush(stdout);}
	memset(&msg, 0, sizeof(msg));
     msg.data.addr = srcbuf;
	msg.data.size = bytes_ack;
	if(DEBUG2){printf("\n%d: in server_send_direct writing to auth=%d  %p\n",armci_me,auth,srcbuf);fflush(stdout);}
	rc = combuf_send(&msg,clauth[proc], dstoffset, 0, &ev); 
	combuf_send_complete(ev,-1,&complet);	
	//armci_rdma_put_wait(proc,(char *)flag_array+SMALL_MSG_NUM*FLAGSIZE);	
	if (rc != COMBUF_SUCCESS) armci_die("armci_server_direct_send combuf_send failed",rc);	
//	armci_rdma_put(srcbuf,dstoffset, bytes_ack,desc);
	
}



static void armci_pipe_advance_buf(int strides, int count[],
                                   char **buf, long **ack, int *bytes )
{
int i, extra;
     for(i=0, *bytes=1; i<=strides; i++)*bytes*=count[i]; 
     extra = ALIGN64ADD((*buf));
     (*buf) +=extra;                  
     if(DEBUG2){ printf("%d: pipe advancing %d %d\n",armci_me, *bytes,extra); fflush(stdout);
     }
     *ack = (long*)((*buf) + *bytes); 
}
 
 
/*\ prepost buffers for receiving data from server (pipeline)
\*/
void armcill_pipe_post_bufs(void *ptr, int stride_arr[], int count[],
                            int strides, void* argvoid)
{
int bytes;
buf_arg_t *arg = (buf_arg_t*)argvoid;
long *ack;
	/*we want to use pipe get buf */	
	if(arg->count==0){
		int dscrlen=((request_header_t*)_sr8k_armci_getbuf_ptr)->dscrlen;
		arg->buf_posted = _sr8k_armci_getbuf_ptr+sizeof(request_header_t)+dscrlen;
		((request_header_t*)_sr8k_armci_getbuf_ptr)->tag.ack=_sr8k_armci_getbuf_ofs\
				+sizeof(request_header_t)+dscrlen;
		((request_header_t*)_sr8k_armci_getbuf_ptr)->tag.data_ptr=_sr8k_armci_getbuf_ptr\
				+sizeof(request_header_t)+dscrlen;
		if(DEBUG2){printf("\ndeed-done count=%d\n",arg->count);fflush(stdout);}
	} 
     armci_pipe_advance_buf(strides, count, &arg->buf_posted, &ack, &bytes);
 
     *ack = 0L;                      /*** clear ACK flag ***/
     arg->buf_posted += bytes+sizeof(long);/* advance pointer for next chunk */
     arg->count++;
     if(DEBUG2){ printf("%d: posting %d pipe receive %d b=%d (%p,%p) ack=%p\n",
          armci_me,arg->count,arg->proc,bytes,arg->buf, arg->buf_posted,ack);
          fflush(stdout);
     }
}
 
 
void armcill_pipe_extract_data(void *ptr, int stride_arr[], int count[],
                               int strides, void* argvoid)
{
int bytes,rc; unsigned int rcvd;
long *ack;
buf_arg_t *arg = (buf_arg_t*)argvoid;
 
     /* for first chunk, wait for the request message to complete */
     if(!arg->count) {
		int dscrlen=((request_header_t*)_sr8k_armci_getbuf_ptr)->dscrlen;
		arg->buf_posted = _sr8k_armci_getbuf_ptr+sizeof(request_header_t)+dscrlen;
		if(DEBUG2){printf("\ndeed-done int extracting count=%d\n",arg->count);fflush(stdout);}
     }
 
     armci_pipe_advance_buf(strides, count, &arg->buf_posted, &ack, &bytes);
 
     if(DEBUG2){printf("%d:extracting pipe  data from %d %d b=%d %p ack=%p\n",
            armci_me,arg->proc,arg->count,bytes,arg->buf_posted,ack); fflush(stdout);
     }
//	rc=combuf_spin_wait(clientgetdirectbufdesc,-1,&rcvd);	
     armci_wait_long_flag_updated(ack, 1); /********* wait for data ********/
     /* copy data to the user buffer identified by ptr */
     armci_read_strided(ptr, strides, stride_arr, count, arg->buf_posted);
     if(DEBUG2 ){printf("%d(c):extracting: data %p first=%f %f %f %f\n",armci_me,
                arg->buf_posted,((double*)arg->buf_posted)[0],((double*)arg->buf_posted)[1],((double*)arg->buf_posted-1)[0],((double*)arg->buf_posted-2)[0]);
                fflush(stdout);
     }
 
     arg->buf_posted += bytes+sizeof(long);/* advance pointer for next chunk */
     arg->count++;
}
 
static int dst_offset; 
void armcill_pipe_send_chunk(void *data, int stride_arr[], int count[],
                             int strides, void* argvoid){
int bytes, bytes_ack;
buf_arg_t *arg = (buf_arg_t*)argvoid;
long *ack;
	if(!arg->count){
		/*first send, reset the destination offset*/
		if(DEBUG2){printf("\n*****resetting destionation offset****\n");fflush(stdout);}
		dst_offset = _sr8k_armci_getbuf_ofs;
	} 
    dst_offset+=ALIGN64ADD(arg->buf_posted);
	armci_pipe_advance_buf(strides, count, &arg->buf_posted, &ack, &bytes);
    armci_pipe_advance_buf(strides, count, &arg->buf, &ack, &bytes);
    bytes_ack = bytes+sizeof(long);
 
	if(DEBUG2){ 
		printf("%d:SENDING pipe data=%p %d to %d %p buf[0]=%f b=%d %p)\n",armci_me,
                 data,arg->count, arg->proc, arg->buf,((double *)arg->buf)[0], bytes, ack); 
		fflush(stdout);
     }
 
     /* copy data to buffer */
    armci_write_strided(data, strides, stride_arr, count, arg->buf);
    *ack=1;

    armci_server_direct_send(arg->proc, arg->buf, dst_offset, bytes_ack);
 
    if(DEBUG2){ printf("%d:  out of send %d bytes=%d first=%f\n",armci_me,
               arg->count,bytes,((double*)arg->buf)[0]); fflush(stdout);
    }
 
    arg->buf += bytes+sizeof(long);        /* advance pointer for next chunk */
    arg->buf_posted += bytes+sizeof(long); /* advance pointer for next chunk */
	dst_offset += bytes+sizeof(long);      /* advance destination offser*/
    arg->count++;
}

//extern void armci_util_spin(int n,void *notused);
 
void armci_pipe_send_req(int proc, void *vbuf, int len)
{
int rc,k=0;
unsigned int recvd;
int dst=armci_clus_id(proc);
	//((request_header_t*)vbuf)->tag.data_ptr = _sr8k_armci_getbuf_ptr;
	//((request_header_t*)vbuf)->tag.ack=_sr8k_armci_getbuf_ofs;
	if(_sr8k_using_smallbuf)k=-1;
	_sr8k_wait_largebuf_ack(dst);
	while(server_smallbuf_pending_count[dst]!=(client_smallbuf_pending_count[dst]+k))
          rc=combuf_spin_wait(clientpendingopbufdesc,-1,&recvd); 
	rc=-1;
#if 0
	ackcount=0;
	while((rc=combuf_cswap(client_auths[armci_clus_id(proc)].server_large_getbuf_auth,0,10+armci_me,0,0))) {
		printf("\n%d: in cswap value at server=%d\n",armci_me,rc);
		armci_util_spin(10000+armci_me*armci_me*100,NULL);	
		ackcount++;
		if(ackcount==10000)armci_die("cswap not completing",ackcount);
	}
	ackcount=0;
#endif
	if(!_sr8k_using_smallbuf)
		_sr8k_pipeget_req=1;
 	armci_send_req_msg(proc, vbuf, len);
	_sr8k_pipeget_req=0;
}

