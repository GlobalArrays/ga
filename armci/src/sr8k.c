/* $Id: sr8k.c,v 1.3 2002-01-29 23:17:34 vinod Exp $
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
#include <hmpp/nalloc.h>
#include <hxb/combuf.h>
#include <hxb/combuf_node.h>
#include <hxb/combuf_returns.h>

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

 
static long reg_idlist[MAX_REGIONS]; /* keys for rdma/shmem segments on this node */
static int  reg_num=0;          /* number of rdma/shmem segments on this node */
static reg_auth_t *rem_reg_list;/* info about rdma segments on every node */
static reg_auth_t **rem_reg;    /* info about rdma segments on every node */
static long *reg_id;            /* buffer to exchange info about new segments */
static int  *proc_reg;          /* count of shmem/rdma segments on every node */
extern ndes_t _armci_group;     /* processor group -- set in clusterinfo */
double *armci_internal_buffer;  /* work buffer for accumulate -- in RDMA area */
static char *bufstart, *bufend; /* address range for local rdma area */
static int bufdesc;             /* descriptor for local rdma buffer */

static int descs[2];            /* descriptors for get pipeline buffers */
static char *getbuf[2];         /* the get pipeline buffers themselves */

#define BUF_KEY 2020L
#define PAGE_SIZE       0x1000
#define ROUND_UP_PAGE(size)  ((size + (PAGE_SIZE-1)) & ~(PAGE_SIZE-1))
#define NOOPTION        0
#define NOFLAG          0
#define NOATOMIC        0

#define DEBUG0 0
#define DEBUG1 1

/* There is scope for "tuning" these two... */
#define LPUT 180000
#define LGET 450000
#define FLAGSIZE 8
#define ALIGNUP(x) (4*((3+x)/4))

/**TCW manipulation routines****************************************/

      /**** Routines for PUT ****/

static char *tcw_flag;

static int armci_rdma_make_tcw(void *src, Cb_size_t off, int bytes, int desc, char *flag)
{
int rc;
static Cb_msg msg;
int tcwd;
     memset(&msg, 0, sizeof(msg));
     msg.data.addr = src;
     msg.data.size = (Cb_size_t)bytes;
     if ( (rc=combuf_make_tcw(&msg, desc, off, flag, &tcwd)) != COMBUF_SUCCESS)
         armci_die("combuf_make_tcw failed",rc);
     if(DEBUG0){printf("%d:put dsc=%d off=%d\n",armci_me,desc,off); fflush(stdout);}
     return tcwd;
}

static void armci_rdma_modify_tcw(int tcwd, char *src, Cb_size_t off, int bytes)
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

static void armci_rdma_kick_tcw_put(int tcwd)
{
    int rc ;
    if ( (rc=combuf_kick_tcw(tcwd, COMBUF_SEND_NOBLOCK)) != COMBUF_SUCCESS)
        armci_die("combuf_kick_tcw failed",rc);
}

static void armci_rdma_put_wait(int tcwd, char *flag)
{
    while(!combuf_check_sendflag(flag)); /* Wait */
}

      /**** Routines for GET ****/

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

static void armci_rdma_get_wait(int desc)
{
    int rc;
    unsigned int recd;
    if ((rc=combuf_spin_wait(desc,-1,&recd)) != COMBUF_SUCCESS)
	                          armci_die("combuf_spin_wait failed",rc);
}

      /**** Other routines ****/

static void armci_rdma_free_tcw(int tcwd)
{
    int rc;
    if ( (rc=combuf_free_tcw(tcwd)) != COMBUF_SUCCESS)
        armci_die("combuf_free_tcw failed",rc);
}

/*******************************************************************/

/*\ intialization of data structures 
 *  called by armci_register_shmem in 1st ARMCI_Malloc call in  ARMCI_Init
\*/ 
void armci_init_sr8k()
{
     int rc;
     Cb_object_t oid;
     int i, bytes = armci_nclus*MAX_REGIONS*sizeof(reg_auth_t);
     long key;
     Cb_opt_region options;

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
     printf("BUFSIZE = %d, LPUT = %d, LGET = %d\n",BUFSIZE,LPUT,LGET);
     if (2*LPUT > BUFSIZE) armci_die("LPUT must be <= BUFSIZE/2",0);

     /* We need space for the work buffer (BUFSIZE) and two receive
      * fields (LGET) along with a flag for each of them (FLAGSIZE) */

     bytes = ROUND_UP_PAGE(BUFSIZE+FLAGSIZE+2*(LGET+FLAGSIZE));
     key   = BUF_KEY + armci_me-armci_master;
     if(combuf_object_get(key, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &oid)
                       != COMBUF_SUCCESS) armci_die("combufget buf failed",0);
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

     memset(&options, 0, sizeof(options));
     options.flag.addr = getbuf[0]+LGET;   /* set up the spin_wait flag */
     options.flag.size = FLAGSIZE;
     if( (rc=combuf_create_field(oid, getbuf[0], LGET, 2,
                            &options, COMBUF_CHECK_FLAG, &descs[0])) != COMBUF_SUCCESS)
                                          armci_die("combufget field 2 failed",rc);
     options.flag.addr = getbuf[1]+LGET;   /* set up the spin_wait flag */
     options.flag.size = FLAGSIZE;
     if( (rc=combuf_create_field(oid, getbuf[1], LGET, 3,
                            &options, COMBUF_CHECK_FLAG, &descs[1])) != COMBUF_SUCCESS)
                                          armci_die("combufget field 3 failed",rc);

     if(DEBUG0){printf("%d:armci_init_sr8k initialization done\n",armci_me);
                fflush(stdout);}
}
     


/*\ registers new rdma area - called in every call to ARMCI_Malloc
\*/
void armci_register_shmem(void *my_ptr, long size, long *idlist, long off)
{
     int i,dst,found=0;
     long id = idlist[2];
     long reg_size=0;

     if(DEBUG0){printf("%d: registering id=%ld size=%ld\n",armci_me,id,size);
                fflush(stdout);}

     /* init data structures when called first time */
     if(!reg_num) armci_init_sr8k();

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
        printf("%d: regist id=%ld found=%d size=%ld\n",armci_me,id,found,reg_size);
        fflush(stdout);}

     bzero(reg_id,3*armci_nclus*sizeof(long));

     /* store id and ptr into array of longs: sizeof(long) must be >=sizeof(void*)*/
     if(armci_me==armci_master){
        if(!found && size){
           reg_id[3*armci_clus_me]=id;
           reg_id[3*armci_clus_me+1]=(long)armci_shmem_reg_ptr(i);
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
int _j;
char *ps=src_ptr, *pd=dst_ptr;

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

  /*
   * There doesn't seem to be much scope for optimising this routine.
   * I cannot find a get equivalent of combuf_stride_send() [BPE]
   */

int _j;
char *ps=src_ptr, *pd=dst_ptr;
      for (_j = 0;  _j < count;  _j++){
          armcill_get(ps, pd, bytes, proc);
          ps += src_stride;
          pd += dst_stride;
      }
}


typedef struct {
       Cb_size_t off;
       int desc;
} sr8k_mutex_t;
static sr8k_mutex_t *_mutex_array;

/*\ allocate the specified number of mutexes on the current SMP node
\*/
void armcill_allocate_locks(int num)
{
int bytes = num*sizeof(int);
int **locks = (int**)malloc(armci_nproc*sizeof(int*));
int rc,i;

    if(armci_me != armci_master)bytes=0;
    _mutex_array = (sr8k_mutex_t*)malloc(sizeof(sr8k_mutex_t)*armci_nclus);
    if(!_mutex_array)
         armci_die("armcill_allocate_locks: malloc failed",armci_nclus);
    if(!locks) armci_die("armcill_allocate_locks: malloc 2 failed",armci_nproc);

    rc = ARMCI_Malloc((void**)locks, bytes);
    if(rc) armci_die("armcill_allocate_locks:failed to allocate array",rc);
    if(bytes)bzero(locks[armci_me],bytes);

    for(i=0; i< armci_nclus; i++){
        char *ps = (char*)locks[armci_clus_info[i].master];
        _mutex_array[i].off  = ps - rem_reg[i][0].ptr;
        _mutex_array[i].desc = rem_reg[i][0].gauth;
        if(DEBUG0){printf("%d:allocate %d locks %p,%p %p %p %d\n",armci_me,num,
                   locks[0],locks[1], ps, rem_reg[i][0].ptr,rem_reg[i][0].size);
                   fflush(stdout);
                  }
        if(_mutex_array[i].off > rem_reg[i][0].size) /*verify if in 1st region*/
                           armci_die("armcill_allocate_locks:offset error",i);
    }
    free(locks);
}



/*\ lock specified mutex on node where process proc is running
\*/
void armcill_lock(int mutex, int proc)
{
#if 1
int desc,node = armci_clus_id(proc);
Cb_size_t off;

    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
    if(DEBUG0){
      printf("%d: lock %d on %d off=%d\n",armci_me,mutex,proc, off);fflush(stdout);}
//    while(combuf_swap(desc,off,1));
    while(combuf_cswap(desc,off,1,0,2)); /* wait 2ms for condition to be met*/ 
#endif
}


/*\ unlock specified mutex on node where process proc is running
\*/
void armcill_unlock(int mutex, int proc)
{
#if 1
int desc,node = armci_clus_id(proc);
Cb_size_t off;

    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
    combuf_swap(desc,off,0);
#endif
}
