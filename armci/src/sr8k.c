/* $Id: sr8k.c,v 1.2 2001-10-20 05:46:20 d3h325 Exp $
 *
 * Hitachi SR-8000 specific code 
 *
 * *** WE NEED TO OPTIMIZE armcill_put/get AND armcill_put2D/get2D ******* 
 * *** latency by using TCW and combuf_kick_tcw_fast()
 * *** bandwidth by overlapping memory copy with RDMA nonblocking communication
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
}reg_auth_t;

 
static long reg_idlist[MAX_REGIONS]; /* keys for rdma/shmem segments on this node */
static int  reg_num=0;          /* number of rdma/shmem segments on this node */
static reg_auth_t *rem_reg_list;/* info about rdma segments on every node */
static reg_auth_t **rem_reg;    /* info about rdma segments on every node */
static long *reg_id;            /* buffer to exchange info about new segments */
static int  *proc_reg;          /* count of shmem/rdma segments on every node */
extern ndes_t _armci_group;     /* processor group -- set in clusterinfo */
double *armci_internal_buffer;  /* work buffer for accumulate -- in RDMA area */
static char *_internal_buffer2; /* another  work buffer - page size */
static char *bufstart, *bufend; /* address range for local rdma area */
static int bufdesc;             /* descriptor for local rdma buffer */

#define BUF_KEY 2020L
#define PAGE_SIZE       0x1000
#define ROUND_UP_PAGE(size)  ((size + (PAGE_SIZE-1)) & ~(PAGE_SIZE-1))
#define NOOPTION        0
#define NOFLAG          0
#define NOATOMIC        0

#define DEBUG0 0
#define DEBUG1 1
 

/*\ intialization of data structures 
 *  called by armci_register_shmem in 1st ARMCI_Malloc call in  ARMCI_Init
\*/ 
void armci_init_sr8k()
{
     Cb_object_t oid;
     int i, bytes = armci_nclus*MAX_REGIONS*sizeof(reg_auth_t);
     long key;

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
     bytes = ROUND_UP_PAGE(BUFSIZE) + PAGE_SIZE;
     key   = BUF_KEY + armci_me-armci_master;
     if(combuf_object_get(key, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &oid)
                       != COMBUF_SUCCESS) armci_die("combufget buf failed",0);
     if(combuf_map(oid, 0, (Cb_size_t)bytes, 0, (char**)&armci_internal_buffer)
                       != COMBUF_SUCCESS) armci_die("combuf map for buf failed",0);
     if(combuf_create_field(oid, (char*)armci_internal_buffer,bytes,FIELD_NUM,
                            NOFLAG, NOOPTION, &bufdesc) != COMBUF_SUCCESS)
                                          armci_die("combufget field failed",0);

     /* store the range of addresses taken by local buffer in rdma memory */
     bufstart = (char*) armci_internal_buffer;
     bufend   = (char*) (armci_internal_buffer+bytes);
     _internal_buffer2 = bufend - PAGE_SIZE;

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
         int rc, auth;

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
         rem_reg[dst][proc_reg[dst]].pauth = auth;      
         rem_reg[dst][proc_reg[dst]].size  = len;      
         rem_reg[dst][proc_reg[dst]].ptr   = ptr;      

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


/*\  contiguous put  dst(proc) = src
\*/
void armcill_put(void *src, void *dst, int bytes, int proc)
{
Cb_size_t off;
int found =0, i, node = armci_clus_id(proc);
int master = armci_clus_info[node].master;
char *ps, *pe, *ptr=(char*)dst;
int desc;

    if(DEBUG0){
       printf("%d:put s=%p d=%p p=%d\n",armci_me,src,dst,proc); fflush(stdout);}

    /* find shmem region corresponding to dst address */
    for(i=0; i< proc_reg[node]; i++){
       ps = rem_reg[node][i].ptr;
       pe = rem_reg[node][i].size + ps;
       if((ptr>=ps) && (ptr<pe)){ found=1; desc= rem_reg[node][i].pauth; break;}
    }
    if(!found) armci_die("armcill_put: bad dst address for p=",proc);
    off = (Cb_size_t) ( ptr - ps);
    
    if((src >= (void*)bufstart) && (src <(void*)bufend)){

       /* no need to copy - data is in the rdma buffer */
        (void)armci_rdma_put(src, off, bytes, desc);

    } else for(i = 0; i< bytes;){ /* send data piece by piece through rdma buf*/

        int len = ((bytes -i)<BUFSIZE)? (bytes -i): BUFSIZE;
        armci_copy(src,armci_internal_buffer,len);
        (void)armci_rdma_put(armci_internal_buffer, off, len, desc);
        i   +=len;
        off +=len;
        src = len +(char*)src;

    }
}


/*\  contiguous get  src = dst(proc)
\*/
void armcill_get(void *src, void *dst, int bytes, int proc)
{
Cb_size_t off, buf_off;
int found =0, i, node = armci_clus_id(proc);
int master = armci_clus_info[node].master;
char *ps, *pe, *ptr=(char*)src;
int desc;

    if(DEBUG0){
       printf("%d:get s=%p d=%p p=%d\n",armci_me,src,dst,proc); fflush(stdout);}

    /* find shmem region corresponding to dst address */
    for(i=0; i< proc_reg[node]; i++){
       ps = rem_reg[node][i].ptr;
       pe = rem_reg[node][i].size + ps;
       if((ptr>=ps) && (ptr<pe)){ found=1; desc= rem_reg[node][i].gauth; break;}
    }

    if(!found) armci_die("armcill_get: bad src address for p=",proc);
    off = (Cb_size_t) ( ptr - ps);
   
    if((dst >= (void*)bufstart) && (dst <(void*)bufend)){

       /* no need to copy - this is our rdma buffer */
        buf_off = (Cb_size_t)(((char*)dst) - bufstart);
        (void)armci_rdma_get(off, desc, buf_off, bufdesc, bytes);

    } else for(i = 0; i< bytes;){ /* send data piece by piece through rdma buf*/

        int len = ((bytes -i)<BUFSIZE)? (bytes -i): BUFSIZE;
        (void)armci_rdma_get(off, desc, 0, bufdesc, len);
        armci_copy(armci_internal_buffer,dst, len);
        i   +=len;
        off +=len;
        dst = len +(char*)dst;

    }
}


/*\ strided put
\*/
void armcill_put2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j;
char *ps=src_ptr, *pd=dst_ptr;
      for (_j = 0;  _j < count;  _j++){
          armcill_put(ps, pd, bytes, proc);
          ps += src_stride;
          pd += dst_stride;
      }
}


/*\ strided get: source is at proc, destination on calling process 
\*/
void armcill_get2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
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
