/* $Id: elan.c,v 1.21 2003-05-15 16:11:51 edo Exp $ */
#include <elan/elan.h>
#include <elan3/elan3.h>
#include <stdio.h>
#include <stdlib.h>
#include "armcip.h"
#include "copy.h"

#define DEBUG_ 0
#ifdef QSNETLIBS_VERSION
#if QSNETLIBS_VERSION_CODE < QSNETLIBS_VERSION(1,4,6) 
#   define VCALLS 0
#else
static ELAN_PGCTRL *_pgctrl;
static void *_qd;
#   define VCALLS 1
#   define QSNETLIBS_NEWAPI
#endif
#else
#   define VCALLS 0
#endif

#ifdef ELAN_ACC
static int armci_server_terminating=0;
static ELAN_MAIN_QUEUE *mq;
static int armci_request_from=-1;
static int armci_request_to=-1;

typedef unsigned short int ops_t;
static ops_t** armci_elan_fence_arr;
static ops_t *ops_pending_ar;
static ops_t *ops_done_ar;

#define _ELAN_SLOTSIZE 320
#define MSG_DATA_LEN (_ELAN_SLOTSIZE - sizeof(request_header_t))

/* elan 1.3 defined DBG_QUEUE */
#if !defined(QSNETLIBS_VERSION_CODE) && defined(DBG_QUEUE)
#       define OLD_QSNETLIBS
        static ELAN_PGCTRL *armci_pgctrl;
#endif

#ifdef OLD_QSNETLIBS
#define MY_PUT(src,dst,bytes,p) \
          elan_wait(elan_doput(armci_pgctrl,(src),(dst),NULL,(bytes),p,0),\
                    elan_base->waitType)
#define MY_GET(src,dst,len,p)\
        elan_wait(elan_doget(armci_pgctrl,src,dst,len,p,0),elan_base->waitType)
               
#else
#define MY_PUT(src,dst,bytes,p) \
        elan_wait(elan_put(elan_base->state,(src),(dst),(bytes),p),\
                  elan_base->waitType)
#define MY_GET(src,dst,len,p)\
        elan_wait(elan_get(elan_base->state,src,dst,len,p),elan_base->waitType)
#endif


void armci_init_connections()
{

ELAN_QUEUE *q;
int nslots=armci_nproc+562, slotsize=_ELAN_SLOTSIZE;


    if ((q = elan_gallocQueue(elan_base, elan_base->allGroup)) == NULL)
            armci_die( "elan_gallocElan",0 );

#if !defined(QSNETLIBS_NEWAPI)
    if (!(mq = elan_mainQueueInit( elan_base->state, q, nslots, slotsize)))
            armci_die("Failed to to initialise Main Q",0);
#else
    if (!(mq = elan_mainQueueInit( elan_base->state, q, nslots, slotsize,
          0)))armci_die("Failed to initialise Main Q",0);
    _qd = elan_gallocElan(elan_base, elan_base->allGroup, E3_QUEUE_ALIGN,
                              elan_pgvGlobalMemSize(elan_base->state));

    if(!_qd) armci_die("failed elan_gallocElan 1",0);
    elan_gsync(elan_base->allGroup);
    _pgctrl = elan_putgetInit(elan_base->state, _qd, 16, 4096, 4096, 32, ELAN_PGVINIT);
    if(!_pgctrl) armci_die("failed elan_gallocElan 2",0);
    elan_gsync(elan_base->allGroup);
#endif

    if(armci_me == armci_master) {
        if(!(ops_done_ar=(ops_t*)calloc(armci_nproc,sizeof(ops_t))))
             armci_die("malloc failed for ARMCI ops_done_ar",0);
    }

    armci_elan_fence_arr = (ops_t**)malloc(armci_nproc*sizeof(ops_t*));
    if(!armci_elan_fence_arr)armci_die("malloc failed for ARMCI fence array",0);
    if(ARMCI_Malloc((void**)armci_elan_fence_arr, armci_nclus*sizeof(ops_t)))
             armci_die("failed to allocate ARMCI fence array",0);
    bzero(armci_elan_fence_arr[armci_me],armci_nclus*sizeof(ops_t));

    if(!(ops_pending_ar=(ops_t*)calloc(armci_nclus,sizeof(ops_t))))
         armci_die("malloc failed for ARMCI ops_pending_ar",0);

#ifdef OLD_QSNETLIBS
    /* initialize control descriptor for put/get */
    armci_pgctrl = elan_putgetInit(elan_base->state, 32, 8);
    if(!armci_pgctrl) armci_die("armci_init_con: elan_putgetInit failed",0);
#endif

    if(MessageSndBuffer){
      ((request_header_t*)MessageSndBuffer)->tag = (void*)0;
    }else armci_die("armci_init_connections: buf not set",0);
}


/*\ server sends ACK to client when request is processed
\*/
static void armci_send_ack()
{
ops_t val=0;
ops_t *buf = armci_elan_fence_arr[armci_request_from] + armci_clus_me;

#if 0
    if(armci_me==0)
    printf("%d:server sends ack p=%d fence=%p slot %p got=%d\n", armci_me, 
           armci_request_from, armci_elan_fence_arr[armci_request_from],buf,
           ops_done_ar[armci_request_from]+1); fflush(stdout);
#endif

    val = ++ops_done_ar[armci_request_from];

    MY_PUT(&val,buf,sizeof(ops_t),armci_request_from);
}


ops_t armci_check_int_val(ops_t *v)
{
  return (*v);
}


void armci_elan_fence(int p)
{
    long loop=0;
    int cluster = armci_clus_id(p);
    ops_t *buf = armci_elan_fence_arr[armci_me] + cluster;
    long res = ops_pending_ar[cluster] - armci_check_int_val(buf);

#if 0
    if(ops_pending_ar[cluster])
    printf("%d: client fencing proc=%d fence=%p slot %p pending=%d got=%d\n", 
           armci_me, p, armci_elan_fence_arr[armci_me], buf, 
           ops_pending_ar[cluster], armci_check_int_val(buf)); fflush(stdout);
#endif

    while(res){
       if(++loop == 1000) { loop=0; usleep(1);  }
       armci_util_spin(loop, buf);
       res = ops_pending_ar[cluster] - armci_check_int_val(buf);
    }
}


void armci_call_data_server()
{
int usec_to_poll =0;
char buf[_ELAN_SLOTSIZE];

    if(DEBUG_){
        printf("%d(server): waiting for request\n",armci_me); fflush(stdout);
    }
 
    while(1){
        elan_queueWait(mq, buf, usec_to_poll );
        armci_data_server(buf);
        armci_send_ack();
    }

    if(DEBUG_) {printf("%d(server): done! closing\n",armci_me); fflush(stdout);}
}


/*\ server receives request 
\*/
void armci_rcv_req(void *mesg,
                   void *phdr, void *pdescr, void *pdata, int *buflen)
{
    request_header_t *msginfo = (request_header_t *)mesg;
    *(void **)phdr = msginfo;
    armci_request_from = msginfo->from;
    armci_request_to = msginfo->to;

    if(DEBUG_) {
       printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d %p\n",
              armci_me, msginfo->operation, msginfo->dscrlen,
              msginfo->datalen, msginfo->from,msginfo->tag); fflush(stdout);
    }

    *buflen = MSG_BUFLEN - sizeof(request_header_t);
    *(void **)pdescr = msginfo+1;
    *(void **)pdata  = msginfo->dscrlen + (char*)(msginfo+1);

    if(msginfo->bytes){
       if(msginfo->operation != GET){
          int payload = msginfo->datalen;
          int off =0;
          char *rembuf = msginfo->tag;
          if(msginfo->dscrlen > MSG_DATA_LEN){
             payload += msginfo->dscrlen;
             *(void **)pdescr = MessageRcvBuffer;
             off = msginfo->dscrlen;
          }else rembuf += msginfo->dscrlen;

          if((msginfo->dscrlen+msginfo->datalen)> MSG_DATA_LEN){
             void *zero=(void*)0;
             void *flag_to_clear;
             *(void **)pdata  = MessageRcvBuffer + off; 

             MY_GET(rembuf,MessageRcvBuffer,payload, msginfo->from);

             if(DEBUG_){ printf("%d:in serv &tag=%p tag=%p\n",armci_me,
                                flag_to_clear, msginfo->tag); fflush(stdout);
             }

             /* mark sender buffer as free -- flag is before descriptor */
             flag_to_clear = ((void**)msginfo->tag)-1; 
             MY_PUT(&zero,flag_to_clear,sizeof(void*),msginfo->from);
          }
        }
    }else
        *(void**)pdescr = NULL;
}


/*\ server sends data to client buffer
\*/
void armci_WriteToDirect(int dst, request_header_t *msginfo, void *buffer)
{
   armci_die("armci_WriteToDirect: should not be called in this case",0);
}

char *armci_ReadFromDirect(int proc, request_header_t * msginfo, int len)
{
    char *buf = (char*) msginfo;
    return(buf);
}


/*\ send request to server thread
\*/
int armci_send_req_msg(int proc, void *vbuf, int len)
{
    char *buf = (char*)vbuf;
    request_header_t *msginfo = (request_header_t *)buf;
    int cluster = armci_clus_id(proc);
    int size=_ELAN_SLOTSIZE;
    int proc_serv = armci_clus_info[cluster].master;

    ops_pending_ar[cluster]++;

    if((msginfo->dscrlen+msginfo->datalen)> MSG_DATA_LEN){
      /* set message tag -> has pointer to client buffer with descriptor+data */
      msginfo->tag = (void *)(buf + sizeof(request_header_t));
      if(DEBUG_){ printf("%d:in send &tag=%p tag=%p\n",armci_me,&msginfo->tag,
                msginfo->tag); fflush(stdout);
      }
    } else /* null tag means buffer is free -- true after elan_queueReq*/;

    elan_queueReq(mq, proc_serv, vbuf, size); /* vbuf is sent/copied out */

#if 0
    if(armci_me==0){
      printf("%d sent request %d to (%d,%d)\n",armci_me,ops_pending_ar[proc], 
      proc,proc_serv); fflush(stdout);
    }
#endif

    return 0;
}



void armci_wait_for_server()
{
  armci_server_terminating=1;
}

void armci_transport_cleanup() {}
void armci_client_connect_to_servers(){}
void armci_server_initial_connection(){}

#endif


/************************************************************************/
#ifdef _ELAN_LOCK_H

#define MAX_LOCKS 4
static ELAN_LOCK *my_locks, *all_locks;
static int num_locks=0;

/* NOTE that if ELAN is defined the scope of locks is limited to SMP
   and we do not call the interfaces below */


/*\ allocate and initialize num locks on each processor (collective call)
\*/
void armcill_allocate_locks(int num)
{
   char *buf;
   int i,elems;
   long mod;

   if(MAX_LOCKS<num)armci_die2("too many locks",MAX_LOCKS,num);
   num_locks = num;

   /* allocate memory to hold lock info for all the processors */
   buf = malloc(armci_nproc*num *sizeof(ELAN_LOCK) + ELAN_LOCK_ALIGN);
   if(!buf) armci_die("armcill_init_locks: malloc failed",0);

   mod = ((long)buf) %ELAN_LOCK_ALIGN;
   all_locks = (ELAN_LOCK*)(buf +ELAN_LOCK_ALIGN-mod); 
   if(((long)all_locks) %ELAN_LOCK_ALIGN) 
        armci_die2("lock alligment failed",mod,ELAN_LOCK_ALIGN);
   bzero(all_locks,armci_nproc*num *sizeof(ELAN_LOCK));

   /* initialize local locks */
   my_locks = all_locks + armci_me * num;
   for(i=0; i<num; i++)
       elan_lockInit(elan_base->state, my_locks+i, ELAN_LOCK_NORMAL);

   /* now we use all-reduce to exchange locks info among everybody */
   elems = (num*armci_nproc*sizeof(ELAN_LOCK))/sizeof(long);
   if((num*sizeof(ELAN_LOCK))%sizeof(long)) 
       armci_die("armcill_init_locks: size mismatch",sizeof(ELAN_LOCK));
   armci_msg_lgop((long*)all_locks,elems,"+");
#if 0
   if(armci_me == 0){
     for(i=0; i<num*armci_nproc; i++) printf("%d:(%d) master=%d type=%d\n",i,elems,(all_locks+i)->lp_master, (all_locks+i)->lp_type);
   }
#endif
   armci_msg_barrier();
}


void armcill_lock(int m, int proc)
{
ELAN_LOCK *rem_locks = (ELAN_LOCK*)(all_locks + proc*num_locks);

   if(m<0 || m>= num_locks) armci_die2("armcill_lock: bad lock id",m,num_locks);
   if(proc<0 || proc>= armci_nproc) armci_die("armcill_lock: bad proc id",proc);

   elan_lockLock(elan_base->state, rem_locks + m, ELAN_LOCK_BUSY);
}

void armcill_unlock(int m, int proc)
{
ELAN_LOCK *rem_locks = (ELAN_LOCK*)(all_locks + proc*num_locks);

   if(m<0 || m>= num_locks) armci_die2("armcill_unlock:bad lockid",m,num_locks);
   if(proc<0 || proc>=armci_nproc)armci_die("armcill_unlock: bad proc id",proc);

   elan_lockUnLock(elan_base->state, rem_locks + m);
}
     
#endif

/************************************************************************/
#if VCALLS 

#define MAX_VECS 600
static void* _src[MAX_VECS], *_dst[MAX_VECS];
void armcill_get2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j, issued=0;
char *ps=src_ptr, *pd=dst_ptr;

    
#if 0
    printf("%d: getv %d\n", armci_me, count); fflush(stdout);
#endif
    for (_j = 0;  _j < count;  _j++ ){
        _src[issued] = ps;
        _dst[issued] = pd;
        ps += src_stride;
        pd += dst_stride;
        issued++;
        if(issued == MAX_VECS){
           elan_wait(elan_getv(_pgctrl,_src,_dst,bytes,issued,proc),100);
           issued=0;
        } 
    }
    if(issued)elan_wait(elan_getv(_pgctrl,_src,_dst,bytes,issued,proc),100);
}

void armcill_put2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j, issued=0;
char *ps=src_ptr, *pd=dst_ptr;

#if 0
    printf("%d: putv %d\n", armci_me, count); fflush(stdout);
#endif

    for (_j = 0;  _j < count;  _j++ ){
        _src[issued] = ps;
        _dst[issued] = pd;
        ps += src_stride;
        pd += dst_stride;
        issued++;
        if(issued == MAX_VECS){
           elan_wait(elan_putv(_pgctrl,_src,_dst,bytes,issued,proc),100);
           issued=0;
        }
    }
    if(issued)elan_wait(elan_putv(_pgctrl,_src,_dst,bytes,issued,proc),100);
}

void armcill_wait_get(){}
void armcill_wait_put(){}

#else

#ifdef _ELAN_PUTGET_H

/* might have to use MAX_SLOTS<MAX_PENDING due to throttling a problem in Elan*/
#define MAX_PENDING 6 
#define MAX_SLOTS 64
#define ZR  (ELAN_EVENT*)0

static ELAN_EVENT* put_dscr[MAX_SLOTS]= {
ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,
ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR};

static ELAN_EVENT* get_dscr[MAX_SLOTS] = {
ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,
ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR,ZR};

static int cur_get=0;
static int cur_put=0;
static int pending_get=0;
static int pending_put=0;

int kwach=0;
/*\ strided put, nonblocking
\*/
void armcill_put2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j, i, batch, issued=0;
char *ps=src_ptr, *pd=dst_ptr;

#if 1
    for (_j = 0;  _j < count;  ){
      /* how big a batch of requests can we issue */
      batch = (count - _j )<MAX_PENDING ? count - _j : MAX_PENDING; 
      _j += batch;
      for(i=0; i< batch; i++){
        if(put_dscr[cur_put])elan_wait(put_dscr[cur_put],100); 
        else pending_put++;
#if 1
        put_dscr[cur_put]= elan_put(elan_base->state,ps, pd,(size_t)bytes,proc);
#else
        elan_wait(elan_put(elan_base->state, ps, pd, (size_t)bytes, proc),1000);
#endif
        issued++;
        ps += src_stride;
        pd += dst_stride;
        cur_put++;
        if(cur_put>=MAX_PENDING)cur_put=0;
      }
    }

    if(issued != count) 
       armci_die2("armci-elan put:mismatch %d %d \n", count,issued);
#else
     for (_j = 0;  _j < count;  _j++){
       elan_wait(elan_put(elan_base->state, ps, pd, (size_t)bytes, proc),1000);
       ps += src_stride;
       pd += dst_stride;
     }
#endif
}




/*\ strided get, nonblocking
\*/
void armcill_get2D(int proc, int bytes, int count, void* src_ptr,int src_stride,
                                                   void* dst_ptr,int dst_stride)
{
int _j, i, batch, issued=0;
char *ps=src_ptr, *pd=dst_ptr;

#if 1
    for (_j = 0;  _j < count;  ){
      /* how big a batch of requests can we issue */
      batch = (count - _j )<MAX_PENDING ? count - _j : MAX_PENDING;
      _j += batch;
      for(i=0; i< batch; i++){
#if 1
        if(get_dscr[cur_get])elan_wait(get_dscr[cur_get],100); 
        else pending_get++;
        get_dscr[cur_get]=elan_get(elan_base->state,ps,pd, (size_t)bytes, proc);
#else
        elan_wait(elan_get(elan_base->state, ps, pd, (size_t)bytes, proc),1000);
#endif
        issued++;
        ps += src_stride;
        pd += dst_stride;
        cur_get++;
        if(cur_get>=MAX_PENDING)cur_get=0;
      }
    }

    if(issued != count) 
       armci_die2("armci-elan get:mismatch %d %d \n", count,issued);
#else
      for (_j = 0;  _j < count;  _j++){
        elan_wait(elan_get(elan_base->state, ps, pd, (size_t)bytes, proc),1000);
        ps += src_stride;
        pd += dst_stride;
      }
#endif
}

void armcill_wait_get()
{
int i;
    
    if(!pending_get)return;
    else pending_get=0;
    for(i=0; i<MAX_PENDING; i++) if(get_dscr[i]){
        elan_wait(get_dscr[i],100); 
        get_dscr[i]=(ELAN_EVENT*)0;
    }
}


void armcill_wait_put()
{
int i;
    if(!pending_put)return;
    else pending_put=0;
    for(i=0; i<MAX_PENDING; i++) if(put_dscr[i]){
        elan_wait(put_dscr[i],100); 
        put_dscr[i]=(ELAN_EVENT*)0;
    }
}

#endif
#endif

#ifdef MULTI_CTX
void armci_checkMapped(void *buffer, size_t size)
{
	printf("Checking %p %p\n",buffer,size);
  if ( ! elan_addMapping(elan_base->state, buffer, size ) )
	  armci_die("Error, can't add elan mapping",0);
}
#endif


