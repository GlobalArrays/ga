#include <elan/elan.h>
#include <elan3/elan3.h>
#include <stdio.h>
#include <stdlib.h>
#include "armcip.h"
#include "copy.h"

#define DEBUG_ 0

#ifdef ELAN
static int armci_server_terminating=0;
static ELAN_MAIN_QUEUE *mq;
static int armci_request_from=-1;
static int armci_request_to=-1;
static int** armci_elan_fence_arr;

#define _ELAN_SLOTSIZE 320
#define MSG_DATA_LEN (_ELAN_SLOTSIZE - sizeof(request_header_t))

void armci_init_connections()
{
ELAN_QUEUE *q;
int nslots=armci_nproc, slotsize=_ELAN_SLOTSIZE;


    if ((q = elan_gallocQueue(elan_base->galloc, elan_base->allGroup)) == NULL)
        armci_die( "elan_gallocElan",0 );

    if (!(mq = elan_mainQueueInit( elan_base->state, q, nslots, slotsize)))
        armci_die("Failed to to initialise Main Queue",0);

#if 0
    if(!(armci_elan_fence_arr = elan_allocMain(elan_base->state, 4, armci_nproc*sizeof(int))))
        armci_die("failed to to initialise Elan fence array",0);
#endif

    armci_elan_fence_arr = (int**)malloc(armci_nproc*sizeof(int*));
    if(!armci_elan_fence_arr) armci_die("malloc failed for ARMCI fence array",0);
    if(ARMCI_Malloc((void**)armci_elan_fence_arr, armci_nproc*sizeof(int)))
             armci_die("failed to allocate ARMCI fence array",0);
    bzero(armci_elan_fence_arr[armci_me],armci_nproc*sizeof(int));

#if 0
  printf("%d:vp=%d localId=%d SendBuf=%p\n",armci_me,elan_base->state->vp,elan_base->state->localId,        MessageSndBuffer); 
#endif
}


extern void armci_send_data_to_client(int proc, void *buf, int bytes);

static void armci_send_ack()
{
int zero=0;
int *buf = armci_elan_fence_arr[armci_request_from] + armci_request_to;

#if 0
    printf("%d: server sending ack proc=%d fence=%p slot %p\n", armci_me, 
           armci_request_from, armci_elan_fence_arr[armci_request_from],buf);
    fflush(stdout);
#endif
    armci_put(&zero,buf,sizeof(int),armci_request_from);
}


int armci_check_int_val(int *v)
{
return (*v);
}


int __armci_wait_some =20;
double __armci_fake_work=99.0;

void armci_elan_fence(int p)
{
    long spin =0, loop=0;
    int *buf = armci_elan_fence_arr[armci_me] + p;
    int  res = armci_check_int_val(buf);

#if 0
    printf("%d: client fencing proc=%d fence=%p slot %p\n", armci_me, p, armci_elan_fence_arr[armci_me], buf);
    fflush(stdout);
#endif

    while(res){
       if(++loop == 100) { loop=0; usleep(1); }
       for(spin=0; spin<__armci_wait_some; spin++)__armci_fake_work+=0.001;
       res = armci_check_int_val(buf);
    }
    *buf = 0; 
    __armci_fake_work =99.0;
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


/* server receives request */
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
             *(void **)pdata  = MessageRcvBuffer + off; 
             armci_get(rembuf, MessageRcvBuffer, payload, msginfo->from); 
          }

        }
    }else
        *(void**)pdescr = NULL;
}


/*\ server sends data to client buffer
\*/
void armci_WriteToDirect(int dst, request_header_t *msginfo, void *buffer)
{
/* none in acc */
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
    int size=_ELAN_SLOTSIZE;

    *(armci_elan_fence_arr[armci_me]+proc)=1;

    /* set message tag -> contains pointer to client buffer with descriptor+data */
    msginfo->tag = (void *)(buf + sizeof(request_header_t));
    elan_queueReq(mq, proc, vbuf, size);

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


#ifdef _ELAN_LOCK_H

#define MAX_LOCKS 4
static ELAN_LOCK *my_locks, *all_locks;
static int num_locks=0;

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
