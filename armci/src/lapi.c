/* initialization of data structures and setup of lapi internal parameters */ 

#include <pthread.h>
#include <stdio.h>
#include "lapidefs.h"
#include "armcip.h"
#include "copy.h"
#include <sys/atomic_op.h>

#define DEBUG_ 0
#define ERROR(str,val) armci_die((str),(val))


int lapi_max_uhdr_data_sz; /* max  data payload */
lapi_cmpl_t *cmpl_arr;     /* completion state array, dim=NPROC */
lapi_cmpl_t  hdr_cntr;     /* AM header buffer counter  */
lapi_cmpl_t  buf_cntr;     /* AM data buffer counter    */
lapi_cmpl_t  ack_cntr;     /* ACK counter used in handshaking protocols
                              between origin and target */
lapi_cmpl_t  get_cntr;     /* counter used with lapi_get  */

int intr_status;
lapi_info_t     lapi_info;
#ifndef TCG
lapi_handle_t   lapi_handle;
#endif
pthread_mutex_t _armci_mutex_thread=PTHREAD_MUTEX_INITIALIZER;


/************* LAPI Active Message handlers *******************************/

volatile static int hndlcnt=0, header_cnt=0;
static int hhnum=0;
static long num_malloc=0;   /* trace and limit the number malloc calls in HH */
#define MAX_NUM_MALLOC 100

/* trace state of accumulate lock */
int kevin_ok=1; /* "1" indicates that no other thread is holding the lock */


void armci_completion_handler(lapi_handle_t *t_hndl, void *save)
{
lapi_handle_t hndl = *t_hndl;
int need_data;
void *message;
int whofrom, msglen;
request_header_t *msginfo = (request_header_t *)save;
char *descr= (char*)(msginfo+1), *buf=MessageRcvBuffer;
int buflen=MSG_BUFLEN;

   if(DEBUG_)
      fprintf(stderr,"%d:CH:op=%d from=%d datalen=%d dscrlen=%d\n", armci_me,
        msginfo->operation, msginfo->from,msginfo->datalen,msginfo->dscrlen);

   if(  msginfo->dscrlen < 0 || msginfo->datalen <0 ){
     /* for large put/acc/scatter need to get the data */
     int rc;
     lapi_cntr_t req_cntr;    
     int bytes=0;
     char *origin_ptr = msginfo->tag.buf;

     if (msginfo->dscrlen<0) {
         descr =MessageRcvBuffer;
         msginfo->dscrlen = -msginfo->dscrlen;
         buf = descr + msginfo->dscrlen;
         buflen += msginfo->dscrlen;
         bytes += msginfo->dscrlen;

         /* for large gather, compute address of descriptor at origin */
         if(msginfo->operation == GET) origin_ptr +=sizeof(request_header_t);
     }
     if (msginfo->datalen <0){
         msginfo->datalen = -msginfo->datalen;
         bytes += msginfo->datalen;
     }

     if(rc=LAPI_Setcntr(hndl, &req_cntr, 0)) ERROR("CH:setcntr failed",rc);
     if(rc=LAPI_Get(hndl, (uint)msginfo->from, bytes,
                origin_ptr, MessageRcvBuffer,
                msginfo->tag.cntr,&req_cntr))ERROR("CH:LAPI_Get failed",rc);

     if(rc=LAPI_Waitcntr(hndl, &req_cntr,1,NULL))ERROR("CH:Waitcntr failed",rc);


   } else{

     /* desc is in save, data could be but not for GET */
     if(msginfo->operation !=GET)buf = descr + msginfo->dscrlen;
     buflen = MSG_BUFLEN;
   }

/*   fprintf(stderr,"CH: val=%lf\n",*(double*)(buf+msginfo->datalen -8));*/
   if(msginfo->format == STRIDED)
                armci_server(msginfo, descr, buf, buflen);
   else
                armci_server_vector(msginfo, descr, buf, buflen);

   free(msginfo);
   (void)fetch_and_add(&num_malloc,-1);
}


void* armci_header_handler(lapi_handle_t *t_hndl, void *uhdr, uint *t_uhdrlen,
                           uint *msglen, compl_hndlr_t **handler, void** psave)
{
lapi_handle_t hndl = *t_hndl;
uint uhdrlen = *t_uhdrlen;
request_header_t *msginfo = (request_header_t *)uhdr;

   if(DEBUG_)
        fprintf(stderr,"%d:HH: op=%d from %d\n",armci_me,msginfo->operation,
                msginfo->from);

   /* process small requests inside header handler */
   if(msginfo->datalen >0 && msginfo->dscrlen>0 && msginfo->operation!=GET){

        /* If another thread is in accumulate use compl. handler path:
         * Try to avoid blocking inside HH which degrades Lapi performance.
         * The completion handler path requires malloc to save request info.
         * Only up to approx. MAX_NUM_MALLOC requests can be rescheduled to
         * run in CH instead of HH. 
         * MAX_NUM_MALLOC is a soft limit to avoid cost of locking when reading 
         */

         if( msginfo->operation==PUT || num_malloc>MAX_NUM_MALLOC || kevin_ok){

             char *descr = (char*)(msginfo+1);
             char *buf   = descr + msginfo->dscrlen;
             int buflen = uhdrlen - sizeof(request_header_t) - msginfo->dscrlen;

             if(DEBUG_)
                fprintf(stderr,"%d:HH: buf =%lf\n",armci_me,*(double*)buf);
             if(msginfo->format == STRIDED)
                armci_server(msginfo, descr, buf, buflen);
             else
                armci_server_vector(msginfo, descr, buf, buflen);

/*             fprintf(stderr,"%d:HH: getting out of server\n",armci_me);*/
             *psave = NULL;
             *handler = NULL;
             return(NULL);
         }
    }

    (void)fetch_and_add(&num_malloc,1); /* AIX atomic increment */

    msginfo  = (request_header_t*) malloc(uhdrlen); /* recycle pointer */
    if(!msginfo) ERROR("HH: malloc failed in header handler",num_malloc);

    /* save the request info for processing in compl. handler */
    memcpy((char*)msginfo, uhdr, uhdrlen);
    *psave = msginfo;
    *handler = armci_completion_handler;

    return(NULL);
}


void armci_send_req(int proc)
{
request_header_t *msginfo = (request_header_t*)MessageSndBuffer;
int msglen = sizeof(request_header_t);
lapi_cntr_t *pcmpl_cntr, *pcntr = &buf_cntr.cntr;
int rc;

      msginfo->tag.cntr= &buf_cntr.cntr;

      /* starting address is modified depending on the operation */
      msginfo->tag.buf = MessageSndBuffer;
      if(msginfo->operation==GET){

         if(lapi_max_uhdr_data_sz < msginfo->dscrlen){

            msginfo->dscrlen = -msginfo->dscrlen; /* no room for descriptor */
            pcntr = NULL; /* GET(descr) from CH will increment buf_cntr */

         }else msglen += msginfo->dscrlen;

            pcmpl_cntr=NULL; /* don't trace completion status for load ops */
            SET_COUNTER(buf_cntr,1); /* expect data to arrive into same buf*/

      }else{

         if(lapi_max_uhdr_data_sz < (msginfo->datalen + msginfo->dscrlen)){

            msginfo->datalen = -msginfo->datalen;
            msginfo->dscrlen = -msginfo->dscrlen;
            pcntr = NULL; /* GET from CH will increment buf_cntr */

         }else msglen += msginfo->dscrlen+msginfo->datalen;

         /* trace completion of store ops */
         pcmpl_cntr = &cmpl_arr[msginfo->to].cntr; 

         msginfo->tag.buf +=sizeof(request_header_t);
      }

      if(msginfo->operation!=GET) 
                  UPDATE_FENCE_STATE(msginfo->to, msginfo->operation, 1);

      if((rc=LAPI_Amsend(lapi_handle,(uint)msginfo->to,armci_header_handler,
                         MessageSndBuffer, msglen, NULL, 0, 
                         NULL, pcntr, pcmpl_cntr))) armci_die("AM failed",rc);

/*      if(msglen == sizeof(request_header_t))sleep(1);*/

      if(DEBUG_)
        fprintf(stderr,"%d sending req=%d to %d\n",armci_me, msginfo->operation,
                proc);
}
      

void armci_send_data(request_header_t* msginfo, char *data)
{
/*     fprintf(stderr,"%d: sending %d bytes (%lf) to %d adr=(%x,%x)\n",armci_me, msginfo->datalen, *(double*)data, msginfo->from, msginfo->tag.buf, MessageSndBuffer);*/
     armci_lapi_send(msginfo->tag, data, msginfo->datalen, msginfo->from);
}

void armci_rcv_data(int proc)
{
/*     fprintf(stderr,"%d: receiving cntr=%d val=%d\n", armci_me, buf_cntr.cntr, buf_cntr.val);*/
     CLEAR_COUNTER(buf_cntr);
/*     fprintf(stderr,"%d received %lf\n",armci_me, *((double*)MessageSndBuffer));*/
}


/*\ initialization of LAPI related data structures
\*/
void armci_init_lapi()
{
int rc, p;
int lapi_max_uhdr_sz;

#ifndef TCG
    rc = LAPI_Init(&lapi_handle, &lapi_info);
    if(rc) ERROR("lapi_init failed",rc);
#endif

    /* set the max limit for AM header data length */
    rc = LAPI_Qenv(lapi_handle,MAX_UHDR_SZ, &lapi_max_uhdr_sz);
    if(rc) ERROR("armci_init_lapi:  LAPI_Qenv failed", rc); 

    /*     fprintf(stderr,"max header size = %d\n",lapi_max_uhdr_sz);*/

    /* how much data can fit into AM header ? */
     lapi_max_uhdr_data_sz = lapi_max_uhdr_sz - sizeof(request_header_t);

    /* allocate memory for completion state array */
    cmpl_arr = (lapi_cmpl_t*)malloc(armci_nproc*sizeof(lapi_cmpl_t));
    if(cmpl_arr==NULL) ERROR("armci_init_lapi:malloc for cmpl_arr failed",0);
     
    /* initialize completion state array */
    for(p = 0; p< armci_nproc; p++){
        rc = LAPI_Setcntr(lapi_handle, &cmpl_arr[p].cntr, 0);
        if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (arr)",rc);
        cmpl_arr[p].oper = -1;
        cmpl_arr[p].val = 0;
    }

     /* initialize ack/buf/hdr counters */
     rc = LAPI_Setcntr(lapi_handle, &ack_cntr.cntr, 0);
     if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (ack)",rc);
     ack_cntr.val = 0;
     rc = LAPI_Setcntr(lapi_handle, &hdr_cntr.cntr, 0);
     if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (hdr)",rc);
     hdr_cntr.val = 0;
     rc = LAPI_Setcntr(lapi_handle, &buf_cntr.cntr, 0); 
     if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (buf)",rc);
     buf_cntr.val = 0;
     rc = LAPI_Setcntr(lapi_handle, &get_cntr.cntr, 0); 
     if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (get)",rc);
     get_cntr.val = 0;

#if  !defined(LAPI2)

     /* for high performance, disable LAPI internal error checking */
     LAPI_Senv(lapi_handle, ERROR_CHK, 0);

#endif

     /* make sure that LAPI interrupt mode is on */
     LAPI_Senv(lapi_handle, INTERRUPT_SET, 1);
}
       

void armci_term_lapi()
{
     free(cmpl_arr);
}

/* primitive pseudo message-passing on top of lapi */ 

/* send data to remote process using p specified message tag */
/* tag contains address of receive buffer guarded by cntr at process p */
void armci_lapi_send(msg_tag_t tag, void* data, int len, int p)
{
     int rc;
     lapi_cntr_t org_cntr;
     void *buf = tag.buf;
     lapi_cntr_t *cntr = tag.cntr;
     if(!buf)ERROR("armci_lapi_send: NULL tag(buf) error",0);
     if(!cntr)ERROR("armci_lapi_send:  NULL tag(cntr) error",0);

     rc=LAPI_Setcntr(lapi_handle, &org_cntr, 0);
     if(rc) ERROR("armci_lapi_send:setcntr failed",rc);
     rc=LAPI_Put(lapi_handle, (uint)p, (uint)len, buf, data, 
                cntr, &org_cntr, NULL);
     if(rc) ERROR("armci_lapi_send:put failed",rc);
     rc+=LAPI_Waitcntr(lapi_handle, &org_cntr, 1, NULL);
     if(rc) ERROR("armci_lapi_send:waitcntr failed",rc);
}

/* subroutine versions of macros disabling and enabling interrupts */
void intr_off_()
{
	INTR_OFF;
}

void intr_on_()
{
        INTR_ON;
}


void print_counters_()
{
  int i;
  printf("bufcntr: val =%d cntr=%d\n", buf_cntr.val, buf_cntr.cntr);
  for(i=0; i< armci_nproc;i++){
      printf("cmpl_arr: val=%d cntr=%d oper=%d\n",cmpl_arr[i].val,
              cmpl_arr[i].cntr, cmpl_arr[i].oper);
  }
  fflush(stdout);
}



#define LOCKED 1

void armci_lapi_lock(int *lock)
{
atomic_p word_addr = (atomic_p)lock;
int spin = 1;


     while(1){

        if(_check_lock(word_addr, 0, LOCKED) == FALSE )
          break; /* we got the lock */ 

        if(spin){
          armci_waitsome(1);
          spin = 0;
        }else{

         /* yield processor to another thread */
         yield(); /* mark thread as not runnable */
   
         /* call usleep to notify scheduler */
         (void)usleep(5);
       }
    }
}


void armci_lapi_unlock(int *lock)
{
atomic_p word_addr = (atomic_p)lock;

  if(_check_lock(word_addr, LOCKED, 0) == TRUE ) 
      armci_die("somebody else unlocked",0);
}
         


#ifdef LAPI2
#include "lapi2.c"
#endif
