/*---------------------------------------------------------------------------\ 
| File:   ga_handler.c                                                        |
| Purpose:                                                                    |
|   This code defines handlers for interrupt-driven communication.            |
|   It works with Intel NX hrecv, IBM MPL rcvncall and LAPI Active Messages.  |
|   The handlers call ga_SERVER() which does the actual GA work.              |
| Author: Jarek Nieplocha, PNNL                                               |
| Date:   04.29.97                                                            |
\----------------------------------------------------------------------------*/


#if defined(NX) || defined(SP1) || defined(SP) || defined(LAPI)

#define DEBUG 0

#include "global.h"
#include "globalp.h"
#include "message.h"
#include "interrupt.h"
#include <stdio.h>

extern void ga_update_serv_num();
long htype = GA_TYPE_REQ;
void ga_init_handler(char*, long);


/***************************** IBM LAPI **************************************/
#ifdef LAPI

#include <lapi.h>

/******************** lock macros for synchronizing Lapi threads  ************/
#if defined(LAPI_MPI) || defined(LAPI_SPLIT)
   static pthread_mutex_t ga_handler_mutex=PTHREAD_MUTEX_INITIALIZER;
#  define LOCK_THREAD pthread_mutex_lock(&ga_handler_mutex)
#  define UNLOCK_THREAD  pthread_mutex_unlock(&ga_handler_mutex)

#else
   typedef struct {
        unsigned int    value;
        pthread_mutex_t mutex;
        pthread_cond_t  ready;
   }semaphore_t;

  /* initialize binary semaphore to 1 */
  static semaphore_t sem={1,PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER};

   static void P(semaphore_t *sem)
   {
       pthread_mutex_lock(&sem->mutex);
       while ( sem->value  == 0) pthread_cond_wait (&sem->ready, &sem->mutex);
       sem->value--;
       pthread_mutex_unlock(&sem->mutex);
   }

   static void V(semaphore_t *sem)
   {
       pthread_mutex_lock(&sem->mutex);
       if( sem->value == 0) pthread_cond_signal(&sem->ready);
       sem->value++;
       pthread_mutex_unlock(&sem->mutex);
   }

#  define LOCK_THREAD P(&sem)
#  define UNLOCK_THREAD V(&sem)
#endif


/************************** global variables ********************************/ 
volatile static int hndlcnt=0, header_cnt=0;
int hhnum=0;

/* trace and limit the number malloc calls in HH */ 
static int num_malloc=0;
#define MAX_NUM_MALLOC 100

/* trace state of accumulate lock */ 
int kevin_ok=1; /* "1" indicates that no other thread is holdeing the lock */

/***************************************************************************/


void ga_completion_handler(lapi_handle_t *t_hndl, void *save)
{
lapi_handle_t hndl = *t_hndl;
void *message;
int whofrom, msglen;
request_header_t *msginfo = (request_header_t *)save;
extern int lapi_max_uhdr_data_sz; /* max GA data payload in AM header */

     if(DEBUG) fprintf(stderr,"%d: in CMPL HANDLER:msg(%d) tag=%d from %d\n",
                       ga_nodeid_(),hndlcnt, msginfo->req_tag, msginfo->from);

     if(DEBUG)
          if(msginfo->operation == GA_OP_ACC)fprintf(stderr,"%d in CH: acc\n",ga_nodeid_()); 

     /* for put/acc/scatter need to get the data */
     if(msginfo->operation == GA_OP_PUT || msginfo->operation == GA_OP_SCT ||
       (msginfo->operation == GA_OP_GAT && msginfo->bytes>lapi_max_uhdr_data_sz)
       ||(msginfo->operation == GA_OP_ACC && 
          msginfo->bytes>lapi_max_uhdr_data_sz)){
#       if defined(LAPI_MPI)
           LOCK_THREAD; 
           ga_msg_rcv(msginfo->req_tag, MessageRcv->buffer, TOT_MSG_SIZE, 
                     &msglen, msginfo->from, &whofrom);
#       elif defined(LAPI_SPLIT)
        {  /* get data from the requesting node  */
           static lapi_cntr_t req_cnt=(lapi_cntr_t)0;
           double sum;
           int rc;
           if(DEBUG)fprintf(stderr,"%d in CH: from=%d,bytes=%d (%lx,%lx,%lx)\n",
                    ga_nodeid_(), msginfo->from, msginfo->bytes,
                    msginfo->tag.buf, MessageRcv->buffer, msginfo->tag.cntr);


           rc=LAPI_Get(hndl, (uint)msginfo->from, msginfo->bytes,
                      msginfo->tag.buf, MessageRcv->buffer, 
                      msginfo->tag.cntr,&req_cnt);
           if(rc) ga_error("CH: LAPI_Get failed",rc);

           rc = LAPI_Waitcntr(hndl, &req_cnt, 1, NULL);
           if(rc) ga_error("CH: LAPI_Waitcntr failed",rc);

#          ifdef CHECKSUM
             ga_checksum(MessageRcv->buffer, msginfo->bytes, &sum);
             if(sum != msginfo->checksum){ 
                  fprintf(stderr,"%d in CH: checksum error %f != %f\n",
                  ga_nodeid_(), sum, msginfo->checksum); 
                  ga_error("checksum error",-1);
             }
#          endif
        }
#       endif
     } 

     /* for short gather, (i,j) data follows the request info */
     if(msginfo->operation==GA_OP_GAT && msginfo->bytes<=lapi_max_uhdr_data_sz){
        memcpy(MessageRcv->buffer, msginfo+1, msginfo->bytes);
     }

     if(msginfo->operation==GA_OP_ACC && msginfo->bytes<=lapi_max_uhdr_data_sz){
        ga_SERVER(msginfo->from, msginfo);
     }else{
        *(request_header_t *)MessageRcv = *msginfo;
        ga_SERVER(msginfo->from, MessageRcv);
     }

#    if  !defined(LAPI_SPLIT)
        UNLOCK_THREAD; 
#    endif

     free(msginfo);
     num_malloc--;

     if(DEBUG) fprintf(stderr,"%d:leaving CMPL handler:msg(%d) from %d\n",
               ga_nodeid_(),hndlcnt++,MessageRcv->from);
}



void* ga_header_handler(lapi_handle_t *t_hndl, void *uhdr, uint *t_uhdrlen,
                        uint *msglen, compl_hndlr_t **handler, void** psave)
{
     lapi_handle_t hndl = *t_hndl;
     uint uhdrlen = *t_uhdrlen;
     request_header_t *msginfo = (request_header_t *)uhdr;
     Integer oper = msginfo->operation;

     if(DEBUG)
  if(ga_nodeid_()==1 && oper == GA_OP_ACC)
  fprintf(stderr,"%d in HH op=%d len=%d [%d:%d,%d:%d]\n",ga_nodeid_(),oper,
                 uhdrlen, msginfo->ilo,msginfo->ihi,msginfo->jlo,msginfo->jhi);

     ga_update_serv_num();

#    ifdef LAPI_SPLIT
       /* process small requests inside header handler */
       if(uhdrlen > MSG_HEADER_SIZE && 
         (oper == GA_OP_ACC || oper==GA_OP_PUT || oper==GA_OP_SCT)){

#         ifdef CHECKSUM
          {
            extern void ga_checksum(void* data, int bytes, double *sum);
            double sum;

            ga_checksum(((struct message_struct*)uhdr)->buffer,
                                          msginfo->bytes,&sum);
            if(DEBUG) 
              fprintf(stderr,"%d HHchecksum=%f(%d)\n",ga_nodeid_(),sum,hhnum++);

            if(sum != ((request_header_t *)uhdr)->checksum){ 
                 fprintf(stderr,"%d in HH: checksum error %f != %f\n",
                 ga_nodeid_(), sum, ((request_header_t *)uhdr)->checksum);
                 ga_error("checksum error",-1);
            }
          }
#         endif

          /* If another thread is in ga_acc_local, use compl. handler path:
           * Try to avoid blocking inside HH which degrades Lapi performance. 
           * The completion handler path requires malloc to save request info.
           * Only up to MAX_NUM_MALLOC requests can be rescheduled to
           * run in CH instead of HH.
           */   
             
          if((oper != GA_OP_ACC) || (num_malloc > MAX_NUM_MALLOC) || kevin_ok){

             ga_SERVER(((request_header_t *)uhdr)->from, uhdr); 

             *psave = NULL;
             *handler = NULL;
             return(NULL);
          }
       }
#    endif
 
     num_malloc++;
     msginfo  = (request_header_t*) malloc(uhdrlen);
     if(!msginfo) ga_error("GA HH: malloc failed in header handler",num_malloc);

     /* save the request info for processing in compl. handler */
     memcpy((char*)msginfo, uhdr, uhdrlen);
     *psave = msginfo;
     *handler = ga_completion_handler;

     return(NULL);
}


/***************************** Intel NX **************************************/
#elif defined(NX) 

void ga_handler(long type, long count, long node, long pid)
{
  long oldmask;
  ga_mask(1L, &oldmask);

  ga_update_serv_num();
  in_handler = 1;
  ga_SERVER(node, MessageRcv);
  in_handler = 0;

  ga_init_handler((char*) MessageRcv, (long)TOT_MSG_SIZE );

  ga_mask(oldmask, &oldmask);
}


void ga_init_handler(char *buffer, long lenbuf) /*Also called in ga_initialize*/
{
  hrecv(htype, buffer, lenbuf, ga_handler); 
}



/******************** SP interrupt receive stuff (MPL) ***********************/
#elif defined(SP1)||defined(SP)

static long  requesting_node;
static long  msgid; 
static long  have_wildcards=0; 
int    dontcare, allmsg, nulltask,allgrp; /*values of MPL/EUIH wildcards*/


/*\ gets values of EUI wildcards
\*/
void wildcards()
{
  int buf[4], qtype, nelem, status;
     qtype = 3; nelem = 4;
     status = mpc_task_query(buf,nelem,qtype);
     if(status==-1) ga_error("wildcards: mpc_task_query error", -1L);

     dontcare = buf[0];
     allmsg   = buf[1];
     nulltask = buf[2];
     allgrp   = buf[3];
     have_wildcards=1; 
}



static void ga_handler(int *pid)
{
size_t msglen;

# ifndef RISKY 
    mpc_wait(pid, &msglen);
# endif

  in_handler = 1;
  ga_update_serv_num();
  /* fprintf(stderr,"in handler: msg from %d\n",requesting_node); */
  ga_SERVER(requesting_node, MessageRcv);
  in_handler = 0;

# ifdef RISKY 
    mpc_wait(pid, &msglen);/*under AIX4 version of MPL can wait after handler*/ 
# endif

  ga_init_handler((char*)MessageRcv, TOT_MSG_SIZE );
  /* fprintf(stderr,"leaving handler\n"); */
}


void ga_init_handler(char *buffer,long lenbuf) /* Also called in ga_initialize*/
{
static long status; 

  if( ! have_wildcards) wildcards();

  requesting_node = dontcare;

  status=mp_rcvncall(buffer, &lenbuf, &requesting_node,
                     &htype, &msgid, ga_handler);
}


#endif


#else

This file should only be linked in under NX/EUIH/MPL/LAPI

#endif
 

