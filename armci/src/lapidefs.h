#ifndef LAPI_DEFS_H
#define LAPI_DEFS_H

#ifdef LAPI2
#include "/u2/d3h325/lapi_vector_beta/lapi.h"
#else
#include <lapi.h>
#endif
#ifdef LAPI_ERR_BAD_NINTH_PARM
#define LAPI2
#endif
#define NB_CMPL_T lapi_cmpl_t   
extern lapi_handle_t lapi_handle;
extern int lapi_max_uhdr_data_sz; /* max data payload in AM header */

typedef struct{
	lapi_cntr_t cntr;	/* counter to trace completion of stores */
	int val;		/* number of pending LAPI store ops */
	int oper;		/* code for last ARMCI store operation */
}lapi_cmpl_t;


typedef struct{                 /* generalized pointer to buffer */
        void *cntr;
        void *buf;
}gp_buf_t;

typedef struct{
        void *buf;
        lapi_cntr_t *cntr;
}msg_tag_t;


extern lapi_cmpl_t *cmpl_arr;	/* completion state array, dim=NPROC */
extern lapi_cmpl_t  ack_cntr;	/* ACK counter used in handshaking protocols
				   between origin and target */
extern lapi_cmpl_t  buf_cntr;	/* AM data buffer counter    */
extern lapi_cmpl_t  get_cntr;	/* lapi_get counter    */
extern lapi_cmpl_t  hdr_cntr;	/* AM header buffer counter  */
extern int intr_status;

extern void armci_init_lapi(void);  /* initialize LAPI data structures*/
extern void armci_term_lapi(void);  /* destroy LAPI data structures */
extern void armci_lapi_send(msg_tag_t, void*, int, int); /* LAPI send */

#define BUF_EXTRA_FIELD_T     lapi_cmpl_t
#define EXTRA_MSG_BUFLEN_DBL  (sizeof(lapi_cmpl_t)>>3)
#define MAX_CHUNKS_SHORT_GET  9
#define SHORT_ACC_THRESHOLD (6 * lapi_max_uhdr_data_sz) 
#define SHORT_PUT_THRESHOLD (6 * lapi_max_uhdr_data_sz) 

#define LONG_PUT_THRESHOLD 4000
#define LONG_GET_THRESHOLD 4000
#define LONG_GET_THRESHOLD_STRIDED LONG_GET_THRESHOLD

#define MSG_BUFLEN_DBL 30000

#define INTR_ON  if(intr_status==1)LAPI_Senv(lapi_handle, INTERRUPT_SET, 1)
#define INTR_OFF {\
        LAPI_Qenv(lapi_handle, INTERRUPT_SET, &intr_status);\
        LAPI_Senv(lapi_handle, INTERRUPT_SET, 0);\
} 


/**** macros to control LAPI modes and ordering of operations ****/
#define WAIT_COUNTER(counter) if((counter).val)\
        for(;;){\
          int _val__;\
          if(LAPI_Getcntr(lapi_handle,&(counter).cntr,&_val__))\
              armci_die("LAPI_Getcntr failed",-1);\
          if(_val__ == (counter).val) break;\
          LAPI_Probe(lapi_handle);\
}

#define CLEAR_COUNTER(counter) if((counter).val) {\
int _val_;\
    if(LAPI_Waitcntr(lapi_handle,&(counter).cntr, (counter).val, &_val_))\
             armci_die("LAPI_Waitcntr failed",-1);\
    if(_val_ != 0) armci_die2("CLEAR_COUNTER: nonzero in file " ## __FILE__,__LINE__,_val_);\
    (counter).val = 0;  \
}


#define INIT_COUNTER(counter,_val) {\
     int _rc = LAPI_Setcntr(lapi_handle, &(counter).cntr, 0);\
     if(_rc)armci_die2("INIT_COUNTER:setcntr failed " ##__FILE__,__LINE__,_rc);\
     (counter).val = (_val);\
}


#define SET_COUNTER(counter, value) (counter).val += (value)

#define INIT_SEND_BUF(_cntr,_snd,_rcv)    INIT_COUNTER(_cntr,1)
#define CLEAR_SEND_BUF_FIELD(_cntr, _s, _r,_t) CLEAR_COUNTER(_cntr)
#define FIRST_INIT_SEND_BUF INIT_COUNTER

#define FENCE_NODE(p) CLEAR_COUNTER(cmpl_arr[(p)])

#define UPDATE_FENCE_STATE(p, opcode, nissued)\
{/* if((opcode)==0)armci_die("op code 0 - buffer overwritten?",(p));*/\
  cmpl_arr[(p)].val += (nissued);\
  cmpl_arr[(p)].oper = (opcode);\
}

#define PENDING_OPER(p) cmpl_arr[(p)].oper


#define WAIT_FOR_GETS CLEAR_COUNTER(get_cntr)
#define WAIT_FOR_PUTS CLEAR_COUNTER(ack_cntr)

#endif
