#ifndef LAPI_DEFS_H
#define LAPI_DEFS_H
#include <lapi.h>
   
extern lapi_handle_t lapi_handle;
extern int lapi_max_uhdr_data_sz; /* max GA data payload in AM header */

typedef struct{
	lapi_cntr_t cntr;	/* counter to trace completion of stores */
	int val;		/* number of pending LAPI store ops */
	int oper;		/* code for last GA store operation */
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

extern void ga_init_lapi(void);     /* initialize LAPI and related data struct*/
extern void ga_term_lapi(void);     /* destroy LAPI related data structures */
extern void ga_lapi_send(msg_tag_t, void*, int, int); /* LAPI version of send */

#define SHORT_ACC_THRESHOLD (6 * lapi_max_uhdr_data_sz) 
#define SHORT_PUT_THRESHOLD (6 * lapi_max_uhdr_data_sz) 
#define LONG_PUT_THRESHOLD 3000
#define LONG_GET_THRESHOLD 2500

extern double t0_fence, t_fence, tcgtime_();
extern double t0_buf, t_buf;


/**** macros to control LAPI modes and ordering of operations ****/

#define INTR_ON  
#define INTR_OFF 
/*
#define INTR_ON  if(intr_status==1)LAPI_Senv(lapi_handle, INTERRUPT_SET, 1) 
#define INTR_OFF {\
        LAPI_Qenv(lapi_handle, INTERRUPT_SET, &intr_status);\
        LAPI_Senv(lapi_handle, INTERRUPT_SET, 0);\
} 
*/

/*  fprintf(stderr,"%d fence cntr (%d,%d)\n",ga_nodeid_(),*_pcntr_, cmpl_arr[(p)].val);\*/

#define WAIT_COUNTER(counter) if((counter).val)\
        for(;;){\
          int _val__;\
          if(LAPI_Getcntr(lapi_handle,&(counter).cntr,&_val__))\
              ga_error("LAPI_Getcntr failed",-1);\
          if(_val__ == (counter).val) break;\
          LAPI_Probe(lapi_handle);\
}


#define CLEAR_COUNTER(counter) if((counter).val) {\
int _val_;\
        if(LAPI_Waitcntr(lapi_handle, &(counter).cntr, (counter).val, &_val_))\
             ga_error("LAPI_Waitcntr failed",-1);\
        if(_val_ != 0) ga_error("CLEAR_COUNTER: nonzero",_val_);\
        (counter).val = 0;  \
}

#define CLEAR_COUNTER2(counter) if((counter).val){\
        if(LAPI_Waitcntr(lapi_handle, &(counter).cntr, (counter).val, NULL))\
             ga_error("LAPI_Waitcntr failed",-1);\
        (counter).val = 0;  \
}

#define SET_COUNTER(counter, value) (counter).val += (value)

#ifndef BROKEN

#define FENCE_NODE(p) CLEAR_COUNTER(cmpl_arr[(p)]) 

#define UPDATE_FENCE_STATE(p, opcode, nissued)\
{\
  cmpl_arr[(p)].val += (nissued);\
  cmpl_arr[(p)].oper = (opcode);\
}

#else
#define FENCE_NODE(p) LAPI_Fence(lapi_handle)
#define UPDATE_FENCE_STATE
#endif

#define PENDING_OPER(p) cmpl_arr[(p)].oper


#endif
