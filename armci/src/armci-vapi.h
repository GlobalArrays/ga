#ifndef _VAPI_H
#define _VAPI_H

/*vapi includes*/
#include "vapi.h"
#include "evapi.h"
#include "mpga.h"
#include "mtl_common.h"

#include "ib_defs.h"
#include "vapi_common.h"

#define MAX_RDMA_SIZE		(8388608)
#define DEFAULT_ADDR_LEN	(8) /* format length of hcalid/qp_num.*/

#define DEFAULT_PORT            (1) /*for vapi*/
#define DEFAULT_QP_OUS_RD_ATOM  (1)
#define DEFAULT_MTU             (MTU1024)
#define DEFAULT_PSN             (0)
#define DEFAULT_PKEY_IX        (0)
#define DEFAULT_P_KEY           (0x0)
#define DEFAULT_MIN_RNR_TIMER  (5)
#define DEFAULT_SERVICE_LEVEL   (0)
#define DEFAULT_TIME_OUT        (5)
#define DEFAULT_STATIC_RATE    (2)
#define DEFAULT_SRC_PATH_BITS  (0)
#define DEFAULT_RETRY_COUNT    (1)
#define DEFAULT_RNR_RETRY      (1)


#define DEFAULT_R_KEY           (0x0)
#define DEFAULT_L_KEY           (0x0)

#define MAX_OUTST_WQS		(2000)


#define DEFAULT_MAX_SG_LIST	(1)
#define DEFAULT_MAX_CQ_SIZE	(40000)

#define MAX_NUM_DHANDLE		(4000) 

#define  DEFAULT_MAX_WQE	(1023)

#define NUM_ADDR_BITS           (64);
#define MASK_ADDR_BITS          (0xffffffff);

#define VBUF_DLEN 2048*1023

typedef struct {
        VAPI_sr_desc_t sdscr;
        VAPI_sg_lst_entry_t    ssg_entry;
        VAPI_rr_desc_t rdscr;
        VAPI_sg_lst_entry_t    rsg_entry;
} armci_vapi_field_t;

extern char * armci_vapi_client_mem_alloc(int);

#define BUF_EXTRA_FIELD_T armci_vapi_field_t 
#define GET_SEND_BUFFER _armci_buf_get
#define FREE_SEND_BUFFER _armci_buf_release
#define INIT_SEND_BUF(_field,_snd,_rcv) _snd=1;_rcv=1;if(operation==GET)_rcv=0
#define BUF_ALLOCATE armci_vapi_client_mem_alloc

#define CLEAR_SEND_BUF_FIELD(_field,_snd,_rcv,_to,_op) armci_vapi_complete_buf((armci_vapi_field_t *)(&(_field)),(_snd),(_rcv),(_to),(_op));_snd=0;_rcv=0;_to=0

#define TEST_SEND_BUF_FIELD(_field,_snd,_rcv,_to,_op,_ret)

#define CLIENT_BUF_BYPASS 1

#define _armci_bypass 1

#define LONG_GET_THRESHOLD 20000000
#define LONG_GET_THRESHOLD_STRIDED 20000000

#endif /* _VAPI_CONST_H */
