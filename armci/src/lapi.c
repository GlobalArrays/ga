/* initialization of data structures and setup of lapi internal parameters */ 

#include <stdio.h>
#include "armcip.h"
#include "lapidefs.h"

#define ERROR(str,val) armci_die((str),(val))


int lapi_max_uhdr_data_sz; /* max  data payload */
lapi_cmpl_t *cmpl_arr;     /* completion state array, dim=NPROC */
lapi_cmpl_t  hdr_cntr;     /* AM header buffer counter  */
lapi_cmpl_t  buf_cntr;     /* AM data buffer counter    */
lapi_cmpl_t  ack_cntr;     /* ACK counter used in handshaking protocols
                              between origin and target */
lapi_cmpl_t  get_cntr;     /* counter used with lapi_get  */

int intr_status;
lapi_info_t   lapi_info;
lapi_handle_t lapi_handle;

void armci_init_lapi()
{
int rc, p;
int lapi_max_uhdr_sz;

    rc = LAPI_Init(&lapi_handle, &lapi_info);
    if(rc) ERROR("lapi_init failed",rc);


     /* set the max limit for AM header data length */

     rc = LAPI_Qenv(lapi_handle,MAX_UHDR_SZ, &lapi_max_uhdr_sz);
     if(rc) ERROR("armci_init_lapi:  LAPI_Qenv failed", rc); 

     /*     fprintf(stderr,"max header size = %d\n",lapi_max_uhdr_sz);*/

     /* how much GA data can fit into AM header ? */
/*     lapi_max_uhdr_data_sz = lapi_max_uhdr_sz - MSG_HEADER_SIZE;*/

     /* allocate memory for completion state array */
     cmpl_arr = (lapi_cmpl_t*)malloc(armci_nproc*sizeof(lapi_cmpl_t));
     if(cmpl_arr== NULL) ERROR("armci_init_lapi:malloc for cmpl_arr failed",0);
     
     /* initialize completion state array */
     for(p = 0; p< armci_nproc; p++){
        rc = LAPI_Setcntr(lapi_handle, &cmpl_arr[p].cntr, 0);
        if(rc) ERROR("armci_init_lapi: LAPI_Setcntr failed (arr)",rc);
        cmpl_arr[p].oper = 0;
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

     /* disable LAPI internal error checking */
     LAPI_Senv(lapi_handle, ERROR_CHK, 0);

     /* make sure that LAPI interrupt mode is on */
     LAPI_Senv(lapi_handle, INTERRUPT_SET, 1);
}
       

void armci_term_lapi()
{
     free(cmpl_arr);
}


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

