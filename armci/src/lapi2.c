/* $Id: lapi2.c,v 1.4 2002-10-21 04:25:09 vinod Exp $ */
#define DEBUG 0
#define CMPL_DSCR_SIZE 4096*8/*given that bufsize=30000*8,conservative,indeed*/
static char * blocking_dscrptr;
#define LAPI_CLEAR_CNTR(ocmpl_) if((ocmpl_)->val) {\
int _val_;\
    if(LAPI_Waitcntr(lapi_handle,&((ocmpl_)->cntr), ((ocmpl_)->val), &_val_))\
             armci_die("LAPI_Waitcntr failed",-1);\
    if(_val_ != 0) armci_die2("CLEAR_COUNTER: nonzero in file " ## __FILE__,__LINE__,_val_);\
    (ocmpl_)->val = 0;  \
} 
 
 

/*\ 2D strided put using lapi_putv strided transfer
\*/
void armcill_put2D(void *src_ptr, int src_stride, void *dst_ptr,int dst_stride,
                 int count, int bytes, int p,lapi_cntr_t *ocntr,char *bufptr)
{
lapi_vec_t *src, *dst;
void **sinfo, **dinfo;
int reqid,rc,dsize=3*sizeof(void*);
int offset = 0;
    
    /*following 4 lines from put2D and get2D can be combined as one function*/
    src    = (lapi_vec_t *)(bufptr+offset);    offset+=sizeof(lapi_vec_t);
    dst    = (lapi_vec_t *)(bufptr+offset);    offset+=sizeof(lapi_vec_t);
    sinfo  = (void **)(bufptr+offset);         offset+=dsize;
    dinfo  = (void **)(bufptr+offset);         offset+=dsize;

    if(DEBUG){
       printf("\n%d:in put2d with p=%d bytes=%d\n",armci_me,p,bytes);
       fflush(stdout);
    }
    src->vec_type = dst->vec_type                   = LAPI_GEN_STRIDED_XFER;

    sinfo[0]      = src_ptr;           dinfo[0]     = dst_ptr;
    sinfo[1]      = (void*)bytes;      dinfo[1]     = (void*)bytes;
    sinfo[2]      = (void*)src_stride; dinfo[2]     = (void*)dst_stride;

    src->num_vecs = (uint)count;       dst->num_vecs= (uint)count;
    src->len      = NULL;              dst->len     = NULL;
    src->info     = sinfo;             dst->info    = dinfo;
    rc = LAPI_Putv(lapi_handle,(uint)p,dst,src,NULL,ocntr,&cmpl_arr[p].cntr);
    if(rc) ERROR("LAPI_putv failed",rc);

    if(DEBUG)printf("\n%d: put completed \n",armci_me);
}



/*\ 2D strided get using lapi_putv strided transfer
\*/
void armcill_get2D(void *src_ptr, int src_stride, void *dst_ptr, int dst_stride,
                   int count, int bytes, int p, lapi_cntr_t *ocntr,char *bufptr)
{
lapi_vec_t *src, *dst;
void **sinfo, **dinfo;
int reqid,rc,dsize=3*sizeof(void*);
int offset = 0;
    
    /*following 4 lines from put2D and get2D can be combined as one function*/
    src    = (lapi_vec_t *)(bufptr+offset);    offset+=sizeof(lapi_vec_t);
    dst    = (lapi_vec_t *)(bufptr+offset);    offset+=sizeof(lapi_vec_t);
    sinfo  = (void **)(bufptr+offset);         offset+=dsize;
    dinfo  = (void **)(bufptr+offset);         offset+=dsize;

    if(DEBUG){
       printf("\n%d:in get2d with p=%d bytes=%d\n",armci_me,p,bytes);
       fflush(stdout);
    }
    sinfo[0]      = src_ptr;           dinfo[0]     = dst_ptr;
    sinfo[1]      = (void*)bytes;      dinfo[1]     = (void*)bytes;
    sinfo[2]      = (void*)src_stride; dinfo[2]     = (void*)dst_stride;


    src->vec_type =                    dst->vec_type = LAPI_GEN_STRIDED_XFER;
    src->num_vecs = (uint)count;       dst->num_vecs = (uint)count;
    src->len      = NULL;              dst->len      = NULL;
    src->info     = sinfo;             dst->info     = dinfo;

    rc = LAPI_Getv(lapi_handle, (uint)p,  src, dst,NULL,ocntr);
    if(rc) ERROR("LAPI_getv failed",rc);
}


/*\ ND strided put packed and sent as vectors get 
\*/
void armcill_putND(void *src_ptr, int src_stride_arr[],void* dst_ptr,
                  int dst_stride_arr[],int count[], int stride_levels, 
                  int proc,lapi_cmpl_t *ocmpl,char *bufptr)
{
char *dst=(char*)dst_ptr;
char *src=(char*)src_ptr;
char *dst1;
char *src1;
int i,j,k,num_xmit=0,lastiovlength,iovlength,n=0,max_iovec,totalsize=0;
int total_of_2D=1;
int index[MAX_STRIDE_LEVEL], unit[MAX_STRIDE_LEVEL];
int offset = 0,dsize,dlen,rc,vecind;
lapi_vec_t *srcv, *dstv;
void **sinfo, **dinfo;
lapi_cntr_t *ocntr=&(ocmpl->cntr);

    if(DEBUG){
       printf("\n%d:in putND count[0] is %d and strarr[0] is%d maxiov=%d\n",
             armci_me,count[0],dst_stride_arr[0],max_iovec);
       fflush(stdout);
    }
    index[2] = 0; unit[2] = 1;
    if(stride_levels>1){
       total_of_2D = count[2];
       for(j=3; j<=stride_levels; j++) {
         index[j] = 0; unit[j] = unit[j-1] * count[j-1];
         total_of_2D *= count[j];
       }
    }

    max_iovec=(CMPL_DSCR_SIZE-2*sizeof(lapi_vec_t))/(2*(sizeof(int)+sizeof(void*)));

    num_xmit = total_of_2D*count[1]/max_iovec;
    lastiovlength = (total_of_2D*count[1])%max_iovec;
    if(num_xmit == 0) num_xmit = 1;
    else if(lastiovlength!=0)num_xmit++;

    k=0;vecind=0;
    if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
    else iovlength=max_iovec;

    /*following 10 lines from put2D and get2D can be combined as one function*/
    dsize = iovlength*sizeof(void*);
    dlen  = iovlength*sizeof(int);
    srcv      = (lapi_vec_t *)(bufptr+offset);  offset+=sizeof(lapi_vec_t);
    dstv      = (lapi_vec_t *)(bufptr+offset);  offset+=sizeof(lapi_vec_t);
    srcv->vec_type = dstv->vec_type             = LAPI_GEN_IOVECTOR;
    srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
    srcv->info= (void **)(bufptr+offset);       offset+=dsize;
    dstv->info= (void **)(bufptr+offset);       offset+=dsize;
    srcv->len = (unsigned int *)(bufptr+offset);offset+=dlen;
    dstv->len = (unsigned int *)(bufptr+offset);offset+=dlen;

    for(i=0; i<total_of_2D; i++) {
       dst = (char *)dst_ptr;
       src = (char *)src_ptr;
       for(j=2; j<=stride_levels; j++) {
         dst += index[j] * dst_stride_arr[j-1];
         src += index[j] * src_stride_arr[j-1];
         if(((i+1) % unit[j]) == 0) index[j]++;
         if(index[j] >= count[j]) index[j] = 0;
       }
       dst1=dst;
       src1=src;
       for(j=0;j<count[1];j++,vecind++){
         if(vecind==iovlength){
           LAPI_CLEAR_CNTR((ocmpl));
           ocmpl->val+=1;
           UPDATE_FENCE_STATE(proc,PUT,1);
           rc = LAPI_Putv(lapi_handle,(uint)proc,dstv,srcv,NULL,ocntr,
                          &cmpl_arr[proc].cntr);
           if(rc) ERROR("LAPI_putv failed",rc);
           vecind = 0; totalsize=0; k++;
           if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
           else iovlength=max_iovec;
           srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
         }

         dstv->info[vecind] = dst1;
         dstv->len[vecind] = count[0];
         srcv->info[vecind] = src1;
         srcv->len[vecind] = count[0];
         totalsize+=count[0];
         dst1+=dst_stride_arr[0];
         src1+=src_stride_arr[0];
       }
       if(vecind==iovlength){
         LAPI_CLEAR_CNTR((ocmpl));
         ocmpl->val+=1;
         UPDATE_FENCE_STATE(proc,PUT,1);
         rc = LAPI_Putv(lapi_handle,(uint)proc,dstv,srcv,NULL,ocntr,
                        &cmpl_arr[proc].cntr);
         if(rc) ERROR("LAPI_putv failed",rc);
         vecind = 0; totalsize=0; k++;
         if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
         else iovlength=max_iovec;
         srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
       }
    }
    if(DEBUG)printf("\n%d: put completed \n",armci_me);
}



/*\ ND strided get packed and sent as vectors get 
\*/
void armcill_getND(void *src_ptr, int src_stride_arr[],void* dst_ptr,
                  int dst_stride_arr[],int count[], int stride_levels, 
                  int proc, lapi_cmpl_t *ocmpl,char *bufptr)
{
char *dst=(char*)dst_ptr;
char *src=(char*)src_ptr;
char *dst1;
char *src1;
int i,j,k,num_xmit=0,lastiovlength,iovlength,n=0,max_iovec,totalsize=0;
int total_of_2D=1;
int index[MAX_STRIDE_LEVEL], unit[MAX_STRIDE_LEVEL];
int offset = 0,dsize,dlen,rc,vecind;
lapi_vec_t *srcv, *dstv;
void **sinfo, **dinfo;
lapi_cntr_t *ocntr=&(ocmpl->cntr);

    if(DEBUG){
       printf("\n%d:in getND count[0] is %d and strarr[0] is%d maxiov=%d\n",
              armci_me,count[0],dst_stride_arr[0],max_iovec);
       fflush(stdout);
    }
    index[2] = 0; unit[2] = 1;
    if(stride_levels>1){
       total_of_2D = count[2];
       for(j=3; j<=stride_levels; j++) {
         index[j] = 0; unit[j] = unit[j-1] * count[j-1];
         total_of_2D *= count[j];
       }
    }

    max_iovec=(CMPL_DSCR_SIZE-2*sizeof(lapi_vec_t))/(2*(sizeof(int)+sizeof(void*)));

    num_xmit = total_of_2D*count[1]/max_iovec;
    lastiovlength = (total_of_2D*count[1])%max_iovec;
    if(num_xmit == 0) num_xmit = 1;
    else if(lastiovlength!=0)num_xmit++;

    k=0;vecind=0;
    if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
    else iovlength=max_iovec;

    /*following 10 lines from putND and getND can be combined as one function*/
    dsize = iovlength*sizeof(void*);
    dlen  = iovlength*sizeof(int);
    srcv      = (lapi_vec_t *)(bufptr+offset);  offset+=sizeof(lapi_vec_t);
    dstv      = (lapi_vec_t *)(bufptr+offset);  offset+=sizeof(lapi_vec_t);
    srcv->vec_type = dstv->vec_type             = LAPI_GEN_IOVECTOR;
    srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
    srcv->info= (void **)(bufptr+offset);       offset+=dsize;
    dstv->info= (void **)(bufptr+offset);       offset+=dsize;
    srcv->len = (unsigned int *)(bufptr+offset);offset+=dlen;
    dstv->len = (unsigned int *)(bufptr+offset);offset+=dlen;

    for(i=0; i<total_of_2D; i++) {
       dst = (char *)dst_ptr;
       src = (char *)src_ptr;
       for(j=2; j<=stride_levels; j++) {
         dst += index[j] * dst_stride_arr[j-1];
         src += index[j] * src_stride_arr[j-1];
         if(((i+1) % unit[j]) == 0) index[j]++;
         if(index[j] >= count[j]) index[j] = 0;
       }
       dst1=dst;
       src1=src;
       for(j=0;j<count[1];j++,vecind++){
         if(vecind==iovlength){
           LAPI_CLEAR_CNTR((ocmpl));
           ocmpl->val+=1;
           rc = LAPI_Getv(lapi_handle,(uint)proc,srcv,dstv,NULL,ocntr);
           if(rc) ERROR("LAPI_getv failed",rc);
           vecind = 0; totalsize=0; k++;
           if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
           else iovlength=max_iovec;
           srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
         }

         dstv->info[vecind] = dst1;
         dstv->len[vecind] = count[0];
         srcv->info[vecind] = src1;
         srcv->len[vecind] = count[0];
         totalsize+=count[0];
         dst1+=dst_stride_arr[0];
         src1+=src_stride_arr[0];
       }
       if(vecind==iovlength){
         LAPI_CLEAR_CNTR((ocmpl));
         ocmpl->val+=1;
         rc = LAPI_Getv(lapi_handle,(uint)proc,srcv,dstv,NULL,ocntr);
         if(rc) ERROR("LAPI_getv failed",rc);
         vecind = 0; totalsize=0; k++;
         if(lastiovlength!=0 && k==(num_xmit-1))iovlength=lastiovlength;
         else iovlength=max_iovec;
         srcv->num_vecs = (uint)iovlength;  dstv->num_vecs= (uint)iovlength;
       }
    }
    if(DEBUG)printf("\n%d: get completed \n",armci_me);
}


void lapi_op_2d(int op, uint proc, void *src_ptr, void *dst_ptr,uint bytes, 
                int count, int src_stride, int dst_stride,lapi_cmpl_t* o_cmpl)
{
int i,rc;
    if(op==PUT)UPDATE_FENCE_STATE(proc, PUT, count); 
    o_cmpl->val+=count; 
    for(i=0;i<count;i++){
       if(op==PUT)
         rc=LAPI_Put(lapi_handle,proc,bytes,(dst_ptr),(src_ptr),
                     NULL,&(o_cmpl->cntr),&cmpl_arr[proc].cntr);
       else
         rc=LAPI_Get(lapi_handle,proc,bytes,(src_ptr),(dst_ptr),NULL,
                     &(o_cmpl->cntr));
       if(rc)ARMCI_Error("LAPI_put failed",0);
       src_ptr+=src_stride;
       dst_ptr+=dst_stride;
    }
}

/*\This function is designed as follows.
 *  CONTIG code breaks ND into 1D chunks a does Lapi_Put on each chunk.
 *  STRIDED code uses strided option in the LAPI_PutV call
 *  VECTOR code packs multi-strided/vector data as vectors as transmits.
 *   ____________________________________ 
 *  | type        small/medium    large |
 *  |------------------------------------ 
 *  | 1D          CONTIG          CONTIG|
 *  | 2D          STRIDED         CONTIG|
 *  | >2D         VECTOR          CONTIG|
 *  |-----------------------------------|
 *  this code uses orig counter from nb_handle for non-blk call
 *  completion counter should always be same for non-blk and blk code to be
 *  able to do ordering/fence.
\*/
void armci_lapi_strided(int op, void* scale, int proc,void *src_ptr,
                   int src_stride_arr[], void* dst_ptr, int dst_stride_arr[],
                   int count[], int stride_levels, armci_hdl_t nb_handle)
{
int rc=0;
lapi_cmpl_t *o_cmpl;
int total_of_2D,i,j;    
char *src = (char*)src_ptr, *dst=(char*)dst_ptr;
char *bufptr;
int index[MAX_STRIDE_LEVEL], unit[MAX_STRIDE_LEVEL];
int dsize=3*sizeof(void*);
    /*pick a counter, default for blocking, from descriptor for non-blocking*/
    if(nb_handle){
       o_cmpl = &(nb_handle->cmpl_info);
    }
    else{
       if(op==GET)
         o_cmpl = &get_cntr;
       else
         o_cmpl = &ack_cntr;
    }
    /*CONTIG protocol: used for 1D(contiguous) or if stride is very large in
      a multi strided case*/
    if(stride_levels==0 || count[0]>LONG_PUT_THRESHOLD){
       switch (stride_levels) {
         case 0: /* 1D op */
           lapi_op_2d(op, (uint)proc, src_ptr, dst_ptr, count[0], 1,
                      0,0,o_cmpl);
           break;
         case 1: /* 2D op */
           lapi_op_2d(op, (uint)proc, src_ptr,dst_ptr, (uint)count[0], count[1],
                      src_stride_arr[0], dst_stride_arr[0], o_cmpl);
           break;
         default: /* N-dimensional */
         {
           index[2] = 0; unit[2] = 1; total_of_2D = count[2];
           for(j=3; j<=stride_levels; j++) {
              index[j] = 0; unit[j] = unit[j-1] * count[j-1];
              total_of_2D *= count[j];
           }
           for(i=0; i<total_of_2D; i++) {
              src = (char *)src_ptr; dst = (char *)dst_ptr;
              for(j=2; j<=stride_levels; j++) {
                  src += index[j] * src_stride_arr[j-1]; 
                  dst += index[j] * dst_stride_arr[j-1];
                  if(((i+1) % unit[j]) == 0) index[j]++;
                  if(index[j] >= count[j]) index[j] = 0;
              }
              lapi_op_2d(op, (uint)proc, src, dst,(uint)count[0], count[1],
                          src_stride_arr[0], dst_stride_arr[0],o_cmpl);
          }
         }
       }
    }              
    else{ /* greated than 1D small/med stride */

       if(stride_levels==1){             /*small/med 2D, use lapi STRIDED */
         o_cmpl->val++;
         if(op==PUT)UPDATE_FENCE_STATE(proc, PUT, 1); 
         bufptr = GET_SEND_BUFFER(2*(sizeof(lapi_vec_t)+dsize),op,proc);
         (BUF_TO_EVBUF(bufptr))->val=0;

         /*if non-blocking, remember the pointer to orig counter in the buf
           this will be needed by complete_buf_index routine in buffers.c*/
         if(nb_handle){
           ((long *)bufptr)[0] = (long)(o_cmpl);
           bufptr+=sizeof(long);
         }
         if(op==GET)
           armcill_get2D(src_ptr,src_stride_arr[0],dst_ptr,dst_stride_arr[0],
                         count[1],count[0],proc,&(o_cmpl->cntr),bufptr);
         else
           armcill_put2D(src_ptr,src_stride_arr[0],dst_ptr,dst_stride_arr[0],
                         count[1],count[0],proc,&(o_cmpl->cntr),bufptr);
                              
       }
       else {                            /*small/med >2D, use lapi VECTOR*/

         bufptr = GET_SEND_BUFFER(CMPL_DSCR_SIZE,op,proc);
         (BUF_TO_EVBUF(bufptr))->val=0;

         if(op==GET){
           armcill_getND(src_ptr,src_stride_arr,dst_ptr, dst_stride_arr,count,
                         stride_levels,proc,o_cmpl,bufptr);
         }
         else {
           armcill_putND(src_ptr,src_stride_arr,dst_ptr, dst_stride_arr,count,
                         stride_levels,proc,o_cmpl,bufptr);
         }
       }

       /*
         for blocking cases, we can free cmpldescr buffer before we wait for op 
         to complete because next step after this in opstrided is WAIT_FOR_OP 
         anyways, so we should be safe. 
       */
         
       if(!nb_handle)FREE_SEND_BUFFER(bufptr);
       
    }
}
