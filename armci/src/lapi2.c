#define DEBUG 0

/*\ nonblocking strided put
\*/
void CopyPatchTo(void *src_ptr, int src_stride, void *dst_ptr, int dst_stride,
                 int count, int bytes, int p)
{
lapi_vec_t src, dst;
void *sinfo[3], *dinfo[3];
int rc;

     if(DEBUG){
        printf("%d: PUT src_stride=%d dst_stride=%d bytes=%d count=%d p=%d\n",
               armci_me,src_stride,dst_stride,bytes,count,p);
        fflush(stdout);
     }

 
     src.vec_type = dst.vec_type                    = LAPI_GEN_STRIDED_XFER;

     sinfo[0]     = src_ptr;           dinfo[0]     = dst_ptr;
     sinfo[1]     = (void*)bytes;      dinfo[1]     = (void*)bytes;
     sinfo[2]     = (void*)src_stride; dinfo[2]     = (void*)dst_stride;

     src.num_vecs = (uint)count;       dst.num_vecs = (uint)count;
     src.len      = NULL;              dst.len      = NULL;
     src.info     = sinfo;             dst.info     = dinfo;
    
     rc = LAPI_Putv(lapi_handle, (uint)p, &dst, &src, NULL,
                    &ack_cntr.cntr, &cmpl_arr[p].cntr);
     if(rc) ERROR("LAPI_putv failed",rc);
}



/*\ nonblocking strided get 
\*/
void CopyPatchFrom(void *src_ptr, int src_stride, void *dst_ptr, int dst_stride,
                   int count, int bytes, int p)
{
lapi_vec_t src, dst;
void* sinfo[3], *dinfo[3];
int rc;

     if(DEBUG){
        printf("%d: GET src_stride=%d dst_stride=%d bytes=%d count=%d p=%d\n",
               armci_me,src_stride,dst_stride,bytes,count,p);
        fflush(stdout);
     }

     sinfo[0]     = src_ptr;           dinfo[0]     = dst_ptr;
     sinfo[1]     = (void*)bytes;      dinfo[1]     = (void*)bytes;
     sinfo[2]     = (void*)src_stride; dinfo[2]     = (void*)dst_stride;

     src.vec_type =                    dst.vec_type = LAPI_GEN_STRIDED_XFER;
     src.num_vecs = (uint)count;       dst.num_vecs = (uint)count;
     src.len      = NULL;              dst.len      = NULL;
     src.info     = sinfo;             dst.info     = dinfo;
    
     rc = LAPI_Getv(lapi_handle, (uint)p,  &src, &dst, NULL, &get_cntr.cntr);
     if(rc) ERROR("LAPI_getv failed",rc);
/*     wait_for_get();*/
/*     sleep(1);*/
/*     LAPI_Waitcntr(lapi_handle, &get_cntr.cntr, 1, NULL); */
/*     printf("%d: GET cntr =%d\n",armci_me,&get_cntr.cntr); fflush(stdout);*/
/*     get_cntr.val =0;*/
/*      CLEAR_COUNTER(get_cntr);*/
}

