#include "armcip.h"
#include <stdio.h>

#if !defined(ACC_COPY) && !defined(CRAY_YMP) && !defined(WIN32)
#   define REMOTE_OP 
#endif

/*\ determine if patch fits in the ARMCI buffer, and if not 
 *  at which stride level (patch dim) need to decompose it
\*/
static void armci_fit_buffer(int count[], int stride_levels, int* fit_level, int *nb)
{
   int bytes=1, sbytes;
   int level;

   /* find out at which stride level BUFFER becomes too small */
   for(level=0; level<= stride_levels; level++){
      sbytes = bytes; /* store #bytes at current level to save div cost later */
      bytes *= count[level];
      if(BUFSIZE < bytes) break;
   }

   /* buffer big enough for entire patch */
   if(BUFSIZE >= bytes){
       *fit_level = stride_levels;
       *nb = count[stride_levels];
       return;
   }

   /* buffer too small */
   switch (level){
   case 0: 
       /* smaller than a single column */
       *fit_level = 0;
       *nb = BUFSIZE;
       break;
   case -1:   /* one column fits */
       *fit_level = 0;
       *nb = sbytes;
       break;
   default:
       /* it could keep nb instances of (level-1)-dimensional patch */
       *fit_level = level;
       *nb = BUFSIZE/sbytes;
   }   
}


/*\ The function decomposes a multi-dimensional patch so that it fits in the
 *  internal ARMCI buffer.
 *  It works by recursively reducing patch dimension until some portion of the
 *  subpatch fits in the buffer.
 *  The recursive process is controlled by "fit_level" and "nb" arguments, 
 *  which have to be set to -1 at the top-level of the recursion tree.
\*/
int armci_pack_strided(int op, void* scale, int proc,
                       void *src_ptr, int src_stride_arr[],
                       void* dst_ptr, int dst_stride_arr[],
                       int count[], int stride_levels, 
                       int fit_level, int nb)
{
    int rc=0, sn;
    void *src, *dst;

    /* determine decomposition of the patch to fit in the buffer */
    if(fit_level<0)armci_fit_buffer(count, stride_levels, &fit_level, &nb);

/*
    if(count[0]>1024){
       printf("%d: count[0]=%d fit_level=%d nb=%d\n",armci_me, count[0], fit_level, nb);
       fflush(stdout);
    }
*/

    if(fit_level == stride_levels){

        /* we can fit subpatch into the buffer */
        int chunk = count[fit_level];
        int dst_stride, src_stride;

        
        if(nb == chunk){ /* take shortcut when whole patch fits in the buffer */
#ifdef REMOTE_OP
           return( armci_rem_strided(op, scale, proc, src_ptr, src_stride_arr,
                          dst_ptr, dst_stride_arr, count, stride_levels, 1));
#else
           return(armci_op_strided(op, scale, proc, src_ptr, src_stride_arr,
                          dst_ptr, dst_stride_arr,count, stride_levels,1));
#endif
        }

        if(fit_level){
           dst_stride = dst_stride_arr[fit_level -1];
           src_stride = src_stride_arr[fit_level -1];
        }else{
           dst_stride = src_stride = 1;
        }

        for(sn = 0; sn < chunk; sn += nb){

           src = (char*)src_ptr + src_stride* sn;
           dst = (char*)dst_ptr + dst_stride* sn;
           count[fit_level] = MIN(nb, chunk-sn); /*modify count for this level*/
#ifdef REMOTE_OP
           rc = armci_rem_strided( op, scale, proc, src, src_stride_arr,
                                   dst, dst_stride_arr, count, fit_level, 1);
#else
           rc = armci_op_strided(op, scale, proc, src, src_stride_arr,
                                 dst, dst_stride_arr, count, fit_level,1);
#endif
           if(rc) break;
        }
        count[fit_level] = chunk; /* restore original count */

    }else for(sn = 0; sn < count[stride_levels]; sn++){
                 src = (char*)src_ptr + src_stride_arr[stride_levels -1]* sn;
                 dst = (char*)dst_ptr + dst_stride_arr[stride_levels -1]* sn;
                 rc = armci_pack_strided(op, scale, proc, src, src_stride_arr,
                                    dst, dst_stride_arr,
                                    count, stride_levels -1, fit_level, nb);
                 if(rc) return rc;
    }
    return rc;
}


/* how much space is needed to move data + reduced descriptor ? */
int armci_vector_bytes( armci_giov_t darr[], int len)
{
int i, bytes=0;
    for(i=0; i<len; i++){                                   
        /*       # elements            * (elem size     + dst address ) */
        bytes += darr[i].ptr_array_len * (darr[i].bytes + sizeof(void*));
        bytes += 2*sizeof(int); /* ptr_array_len + bytes */
    }
    return bytes;
}


#define BUFSIZE10 26000 
#define BUFSIZE1  BUFSIZE

void armci_split_dscr_array( armci_giov_t darr[], int len,
                             armci_giov_t* extra, int *nlen, armci_giov_t* save)
{
int s;
int bytes=0, split=0;

    extra->src_ptr_array=NULL;
    /* go through the sets looking for set to be split */
    for(s=0;s<len;s++){
        int csize;

        csize  = darr[s].ptr_array_len * (darr[s].bytes + sizeof(void*));
        csize += 2*sizeof(int); /* ptr_array_len + bytes */

        if(csize + bytes >BUFSIZE1){

          split =(BUFSIZE1 -bytes-2*sizeof(int))/(darr[s].bytes +sizeof(void*));
          if(split == 0) s--; /* no room available - do not split */
          break;

        }else bytes+=csize;

        if(BUFSIZE1 -bytes < 64) break; /* stop here if almost full */
    }

    if(s==len)s--; /* adjust loop counter should be < number of sets */ 
    *nlen = s+1;

    if(split){

       /* save the value to be overwritten only if "save" is not filled */ 
       if(!save->src_ptr_array)*save= darr[s];

       /* split the set: reduce # of elems, "extra" keeps info for rest of set*/
       *extra = darr[s];
       darr[s].ptr_array_len = split;
       extra->ptr_array_len -= split;
       extra->src_ptr_array  = &extra->src_ptr_array[split];
       extra->dst_ptr_array  = &extra->dst_ptr_array[split];
    }
} 
    
 

int armci_pack_vector(int op, void *scale, armci_giov_t darr[],int len,int proc)
{
armci_giov_t extra; /* keeps data remainder of set to be processed in chunks */
armci_giov_t save;  /* keeps original value of set to be processed in chunks */
armci_giov_t *ndarr; /* points to first array element to be processed now */
int rc, nlen, count=0;

    ndarr = darr;

    save.src_ptr_array=NULL; /* indicates that save slot is empty */
    while(len){

       armci_split_dscr_array(ndarr, len, &extra, &nlen, &save); 

#ifdef REMOTE_OP
       rc = armci_rem_vector(op, scale, ndarr,nlen,proc);
#else
       if(ACC(op))rc=armci_acc_vector(op,scale,ndarr,nlen,proc);
       else rc = armci_copy_vector(op,ndarr,nlen,proc);
#endif
       if(rc) break;

       /* non-NULL pointer indicates that set was split */
       if(extra.src_ptr_array){

          ndarr[nlen-1]=extra; /* set the pointer to remainder of last set */
          nlen--; /* since last set not done in full need to process it again */

       }else{

          if(save.src_ptr_array){
             ndarr[0]=save;
             save.src_ptr_array=NULL; /* indicates that save slot is empty */
          }

          if(nlen==0)
            armci_die("vector packetization problem:buffer too small",BUFSIZE1);
       }

       len -=nlen;
       ndarr +=nlen;
       count ++;
    }

    return rc;
}
