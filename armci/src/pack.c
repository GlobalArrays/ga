#include "armcip.h"
#include <stdio.h>


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

        
        if(nb == chunk) /* take shortcut when whole patch fits in the buffer */
           return(armci_op_strided(op, scale, proc, src_ptr, src_stride_arr,
                                   dst_ptr, dst_stride_arr,count, stride_levels,1));

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
           rc = armci_op_strided(op, scale, proc, src, src_stride_arr,
                                 dst, dst_stride_arr, count, fit_level,1);
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
