#include "armcip.h"
#include "copy.h"
#include "acc.h"
#include <stdio.h>

#define ARMCI_OP_2D(op, scale, proc, src, dst, bytes, count, src_stride, dst_stride)\
if(op == GET || op ==PUT)\
      armci_copy_2D(op, proc, src, dst, bytes, count, src_stride,dst_stride);\
else\
      armci_acc_2D(op, scale, proc, src, dst, bytes, count, src_stride,dst_stride) 



/*\ 2-dimensional array copy
\*/
void armci_copy_2D(int op, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int count, int src_stride, int dst_stride)
{
#if !defined(SYSV) && !defined(WIN32)
  if(proc == armci_me)
#endif
  {
    switch (count){
    case 1: armci_copy(src_ptr, dst_ptr, bytes); break;
    default:
        if(bytes < THRESH){     /* low-latency copy for small data segments */        
          char *ps=(char*)src_ptr;
          char *pd=(char*)dst_ptr;
          int j;
          for (j = 0;  j < count;  j++){
              int i;
              for(i=0;i<bytes;i++) pd[i] = ps[i];
              ps += src_stride;
              pd += dst_stride;
          }
        } else if( bytes % ALIGN_SIZE || ((long)src_ptr) %ALIGN_SIZE || ((long)dst_ptr) %ALIGN_SIZE ){ 

            /* size/address not alligned */
            ByteCopy2D(bytes, count, src_ptr, src_stride, dst_ptr, dst_stride);

        }else { /* segment size alligned -- should be the most efficient copy */

            DCopy2D(bytes/ALIGN_SIZE, count, src_ptr, src_stride/ALIGN_SIZE, 
                                        dst_ptr, dst_stride/ALIGN_SIZE);
        }
      }
  }

#if !defined(SYSV) && !defined(WIN32)
  else {

       if(op==PUT){ 

          UPDATE_FENCE_STATE(proc, PUT, count);

#ifdef LAPI
          SET_COUNTER(ack_cntr,count);
#endif
          if(count==1){
              armci_put(src_ptr, dst_ptr, bytes, proc);
          }else{
              armci_put2D(proc, bytes, count, src_ptr, src_stride,
                                                  dst_ptr, dst_stride);
          }

       }else{

#ifdef LAPI
          SET_COUNTER(get_cntr, count);
#endif
          if(count==1){
              armci_get(src_ptr, dst_ptr, bytes, proc);
          }else{
             armci_get2D(proc, bytes, count, src_ptr, src_stride,
                                            dst_ptr, dst_stride);
          }
       }
  }
#endif

}


/*\ Strided copy
\*/
int armci_op_strided(int op, void* scale, int proc,void *src_ptr, int src_stride_arr[],  
		       void* dst_ptr, int dst_stride_arr[], 
                       int count[], int stride_levels)
{
    char *src = (char*)src_ptr, *dst=(char*)dst_ptr;
    int s2, s3, s4, sn;

    FENCE_NODE(proc); /* ensure ordering */

    switch (stride_levels){
    case 0: /* 1D copy */ 

            ARMCI_OP_2D(op, scale, proc, src_ptr, dst_ptr, count[0], 1, 0,0); 

            break;
    
    case 1: /* 2D op */
            ARMCI_OP_2D(op, scale, proc, src_ptr, dst_ptr, count[0], count[1], 
                         src_stride_arr[0], dst_stride_arr[0]);
            break;
    
    case 2: /* 3D op */
            for (s2= 0; s2  < count[2]; s2++){ /* 2D copy */
              ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1], 
                           dst+s2*dst_stride_arr[1], count[0], count[1], 
                       src_stride_arr[0], dst_stride_arr[0]);
            }
            break;

    case 3: /* 4D op */
            for(s3=0; s3< count[3]; s3++){
               src = (char*)src_ptr + src_stride_arr[2]*s3;
               dst = (char*)dst_ptr + dst_stride_arr[2]*s3;
               for (s2= 0; s2  < count[2]; s2++){ /* 3D copy */
                 ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1],
                       dst+s2*dst_stride_arr[1],
                       count[0], count[1],src_stride_arr[0],dst_stride_arr[0]);
               }
            }
            break;
    
    case 4: /* 5D op */
            for(s4=0; s4< count[4]; s4++){
              for(s3=0; s3< count[3]; s3++){      /* 4D copy */
                 src = (char*)src_ptr + src_stride_arr[2]*s3 + src_stride_arr[3]*s4;
                 dst = (char*)dst_ptr + dst_stride_arr[2]*s3 + dst_stride_arr[3]*s4;
                 for (s2= 0; s2  < count[2]; s2++){ /* 3D copy */
                   ARMCI_OP_2D(op, scale, proc, src+s2*src_stride_arr[1], 
                                dst+s2*dst_stride_arr[1], 
                                count[0], count[1],
                                src_stride_arr[0], dst_stride_arr[0]);
                 }
              }
            }
            break;
    
    default: /* N-dimensional op by recursion */
             for(sn = 0; sn < count[stride_levels]; sn++){
                 int rc;
                 src = (char*)src_ptr + src_stride_arr[stride_levels -1]* sn;
                 dst = (char*)dst_ptr + dst_stride_arr[stride_levels -1]* sn;
                 rc  = armci_op_strided(op, scale, proc, src, src_stride_arr,  
		                          dst, dst_stride_arr, 
                                          count, stride_levels -1);
                 if(rc) return(rc);
             }
    }

#ifdef LAPI
    if(proc != armci_me){

       if(op == GET){
           CLEAR_COUNTER(get_cntr); /* wait for data arrival */
       }else { 
           CLEAR_COUNTER(ack_cntr); /* data must be copied out*/ 
       }
    }
#endif

    return 0;
}



int armci_copy_vector(int op, /* operation code */
                armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int i,s;

    if(proc == armci_me){ /* local copy */

      for(i = 0; i< len; i++){
        for( s=0; s< darr[i].ptr_array_len; s++){
           armci_copy(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes);
        }
      }

    }else {   /********* remote copies **********/

      if(op==PUT) {

        for(i = 0; i< len; i++){

#         ifdef LAPI
                SET_COUNTER(ack_cntr,darr[i].ptr_array_len);
#         endif
          UPDATE_FENCE_STATE(proc, PUT, darr[i].ptr_array_len);
 
          for( s=0; s< darr[i].ptr_array_len; s++){   
              armci_put(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes, proc);
           }
        }

      }else {

        for(i = 0; i< len; i++){

#         ifdef LAPI
                SET_COUNTER(get_cntr,darr[i].ptr_array_len);
#         endif

          for( s=0; s< darr[i].ptr_array_len; s++){   
              armci_get(darr[i].src_ptr_array[s],darr[i].dst_ptr_array[s],darr[i].bytes,proc);
           }
        }

      }
   }

#ifdef LAPI
    if(proc != armci_me){

       if(op == GET) CLEAR_COUNTER(get_cntr); /* wait for data arrival */
       if(op == PUT) CLEAR_COUNTER(ack_cntr); /* data must be copied out*/
    }
#endif

   return 0;
}
