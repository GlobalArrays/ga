#include "armcip.h"
#include "copy.h"
#include "acc.h"
#include <stdio.h>



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
