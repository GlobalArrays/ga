#include <stdio.h>
#include "global.h"
#include "macdecls.h"

/*\ sets values for specified array elements by enumertaing with stride
\*/
void FATR ga_patch_enum_(Integer* g_a, Integer* lo, Integer* hi, 
                         void* start, void* stride)
{
Integer dims[1],lop,hip;
Integer ndim, type, me, off;
register Integer i;

   ga_sync_();
   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_patch_enum");

   ndim = ga_ndim_(g_a);
   if(ndim > 1)ga_error("ga_patch_enum:applicable to 1-dim arrays",ndim);

   nga_inquire_(g_a, &type, &ndim, dims);
   nga_distribution_(g_a, &me, &lop, &hip);

   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */

      /* take product of patch owned and specified by the user */ 
      if(*hi <lop || hip <*lo); /* we got no elements to update */
      else{
        void *ptr;

        if(lop < *lo)lop = *lo;
        if(hip > *hi)hip = *hi;
        off = lop - *lo;

        nga_access_ptr(g_a, &lop, &hip, &ptr, NULL);
        
        switch (type){
          Integer *ia;
          DoublePrecision *da;
          DoubleComplex *ca;

          case MT_F_INT:
             ia = (Integer*)ptr;
             for(i=0; i< hip-lop+1; i++)
                 ia[i] = *(Integer*)start+(off+i)* *(Integer*)stride; 
             break;
          case MT_F_DCPL:
             ca = (DoubleComplex*)ptr;
             for(i=0; i< hip-lop+1; i++){
                 ca[i].real = ((DoubleComplex*)start)->real +
                         (off+i)* ((DoubleComplex*)stride)->real; 
                 ca[i].imag = ((DoubleComplex*)start)->imag +
                         (off+i)* ((DoubleComplex*)stride)->imag; 
             }
             break;
          case MT_F_DBL:
             da = (DoublePrecision*)ptr;
             for(i=0; i< hip-lop+1; i++)
                 da[i] = *(DoublePrecision*)start+
                         (off+i)* *(DoublePrecision*)stride; 
             break;
          default: ga_error("ga_patch_enum:wrong data type ",type);
        }

        nga_release_update_(g_a, &lop, &hip);
      }
   }
   
   ga_sync_();
}
