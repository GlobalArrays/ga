#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "macdecls.h"


static void gai_combine_val(int type, void *ptr, int n, void* val, int add)
{
    int i;

        switch (type){
          Integer *ia;
          DoublePrecision *da;
          DoubleComplex *ca;

          case MT_F_INT:
             ia = (Integer*)ptr;
             if(add)
			   for(i=0; i< n; i++) ia[i] += *(Integer*)val; 
			 else
			   for(i=0; i< n; i++) ia[i] = *(Integer*)val; 
             break;
          case MT_F_DCPL:
             ca = (DoubleComplex*)ptr;
             if(add)
               for(i=0; i< n; i++){
                   ca[i].real += ((DoubleComplex*)val)->real; 
                   ca[i].imag += ((DoubleComplex*)val)->imag; 
               }
			 else
               for(i=0; i< n; i++){
                   ca[i].real = ((DoubleComplex*)val)->real; 
                   ca[i].imag = ((DoubleComplex*)val)->imag; 
               }
             break;
          case MT_F_DBL:
             da = (double*)ptr;
             if(add)
               for(i=0; i< n; i++) da[i] += *(double*)val; 
			 else
               for(i=0; i< n; i++) da[i] = *(double*)val; 
             break;
          default: ga_error("ga_scan/add:wrong data type",type);
        }
}

static void gai_add_val(int type, void *ptr1, void *ptr2, int n, void* val)
{
    int i;
 
        switch (type){
          Integer *ia1, *ia2;
          DoublePrecision *da1, *da2;
          DoubleComplex *ca1, *ca2;
 
          case MT_F_INT:
             ia1 = (Integer*)ptr1;
             ia2 = (Integer*)ptr2;
			 ia2[0] = ia1[0] +  *(Integer*)val; 
             for(i=1; i< n; i++) ia2[i] = ia2[i-1]+ia1[i];
             break;
          case MT_F_DCPL:
             ca1 = (DoubleComplex*)ptr1;
             ca2 = (DoubleComplex*)ptr2;
			 ca2->real = ca1->real +  ((DoubleComplex*)val)->real; 
			 ca2->imag = ca1->imag +  ((DoubleComplex*)val)->imag; 
             for(i=1; i< n; i++){
                   ca2[i].real = ca2[i-1].real + ca1[i].real;
                   ca2[i].imag = ca2[i-1].imag + ca1[i].imag;
             }
             break;
          case MT_F_DBL:
             da1 = (double*)ptr1;
             da2 = (double*)ptr2;
			 da2[0] = da1[0] +  *(double*)val; 
             for(i=1; i< n; i++) da2[i] = da2[i-1]+da1[i];
             break;
          default: ga_error("ga_add_val:wrong data type",type);
        }
}                                                               


static void gai_copy_sbit(int type, void *a, int n, void *b, Integer *sbit, int pack, int mx)
{
    int i, cnt=0;
    Integer         *is, *id;
    DoublePrecision *ds, *dd;
    DoubleComplex   *cs, *cd;

    if(pack)
        switch (type){
         case MT_F_INT:
             is = (Integer*)a; id = (Integer*)b;
             for(i=0; i< n; i++) if(sbit[i]) { 
					 *id = is[i]; id++;
					 cnt++;
			 }
             break;
          case MT_F_DCPL:
             cs = (DoubleComplex*)a; cd = (DoubleComplex*)b;
             for(i=0; i< n; i++)if(sbit[i]){
                 cd->real  = cs[i].real; cd->imag  = cs[i].imag; cd ++;
				 cnt++;
			 }
             break;
          case MT_F_DBL:
             ds = (double*)a; dd = (double*)b;
             for(i=0; i< n; i++)if(sbit[i]){ *dd = ds[i]; dd++; cnt++; }
             break;
          default: ga_error("ga_copy_sbit:wrong data type",type);
        }
    else
        switch (type){
          case MT_F_INT:
             is = (Integer*)b; id = (Integer*)a;
             for(i=0; i< n; i++) if(sbit[i]) { id[i] = *is; is++;  cnt++; }
             break;
          case MT_F_DCPL:
             cs = (DoubleComplex*)b; cd = (DoubleComplex*)a;
             for(i=0; i< n; i++)if(sbit[i]){
                 cd[i].real  = cs->real; cd[i].imag  = cs->imag; cs++; cnt++; }
             break;
          case MT_F_DBL:
             ds = (double*)b; dd = (double*)a;
             for(i=0; i< n; i++)if(sbit[i]){ dd[i] = *ds; ds++; cnt++; }
             break;
          default: ga_error("ga_copy_sbit:wrong data type",type);
        }
	    if(cnt!=mx){
				printf("\nga_copy_sbit: cnt=%d should be%d\n",cnt,mx);
				ga_error("ga_copy_sbit mismatch",0);
		}
}



/*\ sets values for specified array elements by enumerating with stride
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



static void gai_scan_copy_add(Integer* g_a, Integer* g_b, Integer* g_sbit, 
    	   Integer* lo, Integer* hi, int add)
{
   Integer *lim=NULL, handle, idx, nproc, me;
   Integer lop, hip, ndim, dims, type;
   double buf[2];
   Integer *ia, elems, i;

   nproc = ga_nnodes_();
      me = ga_nodeid_();

   ga_check_handle(g_a, "ga_scan_copy");
   ga_check_handle(g_b, "ga_scan_copy 2");
   ga_check_handle(g_sbit,"ga_scan_copy 3");

   ga_sync_();

   if(MA_push_get(MT_F_INT, nproc, "ga scan buf", &handle, &idx))
                  MA_get_pointer(handle, &lim);
   if(!lim) ga_error("ga_scan_copy: MA memory alloc failed",nproc);

   ndim = ga_ndim_(g_a);
   if(ndim>1)ga_error("ga_scan_copy: applicable to 1-dim arrays",ndim);

   nga_inquire_(g_a, &type, &ndim, &dims);
   nga_distribution_(g_sbit, &me, &lop, &hip);

   bzero(lim,sizeof(Integer)*nproc);

   if(!ga_compare_distr_(g_a, g_sbit))
       ga_error("ga_scan_copy: different distribution src",0);
   if(!ga_compare_distr_(g_b, g_sbit))
       ga_error("ga_scan_copy: different distribution dst",0);
      
   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */ 

        nga_access_ptr(g_sbit, &lop, &hip, &ia, NULL);
        elems = hip -lop+1;
        /* find last bit set on given process */
        for(i=0; i<elems; i++) if(ia[i]) lim[me]= i+lop;
   }

   ga_igop(GA_TYPE_GOP,lim, nproc,"+");

   /* take product of patch owned and specified by the user */ 
   if(*hi <lop || hip <*lo); /* we got no elements to update */
   else{
       Integer k, *ip=ia, lops=lop, hips=hip;
       Integer startp=0;
       void *ptr_b;

       /* what part of local data we should be working on */
       if(lop < *lo){
           /* user specified patch starts in the middle */
           lop = *lo;
           ip = ia + (*lo-lop);
       } 
       if(hip > *hi)hip = *hi;
      
       /* access the data */
       nga_access_ptr(g_b, &lop, &hip, &ptr_b, NULL);

       /* find start bit corresponding to my patch */
       /* case 1: sbit set for the first patch element and check earlier elems*/
       for(k=lop, i=0; k >= lops; i--, k--) if(ip[i]){ startp = k; break; }
       if(!startp){
          /* case2: scan lim to find sbit set on lower numbered processors */ 
          for(k=me-1; k >=0; k--)if(lim[k]) {startp =lim[k]; break; }
       }
       if(!startp) ga_error("sbit not found for",lop); /*nothing was found*/

       /* copy the data */
       i = 0;
       for(k=lop; k<= hip; ){ 
           int indx=i, one=1;
		   int elemsize = GAsizeofM(type);
           
           /* find where sbit changes */ 
           for(; i< hip-lop; indx=++i) if(ip[i+1]) {i++; break;}

           elems = indx- k+lop +1; /* that many elements will be updating now*/

           /* get the current value of A */
           nga_get_(g_a, &startp, &startp, buf, &one);

           /* assign it to "elems" elements of B */
           gai_combine_val(type, ptr_b, elems, buf,add); 

		   ptr_b = (char*)ptr_b + elems*elemsize;
           k += elems;
           startp = k;
       }

       /* release local access to arrays */
       nga_release_(g_b, &lop, &hip);
       nga_release_(g_sbit, &lops, &hips);
   }


   ga_sync_();

   if(!MA_pop_stack(handle)) ga_error("MA_pop_stack failed",0);
}


void ga_scan_copy_(Integer* g_a, Integer* g_b, Integer* g_sbit,
				           Integer* lo, Integer* hi)
{       
		gai_scan_copy_add(g_a, g_b, g_sbit, lo, hi, 0);
}


void ga_scan_add_(Integer* g_a, Integer* g_b, Integer* g_sbit,
				           Integer* lo, Integer* hi)
{       
		gai_scan_copy_add(g_a, g_b, g_sbit, lo, hi, 1);
}


static void gai_pack_unpack(Integer* g_a, Integer* g_b, Integer* g_sbit,
              Integer* lo, Integer* hi, Integer* icount, int pack)
{
   Integer *lim=NULL, handle, idx, nproc, me;
   Integer lop, hip, ndim, dims, type;
   void *ptr;
   double buf[2];
   Integer *ia, elems, i, first, myplace =0, counter=0;

   nproc = ga_nnodes_();
      me = ga_nodeid_();

   ga_check_handle(g_a, "ga_pack");
   ga_check_handle(g_b, "ga_pack 2");
   ga_check_handle(g_sbit,"ga_pack 3");

   ga_sync_();

   if(MA_push_get(MT_F_INT, nproc, "ga_pack lim buf", &handle, &idx))
                  MA_get_pointer(handle, &lim);
   if(!lim) ga_error("ga_pack: MA memory alloc failed",nproc);

   bzero(lim,sizeof(Integer)*nproc);
   nga_inquire_(g_a, &type, &ndim, &dims);
   if(ndim>1) ga_error("ga_pack: supports 1-dim arrays only",ndim);
   nga_distribution_(g_sbit, &me, &lop, &hip);

   /* how many elements we have to copy? */
   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */

        /* adjust the range of elements to consider to be within <lo,hi> */
        if(lop < *lo) lop = *lo;
        if(hip > *hi) hip = *hi;

        if(*hi <lop || hip <*lo); /* we got no elements to update */
        else{

          nga_access_ptr(g_sbit, &lop, &hip, &ptr, NULL);
          ia    = (Integer*)ptr;
          elems = hip -lop+1;

          /* find number of elements to be contributed */
          for(i=counter=0,first=-1; i<elems; i++) if(ia[i]){
              counter++;
              if(first==-1)first=i;
          }
          lim[me] = counter;
        }
   }

   /* find number of elements everybody else is contributing */
   ga_igop(GA_TYPE_GOP, lim, nproc,"+");

   for(i= myplace= *icount= 0; i<nproc; i++){
        if( i<me && lim[i]) myplace += lim[i];
        *icount += lim[i];
   }
   if(!MA_pop_stack(handle)) ga_error("pack:MA_pop_stack failed",0);

   if(*hi <lop || hip <*lo || counter ==0 ); /* we got no elements to update */
   else{

     void *buf;
     Integer start=lop+first; /* the first element for which sbit is set */
     Integer dst_lo =myplace+1, dst_hi = myplace + counter;

     nga_access_ptr(g_a, &start, &hip, &ptr, NULL);

     if(MA_push_get(type, counter, "ga pack buf", &handle, &idx))
           MA_get_pointer(handle, &buf);
     if(!buf) ga_error("ga_pack: MA memory alloc for data failed ",counter);

     /* stuff data selected by sbit into(pack) or from(unpack) buffer */
     if(pack){

        gai_copy_sbit(type, ptr, hip-lop+1 , buf, ia+first, pack,counter); /* pack data to buf */
        nga_put_(g_b, &dst_lo, &dst_hi,  buf, &counter); /* put it into destination array */

     }else{

        nga_get_(g_b, &dst_lo, &dst_hi,  buf, &counter); /* get data to buffer */
        gai_copy_sbit(type, ptr, hip-lop+1 , buf, ia+first, pack,counter);  /* copy data to array*/

     }

     if(!MA_pop_stack(handle)) ga_error("pack:MA_pop_stack failed",0);

   }

   ga_sync_();
}



void ga_pack_(Integer* g_a, Integer* g_b, Integer* g_sbit,
              Integer* lo, Integer* hi, Integer* icount)
{
     gai_pack_unpack( g_a, g_b, g_sbit, lo, hi, icount, 1);
}


void ga_unpack_(Integer* g_a, Integer* g_b, Integer* g_sbit,
              Integer* lo, Integer* hi, Integer* icount)
{
     gai_pack_unpack( g_a, g_b, g_sbit, lo, hi, icount, 0);
}

