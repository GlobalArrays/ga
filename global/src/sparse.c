#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "macdecls.h"
#include "message.h"


static void gai_combine_val(Integer type, void *ptr, Integer n, void* val, Integer add)
{
    int i;

    switch (type){
       Integer *ia;
       DoublePrecision *da;
       DoubleComplex *ca;
       float *fa;

       case MT_F_INT:
            ia = (Integer*)ptr;
            if(add) for(i=0; i< n; i++) {
                    if(i==0) 
                       ia[i] += *(Integer*)val; 
                    else
                       ia[i] = ia[i-1] + *(Integer*)val; 
            }
            else
                    for(i=0; i< n; i++) ia[i] = *(Integer*)val; 
            break;
       case MT_F_DCPL:
            ca = (DoubleComplex*)ptr;
            if(add) for(i=0; i< n; i++){
                    if(i==0) {
                       ca[i].real += ((DoubleComplex*)val)->real; 
                       ca[i].imag += ((DoubleComplex*)val)->imag; 
                    }  else {
                       ca[i].real = ca[i-1].real + ((DoubleComplex*)val)->real; 
                       ca[i].imag = ca[i-1].imag + ((DoubleComplex*)val)->imag; 
                    }
                }
            else
                for(i=0; i< n; i++){
                    ca[i].real = ((DoubleComplex*)val)->real; 
                    ca[i].imag = ((DoubleComplex*)val)->imag; 
                }
            break;
       case MT_F_DBL:
            da = (double*)ptr;
            if(add) for(i=0; i< n; i++) {
                    if(i==0) 
                       da[i] += *(double*)val; 
                    else
                       da[i] = da[i-1] + *(double*)val; 
            } else
               for(i=0; i< n; i++) da[i] = *(double*)val; 
            break;
       case MT_F_REAL:
            fa = (float*)ptr;
            if(add) for(i=0; i< n; i++) {
                    if(i==0)
                       fa[i] += *(float*)val;
                    else
                       fa[i] = fa[i-1] + *(float*)val;
            }
            else
                    for(i=0; i< n; i++) fa[i] = *(float*)val;
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
          float *fa1, *fa2;
 
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
          case MT_F_REAL:
             fa1 = (float*)ptr1;
             fa2 = (float*)ptr2;
             fa2[0] = fa1[0] +  *(float*)val;
             for(i=1; i< n; i++) fa2[i] = fa2[i-1]+fa1[i];
             break;   
          default: ga_error("ga_add_val:wrong data type",type);
        }
}                                                               


static void gai_copy_sbit(Integer type, void *a, Integer n, void *b, Integer *sbit, Integer pack, Integer mx)
{
    int i, cnt=0;
    Integer         *is, *id;
    DoublePrecision *ds, *dd;
    DoubleComplex   *cs, *cd;
    float           *fs, *fd;

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
          case MT_F_REAL:
             fs = (float*)a; fd = (float*)b;
             for(i=0; i< n; i++) if(sbit[i]) {
                     *fd = fs[i]; fd++; cnt++;
          }
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
          case MT_F_REAL:
             fs = (float*)b; fd = (float*)a;
             for(i=0; i< n; i++) if(sbit[i]) { fd[i] = *fs; fs++;  cnt++; }
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
          float *fa;

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
          case MT_F_REAL:
             fa = (float*)ptr;
             for(i=0; i< hip-lop+1; i++)
                 fa[i] = *(float*)start+(off+i)* *(float*)stride;
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
   Integer *ia, elems;
   int i;

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
           int indx=i;
           Integer one=1;
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
   void *ptr;
   Integer *lim=NULL, handle, idx, nproc, me;
   Integer lop, hip, ndim, dims, type;
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

        gai_copy_sbit(type, ptr, hip-lop+1-first , buf, ia+first, pack,counter); /* pack data to buf */
        nga_put_(g_b, &dst_lo, &dst_hi,  buf, &counter); /* put it into destination array */

     }else{

        nga_get_(g_b, &dst_lo, &dst_hi,  buf, &counter); /* get data to buffer*/
        gai_copy_sbit(type, ptr, hip-lop+1-first , buf, ia+first, pack,counter);  /* copy data to array*/

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



#define NWORK 2000
int workR[NWORK], workL[NWORK];

/*\ compute offset for each of n bins for the given processor to contribute its
 *  elements, number of which for each bin is specified in x
\*/ 
void gai_bin_offset(int scope, int *x, int n, int *offset)
{
int root, up, left, right;
int len, lenmes, tag=32100, i, me=armci_msg_me();

    if(!x)armci_die("armci_bin_offset: NULL pointer", n);
    if(n>NWORK)armci_die("armci_bin_offset: >NWORK", n);
    len = sizeof(int)*n;

    armci_msg_bintree(scope, &root, &up, &left, &right);

    /* up-tree phase: collect number of elements */
    if (left > -1) armci_msg_rcv(tag, workL, len, &lenmes, left);
    if (right > -1) armci_msg_rcv(tag, workR, len, &lenmes, right);

    /* add number of elements in each bin */
    if((right > -1) && left>-1) for(i=0;i<n;i++)workL[i] += workR[i] +x[i];
    else if(left > -1) for(i=0;i<n;i++)workL[i] += x[i];
    else for(i=0;i<n;i++)workL[i] = x[i]; 

    /* now, workL on root contains the number of elements in each bin*/
         
    if (me != root && up!=-1) armci_msg_snd(tag, workL, len, up);

    /* down-tree: compute offset subtracting elements for self and right leaf*/
    if (me != root && up!=-1){
             armci_msg_rcv(tag, workL, len, &lenmes, up);
    }
    for(i=0; i<n; i++) offset[i] = workL[i]-x[i];

    if (right > -1) armci_msg_snd(tag, offset, len, right);
    if (left > -1) {
            /* we saved num elems for right subtree to adjust offset for left*/
            for(i=0; i<n; i++) workR[i] = offset[i] -workR[i]; 
            armci_msg_snd(tag, workR, len, left);
    }
/*    printf("%d:left=%d right=%d up=%d root=%d off=%d\n",me,left, right,up,root,offset[0]);
    fflush(stdout);
*/
}

static 
Integer gai_match_bin2proc(Integer blo, Integer bhi, Integer plo, Integer phi)
{
int rc=0;
       if(blo == plo) rc=1;
       if(bhi == phi) rc+=2; 
       return rc; /* 1 - first 2-last 3-last+first */
}


logical ga_create_bin_range_(Integer *g_bin, Integer *g_cnt, Integer *g_off, Integer *g_range)
{
Integer type, ndim, nbin, lobin, hibin, me=ga_nodeid_();
Integer dims[2], nproc=ga_nnodes_(),chunk[2];

    nga_inquire_(g_bin, &type, &ndim, &nbin);
    if(ndim !=1) ga_error("ga_bin_index: 1-dim array required",ndim);
    if(type!= MT_F_INT)ga_error("ga_bin_index: not integer type",type);

    chunk[0]=dims[0]=2; dims[1]=nproc; chunk[1]=1;
    if(!nga_create(MT_F_INT, 2, dims, "bin_proc",chunk,g_range)) return FALSE;

    nga_distribution_(g_off,&me, &lobin,&hibin);

    if(lobin>0){ /* enter this block when we have data */
      Integer first_proc, last_proc, p;
      Integer first_off, last_off;
      Integer *myoff, bin;

      /* get offset values stored on my processor to first and last bin */
      nga_access_ptr(g_off, &lobin, &hibin, &myoff, NULL);
      first_off = myoff[0]; last_off = myoff[hibin-lobin];
/*
      nga_get_(g_off,&lobin,&lobin,&first_off,&lo);
      nga_get_(g_off,&hibin,&hibin,&last_off,&hi);
*/

      /* since offset starts at 0, add 1 to get index to g_bin */
      first_off++; last_off++;

      /* find processors on which these bins are located */
      if(!nga_locate_(g_bin, &first_off, &first_proc))
          ga_error("ga_bin_sorter: failed to locate region f",first_off);
      if(!nga_locate_(g_bin, &last_off, &last_proc))
          ga_error("ga_bin_sorter: failed to locate region l",last_off);

      /* inspect range of indices to bin elements stored on these processors */
      for(p=first_proc, bin=lobin; p<= last_proc; p++){
          Integer lo, hi, buf[2], off, cnt; 
          buf[0] =-1; buf[1]=-1;

          nga_distribution_(g_bin,&p,&lo,&hi);

          for(/* start from current bin */; bin<= hibin; bin++, myoff++){ 
              Integer blo,bhi,stat;

              blo = *myoff +1;
              if(bin == hibin){
                 nga_get_(g_cnt, &hibin, &hibin, &cnt, &hibin); /* local */
                 bhi = blo + cnt-1; 
              }else
                 bhi = myoff[1]; 

              stat= gai_match_bin2proc(blo, bhi, lo, hi);

              switch (stat) {
              case 0:  /* bin in a middle */ break;
              case 1:  /* first bin on that processor */
                       buf[0] =bin; break;
              case 2:  /* last bin on that processor */
                       buf[1] =bin; break;
              case 3:  /* first and last bin on that processor */
                       buf[0] =bin; buf[1] =bin; break;
              }

              if(stat>1)break; /* found last bin on that processor */
          }
          
          /* set range of bins on processor p */
          cnt =0; off=1;
          if(buf[0]!=-1){cnt=1; off=0;} 
          if(buf[1]!=-1)cnt++; 
          if(cnt){
                 Integer p1 = p+1;
                 lo = 1+off; hi = lo+cnt-1;
                 ga_put_(g_range,&lo,&hi,&p1, &p1, buf+off, &cnt);
          }
      }
   }
/*
   ga_print_(g_range);
   ga_print_(g_bin);
   ga_print_distribution_(g_bin);
*/
   return TRUE;
}


void ga_bin_sorter_(Integer *g_bin, Integer *g_cnt, Integer *g_off)
{
extern void gai_hsort(Integer *list, int n);
Integer nbin,totbin,type,ndim,lo,hi,me=ga_nodeid_();
Integer g_range;

    if(FALSE==ga_create_bin_range_(g_bin, g_cnt, g_off, &g_range))
        ga_error("ga_bin_sorter: failed to create temp bin range array",0); 

    nga_inquire_(g_bin, &type, &ndim, &totbin);
    if(ndim !=1) ga_error("ga_bin_sorter: 1-dim array required",ndim);
     
    nga_distribution_(g_bin, &me, &lo, &hi);
    if (lo > 0 ){ /* we get 0 if no elements stored on this process */
        Integer bin_range[2], rlo[2],rhi[2];
        Integer *bin_cnt, *ptr, i;

        /* get and inspect range of bins stored on current processor */
        rlo[0] = 1; rlo[1]= me+1; rhi[0]=2; rhi[1]=rlo[1];
        nga_get_(&g_range, rlo, rhi, bin_range, rhi); /* local */
        nbin = bin_range[1]-bin_range[0]+1;
        if(nbin<1 || nbin> totbin || nbin>(hi-lo+1))
           ga_error("ga_bin_sorter:bad nbin",nbin);

        /* get count of elements in each bin stored on this task */
        if(!(bin_cnt = (Integer*)malloc(nbin*sizeof(Integer))))
           ga_error("ga_bin_sorter:memory allocation failed",nbin);
        nga_get_(g_cnt,bin_range,bin_range+1,bin_cnt,&nbin);

        /* get access to local bin elements */
        nga_access_ptr(g_bin, &lo, &hi, &ptr, NULL);
        
        for(i=0;i<nbin;i++){ 
            int elems =(int) bin_cnt[i];
            gai_hsort(ptr, elems);
            ptr+=elems;
        }
        nga_release_update_(g_bin, &lo, &hi);             
    }

    ga_sync_();
}


/*\ note that subs values must be sorted; bins numbered from 1
\*/
void ga_bin_index_(Integer *g_bin, Integer *g_cnt, Integer *g_off, 
                   Integer *values, Integer *subs, Integer *n, Integer *sortit)
{
int i, my_nbin=0;
int *all_bin_contrib, *offset;
Integer type, ndim, nbin;

    nga_inquire_(g_bin, &type, &ndim, &nbin);
    if(ndim !=1) ga_error("ga_bin_index: 1-dim array required",ndim);
    if(type!= MT_F_INT)ga_error("ga_bin_index: not integer type",type);

    all_bin_contrib = (int*)calloc(nbin,sizeof(int));
    if(!all_bin_contrib)ga_error("ga_binning:calloc failed",nbin);
    offset = (int*)malloc(nbin*sizeof(int));
    if(!offset)ga_error("ga_binning:malloc failed",nbin);

    /* count how many elements go to each bin */
    for(i=0; i< *n; i++){
       int selected = subs[i];
       if(selected <1 || selected> nbin) ga_error("wrong bin",selected);

       if(all_bin_contrib[selected-1] ==0) my_nbin++; /* new bin found */
       all_bin_contrib[selected-1]++;
    }

    /* process bins in chunks to match available buffer space */
    for(i=0; i<nbin; i+=NWORK){
        int cnbin = ((i+NWORK)<nbin) ? NWORK: nbin -i;
        gai_bin_offset(SCOPE_ALL, all_bin_contrib+i, cnbin, offset+i);
    }

    for(i=0; i< *n; ){
       Integer lo, hi;
       Integer selected = subs[i];
       int elems = all_bin_contrib[selected-1];

       nga_get_(g_off,&selected,&selected, &lo, &selected);
       lo += offset[selected-1]+1;
       hi = lo + elems -1;
/*
       printf("%d: elems=%d lo=%d sel=%d off=%d contrib=%d nbin=%d\n",ga_nodeid_(), elems, lo, selected,offset[selected-1],all_bin_contrib[0],nbin);
*/
       nga_put_(g_bin, &lo, &hi, values+i, &selected); 
       i+=elems;
    }
    
    free(offset);
    free(all_bin_contrib);

    if(*sortit)ga_bin_sorter_(g_bin, g_cnt, g_off);
    else ga_sync_();
}

