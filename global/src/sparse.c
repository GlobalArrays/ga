#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STRINGS_H
#   include <strings.h>
#endif
#include "globalp.h"
#include "macdecls.h"
#include "message.h"
#include "papi.h"
#include "wapi.h"

static void sgai_combine_val(Integer type, void *ptra, void *ptrb, Integer n, void* val,
                            Integer add, Integer excl)
{
  int i=0;
  switch (type){
    int *ia, *ib;
    double *da, *db;
    DoubleComplex *ca, *cb;
    SingleComplex *cfa, *cfb;
    float *fa, *fb;
    long *la, *lb;
    long long *lla, *llb;
  case C_INT:
    ia = (int*)ptra;
    ib = (int*)ptrb;
    if(add) {
      if (excl) {
        for (i=0; i<n; i++) {
          if (i==0) {
            ib[i] = 0;
          } else {
            ib[i] = ib[i-1] + ia[i-1]; 
          }
        }
      } else {
        for(i=0; i< n; i++) {
          if(i==0) 
            ib[i] = ia[i];
          else
            ib[i] = ib[i-1] + ia[i]; 
        }
      }
    }
    else {
      for(i=0; i< n; i++) ib[i] = *(int*)val; 
    }
    break;
  case C_DCPL:
    ca = (DoubleComplex*)ptra;
    cb = (DoubleComplex*)ptrb;
    if(add) {
      if (excl) {
        for(i=0; i< n; i++) {
          if (i==0) {
            cb[i].real = 0.0;
            cb[i].imag = 0.0;
          } else {
            cb[i].real = cb[i-1].real + ca[i-1].real; 
            cb[i].imag = cb[i-1].imag + ca[i-1].imag; 
          }
        }
      } else {
        for(i=0; i< n; i++){
          if(i==0) {
            cb[i].real = ca[i].real;
            cb[i].imag = ca[i].imag;
          }  else {
            cb[i].real = cb[i-1].real + ca[i].real; 
            cb[i].imag = cb[i-1].imag + ca[i].imag; 
          }
        }
      }
    }
    else
      for(i=0; i< n; i++){
        cb[i].real = ((DoubleComplex*)val)->real; 
        cb[i].imag = ((DoubleComplex*)val)->imag; 
      }
    break;
    case C_SCPL:
    cfa = (SingleComplex*)ptra;
    cfb = (SingleComplex*)ptrb;
    if(add) {
      if (excl) {
        for(i=0; i< n; i++){
          if (i==0) {
            cfb[i].real = 0.0;
            cfb[i].imag = 0.0;
          } else {
            cfb[i].real = cfb[i-1].real + cfa[i-1].real; 
            cfb[i].imag = cfb[i-1].imag + cfa[i-1].imag; 
          }
        }
      } else {
        for(i=0; i< n; i++){
          if(i==0) {
            cfb[i].real = cfa[i].real;
            cfb[i].imag = cfa[i].imag;
          }  else {
            cfb[i].real = cfb[i-1].real + cfa[i].real; 
            cfb[i].imag = cfb[i-1].imag + cfa[i].imag; 
          }
        }
      }
    }
    else
      for(i=0; i< n; i++){
        cfb[i].real = ((SingleComplex*)val)->real; 
        cfb[i].imag = ((SingleComplex*)val)->imag; 
      }
    break;
    case C_DBL:
    da = (double*)ptra;
    db = (double*)ptrb;
    if(add) {
      if (excl) {
        for(i=0; i< n; i++) {
          if (i==0) {
            db[i] = 0.0;
          } else {
            db[i] = db[i-1] + da[i-1]; 
          }
        }
      } else {
        for(i=0; i< n; i++) {
          if(i==0) 
            db[i] = da[i];
          else
            db[i] = db[i-1] + da[i]; 
        }
      }
    } else
      for(i=0; i< n; i++) db[i] = *(double*)val; 
    break;
    case C_FLOAT:
    fa = (float*)ptra;
    fb = (float*)ptrb;
    if(add) {
      if (excl) {
        if (i==0) {
            fb[i] = 0.0;
        } else {
            fb[i] = fb[i-1] + fa[i-1];
        }
      } else {
        for(i=0; i< n; i++) {
          if(i==0)
            fb[i] = fa[i];
          else
            fb[i] = fb[i-1] + fa[i];
        }
      }
    }
    else
      for(i=0; i< n; i++) fb[i] = *(float*)val;
    break; 
    case C_LONG:
    la = (long*)ptra; 
    lb = (long*)ptrb; 
    if(add) {
      if (excl) {
        for(i=0; i< n; i++) {
          if (i==0) {
            lb[i] = 0;
          } else {
            lb[i] = lb[i-1] + la[i-1];
          }
        }
      } else {
        for(i=0; i< n; i++) {
          if(i==0)
            lb[i] = la[i];
          else
            lb[i] = lb[i-1] + la[i];
        }
      }
    }
    else
      for(i=0; i< n; i++) lb[i] = *(long*)val;
    break;                                                         
    case C_LONGLONG:
    lla = (long long*)ptra; 
    llb = (long long*)ptrb; 
    if(add) {
      if (excl) {
        for(i=0; i< n; i++) {
          if (i==0) {
            llb[i] = 0;
          } else {
            llb[i] = llb[i-1] + lla[i-1];
          }
        }
      } else {
        for(i=0; i< n; i++) {
          if(i==0)
            llb[i] = lla[i];
          else
            llb[i] = llb[i-1] + lla[i];
        }
      }
    }
    else
      for(i=0; i< n; i++) llb[i] = *(long long*)val;
    break;                                                         
    default: pnga_error("ga_scan/add:wrong data type",type);
  }
}

#if 0
static void gai_add_val(int type, void *ptr1, void *ptr2, int n, void* val)
{
    int i;
 
        switch (type){
          int *ia1, *ia2;
          double *da1, *da2;
          DoubleComplex *ca1, *ca2;
          SingleComplex *cfa1, *cfa2;
          float *fa1, *fa2;
          long *la1, *la2; 
          long long *lla1, *lla2; 
          case C_INT:
             ia1 = (int*)ptr1;
             ia2 = (int*)ptr2;
             ia2[0] = ia1[0] +  *(int*)val; 
             for(i=1; i< n; i++) ia2[i] = ia2[i-1]+ia1[i];
             break;
          case C_DCPL:
             ca1 = (DoubleComplex*)ptr1;
             ca2 = (DoubleComplex*)ptr2;
             ca2->real = ca1->real +  ((DoubleComplex*)val)->real; 
             ca2->imag = ca1->imag +  ((DoubleComplex*)val)->imag; 
             for(i=1; i< n; i++){
                   ca2[i].real = ca2[i-1].real + ca1[i].real;
                   ca2[i].imag = ca2[i-1].imag + ca1[i].imag;
             }
             break;
          case C_SCPL:
             cfa1 = (SingleComplex*)ptr1;
             cfa2 = (SingleComplex*)ptr2;
             cfa2->real = cfa1->real +  ((SingleComplex*)val)->real; 
             cfa2->imag = cfa1->imag +  ((SingleComplex*)val)->imag; 
             for(i=1; i< n; i++){
                   cfa2[i].real = cfa2[i-1].real + cfa1[i].real;
                   cfa2[i].imag = cfa2[i-1].imag + cfa1[i].imag;
             }
             break;
          case C_DBL:
             da1 = (double*)ptr1;
             da2 = (double*)ptr2;
             da2[0] = da1[0] +  *(double*)val; 
             for(i=1; i< n; i++) da2[i] = da2[i-1]+da1[i];
             break;
          case C_FLOAT:
             fa1 = (float*)ptr1;
             fa2 = (float*)ptr2;
             fa2[0] = fa1[0] +  *(float*)val;
             for(i=1; i< n; i++) fa2[i] = fa2[i-1]+fa1[i];
             break;   
          case C_LONG:
             la1 = (long*)ptr1;
             la2 = (long*)ptr2;
             la2[0] = la1[0] +  *(long*)val;
             for(i=1; i< n; i++) la2[i] = la2[i-1]+la1[i];
             break;
          case C_LONGLONG:
             lla1 = (long long*)ptr1;
             lla2 = (long long*)ptr2;
             lla2[0] = lla1[0] +  *(long long*)val;
             for(i=1; i< n; i++) lla2[i] = lla2[i-1]+lla1[i];
             break;
          default: pnga_error("ga_add_val:wrong data type",type);
        }
}                                                               
#endif


static void sgai_copy_sbit(Integer type, void *a, Integer n, void *b, Integer *sbit, Integer pack, Integer mx)
{
    int i, cnt=0;
    int         *is, *id;
    double *ds, *dd;
    DoubleComplex   *cs, *cd;
    SingleComplex   *cfs, *cfd;
    float           *fs, *fd;
    long            *ls, *ld;
    long long      *lls, *lld;
    if(pack)
        switch (type){
         case C_INT:
             is = (int*)a; id = (int*)b;
             for(i=0; i< n; i++) if(sbit[i]) { 
                     *id = is[i]; id++;
                     cnt++;
          }
             break;
          case C_DCPL:
             cs = (DoubleComplex*)a; cd = (DoubleComplex*)b;
             for(i=0; i< n; i++)if(sbit[i]){
                 cd->real  = cs[i].real; cd->imag  = cs[i].imag; cd ++;
                 cnt++;
         }
             break;
          case C_SCPL:
             cfs = (SingleComplex*)a; cfd = (SingleComplex*)b;
             for(i=0; i< n; i++)if(sbit[i]){
                 cfd->real  = cfs[i].real; cfd->imag  = cfs[i].imag; cfd ++;
                 cnt++;
         }
             break;
          case C_DBL:
             ds = (double*)a; dd = (double*)b;
             for(i=0; i< n; i++)if(sbit[i]){ *dd = ds[i]; dd++; cnt++; }
             break;
          case C_FLOAT:
             fs = (float*)a; fd = (float*)b;
             for(i=0; i< n; i++) if(sbit[i]) {
                     *fd = fs[i]; fd++; cnt++;
          }
             break;   
          case C_LONG:
             ls = (long*)a; ld = (long*)b;
             for(i=0; i< n; i++) if(sbit[i]) {
                     *ld = ls[i]; ld++; cnt++;
          }
             break;    
          case C_LONGLONG:
             lls = (long long*)a; lld = (long long*)b;
             for(i=0; i< n; i++) if(sbit[i]) {
                     *lld = lls[i]; lld++; cnt++;
          }
             break;    
          default: pnga_error("ga_copy_sbit:wrong data type",type);
        }
    else
        switch (type){
          case C_INT:
             is = (int*)b; id = (int*)a;
             for(i=0; i< n; i++) if(sbit[i]) { id[i] = *is; is++;  cnt++; }
             break;
          case C_DCPL:
             cs = (DoubleComplex*)b; cd = (DoubleComplex*)a;
             for(i=0; i< n; i++)if(sbit[i]){
                 cd[i].real  = cs->real; cd[i].imag  = cs->imag; cs++; cnt++; }
             break;
          case C_SCPL:
             cfs = (SingleComplex*)b; cfd = (SingleComplex*)a;
             for(i=0; i< n; i++)if(sbit[i]){
                 cfd[i].real  = cfs->real; cfd[i].imag  = cfs->imag; cfs++; cnt++; }
             break;
          case C_DBL:
             ds = (double*)b; dd = (double*)a;
             for(i=0; i< n; i++)if(sbit[i]){ dd[i] = *ds; ds++; cnt++; }
             break;
          case C_FLOAT:
             fs = (float*)b; fd = (float*)a;
             for(i=0; i< n; i++) if(sbit[i]) { fd[i] = *fs; fs++;  cnt++; }
             break;   
          case C_LONG:
             ls = (long*)b; ld = (long*)a;
             for(i=0; i< n; i++) if(sbit[i]) { ld[i] = *ls; ls++;  cnt++; }
             break;     
          case C_LONGLONG:
             lls = (long long*)b; lld = (long long*)a;
             for(i=0; i< n; i++) if(sbit[i]) { lld[i] = *lls; lls++;  cnt++; }
             break; 
          default: pnga_error("ga_copy_sbit:wrong data type",type);
        }
    if(cnt!=mx){
        printf("\nga_copy_sbit: cnt=%d should be%ld\n",cnt,(long)mx);
        pnga_error("ga_copy_sbit mismatch",0);
    }
}



/*\ sets values for specified array elements by enumerating with stride
\*/
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_patch_enum = pnga_patch_enum
#endif
void pnga_patch_enum(Integer g_a, Integer lo, Integer hi, void* start, void* stride)
{
Integer dims[1],lop,hip;
Integer ndim, type, me, off;
register Integer i;
register Integer nelem;

   pnga_sync();
   me = pnga_nodeid();

   pnga_check_handle(g_a, "ga_patch_enum");

   ndim = pnga_ndim(g_a);
   if (ndim > 1) pnga_error("ga_patch_enum:applicable to 1-dim arrays",ndim);

   pnga_inquire(g_a, &type, &ndim, dims);
   pnga_distribution(g_a, me, &lop, &hip);

   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */

      /* take product of patch owned and specified by the user */ 
      if(hi <lop || hip <lo); /* we got no elements to update */
      else{
        void *ptr;
        Integer ld;
        nelem = hip-lop+1;

        if(lop < lo)lop = lo;
        if(hip > hi)hip = hi;
        off = lop - lo;
        pnga_access_ptr(g_a, &lop, &hip, &ptr, &ld);
        
        switch (type) {
#define ga_patch_enum_reg aptr[i] = astart + ((off+i)*astride)
#define ga_patch_enum_cpl aptr[i].real = astart.real + ((off+i)*astride.real); \
                          aptr[i].imag = astart.imag + ((off+i)*astride.imag)
#define ga_patch_enum_case(MT,T,INNER) \
            case MT: \
                { \
                    T *aptr = (T*)ptr; \
                    T astart = *((T*)start); \
                    T astride = *((T*)stride); \
                    for (i=0; i<nelem; i++) { \
                        ga_patch_enum_##INNER; \
                    } \
                    break; \
                }
            ga_patch_enum_case(C_INT,int,reg)
            ga_patch_enum_case(C_LONG,long,reg)
            ga_patch_enum_case(C_LONGLONG,long long,reg)
            ga_patch_enum_case(C_FLOAT,float,reg)
            ga_patch_enum_case(C_DBL,double,reg)
            ga_patch_enum_case(C_SCPL,SingleComplex,cpl)
            ga_patch_enum_case(C_DCPL,DoubleComplex,cpl)
#undef ga_patch_enum_case
#undef ga_patch_enum_reg
#undef ga_patch_enum_cpl
            default: pnga_error("ga_patch_enum:wrong data type ",type);
        }

        pnga_release_update(g_a, &lop, &hip);
      }
   }
   
   pnga_sync();
}



static void sgai_scan_copy_add(Integer g_a, Integer g_b, Integer g_sbit, 
           Integer lo, Integer hi, int add, Integer excl)
{
   Integer *lim=NULL, *lom=NULL, nproc, me;
   Integer lop, hip, ndim, dims, type, ioff;
   double buf[2];
   Integer *ia=NULL, *ip=NULL, elems,ld;
   int i, k;
   void *ptr_b=NULL;
   void *ptr_a=NULL;

   nproc = pnga_nnodes();
      me = pnga_nodeid();

   pnga_check_handle(g_a, "ga_scan_copy");
   pnga_check_handle(g_b, "ga_scan_copy 2");
   pnga_check_handle(g_sbit,"ga_scan_copy 3");

   pnga_sync();


   ndim = pnga_ndim(g_a);
   if(ndim>1)pnga_error("ga_scan_copy: applicable to 1-dim arrays",ndim);

   pnga_inquire(g_a, &type, &ndim, &dims);
   pnga_distribution(g_sbit, me, &lop, &hip);

   /* create arrays to hold first and last bits set on a given process */
   lim = (Integer *) ga_malloc(2*nproc, MT_F_INT, "ga scan buf");
   bzero(lim,2*sizeof(Integer)*nproc);

   lom = lim + nproc;

   if(!pnga_compare_distr(g_a, g_sbit))
       pnga_error("ga_scan_copy: different distribution src",0);
   if(!pnga_compare_distr(g_b, g_sbit))
       pnga_error("ga_scan_copy: different distribution dst",0);
      
   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */ 

        pnga_access_ptr(g_sbit, &lop, &hip, &ia, &ld);
        elems = hip - lop + 1;
        /* find last bit set on given process (store as global index) */
        for(i=0; i<elems; i++) {
          if(ia[i]) {
            ioff = i + lop;
            if (ioff >= lo && ioff <= hi) {
              lim[me]= ioff;
            }
            /* find first bit set on given process (store as local index) */
            if (!lom[me]) {
              lom[me] = i;
            }
          }
        }
   } else {
     /* if processor has no data then set value to -1 */
     lim[me] = -1;
   }

   pnga_gop(pnga_type_f2c(MT_F_INT),lim, 2*nproc,"+");

   /* take intersection of patch owned by process and patch
      specified by the user */ 
   if(hi <lop || hip <lo); /* we have no elements to update */
   else{
       Integer lops=lop, hips=hip;
       Integer startp=0;

       /* what part of local data we should be working on */
       ip = ia;
       if(lop < lo){
           /* user specified patch starts in the middle */
           ip = ia + (lo-lop); /*set pointer to first value in sbit array*/
           lop = lo;
       } 
       if(hip > hi) hip = hi;
      
       /* access the data. g_a is source, g_b is destination */
       pnga_access_ptr(g_b, &lop, &hip, &ptr_b, &ld);
       pnga_access_ptr(g_a, &lop, &hip, &ptr_a, &ld);

       /* find start bit corresponding to my patch */
       /* case 1: sbit set for the first patch element and check earlier elems*/
       for(k=lop, i=0; k >= lops; i--, k--) if (ip[i]) { startp = k; break; }
       if(!startp){
          /* case2: scan lim to find sbit set on lower numbered processors */ 
          for(k=me-1; k >=0; k--)if(lim[k]>0) {startp =lim[k]; break; }
       }
       if(!startp) pnga_error("sbit not found for",lop); /*nothing was found*/

       /* copy or scan the data */
       i = 0;
       for(k=lop; k<= hip; ){ 
           int indx=i;
           Integer one=1;
           int elemsize = GAsizeofM(type);
           
           /* find where sbit changes */ 
           for(; i< hip-lop; indx=++i) if(ip[i+1]) {i++; break;}
           /* at this point, i equals the location of the next non-zero value in
            * sbit, indx equals the location of the last entry before this bit
            * (unless there are two consecutive non-zero values in sbit, this
            * will point to a zero in sbit) */

           elems = indx- k+lop +1; /* the number of elements that will be updated*/

           /* get the current value of A */
           pnga_get(g_a, &startp, &startp, buf, &one);

           /* assign elements of B
              If add then assign ptr_b[i] = ptr_b[i-1]+ptr_a[i]
              If add and excl then ptr_b[i] = ptr_b[i-1] + ptr_a[i-1]
              If !add then ptr_b[i] = *buf */
           sgai_combine_val(type, ptr_a, ptr_b, elems, buf, add, excl); 

           ptr_a = (char*)ptr_a + elems*elemsize;
           ptr_b = (char*)ptr_b + elems*elemsize;
           k += elems;
           startp = k;
       }
       /* release local access to arrays */
       pnga_release(g_a, &lop, &hip);
       pnga_release(g_b, &lop, &hip);
       if (lops > 0) pnga_release(g_sbit, &lops, &hips);

    }

    /* fix up scan_add values for segments that cross processor boundaries */
    if (add) {
      Integer ichk = 1;
      pnga_access_ptr(g_b, &lop, &hip, &ptr_b, &ld);
      if (excl) pnga_access_ptr(g_a, &lop, &hip, &ptr_a, &ld);
      ioff = hip - lop;
      switch (type) {
        int *ilast;
        DoubleComplex *cdlast;
        SingleComplex *cflast;
        double *dlast;
        float *flast;
        long *llast;
        long long *lllast;
        case C_INT:
          ilast = (int*) ga_malloc(nproc, C_INT, "ga add buf");
          bzero(ilast,sizeof(int)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            ilast[me] = ((int*)ptr_b)[ioff];
            if (excl) {
              if (lim[me] - lop == ioff) {
                ilast[me] = ((int*)ptr_a)[ioff];
              } else {
                ilast[me] += ((int*)ptr_a)[ioff];
              }
            }
          }
          pnga_gop(MT_C_INT,ilast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me]; 
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((int*)ptr_b)[i] += (int)ilast[k];
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(ilast);
          break;
        case C_DCPL:
          cdlast = (DoubleComplex*) ga_malloc(nproc, C_DCPL, "ga add buf");
          bzero(cdlast,sizeof(DoubleComplex)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            cdlast[me].real = ((DoubleComplex*)ptr_b)[ioff].real;
            cdlast[me].imag = ((DoubleComplex*)ptr_b)[ioff].imag;
            if (excl) {
              if (lim[me] - lop == ioff) {
                cdlast[me].real = ((DoubleComplex*)ptr_a)[ioff].real;
                cdlast[me].imag = ((DoubleComplex*)ptr_a)[ioff].imag;
              } else {
                cdlast[me].real += ((DoubleComplex*)ptr_a)[ioff].real;
                cdlast[me].imag += ((DoubleComplex*)ptr_a)[ioff].imag;
              }
            }
          }
          pnga_gop(MT_C_DCPL,cdlast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((DoubleComplex*)ptr_b)[i].real += cdlast[k].real;
                ((DoubleComplex*)ptr_b)[i].imag += cdlast[k].imag;
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(cdlast);
          break;
        case C_SCPL:
          cflast = (SingleComplex*) ga_malloc(nproc, C_SCPL, "ga add buf");
          bzero(cflast,sizeof(SingleComplex)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            cflast[me].real = ((SingleComplex*)ptr_b)[ioff].real;
            cflast[me].imag = ((SingleComplex*)ptr_b)[ioff].imag;
            if (excl) {
              if (lim[me] - lop == ioff) {
                cflast[me].real = ((SingleComplex*)ptr_a)[ioff].real;
                cflast[me].imag = ((SingleComplex*)ptr_a)[ioff].imag;
              } else {
                cflast[me].real += ((SingleComplex*)ptr_a)[ioff].real;
                cflast[me].imag += ((SingleComplex*)ptr_a)[ioff].imag;
              }
            }
          }
          pnga_gop(MT_C_SCPL,cflast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((SingleComplex*)ptr_b)[i].real += cflast[k].real;
                ((SingleComplex*)ptr_b)[i].imag += cflast[k].imag;
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(cflast);
          break;
        case C_DBL:
          dlast = (double*) ga_malloc(nproc, C_DBL, "ga add buf");
          bzero(dlast,sizeof(double)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            dlast[me] = ((double*)ptr_b)[ioff];
            if (excl) {
              if (lim[me] - lop == ioff) {
                dlast[me] = ((double*)ptr_a)[ioff];
              } else {
                dlast[me] += ((double*)ptr_a)[ioff];
              }
            }
          }
          pnga_gop(MT_C_DBL,dlast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((double*)ptr_b)[i] += dlast[k];
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(dlast);
          break;
        case C_FLOAT:
          flast = (float*) ga_malloc(nproc, C_FLOAT, "ga add buf");
          bzero(flast,sizeof(float)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            flast[me] = ((float*)ptr_b)[ioff];
            if (excl) {
              if (lim[me] - lop == ioff) {
                flast[me] = ((float*)ptr_a)[ioff];
              } else {
                flast[me] += ((float*)ptr_a)[ioff];
              }
            }
          }
          pnga_gop(MT_C_FLOAT,flast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((float*)ptr_b)[i] += flast[k];
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(flast);
          break;
        case C_LONG:
          llast = (long*) ga_malloc(nproc, C_LONG, "ga add buf");
          bzero(llast,sizeof(long)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            llast[me] = ((long*)ptr_b)[ioff];
            if (excl) {
              if (lim[me] - lop == ioff) {
                llast[me] = ((long*)ptr_a)[ioff];
              } else {
                llast[me] += ((long*)ptr_a)[ioff];
              }
            }
          }
          pnga_gop(MT_C_LONGINT,llast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((long*)ptr_b)[i] += llast[k];
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(llast);
          break;
        case C_LONGLONG:
          lllast = (long long*) ga_malloc(nproc, C_LONGLONG, "ga add buf");
          bzero(lllast,sizeof(long long)*nproc);
          if (lim[me] >= 0) { /* This processor contains data */
            lllast[me] = ((long long*)ptr_b)[ioff];
            if (excl) {
              if (lim[me] - lop == ioff) {
                lllast[me] = ((long long*)ptr_a)[ioff];
              } else {
                lllast[me] += ((long long*)ptr_a)[ioff];
              }
            }
          }
          pnga_gop(MT_C_LONGLONG,lllast,nproc,"+");
          if (!ip[0]) {
            Integer iup;
            if (lim[me] > 0) { /* There is a bit set on this processor */
              iup = lom[me];
            } else {
              iup = hip - lop + 1;
            }
            for (k=me-1; k>=0 && ichk; k--) {
              for (i=0; i<iup; i++) {
                ((long long*)ptr_b)[i] += lllast[k];
              }
              if (lim[k] > 0) ichk = 0;
            }
          }
          ga_free(lllast);
          break;
        default: pnga_error("ga_scan/add:wrong data type",type);
      }
      pnga_release(g_b, &lop, &hip);
      if (excl) pnga_release(g_a, &lop, &hip);

   }

   pnga_sync();
   ga_free(lim);
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_scan_copy = pnga_scan_copy
#endif
void pnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit,
                           Integer lo, Integer hi)
{       
        Integer zero = 0;
        sgai_scan_copy_add(g_a, g_b, g_sbit, lo, hi, 0, zero);
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_scan_add = pnga_scan_add
#endif
void pnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit,
                           Integer lo, Integer hi, Integer excl)
{       
        sgai_scan_copy_add(g_a, g_b, g_sbit, lo, hi, 1, excl);
}

/**
 * pack/unpack data from g_a into g_b based on the mask array g_sbit
 * Return total number of bits set in variable icount.
 */
static void sgai_pack_unpack(Integer g_a, Integer g_b, Integer g_sbit,
              Integer lo, Integer hi, Integer* icount, int pack)
{
   void *ptr;
   Integer *lim=NULL, nproc, me;
   Integer lop, hip, ndim, dims, type,crap;
   Integer *ia=NULL, elems=0, i=0, first=0, myplace =0, counter=0;

   nproc = pnga_nnodes();
      me = pnga_nodeid();

   pnga_check_handle(g_a, "ga_pack");
   pnga_check_handle(g_b, "ga_pack 2");
   pnga_check_handle(g_sbit,"ga_pack 3");

   pnga_sync();

   lim = (Integer *) ga_malloc(nproc, MT_F_INT, "ga_pack lim buf");

   bzero(lim,sizeof(Integer)*nproc);
   pnga_inquire(g_a, &type, &ndim, &dims);
   if(ndim>1) pnga_error("ga_pack: supports 1-dim arrays only",ndim);
   pnga_distribution(g_sbit, me, &lop, &hip);

   /* how many elements we have to copy? */
   if ( lop > 0 ){ /* we get 0 if no elements stored on this process */

        /* adjust the range of elements to be within <lo,hi> */
        if(lop < lo) lop = lo;
        if(hip > hi) hip = hi;

        if(hi <lop || hip <lo); /* we have no elements to update */
        else{

          pnga_access_ptr(g_sbit, &lop, &hip, &ptr, &elems);
          ia    = (Integer*)ptr;
          elems = hip -lop+1;

          /* find number of elements to be contributed */
          for(i=counter=0,first=-1; i<elems; i++) if(ia[i]){
              counter++;
              if(first==-1) first=i;
          }
          lim[me] = counter;
        }
   }

   /* find number of elements everybody else is contributing */
   pnga_gop(pnga_type_f2c(MT_F_INT), lim, nproc,"+");

   for(i= myplace= *icount= 0; i<nproc; i++){
        if( i<me && lim[i]) myplace += lim[i];
        *icount += lim[i];
   }
   ga_free(lim);

   if(hi <lop || hip <lo || counter ==0 ); /* we have no elements to update */
   else{

     void *buf;
     Integer start=lop+first; /* the first element for which sbit is set */
     Integer dst_lo =myplace+1, dst_hi = myplace + counter;

     pnga_access_ptr(g_a, &start, &hip, &ptr, &crap);

     buf = ga_malloc(counter, type, "ga pack buf");

     /* stuff data selected by sbit into(pack) or from(unpack) buffer */
     if(pack){

        sgai_copy_sbit(type, ptr, hip-lop+1-first , buf, ia+first, pack,counter); /* pack data to buf */
        pnga_put(g_b, &dst_lo, &dst_hi,  buf, &counter); /* put it into destination array */

     }else{

        pnga_get(g_b, &dst_lo, &dst_hi,  buf, &counter); /* get data to buffer*/
        sgai_copy_sbit(type, ptr, hip-lop+1-first , buf, ia+first, pack,counter);  /* copy data to array*/

     }

     ga_free(buf); 

   }

   pnga_sync();
}



#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_pack = pnga_pack
#endif
void pnga_pack(Integer g_a, Integer g_b, Integer g_sbit,
              Integer lo, Integer hi, Integer* icount)
{
     sgai_pack_unpack( g_a, g_b, g_sbit, lo, hi, icount, 1);
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_unpack = pnga_unpack
#endif
void pnga_unpack(Integer g_a, Integer g_b, Integer g_sbit,
              Integer lo, Integer hi, Integer* icount)
{
     sgai_pack_unpack( g_a, g_b, g_sbit, lo, hi, icount, 0);
}



#define NWORK 2000
int workR[NWORK], workL[NWORK];

/*\ compute offset for each of n bins for the given processor to contribute its
 *  elements, number of which for each bin is specified in x
\*/ 
static void sgai_bin_offset(int scope, int *x, int n, int *offset)
{
int root, up, left, right;
int len, lenmes, tag=32100, i, me=armci_msg_me();

    if(!x)pnga_error("sgai_bin_offset: NULL pointer", n);
    if(n>NWORK)pnga_error("sgai_bin_offset: >NWORK", n);
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
Integer sgai_match_bin2proc(Integer blo, Integer bhi, Integer plo, Integer phi)
{
int rc=0;
       if(blo == plo) rc=1;
       if(bhi == phi) rc+=2; 
       return rc; /* 1 - first 2-last 3-last+first */
}


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_create_bin_range = pnga_create_bin_range
#endif
logical pnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
Integer type, ndim, nbin, lobin, hibin, me=pnga_nodeid(),crap;
Integer dims[2], nproc=pnga_nnodes(),chunk[2];
Integer tlo[2], thi[2];

    pnga_inquire(g_bin, &type, &ndim, &nbin);
    if(ndim !=1) pnga_error("ga_bin_index: 1-dim array required",ndim);
    if(type!= C_INT && type!=C_LONG && type!=C_LONGLONG)
       pnga_error("ga_bin_index: not integer type",type);

    chunk[0]=dims[0]=2; dims[1]=nproc; chunk[1]=1;
    if(!pnga_create(MT_F_INT, 2, dims, "bin_proc",chunk,g_range)) return FALSE;

    pnga_distribution(g_off,me, &lobin,&hibin);

    if(lobin>0){ /* enter this block when we have data */
      Integer first_proc, last_proc, p;
      Integer first_off, last_off;
      Integer *myoff, bin;

      /* get offset values stored on my processor to first and last bin */
      pnga_access_ptr(g_off, &lobin, &hibin, &myoff, &crap);
      first_off = myoff[0]; last_off = myoff[hibin-lobin];
/*
      pnga_get(g_off,&lobin,&lobin,&first_off,&lo);
      pnga_get(g_off,&hibin,&hibin,&last_off,&hi);
*/

      /* since offset starts at 0, add 1 to get index to g_bin */
      first_off++; last_off++;

      /* find processors on which these bins are located */
      if(!pnga_locate(g_bin, &first_off, &first_proc))
          pnga_error("ga_bin_sorter: failed to locate region f",first_off);
      if(!pnga_locate(g_bin, &last_off, &last_proc))
          pnga_error("ga_bin_sorter: failed to locate region l",last_off);

      /* inspect range of indices to bin elements stored on these processors */
      for(p=first_proc, bin=lobin; p<= last_proc; p++){
          Integer lo, hi, buf[2], off, cnt; 
          buf[0] =-1; buf[1]=-1;

          pnga_distribution(g_bin,p,&lo,&hi);

          for(/* start from current bin */; bin<= hibin; bin++, myoff++){ 
              Integer blo,bhi,stat;

              blo = *myoff +1;
              if(bin == hibin){
                 pnga_get(g_cnt, &hibin, &hibin, &cnt, &hibin); /* local */
                 bhi = blo + cnt-1; 
              }else
                 bhi = myoff[1]; 

              stat= sgai_match_bin2proc(blo, bhi, lo, hi);

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
                 tlo[0] = lo;
                 tlo[1] = p1;
                 thi[0] = hi;
                 thi[1] = p1;
                 pnga_put(*g_range, tlo, thi, buf+off, &cnt);
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


#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_bin_sorter = pnga_bin_sorter
#endif
void pnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
Integer nbin,totbin,type,ndim,lo,hi,me=pnga_nodeid(),crap;
Integer g_range;

    if(FALSE==pnga_create_bin_range(g_bin, g_cnt, g_off, &g_range))
        pnga_error("ga_bin_sorter: failed to create temp bin range array",0); 

    pnga_inquire(g_bin, &type, &ndim, &totbin);
    if(ndim !=1) pnga_error("ga_bin_sorter: 1-dim array required",ndim);
     
    pnga_distribution(g_bin, me, &lo, &hi);
    if (lo > 0 ){ /* we get 0 if no elements stored on this process */
        Integer bin_range[2], rlo[2],rhi[2];
        Integer *bin_cnt, *ptr, i;

        /* get and inspect range of bins stored on current processor */
        rlo[0] = 1; rlo[1]= me+1; rhi[0]=2; rhi[1]=rlo[1];
        pnga_get(g_range, rlo, rhi, bin_range, rhi); /* local */
        nbin = bin_range[1]-bin_range[0]+1;
        if(nbin<1 || nbin> totbin || nbin>(hi-lo+1))
           pnga_error("ga_bin_sorter:bad nbin",nbin);

        /* get count of elements in each bin stored on this task */
        if(!(bin_cnt = (Integer*)malloc(nbin*sizeof(Integer))))
           pnga_error("ga_bin_sorter:memory allocation failed",nbin);
        pnga_get(g_cnt,bin_range,bin_range+1,bin_cnt,&nbin);

        /* get access to local bin elements */
        pnga_access_ptr(g_bin, &lo, &hi, &ptr, &crap);
        
        for(i=0;i<nbin;i++){ 
            int elems =(int) bin_cnt[i];
            gai_hsort(ptr, elems);
            ptr+=elems;
        }
        pnga_release_update(g_bin, &lo, &hi);             
    }

    pnga_sync();
}


/*\ note that subs values must be sorted; bins numbered from 1
\*/
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_bin_index = pnga_bin_index
#endif
void pnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, 
                   Integer *values, Integer *subs, Integer n, Integer sortit)
{
int i, my_nbin=0;
int *all_bin_contrib, *offset;
Integer type, ndim, nbin;

    pnga_inquire(g_bin, &type, &ndim, &nbin);
    if(ndim !=1) pnga_error("ga_bin_index: 1-dim array required",ndim);
    if(type!= C_INT && type!=C_LONG && type!=C_LONGLONG)
       pnga_error("ga_bin_index: not integer type",type);

    all_bin_contrib = (int*)calloc(nbin,sizeof(int));
    if(!all_bin_contrib)pnga_error("ga_binning:calloc failed",nbin);
    offset = (int*)malloc(nbin*sizeof(int));
    if(!offset)pnga_error("ga_binning:malloc failed",nbin);

    /* count how many elements go to each bin */
    for(i=0; i< n; i++){
       int selected = subs[i];
       if(selected <1 || selected> nbin) pnga_error("wrong bin",selected);

       if(all_bin_contrib[selected-1] ==0) my_nbin++; /* new bin found */
       all_bin_contrib[selected-1]++;
    }

    /* process bins in chunks to match available buffer space */
    for(i=0; i<nbin; i+=NWORK){
        int cnbin = ((i+NWORK)<nbin) ? NWORK: nbin -i;
        sgai_bin_offset(SCOPE_ALL, all_bin_contrib+i, cnbin, offset+i);
    }

    for(i=0; i< n; ){
       Integer lo, hi;
       Integer selected = subs[i];
       int elems = all_bin_contrib[selected-1];

       pnga_get(g_off,&selected,&selected, &lo, &selected);
       lo += offset[selected-1]+1;
       hi = lo + elems -1;
/*
       printf("%d: elems=%d lo=%d sel=%d off=%d contrib=%d nbin=%d\n",pnga_nodeid(), elems, lo, selected,offset[selected-1],all_bin_contrib[0],nbin);
*/
       if(lo > nbin) {
	      printf("Writing off end of bins array: index=%d elems=%d lo=%ld hi=%ld values=%ld nbin=%ld\n",
                i,elems,(long)lo,(long)hi,(long)values+i,(long)nbin);
         break;   
       }else{
          pnga_put(g_bin, &lo, &hi, values+i, &selected); 
       }
       i+=elems;
    }
    
    free(offset);
    free(all_bin_contrib);

    if(sortit)pnga_bin_sorter(g_bin, g_cnt, g_off);
    else pnga_sync();
}

