/*$Id: matmul.c,v 1.16 2003-05-07 15:57:14 d3g293 Exp $*/
#include "global.h"
#include "globalp.h"
#include <math.h>
#ifdef KSR
#  define dgemm_ sgemm_
#  define zgemm_ cgemm_
#endif

#ifdef CRAY
#      include <fortran.h>
#      define  DGEMM SGEMM
#      define  ZGEMM CGEMM
#elif defined(WIN32)
   extern void FATR DGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
                     void*, void*, Integer*, void*, Integer*, void*,
                     void*, Integer*);
   extern void FATR ZGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
                     DoubleComplex*, DoubleComplex*, Integer*, DoubleComplex*,
                     Integer*, DoubleComplex*, DoubleComplex*, Integer*);
#elif defined(F2C2__)
#      define DGEMM dgemm__
#      define ZGEMM zgemm__
#elif defined(HITACHI)
#      define dgemm_ DGEMM
#      define zgemm_ ZGEMM
#endif

#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif
#ifdef STATBUF
#  define C_CHUNK  92 
#  define D_CHUNK  64
#  define ICHUNK C_CHUNK
#  define JCHUNK C_CHUNK
#  define KCHUNK C_CHUNK
#else
   /* min acceptable amount of memory (in elements) and default chunk size */
#  define MINMEM 64
#  define CHUNK_SIZE 128
#  define EXTRA 4 /* Extra elements for safety reasons */
#endif

#define VECTORCHECK(rank,dims,dim1,dim2, ilo, ihi, jlo, jhi) \
  if(rank>2)  ga_error("rank is greater than 2",rank); \
  else if(rank==2) {dim1=dims[0]; dim2=dims[1];} \
  else if(rank==1) {if((ihi-ilo)>0) { dim1=dims[0]; dim2=1;} \
                    else { dim1=1; dim2=dims[0];}} \
  else ga_error("rank must be atleast 1",rank); 

static int max3(int ichunk, int jchunk, int kchunk) {
  if(ichunk>jchunk) return MAX(ichunk,kchunk);
  else return MAX(jchunk, kchunk);
} 


/*\ MATRIX MULTIPLICATION for patches 
 *  
 *  C[lo:hi,lo:hi] = alpha*op(A)[lo:hi,lo:hi] * op(B)[lo:hi,lo:hi]        
 *                 + beta *C[lo:hi,lo:hi]
 *
 *  where:
 *          op(A) = A or A' depending on the transpose flag
 *  [lo:hi,lo:hi] - patch indices _after_ op() operator was applied
 *
\*/
void ga_matmul_patch(transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     void    *alpha, *beta;
     char    *transa, *transb;
{
#ifdef STATBUF
  /* approx. sqrt(2) ratio in chunk size to use the same buffer space */
   DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
#else
   DoubleComplex *a, *b, *c;
#endif
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2, dims[2], rank;
Integer me= ga_nodeid_(), nproc;
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim;
Integer Ichunk, Kchunk, Jchunk;
DoubleComplex ONE, ZERO;

DoublePrecision chunk_cube;
Integer min_tasks = 10, max_chunk;
int need_scaling=1;
Integer ZERO_I = 0, inode, iproc;
float ONE_F = 1.0, ZERO_F = 0.0;
double ZERO_D = 0.0;
Integer get_new_B;
int local_sync_begin,local_sync_end;
int idim_t, jdim_t, kdim_t, adim_t, bdim_t, cdim_t;

   ONE.real =1.; ZERO.real =0.;
   ONE.imag =0.; ZERO.imag =0.;

   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(local_sync_begin)ga_sync_();

   GA_PUSH_NAME("ga_matmul_patch");

   /* Check to make sure all global arrays are of the same type */
   if (!(ga_is_mirrored_(g_a) == ga_is_mirrored_(g_b) &&
        ga_is_mirrored_(g_a) == ga_is_mirrored_(g_c))) {
     ga_error_("Processors do not match for all arrays",ga_nnodes_());
   }
   if (ga_is_mirrored_(g_a)) {
     inode = ga_cluster_nodeid_();
     nproc = ga_cluster_nprocs_(&inode);
     iproc = me - ga_cluster_procid_(&inode, &ZERO_I);
   } else {
     nproc = ga_nnodes_();
     iproc = me;
   }

   nga_inquire_internal_(g_a, &atype, &rank, dims); 
   VECTORCHECK(rank, dims, adim1, adim2, *ailo, *aihi, *ajlo, *ajhi);
   nga_inquire_internal_(g_b, &btype, &rank, dims); 
   VECTORCHECK(rank, dims, bdim1, bdim2, *bilo, *bihi, *bjlo, *bjhi);
   nga_inquire_internal_(g_c, &ctype, &rank, dims); 
   VECTORCHECK(rank, dims, cdim1, cdim2, *cilo, *cihi, *cjlo, *cjhi);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
   if(atype != C_DCPL && atype != C_DBL && atype != C_FLOAT) 
     ga_error(" type error",atype);
   
   
   
   /* check if patch indices and dims match */
   if (*transa == 'n' || *transa == 'N'){
     if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
       ga_error("  g_a indices out of range ", *g_a);
   }else
     if (*ailo <= 0 || *aihi > adim2 || *ajlo <= 0 || *ajhi > adim1)
       ga_error("  g_a indices out of range ", *g_a);
   
   if (*transb == 'n' || *transb == 'N'){
     if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error("  g_b indices out of range ", *g_b);
   }else
     if (*bilo <= 0 || *bihi > bdim2 || *bjlo <= 0 || *bjhi > bdim1)
       ga_error("  g_b indices out of range ", *g_b);
   
   if (*cilo <= 0 || *cihi > cdim1 || *cjlo <= 0 || *cjhi > cdim2)
     ga_error("  g_c indices out of range ", *g_c);
   
   /* verify if patch dimensions are consistent */
   m = *aihi - *ailo +1;
   n = *bjhi - *bjlo +1;
   k = *ajhi - *ajlo +1;
   if( (*cihi - *cilo +1) != m) ga_error(" a & c dims error",m);
   if( (*cjhi - *cjlo +1) != n) ga_error(" b & c dims error",n);
   if( (*bihi - *bilo +1) != k) ga_error(" a & b dims error",k);
   

   /* In 32-bit platforms, k*m*n might exceed the "long" range(2^31), 
      eg:k=m=n=1600. So casting the temporary value to "double" helps */
   chunk_cube = (k*(double)(m*n)) / (min_tasks * nproc);
   max_chunk = (Integer)pow(chunk_cube, (DoublePrecision)(1.0/3.0) );
   if (max_chunk < 32) max_chunk = 32;

#ifdef STATBUF
   if(atype ==  C_DBL || atype == C_FLOAT){
      Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
   }else{
      Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
   }
#else
   {
     /**
      * Find out how much memory we can grab.  It will be used in
      * three chunks, and the result includes only the first one.
      */
     
     Integer elems, factor = sizeof(DoubleComplex)/GAsizeofM(atype);
     Ichunk = Jchunk = Kchunk = CHUNK_SIZE;
     
     if ( max_chunk > Ichunk) {       
       /*if memory if very limited, performance degrades for large matrices
	 as chunk size is very small, which leads to communication overhead)*/
       Integer avail = ga_memory_avail(atype);
       ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
       if(avail<MINMEM && ga_nodeid_()==0) ga_error("NotEnough memory",avail);
       elems = (Integer)(avail*0.9); /* Donot use every last drop */
       
       max_chunk=MIN(max_chunk, (Integer)(sqrt( (double)((elems-EXTRA)/3))));
       Ichunk = MIN(m,max_chunk);
       Jchunk = MIN(n,max_chunk);
       Kchunk = MIN(k,max_chunk);
     }
     else /* "EXTRA" elems for safety - just in case */
       elems = 3*Ichunk*Jchunk + EXTRA*factor;
     
     a = (DoubleComplex*) ga_malloc(elems, atype, "GA mulmat bufs");
     b = a + (Ichunk*Kchunk)/factor + 1; 
     c = b + (Kchunk*Jchunk)/factor + 1;
   }
#endif

   if(atype==C_DCPL){if((((DoubleComplex*)beta)->real == 0) &&
	       (((DoubleComplex*)beta)->imag ==0)) need_scaling =0;} 
   else if((atype==C_DBL)){if(*(DoublePrecision *)beta == 0) need_scaling =0;}
   else if( *(float*)beta ==0) need_scaling =0;

   if(need_scaling) ga_scale_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
   else  ga_fill_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);

   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;

       for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
	 khi = MIN(k-1, klo+Kchunk-1);          /* loop rows of g_b patch */
	 kdim= khi - klo +1;                                     
	 
	 /** Each pass through the outer two loops means we need a
	     different patch of B.*/
	 get_new_B = TRUE;
	 
	 for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
	   
	   if(ijk%nproc == iproc){

	     ihi = MIN(m-1, ilo+Ichunk-1);
	     idim= cdim = ihi - ilo +1;
	     
	     if(atype == C_FLOAT) 
	       for (i = 0; i < idim*jdim; i++) *(((float*)c)+i)=0;
	     else if(atype ==  C_DBL)
	       for (i = 0; i < idim*jdim; i++) *(((double*)c)+i)=0;
	     else
	       for (i = 0; i < idim*jdim; i++){ c[i].real=0;c[i].imag=0;}
	     
	     if (*transa == 'n' || *transa == 'N'){ 
	       adim = idim;
	       i0= *ailo+ilo; i1= *ailo+ihi;   
	       j0= *ajlo+klo; j1= *ajlo+khi;
	       ga_get_(g_a, &i0, &i1, &j0, &j1, a, &idim);
	     }else{
	       adim = kdim;
	       i0= *ajlo+klo; i1= *ajlo+khi;   
	       j0= *ailo+ilo; j1= *ailo+ihi;
	       ga_get_(g_a, &i0, &i1, &j0, &j1, a, &kdim);
	     }


	     /* Avoid rereading B if it is the same patch as last time. */
	     if(get_new_B) { 
	       if (*transb == 'n' || *transb == 'N'){ 
		 bdim = kdim;
		 i0= *bilo+klo; i1= *bilo+khi;   
		 j0= *bjlo+jlo; j1= *bjlo+jhi;
		 ga_get_(g_b, &i0, &i1, &j0, &j1, b, &kdim);
	       }else{
		 bdim = jdim;
		 i0= *bjlo+jlo; i1= *bjlo+jhi;   
		 j0= *bilo+klo; j1= *bilo+khi;
		 ga_get_(g_b, &i0, &i1, &j0, &j1, b, &jdim);
	       }
	       get_new_B = FALSE; /* Until J or K change again */
	     }

	     
	     idim_t=idim; jdim_t=jdim; kdim_t=kdim;
	     adim_t=adim; bdim_t=bdim; cdim_t=cdim;

#	   if (defined(CRAY) || defined(WIN32)) && !defined(GA_C_CORE)
	     switch(atype) {
	     case C_FLOAT:
	       xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(float *)alpha, (float *)a, &adim_t, (float *)b, 
			&bdim_t, &ZERO_F,  (float *)c, &cdim_t);
	       break;
	     case C_DBL:
	       DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		     alpha, (double*)a, &adim, (double*)b, &bdim, &ONE, 
		     (double*)c, &cdim);
	       break;
	     case C_DCPL:
	       ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		     (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE,c,&cdim);
	       break;
	     default:
	       ga_error("ga_matmul_patch: wrong data type", atype);
	     }
#          else 
	     switch(atype) {
	     case C_FLOAT:
	       xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(float *)alpha, (float *)a, &adim_t, (float *)b, 
			&bdim_t, &ZERO_F,  (float *)c, &cdim_t);
	       break;
	     case C_DBL:
#            ifdef GA_C_CORE
	       xb_dgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			alpha, (double *)a, &adim_t, (double *)b, &bdim_t, 
			&ZERO_D,  (double *)c, &cdim_t);
#            else
	       dgemm_(transa, transb, &idim, &jdim, &kdim,
		      alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#            endif
	       break;
	     case C_DCPL:
#            ifdef GA_C_CORE
	       xb_zgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(DoubleComplex *)alpha, a, &adim_t, b, &bdim_t, 
			&ZERO,  c, &cdim_t);
#            else
	       zgemm_(transa, transb, &idim, &jdim, &kdim,
		      (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE, c, 
		      &cdim, 1, 1);
#            endif
	       break;
	     default:
	       ga_error("ga_matmul_patch: wrong data type", atype);
	     }
#          endif
	     
	     i0= *cilo+ilo; i1= *cilo+ihi;   j0= *cjlo+jlo; j1= *cjlo+jhi;
	     if(atype == C_FLOAT) 
	       ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, 
		       &cdim, &ONE_F);
	     else
	       ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
		       &cdim, (DoublePrecision*)&ONE);
	   }
	   ++ijk;
	 }
       }
   }
   
#ifndef STATBUF
   ga_free(a);
#endif

   GA_POP_NAME;
   if(local_sync_end)ga_sync_();
}


/*\ MATRIX MULTIPLICATION for patches 
 *  Fortran interface
\*/



/*\ select the 2d plane to be used in matrix multiplication
\*/
static void  gai_setup_2d_patch(Integer rank, Integer dims[], 
		                Integer lo[], Integer hi[], 
		                Integer* ilo, Integer* ihi, 
				Integer* jlo, Integer* jhi, 
				Integer* dim1, Integer* dim2, 
				int* ipos, int* jpos)
{
    int d,e=0;

    for(d=0; d<rank; d++)
      if( (hi[d]-lo[d])>0 && ++e>2 ) ga_error("3-D Patch Detected", 0L);    
    *ipos = *jpos = -1;
    for(d=0; d<rank; d++){
	   if( (*ipos <0) && (hi[d]>lo[d]) ) { *ipos =d; continue; }
	   if( (*ipos >=0) && (hi[d]>lo[d])) { *jpos =d; break; }
    }
    
/*    if(*ipos >*jpos){Integer t=*ipos; *ipos=*jpos; *jpos=t;} 
*/

    /* single element case (trivial) */
    if((*ipos <0) && (*jpos <0)){ *ipos =0; *jpos=1; }
    else{

      /* handle almost trivial case of only one dimension with >1 elements */
      if(*ipos == rank-1) (*ipos)--; /* i cannot be the last dimension */
      if(*ipos <0) *ipos = *jpos-1; /* select i dimension based on j */ 
      if(*jpos <0) *jpos = *ipos+1; /* select j dimenison based on i */

    }

    *ilo = lo[*ipos]; *ihi = hi[*ipos];
    *jlo = lo[*jpos]; *jhi = hi[*jpos];
    *dim1 = dims[*ipos];
    *dim2 = dims[*jpos];
}

#define  SETINT(tmp,val,n) {int _i; for(_i=0;_i<n; _i++)tmp[_i]=val;}

/*\ MATRIX MULTIPLICATION for 2d patches of multi-dimensional arrays 
 *  
 *  C[lo:hi,lo:hi] = alpha*op(A)[lo:hi,lo:hi] * op(B)[lo:hi,lo:hi]        
 *                 + beta *C[lo:hi,lo:hi]
 *
 *  where:
 *          op(A) = A or A' depending on the transpose flag
 *  [lo:hi,lo:hi] - patch indices _after_ op() operator was applied
 *
\*/
void nga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, 
		      Integer *g_a, Integer alo[], Integer ahi[], 
                      Integer *g_b, Integer blo[], Integer bhi[], 
		      Integer *g_c, Integer clo[], Integer chi[])
{
#ifdef STATBUF
   DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
#else
   DoubleComplex *a, *b, *c;
#endif
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer me= ga_nodeid_(), nproc, inode, iproc;
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim, arank, brank, crank;
int aipos, ajpos, bipos, bjpos,cipos, cjpos, need_scaling=1;
Integer Ichunk, Kchunk, Jchunk;
Integer ailo, aihi, ajlo, ajhi;    /* 2d plane of g_a */
Integer bilo, bihi, bjlo, bjhi;    /* 2d plane of g_b */
Integer cilo, cihi, cjlo, cjhi;    /* 2d plane of g_c */
Integer adims[GA_MAX_DIM],bdims[GA_MAX_DIM],cdims[GA_MAX_DIM],tmpld[GA_MAX_DIM];
Integer *tmplo = adims, *tmphi =bdims; 
DoubleComplex ONE, ZERO;
float ONE_F = 1.0, ZERO_F = 0.0;
double ZERO_D = 0.0;
Integer ZERO_I = 0;
Integer get_new_B;
DoublePrecision chunk_cube;
Integer min_tasks = 10, max_chunk;
int local_sync_begin,local_sync_end;
int idim_t, jdim_t, kdim_t, adim_t, bdim_t, cdim_t;

   ONE.real =1.; ZERO.real =0.;
   ONE.imag =0.; ZERO.imag =0.;
   
   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(local_sync_begin)ga_sync_();

   GA_PUSH_NAME("nga_matmul_patch");

   /* Check to make sure all global arrays are of the same type */
   if (!(ga_is_mirrored_(g_a) == ga_is_mirrored_(g_b) &&
        ga_is_mirrored_(g_a) == ga_is_mirrored_(g_c))) {
     ga_error_("Processors do not match for all arrays",ga_nnodes_());
   }
   if (ga_is_mirrored_(g_a)) {
     inode = ga_cluster_nodeid_();
     nproc = ga_cluster_nprocs_(&inode);
     iproc = me - ga_cluster_procid_(&inode, &ZERO_I);
   } else {
     nproc = ga_nnodes_();
     iproc = me;
   }

   nga_inquire_internal_(g_a, &atype, &arank, adims);
   nga_inquire_internal_(g_b, &btype, &brank, bdims);
   nga_inquire_internal_(g_c, &ctype, &crank, cdims);

   if(arank<2)  ga_error("rank of A must be at least 2",arank);
   if(brank<2)  ga_error("rank of B must be at least 2",brank);
   if(crank<2)  ga_error("rank of C must be at least 2",crank);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
   if(atype != C_DCPL && atype != C_DBL && atype != C_FLOAT) 
     ga_error(" type error",atype);
   
   gai_setup_2d_patch(arank, adims, alo, ahi, &ailo, &aihi, &ajlo, &ajhi, 
		                  &adim1, &adim2, &aipos, &ajpos);
   gai_setup_2d_patch(brank, bdims, blo, bhi, &bilo, &bihi, &bjlo, &bjhi, 
		                  &bdim1, &bdim2, &bipos, &bjpos);
   gai_setup_2d_patch(crank, cdims, clo, chi, &cilo, &cihi, &cjlo, &cjhi, 
		                  &cdim1, &cdim2, &cipos, &cjpos);

   /* check if patch indices and dims match */
   if (*transa == 'n' || *transa == 'N'){
      if (ailo <= 0 || aihi > adim1 || ajlo <= 0 || ajhi > adim2)
         ga_error("  g_a indices out of range ", *g_a);
   }else
      if (ailo <= 0 || aihi > adim2 || ajlo <= 0 || ajhi > adim1)
         ga_error("  g_a indices out of range ", *g_a);

   if (*transb == 'n' || *transb == 'N'){
      if (bilo <= 0 || bihi > bdim1 || bjlo <= 0 || bjhi > bdim2)
          ga_error("  g_b indices out of range ", *g_b);
   }else
      if (bilo <= 0 || bihi > bdim2 || bjlo <= 0 || bjhi > bdim1)
          ga_error("  g_b indices out of range ", *g_b);

   if (cilo <= 0 || cihi > cdim1 || cjlo <= 0 || cjhi > cdim2)
       ga_error("  g_c indices out of range ", *g_c);

  /* verify if patch dimensions are consistent */
   m = aihi - ailo +1;
   n = bjhi - bjlo +1;
   k = ajhi - ajlo +1;
   if( (cihi - cilo +1) != m) ga_error(" a & c dims error",m);
   if( (cjhi - cjlo +1) != n) ga_error(" b & c dims error",n);
   if( (bihi - bilo +1) != k) ga_error(" a & b dims error",k);

   
   chunk_cube = (k*(double)(m*n)) / (min_tasks * nproc);
   max_chunk = (Integer)pow(chunk_cube, (DoublePrecision)(1.0/3.0) );
   if (max_chunk < 32) max_chunk = 32;
   
#ifdef STATBUF
   if(atype ==  C_DBL || atype == C_FLOAT){
      Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
   }else{
      Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
   }
#else
   {
     Integer elems, factor = sizeof(DoubleComplex)/GAsizeofM(atype);
     Ichunk = Jchunk = Kchunk = CHUNK_SIZE;
     
     if ( max_chunk > Ichunk) {       
       /*if memory if very limited, performance degrades for large matrices
	 as chunk size is very small, which leads to communication overhead)*/
       Integer avail = ga_memory_avail(atype);
       ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
       if(avail<MINMEM && ga_nodeid_()==0) ga_error("Not enough memory",avail);
       elems = (Integer)(avail*0.9);/* Donot use every last drop */
       
       max_chunk=MIN(max_chunk, (Integer)(sqrt( (double)((elems-EXTRA)/3))));
       Ichunk = MIN(m,max_chunk);
       Jchunk = MIN(n,max_chunk);
       Kchunk = MIN(k,max_chunk);
     }
     else /* "EXTRA" elems for safety - just in case */
       elems = 3*Ichunk*Jchunk + EXTRA*factor;

     a = (DoubleComplex*) ga_malloc(elems, atype, "GA mulmat bufs");     
     b = a + (Ichunk*Kchunk)/factor + 1; 
     c = b + (Kchunk*Jchunk)/factor + 1;
   }
#endif

   if(atype==C_DCPL){if((((DoubleComplex*)beta)->real == 0) &&
	       (((DoubleComplex*)beta)->imag ==0)) need_scaling =0;} 
   else if((atype==C_DBL)){if(*(DoublePrecision *)beta == 0)need_scaling =0;}
   else if( *(float*)beta ==0) need_scaling =0;

   if(need_scaling) nga_scale_patch_(g_c, clo, chi, beta);
   else      nga_fill_patch_(g_c, clo, chi, beta);
  
   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;
       
       for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
	 khi = MIN(k-1, klo+Kchunk-1);        /* loop rows of g_b patch */
	 kdim= khi - klo +1;               

	 get_new_B = TRUE;
	 
	 for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
	   
	   if(ijk%nproc == iproc){
	     ihi = MIN(m-1, ilo+Ichunk-1);
	     idim= cdim = ihi - ilo +1;
	     
	     if(atype == C_FLOAT) 
	       for (i = 0; i < idim*jdim; i++) *(((float*)c)+i)=0;
	     else if(atype ==  C_DBL)
	       for (i = 0; i < idim*jdim; i++) *(((double*)c)+i)=0;
	     else
	       for (i = 0; i < idim*jdim; i++){ c[i].real=0;c[i].imag=0;}
	     
	     if (*transa == 'n' || *transa == 'N'){ 
	       adim = idim;
	       i0= ailo+ilo; i1= ailo+ihi;   
	       j0= ajlo+klo; j1= ajlo+khi;
	     }else{
	       adim = kdim;
	       i0= ajlo+klo; i1= ajlo+khi;   
	       j0= ailo+ilo; j1= ailo+ihi;
	     }

	     /* ga_get_(g_a, &i0, &i1, &j0, &j1, a, &adim); */
	     memcpy(tmplo,alo,arank*sizeof(Integer));
	     memcpy(tmphi,ahi,arank*sizeof(Integer));
	     SETINT(tmpld,1,arank-1);
	     tmplo[aipos]=i0; tmphi[aipos]=i1;
	     tmplo[ajpos]=j0; tmphi[ajpos]=j1;
	     tmpld[aipos]=i1-i0+1;
	     nga_get_(g_a,tmplo,tmphi,a,tmpld);
	     
	     if(get_new_B) {
	       if (*transb == 'n' || *transb == 'N'){ 
		 bdim = kdim;
		 i0= bilo+klo; i1= bilo+khi;   
		 j0= bjlo+jlo; j1= bjlo+jhi;
	       }else{
		 bdim = jdim;
		 i0= bjlo+jlo; i1= bjlo+jhi;   
		 j0= bilo+klo; j1= bilo+khi;
	       }
	       /* ga_get_(g_b, &i0, &i1, &j0, &j1, b, &bdim); */
	       memcpy(tmplo,blo,brank*sizeof(Integer));
	       memcpy(tmphi,bhi,brank*sizeof(Integer));
	       SETINT(tmpld,1,brank-1);
	       tmplo[bipos]=i0; tmphi[bipos]=i1;
	       tmplo[bjpos]=j0; tmphi[bjpos]=j1;
	       tmpld[bipos]=i1-i0+1;
	       nga_get_(g_b,tmplo,tmphi,b,tmpld);
	       get_new_B = FALSE;
	     }

	     idim_t=idim; jdim_t=jdim; kdim_t=kdim;
	     adim_t=adim; bdim_t=bdim; cdim_t=cdim;

#	     if (defined(CRAY) || defined(WIN32)) && !defined(GA_C_CORE)
		  switch(atype) {
		  case C_FLOAT:
		    xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (float *)alpha, (float *)a, &adim_t, (float *)b, 
			     &bdim_t, &ZERO_F,  (float *)c, &cdim_t);
		    break;		    
		  case C_DBL:
                    DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          alpha, (double*)a, &adim, (double*)b, &bdim, &ONE, 
			  (double*)c, &cdim);
		    break;
		  case C_DCPL:
                    ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE,c,&cdim);
		    break;
		  default:
		    ga_error("ga_matmul_patch: wrong data type", atype);
		  }
#            else 
		  switch(atype) {
		  case C_FLOAT:
		    xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (float *)alpha, (float *)a, &adim_t, (float *)b, &bdim_t, 
			     &ZERO_F,  (float *)c, &cdim_t);
		    break;
		  case C_DBL:
#                 ifdef GA_C_CORE
		    
		    xb_dgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     alpha, (double *)a, &adim_t, (double *)b, &bdim_t, 
			     &ZERO_D,  (double *)c, &cdim_t);
#                 else
		    dgemm_(transa, transb, &idim, &jdim, &kdim,
			   alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#                 endif
		    break;
		  case C_DCPL:
#                 ifdef GA_C_CORE
		    xb_zgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (DoubleComplex *)alpha, a, &adim_t, b, &bdim_t, 
			     &ZERO,  c, &cdim_t);
#                 else
		    zgemm_(transa, transb, &idim, &jdim, &kdim,
			   (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE, c, 
			   &cdim, 1, 1);
#                 endif
		    break;
		  default:
		    ga_error("ga_matmul_patch: wrong data type", atype);
		  }
#            endif

                  i0= cilo+ilo; i1= cilo+ihi;   j0= cjlo+jlo; j1= cjlo+jhi;
                  /* ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
                                            &cdim, (DoublePrecision*)&ONE); */
		  memcpy(tmplo,clo,crank*sizeof(Integer));
		  memcpy(tmphi,chi,crank*sizeof(Integer));
		  SETINT(tmpld,1,crank-1);
		  tmplo[cipos]=i0; tmphi[cipos]=i1;
		  tmplo[cjpos]=j0; tmphi[cjpos]=j1;
		  tmpld[cipos]=i1-i0+1;
		  if(atype == C_FLOAT) 
		    nga_acc_(g_c,tmplo,tmphi,(float *)c,tmpld, &ONE_F);
		  else
		    nga_acc_(g_c,tmplo,tmphi,c,tmpld,(DoublePrecision*)&ONE);
               }
	   ++ijk;
	 }
       }
   }

#ifndef STATBUF
   ga_free(a);
#endif
   
   GA_POP_NAME;
   if(local_sync_end)ga_sync_(); 
}


/*\ MATRIX MULTIPLICATION for patches 
 *  Fortran interface
\*/
void FATR nga_matmul_patch_(transa, transb, alpha, beta, g_a, alo, ahi, 
                       g_b, blo, bhi, g_c, clo, chi)

                      void *alpha, *beta;
		      Integer *g_a, alo[], ahi[]; 
                      Integer *g_b, blo[], bhi[]; 
		      Integer *g_c, clo[], chi[];

#if defined(CRAY) || defined(WIN32)
     _fcd   transa, transb;
{    
     nga_matmul_patch(_fcdtocp(transa), _fcdtocp(transb), alpha, beta, g_a, alo,
                      ahi, g_b, blo, bhi, g_c, clo, chi);
#else
     char    *transa, *transb;
{    
	nga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi,
                         g_b, blo, bhi, g_c, clo, chi);
#endif
}

void FATR ga_matmul_patch_(transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;

#if defined(CRAY) || defined(WIN32)
     _fcd   transa, transb;
{    ga_matmul_patch(_fcdtocp(transa), _fcdtocp(transb), alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi);}
#else
     char    *transa, *transb;
{    
#if 0
Integer alo[2], ahi[2]; 
Integer blo[2], bhi[2];
Integer clo[2], chi[2];
        alo[0]=*ailo; ahi[0]=*aihi; alo[1]=*ajlo; ahi[1]=*ajhi;
        blo[0]=*bilo; bhi[0]=*bihi; blo[1]=*bjlo; bhi[1]=*bjhi;
        clo[0]=*cilo; chi[0]=*cihi; clo[1]=*cjlo; chi[1]=*cjhi;
	nga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi,
                         g_b, blo, bhi, g_c, clo, chi);
#else
	ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi);
#endif
}
#endif


/*********************** Fortran warppers for ga_Xgemm ***********************/


#ifdef USE_SUMMA
void ga_dgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
               double *alpha, Integer *g_a, Integer *g_b,
               double *beta, Integer *g_c) {
  /**
   * ga_summa calls ga_ga_dgemm to handle cases it does not cover
   */
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  ga_summa_(transa, transb, m, n, k, alpha, g_a, g_b, beta, g_c);
}
#  define GA_DGEMM ga_ga_dgemm_
#else
#  define GA_DGEMM ga_dgemm_
#endif


#define  SET_GEMM_INDICES\
  Integer ailo = 1;\
  Integer aihi = *m;\
  Integer ajlo = 1;\
  Integer ajhi = *k;\
\
  Integer bilo = 1;\
  Integer bihi = *k;\
  Integer bjlo = 1;\
  Integer bjhi = *n;\
\
  Integer cilo = 1;\
  Integer cihi = *m;\
  Integer cjlo = 1;\
  Integer cjhi = *n

#if defined(CRAY) || defined(WIN32)
void FATR GA_DGEMM(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR GA_DGEMM(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif
 
  ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, &ailo, &aihi, &ajlo, &ajhi,
                      g_b, &bilo, &bihi, &bjlo, &bjhi,
                      g_c, &cilo, &cihi, &cjlo, &cjhi);
}

#if defined(CRAY) || defined(WIN32)
void FATR ga_sgemm_(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR ga_sgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif


  ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, &ailo, &aihi, &ajlo, &ajhi,
                      g_b, &bilo, &bihi, &bjlo, &bjhi,
                      g_c, &cilo, &cihi, &cjlo, &cjhi);
}


#if defined(CRAY) || defined(WIN32)
void FATR ga_zgemm_(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR ga_zgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif


  ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, &ailo, &aihi, &ajlo, &ajhi,
                      g_b, &bilo, &bihi, &bjlo, &bjhi,
                      g_c, &cilo, &cihi, &cjlo, &cjhi);
}

