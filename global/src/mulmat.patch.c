/*$Id: mulmat.patch.c,v 1.1 1999-11-18 21:36:54 d3h325 Exp $*/
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
#endif

#ifdef WIN32
   extern void FATR DGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
                     void*, void*, Integer*, void*, Integer*, void*,
                     void*, Integer*);
   extern void FATR ZGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
                     DoubleComplex*, DoubleComplex*, Integer*, DoubleComplex*, Integer*,
                     DoubleComplex*, DoubleComplex*, Integer*);
#endif

#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif



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
     DoublePrecision    *alpha, *beta;
     char    *transa, *transb;
{
#ifdef STATBUF
  /* approx. sqrt(2) ratio in chunk size to use the same buffer space */
#  define C_CHUNK  92 
#  define D_CHUNK  64
#  define ICHUNK C_CHUNK
#  define JCHUNK C_CHUNK
#  define KCHUNK C_CHUNK
   DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
#else
   /* min acceptable and max amount of memory (in elements) */
#  define MINMEM 400
#  define MAXMEM 100000 
   DoubleComplex *a, *b, *c;
   Integer handle, idx;
#endif
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer me= ga_nodeid_(), nproc=ga_nnodes_();
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim;
Integer Ichunk, Kchunk, Jchunk;
DoubleComplex ONE;

   ONE.real =1.;
   ONE.imag =0.;

   ga_sync_();
   GA_PUSH_NAME("ga_matmul_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);
   ga_inquire_(g_c, &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
   if(atype != MT_F_DCPL && atype != MT_F_DBL) ga_error(" type error",atype);

#ifdef STATBUF
   if(atype ==  MT_F_DBL){
      Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
   }else{
      Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
   }
#else
   {
            Integer avail = MA_inquire_avail(atype),elems, used ;
            ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
            if(avail < MINMEM && ga_nodeid_() == 0)
              ga_error("Not enough memory for buffers",avail);
            elems = MIN((Integer)(avail*0.9), MAXMEM);
            if(MA_push_get(atype, elems, "GA mulmat bufs", &handle, &idx))
                MA_get_pointer(handle, &a);
            else
                ga_error("ma_alloc_get failed",avail);
            Ichunk = Kchunk = Jchunk = (Integer) sqrt((double)(elems-2)/3.0);
            used = Ichunk * Kchunk;
            if(atype ==  MT_F_DBL) used = 1+used/2; 
            b = a+ used;
            used = Kchunk*Jchunk;
            if(atype ==  MT_F_DBL) used = 1+used/2; 
            c = b+ used;
   }
#endif

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

   if(*beta) ga_scale_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
   else      ga_fill_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
  
   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;
       for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
           ihi = MIN(m-1, ilo+Ichunk-1);
           idim= cdim = ihi - ilo +1;
           for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
                                                    /* loop rows of g_b patch */
               if(ijk%nproc == me){
                  if(atype ==  MT_F_DBL)
                     for (i = 0; i < idim*jdim; i++) *(((double*)c)+i)=0;
                  else
                     for (i = 0; i < idim*jdim; i++){ c[i].real=0;c[i].imag=0;}
                  khi = MIN(k-1, klo+Kchunk-1);
                  kdim= khi - klo +1;
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
   if(atype ==  MT_F_DBL){
#                 if defined(CRAY) || defined(WIN32)
                    DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          alpha, (double*)a, &adim, (double*)b, &bdim, &ONE, (double*)c, &cdim);
#                 else
                    dgemm_(transa, transb, &idim, &jdim, &kdim,
                           alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#                 endif
   }else{
#                 if defined(CRAY) || defined(WIN32)
                    ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          alpha, a, &adim, b, &bdim, &ONE, c, &cdim);
#                 else
                    zgemm_(transa, transb, &idim, &jdim, &kdim,
                           alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#                 endif
   }
                  i0= *cilo+ilo; i1= *cilo+ihi;   j0= *cjlo+jlo; j1= *cjlo+jhi;
                  ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
                                            &cdim, (DoublePrecision*)&ONE);
               }
               ijk++;
          }
      }
   }

#ifndef STATBUF
   if(!MA_pop_stack(handle)) ga_error("MA_pop_stack failed",0);
#endif
 
   GA_POP_NAME;
   ga_sync_();
}


/*\ MATRIX MULTIPLICATION for patches 
 *  Fortran interface
\*/
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
{    ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi);}
#endif

