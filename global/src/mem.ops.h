/***************************************************************************\
* Memory copy operations for local, shared and globally adressable memory   * 
* plus accumulate operation                                                 *
*                                                                           *
* Jarek Nieplocha, 08.15.95                                                 *
\***************************************************************************/

static Integer ONE=1;
static DoublePrecision DPzero=0.;

/***************************** 1-Dimensional copy ************************/

#ifdef CRAY_T3D
       extern Integer GAme;
#      define Copy(src,dst,n)          memcpy((dst), (src), (n))
#      define CopyElemTo(src,dst,n,proc){   \
                  if(proc==GAme)memwcpy((long*)(dst), (long*)(src), (int)(n));\
                  else shmem_put((long*) (dst), (long*)(src), (int)(n),(proc));\
              }
#      define CopyElemFrom(src,dst,n,proc){ \
                  if(proc==GAme)memwcpy((long*)(dst), (long*)(src), (int)(n));\
                  else shmem_get((long*) (dst), (long*)(src), (int)(n),(proc));\
              }

#elif  defined(FUJITSU)
#      include "../../config/fujitsu-vpp.h"
#      define Copy(src,dst,n)          _MmCopy((dst), (src), (n))

#elif  defined(LAPI)
#      include <lapi.h>
       extern Integer GAme;
       extern lapi_handle_t lapi_handle;
#      define Copy(src,dst,n)          memcpy((dst), (src), (n))
#      define CopyElemTo(src,dst,n,proc){   \
                  if(proc==GAme)memcpy((long*)(dst), (long*)(src), \
                                       (int)((n)*sizeof(Integer)));\
                  else if(LAPI_Put(lapi_handle, (uint)proc, (uint) n*sizeof(Integer), (void*) (dst), (void*)(src), NULL, &ack_cntr.cntr, &cmpl_arr[proc].cntr))\
                  ga_error("LAPI_put failed",0);\
              }

       /**** this copy is nonblocking and requires fence to complete!!! ****/
#      define CopyElemFrom(src,dst,n,proc){ \
                  if(proc==GAme)memcpy((long*)(dst), (long*)(src), \
                                (int)((n)*sizeof(Integer)));\
                  else {\
                    if(LAPI_Get(lapi_handle, (uint)proc, \
                        (uint) n*sizeof(Integer), (void*) (src), (void*)(dst),\
                         NULL, NULL))\
                    ga_error("LAPI_Get failed",0);\
                  }\
              }
#elif  defined(KSR)
#      define Copy(src,dst,n)      memcpy((char*)(dst),(char*)(src),(n))
       void   CopyTo(char*, char*, Integer);
       void   CopyFrom(char*, char*, Integer);
#      define CopyElemFrom(src, dst, n,proc ) CopyFrom((src),(dst),8*(n));
#      define CopyElemTo(src, dst, n, proc) CopyTo((src),(dst),8*(n));
#else
#      define Copy(src,dst,n)           memcpy((dst),(src),(n))
#      define CopyTo(src,dst,n)         Copy((src),(dst),(n))
#      define CopyFrom(src,dst,n)       Copy((src),(dst),(n))
#      define CopyElemFrom(src, dst, n,proc ) CopyFrom((src),(dst),sizeof(Integer)*(n));
#      define CopyElemTo(src, dst, n, proc) CopyTo((src),(dst),sizeof(Integer)*(n));
#endif

/***************************** 2-Dimensional copy ************************/

#ifdef CRAY
#  define dcopy2d_ DCOPY2D
#  define icopy2d_ ICOPY2D
#  define d_accumulate_ D_ACCUMULATE
#  define z_accumulate_ Z_ACCUMULATE
#  define i_accumulate_ I_ACCUMULATE
#  define XX_DAXPY SAXPY
#  define XX_ZAXPY CAXPY
#  define XX_ICOPY SCOPY
#  define XX_DCOPY SCOPY
#elif defined(KSR)
#  define XX_DAXPY saxpy_
#  define XX_ZAXPY caxpy_
#  define XX_ICOPY scopy_
#  define XX_DCOPY scopy_
#else
#  define XX_DAXPY daxpy_
#  define XX_ZAXPY zaxpy_
#  define XX_ICOPY scopy_
#  define XX_DCOPY dcopy_
#endif

#  define THRESH   32


void dcopy2d_(), icopy2d_(), d_accumulate(), z_accumulate(), i_accumulate(); 

/******************** 2D copy from local to local memory ******************/
#if defined(SGI64)||defined(DECOSF)||defined(SGI)
    /* C pointer version faster */
#   define Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    Integer nbytes = item_size* *rows;\
    char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
          Copy(ps, pd, nbytes);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }

#elif defined(SP1__)
    /* call BLAS version if more than THRESH rows */
#   define Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    Integer nbytes = item_size* *rows;\
    char *ps=ptr_src, *pd=ptr_dst;\
    if((*rows < THRESH) || type==MT_F_INT)\
      for (j = 0;  j < *cols;  j++){\
          Copy(ps, pd, nbytes);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    else\
      for (j = 0;  j < *cols;  j++){\
          XX_DCOPY(rows, ps, &ONE, pd,  &ONE);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }


#elif defined(PARAGON)
     /* call vectorized version if more than THRESH rows */
static void Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)
Integer type, *rows, *cols, *ld_src, *ld_dst;
char *ptr_src, *ptr_dst;
{
Integer rrows, ldd, lds;
    if(type!=MT_F_DCPL){
       rrows = *rows;
       lds =   *ld_src;
       ldd =   *ld_dst;
    }else{
       rrows = 2* *rows;
       lds = 2* *ld_src;
       ldd = 2* *ld_dst;
    }
    if(rrows<THRESH){
      if(type!=MT_F_INT)dcopy2d_(&rrows, cols, ptr_src, &lds,ptr_dst,&ldd);\
      else icopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
    }else {
         if(in_handler){\
           Integer item_size=GAsizeofM(type), j;\
           Integer nbytes = item_size* *rows;\
           char *ps=ptr_src, *pd=ptr_dst;\
           for (j = 0;  j < *cols;  j++){\
               bcopy_i(ps, pd, nbytes);\
               ps += item_size* *ld_src;\
               pd += item_size* *ld_dst;\
           }\
        }else{
           if(type!=MT_F_INT)\
             dcopy2d_v_(&rrows, cols, ptr_src, &lds,ptr_dst,&ldd);\
           else icopy2d_v_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
        }\
    }\
}



#else
    /* fortran array version faster */
#   define Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
      Integer rrows, ldd, lds;\
      if(type!=MT_F_DCPL){\
             rrows = *rows;\
             lds =   *ld_src;\
             ldd =   *ld_dst;\
      }else{\
             rrows = 2* *rows;\
             lds = 2* *ld_src;\
             ldd = 2* *ld_dst;\
      }\
      if(type!=MT_F_INT)dcopy2d_(&rrows, cols, ptr_src, &lds,ptr_dst,&ldd);\
      else icopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
    }
#endif



/***************** 2D copy between local and shared/global memory ***********/
#if defined(CRAY_T3D) || defined(KSR) || defined(FUJITSU___)
    /* special copy routines for moving words */
#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    Integer words =  *rows * item_size/sizeof(Integer); \
    char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
          CopyElemTo(ps, pd, words, proc);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    Integer words =  *rows * item_size/sizeof(Integer); \
    char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
          CopyElemFrom(ps, pd, words, proc);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }

#elif defined(FUJITSU)
#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    if(proc==GAme){\
      Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      }\
    else {\
      int item_size=GAsizeofM(type);\
      int bytes =  *rows * item_size; \
      CopyPatchTo((ptr_src),(*ld_src *item_size),(ptr_dst),(*ld_dst *item_size),(*cols),bytes,(proc));\
    }\
  }

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    if(proc==GAme){\
      Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      }\
    else {\
      int item_size = GAsizeofM(type);\
      int bytes = item_size* *rows;\
      CopyPatchFrom((ptr_src),(*ld_src *item_size),(ptr_dst),(*ld_dst *item_size),(*cols),bytes,(proc));\
    }\
  }



#elif defined(LAPI)
#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    if(proc==GAme){\
      Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      }\
    else {\
      uint item_size=GAsizeofM(type), j;\
      uint bytes =  *rows * item_size; \
      char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
         if(LAPI_Put(lapi_handle, (uint)proc, bytes, pd, ps, NULL,\
                     &ack_cntr.cntr, &cmpl_arr[proc].cntr))\
                     ga_error("LAPI_put (2D) failed when puting to",proc);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }\
  }

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    if(proc==GAme){\
      Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      }\
    else {\
         uint item_size=GAsizeofM(type), j;\
         uint bytes = item_size* *rows;\
         char *ps = ptr_src, *pd=ptr_dst;\
         static int val;\
         SET_COUNTER(get_cntr, *cols); \
         for (j = 0;  j < *cols;  j++){\
              if(LAPI_Get(lapi_handle, (uint)proc, \
                 bytes, ps, pd, NULL, &get_cntr.cntr))\
                      ga_error("LAPI_get failed when geting from",proc);\
              ps += item_size* *ld_src; \
              pd += item_size* *ld_dst;\
         }\
      }\
    }


#else

#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)\
           Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)\
           Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)
#endif



/**************************** accumulate operation **************************/
#if defined(CRAY) || defined(FUJITSU)
static void dacc_column(alpha, a, b,n)
Integer n;
DoublePrecision *alpha, *a, *b;
{
  int i;
  if(n< THRESH) for (i=0;i<n;i++) a[i] += *alpha * b[i];
  else XX_DAXPY(&(n), alpha, b, &ONE, a, &ONE);
}


static void zacc_column(alpha, a, b,n)
Integer n;
DoubleComplex *alpha, *a, *b;
{
  int i;
  if(n<-1) for (i=0;i<n;i++){ 
       a[i].real  += alpha->real * b[i].real - alpha->imag * b[i].imag;
       a[i].imag  += alpha->imag * b[i].real + alpha->real * b[i].imag;}
  else XX_ZAXPY(&(n), alpha, b, &ONE, a, &ONE);
}

static void iacc_column(alpha, a, b,n)
Integer n,  *alpha, *a, *b;
{
  int i;
  for (i=0;i<n;i++) a[i] += *alpha * b[i];
}

#endif

#ifdef KSR
#  define accumulate(alpha, rows, cols, A, ald, B, bld)\
   {\
   Integer c;\
      void Accum(DoublePrecision, DoublePrecision*, DoublePrecision*, Integer);\
      /* A and B are Fortran arrays! */\
      for(c=0;c<(cols);c++)\
          Accum(*(DoublePrecision*)(alpha), (B)+c*(bld), (A)+c*(ald), (rows));\
   }
#elif defined(CRAY_T3D) || defined(FUJITSU)
#  define accumulate(alpha, rows, cols, A, ald, B, bld) {\
   register Integer c,r;\
   DoublePrecision *AA = (DoublePrecision*)(A), *BB= (DoublePrecision*)(B);\
   DoublePrecision Alpha=*(DoublePrecision*)alpha;\
   if(rows< THRESH)\
      for(c=0;c<(cols);c++)\
         for(r=0;r<(rows);r++)\
           *((AA)+c*(ald)+ r) += Alpha * *((BB)+c*(bld)+r);\
    else for(c=0;c<(cols);c++)\
           XX_DAXPY(&(rows), alpha, (B)+c*(bld), &ONE, (A)+c*(ald), &ONE);\
   }

#elif defined(PARAGON)
   /* call vectorized version if more than THRESH rows */
/*      if((rows< THRESH) || in_handler)accumulatef_(&alpha, &r, &c, A, &a_ld, B, &b_ld);\*/
#  define accumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      if(in_handler)d_accumulate_(alpha, &r, &c, A, &a_ld, B, &b_ld);\
      else accumulatef_v_(alpha, &r, &c, A, &a_ld, B, &b_ld);\
   }

#elif defined(C_ACC)
#  define accumulate(alpha, rows, cols, A, ald, B, bld)\
   {\
   register Integer c,r;\
      /* A and B are Fortran arrays! */\
      for(c=0;c<(cols);c++)\
           for(r=0;r<(rows);r++)\
                *((A) +c*(ald) + r) += *alpha * *((B) + c*(bld) +r);\
   }
#else
#  define accumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      d_accumulate_(alpha, &r, &c, A, &a_ld, B, &b_ld);\
   }
#endif


/******************* complex accumulate operation **************************/

#  define zaccumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      z_accumulate_(alpha, &r, &c, A, &a_ld, B, &b_ld);\
}
/******************* integer accumulate operation **************************/

#  define iaccumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      i_accumulate_(alpha, &r, &c, A, &a_ld, B, &b_ld);\
}

