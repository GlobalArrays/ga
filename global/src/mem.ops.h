/***************************************************************************\
* Memory copy operations for local, shared and globally adressable memory   * 
* plus accumulate operation                                                 *
*                                                                           *
* Jarek Nieplocha, 08.15.95                                                 *
\***************************************************************************/

Integer ONE=1;
DoublePrecision DPzero=0.;

/***************************** 1-Dimensional copy ************************/

#ifdef CRAY_T3D
       extern Integer GAme;
#      define Copy(src,dst,n)          memcpy((dst), (src), (n))
#      define CopyElemTo(src,dst,n,proc){   \
                  if(proc==GAme)memwcpy((dst), (src), (n));\
                  else shmem_put((long*) (dst), (long*)(src), (n), (proc));\
              }
#      define CopyElemFrom(src,dst,n,proc){ \
                  if(proc==GAme)memwcpy((dst), (src), (n));\
                  else shmem_get((long*) (dst), (long*)(src), (n), (proc));\
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
#endif

/***************************** 2-Dimensional copy ************************/

#ifdef CRAY_T3D
#  define dcopy2d_ DCOPY2D
#  define icopy2d_ ICOPY2D
#  define accumulatef_ ACCUMULATEF
#  define XX_DAXPY SAXPY
#elif defined(KSR)
#  define XX_DAXPY saxpy_
#else
#  define XX_DAXPY daxpy_
#endif

#  define THRESH   32


void dcopy2d_(), icopy2d_(), accumulatef_();

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
#elif defined(PARAGON____)
    /* call vectorized version if more than THRESH rows */
#   define Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    if((*rows < THRESH) || in_handler){\
      if(type==MT_F_DBL)dcopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      else icopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
    }else {\
      if(type==MT_F_DBL)dcopy2d_v_(rows, cols, ptr_src, ld_src,ptr_dst,ld_dst);\
      else icopy2d_v_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
    }\
    }

#elif defined(PARAGON)
    /* call vectorized version if more than THRESH rows */
void Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)
Integer type, *rows, *cols, *ld_src, *ld_dst;
char *ptr_src, *ptr_dst;
{
    if(*rows<THRESH){
      if(type==MT_F_DBL)dcopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
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
           if(type==MT_F_DBL)\
             dcopy2d_v_(rows, cols, ptr_src, ld_src,ptr_dst,ld_dst);\
           else icopy2d_v_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
        }\
    }\
}



#else
    /* fortran array version faster */
#   define Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
      if(type==MT_F_DBL)dcopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
      else icopy2d_(rows, cols, ptr_src, ld_src, ptr_dst,ld_dst);\
    }
#endif



/***************** 2D copy between local and shared/global memory ***********/
#if defined(CRAY_T3D) || defined(KSR)
    /* special copy routines for moving words */
#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    if(sizeof(Integer) != sizeof(DoublePrecision))\
              ga_error("Copy broken", sizeof(Integer));\
    char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
          CopyElemTo(ps, pd, *rows, proc);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst){\
    Integer item_size=GAsizeofM(type), j;\
    Integer nbytes = item_size* *rows;\
    if(sizeof(Integer) != sizeof(DoublePrecision))\
              ga_error("Copy broken", sizeof(Integer));\
    char *ps=ptr_src, *pd=ptr_dst;\
      for (j = 0;  j < *cols;  j++){\
          CopyElemFrom(ps, pd, *rows, proc);\
          ps += item_size* *ld_src;\
          pd += item_size* *ld_dst;\
      }\
    }


#else

#   define Copy2DTo(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)\
           Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)

#   define Copy2DFrom(type, proc, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)\
           Copy2D(type, rows, cols, ptr_src, ld_src, ptr_dst,ld_dst)
#endif



/**************************** accumulate operation **************************/
void acc_column(alpha, a, b,n)
Integer n;
DoublePrecision alpha, *a, *b;
{
  int i;
  if(n< THRESH) for (i=0;i<n;i++) a[i] += alpha* b[i];
  else XX_DAXPY(&(n), &alpha, b, &ONE, a, &ONE);
}


#ifdef KSR
#  define accumulate(alpha, rows, cols, A, ald, B, bld)\
   {\
   Integer c;\
      void Accum(DoublePrecision, DoublePrecision*, DoublePrecision*, Integer);\
      /* A and B are Fortran arrays! */\
      for(c=0;c<(cols);c++)\
           Accum((alpha), (B) + c*(bld), (A) + c*(ald), (rows));\
   }
#elif defined(CRAY_T3D) || defined(SGI)
#  define accumulate(alpha, rows, cols, A, ald, B, bld) {\
   register Integer c,r;\
   if(rows< THRESH)\
      for(c=0;c<(cols);c++)\
           for(r=0;r<(rows);r++)\
                *((A) +c*(ald) + r) += (alpha) * *((B) + c*(bld) +r);\
    else for(c=0;c<(cols);c++)\
           XX_DAXPY(&(rows), &(alpha), (B)+c*(bld), &ONE, (A)+c*(ald), &ONE);\
   }

#elif defined(PARAGON)
   /* call vectorized version if more than THRESH rows */
/*      if((rows< THRESH) || in_handler)accumulatef_(&alpha, &r, &c, A, &a_ld, B, &b_ld);\*/
#  define accumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      if(in_handler)accumulatef_(&alpha, &r, &c, A, &a_ld, B, &b_ld);\
      else accumulatef_v_(&alpha, &r, &c, A, &a_ld, B, &b_ld);\
   }

#elif defined(C_ACC)
#  define accumulate(alpha, rows, cols, A, ald, B, bld)\
   {\
   register Integer c,r;\
      /* A and B are Fortran arrays! */\
      for(c=0;c<(cols);c++)\
           for(r=0;r<(rows);r++)\
                *((A) +c*(ald) + r) += (alpha) * *((B) + c*(bld) +r);\
   }
#else
#  define accumulate(alpha, rows, cols, A, ald, B, bld){\
      Integer r=rows, c=cols, a_ld=ald, b_ld=bld;\
      accumulatef_(&alpha, &r, &c, A, &a_ld, B, &b_ld);\
   }
#endif

