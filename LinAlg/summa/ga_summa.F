#define GA_GA_DGEMM ga_dgemm

      subroutine ga_summa(transa, transb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)

      implicit none

#include "global.fh"

      Character*1        transa, transb
      Integer            m, n, k
      Double precision   alpha, beta
      Integer            g_a, g_b, g_c

c     
c     GA_SUMMA  performs one of the matrix-matrix operations:
c     
c     C := alpha*op( A )*op( B ) + beta*C,
c     
c     where  op( X ) is one of
c     
c     op( X ) = X   or   op( X ) = X`,
c     
c     alpha and beta are scalars, and A, B and C are matrices, with op( A )
c     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
c     
c     On entry, TRANSA specifies the form of op( A ) to be used in
c     the matrix multiplication as follows:
c     transa = 'N' or 'n',  op( A ) = A.
c     transa = 'T' or 't',  op( A ) = A`.
c     
c     M      - On entry,  M  specifies  the number  of rows  of the  matrix
c     op( A )  and of the  matrix  C.  M  must  be at least  zero.
c     N      - On entry,  N  specifies the number  of columns of the matrix
c     op( B ) and the number of columns of the matrix C. N must be
c     at least zero.
c     K      - On entry,  K  specifies  the number of columns of the matrix
c     op( A ) and the number of rows of the matrix op( B ). K must
c     be at least  zero.
c     
      external ga_ga_dgemm

      integer itransa, itransb, istat

      integer ct, c1, c2

      integer  ga_summa_c, ga_summa_cc
      external ga_summa_c, ga_summa_cc


      if( transa .eq. 't'  .or.  transa .eq. 'T' ) then
         itransa = 1
      else
         itransa = 0
      endif

      if( transb .eq. 't'  .or.  transb .eq. 'T' ) then
         itransb = 1
      else
         itransb = 0
      endif

*     trying "direct" version of ga_summa
*     this version does not accept n != dim2(c)

      istat = 1
      call ga_inquire(g_c, ct, c1, c2)
      if (n .ne. c2) goto 1000

      istat = ga_summa_c( itransa, itransb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)

*     try "copying" version of ga_summa if ga arrays were
*     wrong shape for "direct" version.
*     
*     this version gives wrong numbers for n != dim2(c)
*     never use it
*     
*     if( istat .eq. 1 )  then
*     istat = ga_summa_cc( itransa, itransb, m, n, k, alpha, g_a, 
*     $                       g_b, beta, g_c)
*     endif

*     try ga_ga_dgemm if could not use a ga_summa routine.

 1000 if( istat .ne. 0 ) then
         if( ga_nodeid() .eq. 0 )
     $        write( *, *)
     $        ' calling ga_ga_dgemm, ga_summa_c(c) returned ', istat
         call GA_GA_DGEMM( transa, transb, m, n, k, alpha,
     $        g_a, g_b, beta, g_c)
      endif
c     
      end

