      subroutine ga_dgemm(transa, transb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)
      implicit none
      Character*1        transa, transb
      Integer            m, n, k
      Double precision   alpha, beta
      Integer            g_a, g_b, g_c
#include "mafdecls.h"
#include "global.fh"
c     
c     GA_DGEMM  performs one of the matrix-matrix operations:
c           C := alpha*op( A )*op( B ) + beta*C,
c     where  op( X ) is one of
c           op( X ) = X   or   op( X ) = X',
c     
c     alpha and beta are scalars, and A, B and C are matrices, with op( A )
c     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
c     
c     On entry, TRANSA specifies the form of op( A ) to be used in
c     the matrix multiplication as follows:
c           transa = 'N' or 'n',  op( A ) = A.
c           transa = 'T' or 't',  op( A ) = A'.
c     
c     M      - On entry,  M  specifies  the number  of rows  of the  matrix
c              op( A )  and of the  matrix  C.  M  must  be at least  zero.
c     N      - On entry,  N  specifies the number  of columns of the matrix
c              op( B ) and the number of columns of the matrix C. N must be
c              at least zero.
c     K      - On entry,  K  specifies  the number of columns of the matrix
c              op( A ) and the number of rows of the matrix op( B ). K must
c              be at least  zero.
c
      integer ilo, ihi, jlo, jhi, klo, khi, ichunk, jchunk, kchunk
      integer idim, jdim, kdim, adim, bdim, cdim, ijk, me, nproc
#ifdef DATA_SERVER
      parameter (ichunk=500, jchunk=500, kchunk=500)
#else
      parameter (ichunk=128, jchunk=128, kchunk=128)
#endif
      double precision c(ichunk*jchunk), a(ichunk*kchunk),
     $     b(kchunk*jchunk)
c
      me = ga_nodeid()
      nproc = ga_nnodes()
c
      if (beta .eq. 0.0d0) then
         call ga_zero(g_c)
      else if (beta .ne. 1.0d0) then
         call ga_dscal(g_c, beta)
      endif
c
      ijk = 0
      do jlo = 1, n, jchunk
         jhi = min(n, jlo+jchunk-1)
         jdim = jhi - jlo + 1
         do ilo = 1, m, ichunk
            ihi = min(m, ilo+ichunk-1)
            idim = ihi - ilo + 1
            cdim = idim
            do klo = 1, k, kchunk
               if (mod(ijk,nproc) .eq. me) then
                  call dfill(idim*jdim, 0.0d0, c, 1)
                  khi = min(k, klo+kchunk-1)
                  kdim = khi - klo + 1
                  if (transa.eq.'n' .or. transa.eq.'N') then
                     adim = idim
                     call ga_get(g_a, ilo, ihi, klo, khi, a, adim)
                  else
                     adim = kdim
                     call ga_get(g_a, klo, khi, ilo, ihi, a, adim)
                  endif
                  if (transb.eq.'n' .or. transb.eq.'N') then
                     bdim = kdim
                     call ga_get(g_b, klo, khi, jlo, jhi, b, bdim)
                  else
                     bdim = jdim
                     call ga_get(g_b, jlo, jhi, klo, khi, b, bdim)
                  endif
                  call dgemm(transa, transb, idim, jdim, kdim,
     $                 alpha, a, adim, b, bdim, 1.0d0, c, cdim)
                  call ga_acc(g_c, ilo, ihi, jlo, jhi, c, cdim, 1.0d0)
               endif
               ijk = ijk + 1
            enddo
         enddo
      enddo
      call ga_sync()
c
      end
