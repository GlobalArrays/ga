c     ijk loops rearanged into ikj
c     Thu Aug 25 14:08:17 PDT 1994, JN
c
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
c           op( X ) = X   or   op( X ) = X`,
c     
c     alpha and beta are scalars, and A, B and C are matrices, with op( A )
c     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
c     
c     On entry, TRANSA specifies the form of op( A ) to be used in
c     the matrix multiplication as follows:
c           transa = 'N' or 'n',  op( A ) = A.
c           transa = 'T' or 't',  op( A ) = A`.
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
c     parameter (ichunk=64, jchunk=64, kchunk=64)
      parameter (ichunk=128, jchunk=128, kchunk=128)
#endif
*
* rather than put these on the system stack put them on the MA stack
*
*      double precision c(ichunk*jchunk), a(ichunk*kchunk),
*     $     b(kchunk*jchunk)
      integer l_a, k_a, l_b, k_b, l_c, k_c
      logical status
c
      status = .true.
      status = ma_push_get(MT_DBL, ichunk*kchunk, 'ga_dgemm:a', l_a,k_a)
     $     .and. status
      status = ma_push_get(MT_DBL, kchunk*jchunk, 'ga_dgemm:b', l_b,k_b)
     $     .and. status
      status = ma_push_get(MT_DBL, ichunk*jchunk, 'ga_dgemm:c', l_c,k_c)
     $     .and. status
      if (.not. status) call ga_error('ga_dgemm: insufficent memory?',0)
c
      me = ga_nodeid()
      nproc = ga_nnodes()
c
      if (beta .eq. 0.0d0) then
         call ga_zero(g_c)
      else
         call ga_dscal(g_c, beta)
      endif
c
      ijk = 0
      do jlo = 1, n, jchunk
         jhi = min(n, jlo+jchunk-1)
         jdim = jhi - jlo + 1
         do klo = 1, k, kchunk
            khi = min(k, klo+kchunk-1)
            kdim = khi - klo + 1
            do ilo = 1, m, ichunk
               if (mod(ijk,nproc) .eq. me) then
                  ihi = min(m, ilo+ichunk-1)
                  idim = ihi - ilo + 1
                  cdim = idim
                  call dfill(idim*jdim, 0.0d0, dbl_mb(k_c), 1)
                  if (transa.eq.'n' .or. transa.eq.'N') then
                     adim = idim
                     call ga_get(g_a, ilo, ihi, klo, khi,
     $                    dbl_mb(k_a), adim)
                  else
                     adim = kdim
                     call ga_get(g_a, klo, khi, ilo, ihi, 
     $                    dbl_mb(k_a), adim)
                  endif
                  if (transb.eq.'n' .or. transb.eq.'N') then
                     bdim = kdim
                     call ga_get(g_b, klo, khi, jlo, jhi, 
     $                    dbl_mb(k_b), bdim)
                  else
                     bdim = jdim
                     call ga_get(g_b, jlo, jhi, klo, khi, 
     $                    dbl_mb(k_b), bdim)
                  endif
                  call dgemm(transa, transb, idim, jdim, kdim,
     $                 alpha, dbl_mb(k_a), adim, dbl_mb(k_b), bdim,
     $                 1.0d0, dbl_mb(k_c), cdim)
                  call ga_acc(g_c, ilo, ihi, jlo, jhi, dbl_mb(k_c), 
     $                 cdim, 1.0d0)
               endif
               ijk = ijk + 1
            enddo
         enddo
      enddo
      status = .true.
      status = status .and. ma_pop_stack(l_c)
      status = status .and. ma_pop_stack(l_b)
      status = status .and. ma_pop_stack(l_a)
      if (.not. status) call errquit('ga_dgemm: pop of stack failed', 0)
      call ga_sync()
c
      end
