      subroutine ga_cholesky(g_a)
c
c     Cholesky factorization G&VL: alg. 6.6.1
c
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer g_a
      Integer me, nproc
      Integer ilo, ihi, jlo, jhi, n, j, k
      Integer dim1, dim2, type
      double precision alpha
      Logical status

      Integer hv, adrv          !v
      Integer hw, adrw          !w
c
c***  check environment
c
      me = ga_nodeid()
      nproc   = ga_nnodes()
c
      call ga_inquire(g_a,  type, dim1, dim2)
      n = dim1
c
      if (dim1 .ne. dim2 )
     $    call ga_error('ga_cholesky: dimensions do not match', 0)
      if (type .ne. MT_DBL)
     $   call ga_error('ga_cholesky: wrong type ', 0)
c
c***  Find the local distribution
      call ga_distribution(g_a, me, ilo, ihi, jlo, jhi)
c
c***  get temporary local storage for v & w
c
      if(.not. ma_push_get(MT_DBL,n,'v',hv,adrv))
     $    call ga_error('ga_cholesky: v allocation failed ', 0)
      if(.not. ma_push_get(MT_DBL,n,'w',hw,adrw))
     $    call ga_error('ga_cholesky: w allocation failed ', 0)
c
      call ga_sync()
      do k = 1, n
         if(me.eq.0)then
            call ga_get(g_a, k,n, k, k, dbl_mb(adrv+k-1),n)
            alpha = dbl_mb(adrv+k-1)
            if(alpha .le. 0d0)
     &               call ga_error('ga_cholesky: diag<=0',k)
            alpha = 1d0/sqrt(alpha)
            call dscal(n-k+1, alpha, dbl_mb(adrv+k-1), 1)
            call ga_put(g_a, k,n, k, k, dbl_mb(adrv+k-1),n)
            if(k.lt.n) 
     &         call ga_brdcst(32111, dbl_mb(adrv+k), mdtob(n-k),me)
         endif
         call ga_sync()
c
         if(k.lt.n)then
            if(me.ne.0)then
                  call ga_brdcst(32111, dbl_mb(adrv+k), mdtob(n-k),0)
            endif
         endif
c          
         do j = k+me+1, n, nproc
            call ga_get(g_a, j,n, j, j, dbl_mb(adrw),n)
            alpha = -dbl_mb(adrv-1+j)
            call daxpy(n-j+1, alpha, dbl_mb(adrv-1+j), 1,
     &                               dbl_mb(adrw),    1)
            call ga_put(g_a, j,n, j, j, dbl_mb(adrw),n)
         enddo
         call ga_sync()
*        call ga_print(g_a)
      enddo
c
      status = ma_pop_stack(hw)
      status = ma_pop_stack(hv)
      end

