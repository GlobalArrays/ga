      subroutine ga_symmetrize(g_a) 
c
c     Symmetrizes matrix A:  A := .5 * (A+A`)
c     diag(A) remains unchanged
c 
c     J. Nieplocha 08.22.93
c
      implicit none
#include "mafdecls.fh" 
#include "global.fh"
      integer g_a               ! Matrix to symmetrize 
c
      Double Precision alpha   
      Integer myproc, proc 
      Integer ilo, ihi, jlo, jhi, myelem, nrow,ncol, lda
      Integer dim1, dim2, type
      Logical have_data

      Integer     adrA          !A
      Integer hB, adrB          !B
c
c***  check environment
c
      myproc = ga_nodeid() 
      proc   = ga_nnodes()
      alpha = 0.5
c  
      call ga_inquire(g_a,  type, dim1, dim2)
c
      if (dim1 .ne. dim2) then
         call ga_error('ga_sym: can only sym square matrix', 0)
      endif
c
c***  Find the local distribution
      call ga_distribution(g_a, myproc, ilo, ihi, jlo, jhi)
c
      have_data = ihi.gt.0 .and. jhi.gt.0
c
c     if(myproc.eq.0)call ga_print(g_a)
      call ga_sync()
      if (have_data) then
         call ga_access(g_a, ilo, ihi, jlo, jhi, adrA, lda)
         nrow = ihi-ilo+1
         ncol = jhi-jlo+1
         myelem = nrow * ncol
         if (.not. ma_push_get(MT_DBL,myelem,'ga_symtr:B',hB,adrB))
     $      call ga_error('ga_symmetrize: insufficient memory', myelem)
c        
         call GA_GET(g_a, jlo, jhi, ilo, ihi, dbl_mb(adrB),ncol) 
c        print *, myproc,'a, aT', dbl_mb(adrA), dbl_mb(adrB),
c    $            ilo, ihi, jlo, jhi
      endif

      call ga_sync()
      
      if(have_data) then
         call gai_add(nrow, ncol, dbl_mb(adrA), lda, dbl_mb(adrB),
     &                ncol, alpha)
         call ga_release_update(g_a, ilo, ihi, jlo, jhi)
         if (.not. ma_pop_stack(hB))
     $        call ga_error('ga_symmetrize: ma_pop?', 0)
      endif
c
      call ga_sync()
c
      end



      subroutine gai_add(nrow, ncol, A, lda, B, ldb, alpha)
c
c     A := alpha * (A + B^)
c
      implicit none
      Integer nrow, ncol, lda, ldb
      Double Precision A(lda,ncol), B(ldb,nrow), alpha
      Integer i,j
c
      do j = 1, ncol
         do i = 1, nrow
	a(i,j) = alpha*(a(i,j)+b(j,i))
         enddo
      enddo
      END
