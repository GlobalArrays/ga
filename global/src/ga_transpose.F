      subroutine ga_transpose(g_a,g_b) 
c
c     Transposes matrix A:  B := A`
c     A remains unchanged
c 
c     J. Nieplocha 05.17.94
c
      implicit none
#include "mafdecls.h" 
#include "global.fh"
      integer g_a, g_b               ! g_b = trans(g_a) 
c
      Integer myproc, proc 
      Integer ilo, ihi, jlo, jhi, myelem, nrow,ncol, lda, i, j
      Integer adim1, adim2, atype, Abase
      Integer bdim1, bdim2, btype, Bbase
      Logical status, have_data

      Integer     adrA          !A
      Integer hB, adrB          !B
c
c***  check environment
c
      myproc = ga_nodeid() 
      proc   = ga_nnodes()
c  
      call ga_inquire(g_a,  atype, adim1, adim2)
      call ga_inquire(g_b,  btype, bdim1, bdim2)
c
      if (bdim1 .ne. adim2 .or. adim1 .ne. bdim2) then
         call ga_error('ga_trans: dimensions do not mach', 0)
      endif
c
      if (atype .ne. btype .or. ((atype.ne.MT_INT)
     $                     .and. (atype.ne.MT_DBL))) then
         call ga_error('ga_trans: wrong type(s) ', 0)
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
c
c***     get temporary storage for transposed local patch of array 
         if (.not. ma_push_get(MT_DBL,myelem,'B',hB,adrB))
     $        call ga_error('ga_transpose: insufficient memory', myelem)
c
c***     transpose local patch
         Bbase = adrB-1
         Abase = adrA-1
         do i = 1, nrow
            Abase = Abase + 1
            do j = 1, ncol
               dbl_mb(Bbase+j) = dbl_mb(Abase + (j-1)*lda) 
            enddo
            Bbase = Bbase + ncol
         enddo
c 
c***     put the data into g_b
         call GA_PUT(g_b, jlo, jhi, ilo, ihi, dbl_mb(adrB),ncol) 
c
         status = ma_pop_stack(hB) 
      endif
c
      call ga_sync()
c
      end
