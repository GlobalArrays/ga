c**********************************************************************
c* Source : ga_lu_solve
c* Scope  : LU factorization
c*
c* 02/16/94  Jarek Nieplocha 
c* 04/12/96  GVT  Adapted the code to the new version of ScaLAPACK
c*           Giuseppe Vitillaro peppe@unipg.it
c**********************************************************************      
c
c length of scalapack array descriptor
# define DSCLEN 10
c
      subroutine ga_lu_solve(trans,g_a, g_b)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "slface.fh"
c
      character*1 trans         ! transpose or not  
      integer g_a               ! coefficient matrix A 
      integer g_b               ! rhs matrix, B, overwritten on exit by
                                ! the solution vector, X
c
c     solve the set of linear equations 
c
c           AX = B
c
c     with possibly multiple rhs stored as columns of matrix B
c
c     the matrix A is not destroyed
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      integer ha, adra          !A
      integer hb, adrb          !B
      integer hi, adri          !ipiv
c
c
      logical oactive           ! true iff this process participates
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
c
      integer mpA, nqA          ! number of rows/cols of A held by the processor
      integer mpB, nqB          ! number of rows/cols of B held by the processor
      integer info
      integer me
      integer nwork
      integer n,lda, ldb, elemA, elemB
      integer numroc

      integer descA(DSCLEN), descB(DSCLEN) ! descriptors for distr. matrixes A,B
      integer nb                           ! block size for A and rhs
      data nb /64/
c
c***  check environment
c
      me     = ga_nodeid()
c
c***  check GA info for input arrays
c
      call ga_check_handle(g_a, 'ga_lu_solve: a')
      call ga_check_handle(g_b, 'ga_lu_solve: b')
      call ga_inquire(g_a, typeA, dimA1, dimA2)
      call ga_inquire(g_b, typeB, dimB1, dimB2)
c
      if (dimA1 .ne. dimA2) then
         call ga_error('ga_lu_solve: g_a must be square matrix ', 1)
      else if(dimA1 .ne. dimB1) then
         call ga_error('ga_lu_solve: dims of A and B do not match ',1)
      endif
c
      n = dimA1
c
c***  initialize SL interface
c 
      call SLinit()
c
c***  find SBS format parameters
c
      mpA = numroc(n, nb,myrow, 0, nprow)
      nqA = numroc(n, nb,mycol, 0, npcol)
      mpB = numroc(dimB1, nb,myrow, 0, nprow)
      nqB = numroc(dimB2, nb,mycol, 0, npcol) 
      lda = max(1,mpA)
      ldb = max(1,mpB)

      oactive = .true.
      if (oactive) then
c
c***     allocate A,B and ipiv arrays
c     
         elemA= mpA*nqA
         status = .true.
         if(elemA.ne.0)status = ma_push_get(MT_DBL,elemA,'a',ha,adra)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed A ', -1)
c
         elemB= mpB*nqB
         if(elemB.ne.0)status = ma_push_get(MT_DBL,elemB,'b',hb,adrb)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed B ', -1)
c
         status = ma_push_get(MT_INT,mpA+nb,'ipiv',hi,adri)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed ipiv ', -1)
c
c
c***     copy g_a to A and g_b to B using the SBS SL format 
c
         call ga_to_SL(g_a, n, n, nb, nb, dbl_mb(adrA), lda, mpA,nqA)
         call ga_to_SL(g_b, n,dimB2, nb,nb, dbl_mb(adrB), ldb, mpB, nqB)
c     
c***     fill SCALAPACK matrices descriptors
c
	 call descinit(descA, dimA1, dimA2, nb, nb, 0, 0, 
     &                 iSLctxt, lda, info)
         if(info.ne.0) call ga_error(' ga_lu_solve: descinit A failed ',
     &                               -info)
         call descinit(descB, dimB1, dimB2, nb, nb, 0, 0, 
     &                 iSLctxt, ldb, info)
         if(info.ne.0) call ga_error(' ga_lu_solve: descinit B failed ',
     &                               -info)
c
c***     LU factorization
c
         call ga_sync()
	 call pdgetrf(n, n, dbl_mb(adrA), 1, 1, descA, 
     &                int_mb(adrI), info)
         call ga_sync()
c
c
         if(info.eq.0)then
c
c***        solve: forward/backward substitution
c
            call pdgetrs(trans, n, dimB2, dbl_mb(adrA), 1, 1, descA,
     &                   int_mb(adrI), dbl_mb(adrB), 1, 1, descB, info)
            if(info.eq.0)then
c
c***           copy solution matrix back to g_b
c
               call ga_from_SL(g_b, dimB1, dimB2, nb, nb, dbl_mb(adrb),
     &                         mpB, ldb, nqB)
            else
               call ga_error(' ga_lu_solve: pdgetrs failed ', -info)
            endif
c
         else
            call ga_error(' ga_lu_solve: pdgetrf failed ', -info)
         endif
c
c***     deallocate work/SL arrays
c
         status = ma_pop_stack(hi)
         if(elemB.ne.0) status = status .and. ma_pop_stack(hb)
         if(elemA.ne.0) status = status .and. ma_pop_stack(ha)
         if(.not.status)
     &       call ga_error('ga_lu_solve:ma_pop_stack failed',0)
      endif
c
      call ga_sync()
      end

c     alt entry point for C that avoids char args
c
	subroutine ga_lu_solve_alt(tr, g_a, g_b)
	implicit none
        integer g_a, g_b, tr
        if(tr.eq.0) then
           call ga_lu_solve('n',g_a, g_b) 
        else
           call ga_lu_solve('y',g_a, g_b) 
        endif
        end

c     alt entry point for C that avoids char args
c
	subroutine ga_lu_solve_alt(tr, g_a, g_b)
	implicit none
        integer g_a, g_b, tr
        if(tr.eq.0) then
           call ga_lu_solve('n',g_a, g_b) 
        else
           call ga_lu_solve('t',g_a, g_b) 
        endif
        end

