      subroutine ga_lu_solve(trans,g_a, g_b)
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "slface.fh"
c
      character*1 trans         ! transpose or not  
      integer g_a               ! coefficient matrix A 
      integer g_b               ! rhs matrix, B, overwritten on exit by
                                ! the solution vector, X
c
c
c     solve the set of linear equations 
c
c           AX = B
c
c     with possibly multiple rhs stored as columns of matrix B
c
c     the matrix A is not destroyed
c
c
c Jarek Nieplocha, 02.16.94
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      integer ha, adra          !A
      integer hb, adrb          !B
      integer hz, adrz          !workspace
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

      integer nb                ! block size for A and rhs, internal parameter
      data nb /5/
c
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
      call ga_sync()
      if (oactive) then
c
c***     allocate a,b, and work and ipiv arrays
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
c...     amount of Scalapack workspace as specified in pdgetrf - WRONG !
c
c        nwork = max(1,nb * (n/(nb*nprow) +n/(nb*npcol)+2))
         nwork = max( mpA*nqA*2,mpB*nqB*2 )
         status = ma_push_get(MT_DBL,nwork,'z',hz,adrz)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed Z ', -1)
c
         status = ma_push_get(MT_INT,lda+nb,'ipiv',hi,adri)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed ipiv ', -1)
c
c
c***     copy g_a to A and g_b to B using the SBS SL format 
c
         call ga_to_SL(g_a, n, n, nb, nb, dbl_mb(adrA), lda, mpA,nqA)
         call ga_to_SL(g_b, n,dimB2, nb,nb, dbl_mb(adrB), ldb, mpB, nqB)
c     
         call ga_sync()
c
c
c***     LU factorization
c

         call pdgetrf(n, n, nb, dbl_mb(adra), lda, int_mb(adri),
     &                info, dbl_mb(adrz))
c
         if(info.eq.0)then
c
c***        solve: forward/backward substitution
c
            call pdgetrs(trans, n, nb, dimB2, nb, dbl_mb(adra), lda,
     &           int_mb(adri),dbl_mb(adrb), ldb, 0, info, dbl_mb(adrz))
c
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
         status = ma_pop_stack(hz)
         if(elemB.ne.0)status = ma_pop_stack(hb)
         if(elemA.ne.0)status = ma_pop_stack(ha)
      endif
c
      call ga_sync()
      end
