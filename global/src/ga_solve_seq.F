#ifdef CRAY
#define DGETRF SGETRF
#define DGETRS SGETRS
#endif

      subroutine ga_lu_solve_seq(trans,g_a, g_b)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c
      character trans           ! transpose or not  
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
      integer me
      integer info
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
      call ga_inquire(g_a,  typeA, dimA1, dimA2)
      call ga_inquire(g_b,  typeB, dimB1, dimB2)
c
      if (dimA1 .ne. dimA2) then
         call ga_error('ga_lu_solve: g_a must be square matrix ', 1)
      else if(dimA1 .ne. dimB1) then
         call ga_error('ga_lu_solve: dims of A and B do not match ', 1)
      else if(typeA.ne.MT_DBL .or. typeB.ne.MT_DBL)then
         call ga_error('ga_lu_solve: wrong type(s) of A and/or B ', 1)
      endif
c
      call ga_sync()
      oactive = me.eq.0
      if (oactive) then
c
c***     allocate a,b, and work and ipiv arrays
c     
         status = ma_push_get(MT_DBL,dimA1*dimA2,'a',ha,adra)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed A ', -1)
         status = ma_push_get(MT_DBL,dimB1*dimB2,'b',hb,adrb)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed B ', -1)
         status = ma_push_get(MT_INT,min(dimA1,dimA2),'ipiv',hi,adri)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed ipiv ', -1)
c
c***     Fill local arrays from global arrays
c    
         call ga_get(g_a, 1, dimA1, 1, dimA2, dbl_mb(adra), dimA1)
         call ga_get(g_b, 1, dimB1, 1, dimB2, dbl_mb(adrb), dimB1)
c
c***     LU factorization
c
         call DGETRF(dimA1, dimA2, dbl_mb(adra), dimA1, int_mb(adri),
     &                info)
c
         if(info.eq.0)then
c
c***        solve
c
            call DGETRS(trans, dimA1, dimB2, dbl_mb(adra), dimA1, 
     &                  int_mb(adri),dbl_mb(adrb), dimB1, info)
c
            if(info.eq.0)then
               call ga_put(g_b, 1, dimB1, 1, dimB2, dbl_mb(adrb), dimB1)
            else
               call ga_error(' ga_lu_solve: dgetrs failed ', -info)
            endif
c
         else
            call ga_error(' ga_lu_solve: dgetrf failed ', -info)
         endif
c
c***     deallocate work arrays
c
         status = ma_pop_stack(hi)
         status = ma_pop_stack(hb)
         status = ma_pop_stack(ha)
      endif
c
c
      call ga_sync()     
      end
