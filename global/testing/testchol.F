      program test
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
      integer heap
c     
c     Intitialize the TCGMSG package
c
c
        call pbeginf
c       call setdbg(1)            ! Turn TCGMSG debugging on/off
c
c     Initialize the MA package
c
      heap = 400000
      if (.not. ma_init(MT_DBL, heap,heap))
     $     call parerr(1)
c
c     Intitialize the GA package
c
      call ga_initialize()
c
      call testit()
      call ga_terminate()
c
      call pend()
      end


c-----------------

      subroutine testit()
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
c     
      integer n
      parameter (n = 1200)
      double precision a(n), b(n)
*     double precision aa(n,n), bb(n,n), c(n,n)
      integer g_a,g_b,g_c,g_d
      integer  i, j
      integer nproc, me
      double precision sum, t0, t1
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c     a() is a local copy of what the global array should start as
c
c
c***  Create global arrays
      if (.not. ga_create(MT_DBL, n, n, 'a', 1, 1, g_a))
     $     call ga_error(' ga_create failed ',2)
      if (.not. ga_create(MT_DBL, n, n, 'b', 1, 1, g_b))
     $     call ga_error(' ga_create failed ',2)
      if (.not. ga_create(MT_DBL, n, n, 'c', 1, 1, g_c))
     $     call ga_error(' ga_create failed ',2)
      if (.not. ga_create(MT_DBL, n, n, 'd', 1, 1, g_d))
     $     call ga_error(' ga_create failed ',2)
c     
c     
c***  Fill in arrays A & B
      if (me .eq. 0) then
         write(6,21)n 
 21      format(/' preparing input data n=  ',i8)
         call ffflush(6)
      endif
      call ga_zero(g_b)
      call ga_zero(g_a)
c
      do j = 1+me, n, nproc 
         do i = 1, n
            a(i) = .1d0 * (i+j)
            if(j.le.i)then
              b(i) = .1d0 * (i+j)
            else
              b(i)=0d0 
            endif
            if(i.eq.j) then
               a(i) = 10.d0 *n*n
               b(i) = 10.d0 *n*n
            endif
         enddo
c
         call ga_put(g_a, 1,n, j,j, a(1),n)
         call ga_put(g_b, 1,n, j,j, b(1),n)
      enddo
c

*     call ga_print(g_b) 
      if (me .eq. 0) then
         write(6,*)' calling cholesky '
         call ffflush(6)
      endif

      call ga_sync()
      t0 = tcgtime()
      call ga_cholesky(g_b) 
      t1 = tcgtime()
      if (me .eq. 0) then
         write(6,*)' after cholesky --  time: ', t1-t0
         call ffflush(6)
      endif
c
      call ga_dgemm('n','t',n,n,n, 1d0, g_b, g_b, 0d0, g_d) ! d := b*b'
      call ga_dadd(1d0, g_a, -1d0, g_d, g_c) 
      sum = ga_ddot(g_c,g_c)
      if (me .eq. 0) then
 	print *,' '
        print *, ' norm = ', dsqrt(sum)/n
        if(dsqrt(sum)/n.lt.1d-10) print *, ' test passed '
 	print *,' '
        call ffflush(6)
      endif
c
      end

