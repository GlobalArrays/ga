      program test
c***
c***  Last modification: Tue Oct  4 13:27:10 PDT 1994
c***
      implicit none
#include "mafdecls.h"
#include "global.fh"
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
      heap = 200000
      if (.not. ma_init(MT_DBL, heap,heap))
     $     call parerr(1)
c
c     Intitialize the GA package
c
      call ga_initialize()
      if(ga_nnodes().ne.4)
     $   call ga_error('Must be run with 4 GA processes',ga_nnodes())
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
c     not for the KSR
c
      integer n, nn
      parameter (n = 2*352, nn = n*n)
c     parameter (n = 60, nn = n*n)
c     parameter (n = 40, nn = n*n)
      double precisiON a(n),  buf(nn)
c     double precision a(n,n), b(n,n), c(n,n)
      integer g_a,g_b
      integer  i, j
      integer  ilo, ihi, jlo, jhi, ii0, ii1, jj0, jj1, bytes
      integer nproc, me
      integer loop, k,kk, ldb, count
      double precision tp0, tp1, tg0, tg1, ta0, ta1, rk
c     
      nproc = ga_nnodes()
      me = ga_nodeid()

c     
c
c***  Create global arrays
      if (.not. ga_create(MT_DBL, n, n, 'a', 1, 1, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
c     
c***  Fill in arrays A & B
      if (me .eq. 0) then
         write(6,21)n 
 21      format(/' preparing input data n=  ',i8)
         call flush(6)
      endif
c
c     do j = 1+me, n, nproc 
c        do i = 1, n
c           a(i) = 1d0 * (i+j)
c           if(i.eq.j) then
c              a(i) = i
c           endif
c        enddo
c
c        call ga_put(g_a, 1,n, j,j, a(1),n)
c     enddo
c
c

c     this guarantees that everybody touches own data
c
      call ga_dfill_patch(g_a, 1, n, 1,n , 1d0*me)

      call ga_sync()
c****
      if (me .eq. 0) then
        print *,' '
        print *,'> test of ga_get, ga_put & ga_acc '
        print *,' '
        call flush(6)
      endif
c
c
      
      count = 30 
      if (me .eq. 0) then
        print *, 'bytes          get                     put',
     &           '          acc' 

        do k = 1, count 
         call ga_distribution(g_a, me, ilo, ihi, jlo, jhi)

         rk = sqrt((1.* k)/count) 
         ii0 = ilo 
         ii1 = ii0 + int((ihi -ii0)*rk)
         jj0 = jlo
         jj1 = jj0 + int((jhi -jj0)*rk)

         bytes= 8*(ii1-ii0+1)*(jj1-jj0+1)  
         loop = 1
         ldb = ii1-ii0+1
c
	 call ga_get(g_a, ii0,ii1, jj0, jj1, buf,ldb)
         tg0 = tcgtime()
         do kk = 1, loop 
	    call ga_get(g_a, ii0,ii1, jj0, jj1, buf,ldb)
         enddo
         tg1 = tcgtime()
c
         tp0 = tcgtime()
         do kk = 1, loop
            call ga_put(g_a, ii0,ii1, jj0, jj1, buf,ldb)
         enddo
         tp1 = tcgtime()
c
         ta0 = tcgtime()
         do kk = 1, loop
            call ga_acc(g_a, ii0,ii1, jj0, jj1, buf,ldb, .999d0)
         enddo
         ta1 = tcgtime()
c
         write(*,77)bytes,(tg1-tg0)/loop,(tp1-tp0)/loop,(ta1-ta0)/loop 
         call flush(6)
        enddo
      endif
      call ga_sync()

      if (me .eq. 0) then
77      format(i9, 3d16.8)
c
        print *, ' '
      endif
c
      do k = 1, count
        
#ifdef SHMEM
c        everybody touches own data 
         call ga_dfill_patch(g_a, 1, n, 1,n , 1d0)
#endif
      if (me .eq. 0) then
         rk = sqrt((1.* k)/count )
c
         call ga_distribution(g_a, me+1, ilo, ihi, jlo, jhi)

         ii0 = ilo 
         ii1 = ii0 + int((ihi -ii0)*rk)
         jj0 = jlo
         jj1 = jj0 + int((jhi -jj0)*rk)
c
c        ii0 = ihi+1
c        ii1 = ii0 + int((n -ii0)*rk)
c        jj0 = jhi+1
c        jj1 = jj0 + int((n -jj0)*rk)
c

         bytes= 8*(ii1-ii0+1)*(jj1-jj0+1) 
c        loop = 20*k
         loop = 1
         ldb = ii1-ii0+1
c
c        call ga_get(g_a, ii0,ii1, jj0, jj1, buf,ldb)
         tg0 = tcgtime()
c         do kk = 1, loop
            call ga_get(g_a, ii0,ii1, jj0, jj1, buf,ldb)
c         enddo
         tg1 = tcgtime()
 	 call sleep(1)
c
         call ga_distribution(g_a, me+3, ilo, ihi, jlo, jhi)
         ii0 = ilo 
         ii1 = ii0 + int((ihi -ii0)*rk)
         jj0 = jlo
         jj1 = jj0 + int((jhi -jj0)*rk)
	 if(8*(ii1-ii0+1)*(jj1-jj0+1).ne.bytes)call ga_error('dist',0)
c
         tp0 = tcgtime()
         do kk = 1, loop
            call ga_put(g_a, ii0,ii1, jj0, jj1, buf,ldb)
         enddo
         tp1 = tcgtime()
c	 call sleep(1)
c
         call ga_distribution(g_a, me+2, ilo, ihi, jlo, jhi)
         ii0 = ilo 
         ii1 = ii0 + int((ihi -ii0)*rk)
         jj0 = jlo
         jj1 = jj0 + int((jhi -jj0)*rk)
	 if(8*(ii1-ii0+1)*(jj1-jj0+1).ne.bytes)call ga_error('dist',0)
c
c        print *, ii0,ii1, jj0, jj1
         ta0 = tcgtime()
         do kk = 1, loop
            call ga_acc(g_a, ii0,ii1, jj0, jj1, buf,ldb, .999d0)
         enddo
         ta1 = tcgtime()
c
         write(*,77)bytes,(tg1-tg0)/loop,(tp1-tp0)/loop,(ta1-ta0)/loop 
         call flush(6)
c	 call sleep(1)
      else
c         call sleep(1)
      endif
      enddo
c

      end



