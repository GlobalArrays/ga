      program test
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
      integer heap, stack, fudge
      logical status
      parameter (heap=100*100*2, fudge=10240, stack=0)
c     
c     Intitialize the TCGMSG package
c
      call pbeginf
c     call setdbg(1)            ! Turn TCGMSG debugging on/off
c
c     Initialize the MA package
c
      status = ma_init(MT_DBL, stack, 2*heap/nnodes() + fudge)
      if (.not. status) print *, 'ma_init failed' 
c
c     Intitialize the GA package
c
      if(nodeid().eq.0)then
        print *, 'using ',nnodes(), ' process(es)'
         call ffflush(6)
      endif
      call ga_initialize()
      if(ga_nodeid().eq.0)then
         print *,' GA initialized'
         call ffflush(6)
      endif
c
c     Check support for double precision arrays
c
      if (nodeid().eq.0) then
         write(6,*)
         write(6,*) ' CHECKING DOUBLES  '
         write(6,*)
         call ffflush(6)
      endif

      call check_dbl()
c
c     Check support for integer arrays (limited !)
c
      if (nodeid().eq.0) then
         write(6,*)
         write(6,*) ' CHECKING INTEGERS  '
         write(6,*)
         call ffflush(6)
      endif

      call check_int()
c
c     Tidy up the global array package
c
      call ga_terminate()
c
c
c     Tidy up after TCGMSG
c
      call pend()
c
      end


      subroutine check_dbl()
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
c     
      integer n,m
      parameter (n = 100)
      parameter (m = 2*n)
      double precision a(n,n), b(n,n), v(m),w(m)
      integer iv(m), jv(m)
      logical status
      integer g_a, g_b
      integer iran, i, j, loop, nloop, ilo, ihi, jlo, jhi, itmp
      integer nproc, me, int, ij, inc, ii, jj
      parameter (nloop = 100)
      integer maxproc
      parameter (maxproc = 128)
      double precision drand, crap, sum1, sum2, x
      double precision nwords
      external drand
      intrinsic int
      iran(i) = int(drand(0)*dfloat(n)) + 1
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c     a() is a local copy of what the global array should start as
c
      do j = 1, n
         do i = 1, n
            a(i,j) = i-1 + (j-1)*n
         enddo
      enddo
*      write(6,*) ' correct '
*      call output(a, 1, n, 1, n, n, n, 1)
*      call ffflush(6)
c
c     Create a global array
c
*     print *,ga_nodeid(), ' creating array'
*     call ffflush(6)
c     call setdbg(1)
      status = ga_create(MT_DBL, n, n, 'a', 1, 1, g_a)
      if (.not. status) then
         write(6,*) ' ga_create failed'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
*     call ga_distribution(g_a,me,ilo, ihi, jlo, jhi)
*     print *,me, ilo, ihi, jlo, jhi
*     call ffflush(6)
      call ga_sync()
c     
c     Zero the array
c
      if (me .eq. 0) then 
         write(6,21) 
 21      format(/'> Checking zero ... ')
         call ffflush(6)
      endif
      call ga_zero(g_a)
c
c     Check that it is indeed zero
c
      
      call ga_get(g_a, 1, n, 1, n, b, n)
      do i = 1, n
         do j = 1, n
            if (b(i,j) .ne. 0.0d0) then
               write(6,*) ' zero ', me, i, j, b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_zero is OK'
         write(6,*)
      endif
      call ga_sync()
c
c     Each node fills in disjoint sections of the array
c
      if (me .eq. 0) then
         write(6,2) 
 2       format(/'> Checking disjoint put ... ')
         call ffflush(6)
      endif
      call ga_sync()
c
      inc = (n-1)/10 + 1
      ij = 0
      do j = 1, n, inc
         do i = 1, n, inc
            if (mod(ij,nproc) .eq. me) then
               ilo = i
               ihi = min(i+inc, n)
               jlo = j
               jhi = min(j+inc, n)
*               write(6,4) me, ilo, ihi, jlo, jhi
* 4             format(' node ',i2,' checking put ',4i4)
*               call ffflush(6)
               call ga_put(g_a, ilo, ihi, jlo, jhi, a(ilo, jlo), n)
            endif
            ij = ij + 1
         enddo
      enddo
      call ga_sync()
c
c     All nodes check all of a
c
      call dfill(n*n, 0.0d0, b, 1)
      call ga_get(g_a, 1, n, 1, n, b, n)
*      write(6,*) ' after get'
*      call output(b, 1, n, 1, n, n, n, 1)
c
      do i = 1, n
         do j = 1, n
            if (b(i,j) .ne. a(i,j)) then
               write(6,*) ' put ', me, i, j, a(i,j),b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_put is OK'
         write(6,*)
      endif
      call ga_sync()
c
c     Now check nloop random gets from each node
c
      if (me .eq. 0) then
         write(6,5) nloop
 5       format(/'> Checking random get (',i5,' calls)...')
         call ffflush(6)
      endif
      call ga_sync()
c
      nwords = 0
c
      crap = drand(nodeid()*51 + 1) ! Different seed for each process
      do loop = 1, nloop
         ilo = iran(loop)
         ihi = iran(loop)
         if (ihi.lt. ilo) then
            itmp = ihi
            ihi = ilo
            ilo = itmp
         endif
         jlo = iran(loop)
         jhi = iran(loop)
         if (jhi.lt. jlo) then
            itmp = jhi
            jhi = jlo
            jlo = itmp
         endif
c
         nwords = nwords + (ihi-ilo+1)*(jhi-jlo+1)
c     
         call dfill(n*n, 0.0d0, b, 1)
         call ga_get(g_a, ilo, ihi, jlo, jhi, b(ilo, jlo), n)
         if (me .eq. 0 .and. mod(loop-1, max(1,nloop/20)).eq.0) then
            write(6,1) loop, me, ilo, ihi, jlo, jhi, nwords
 1          format(' call ',i5, ' node ',i2,' checking get ',4i4,
     $           ' total ',d9.2)
            call ffflush(6)
         endif
         sum1 = 0.0d0
         do j = jlo, jhi
            do i = ilo, ihi
               sum1 = sum1 + b(i,j)
               if (b(i,j) .ne. a(i,j)) then
                  write(6,*) i, j, b(i,j), a(i,j)
                  call ffflush(6)
                  call ga_error('... exiting ',0)
               endif
            enddo
         enddo
c
c     This test for writing out of range is quite slow ...
c     commented out here only to permit greater contention
c
*         sum2 = 0.0d0
*         do j = 1, n
*            do i = 1, n
*               sum2 = sum2 + b(i,j)
*            enddo
*         enddo
*         if (sum1 .ne. sum2) then
*            write(6,*) ' sums dont match ', sum1, sum2
*            call ffflush(6)
*            call call ga_error('... exiting ',0)
*         endif
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_get is OK'
         write(6,*)
         call ffflush(6)
      endif
      call ga_sync()
c
c     Each node accumulates into disjoint sections of the array
c
      if (me .eq. 0) then
         write(6,9) 
 9       format(/'> Checking disjoint accumulate ... ')
         call ffflush(6)
      endif
      call ga_sync()
c     
      crap = drand(12345)       ! Same seed for each process
      do j = 1, n
         do i = 1, n
            b(i,j) = drand(0)
         enddo
      enddo
c
      inc = (n-1)/10 + 1
      ij = 0
      do j = 1, n, inc
         do i = 1, n, inc
            x = drand(0)
            ilo = i
            ihi = min(i+inc-1, n)
            if(ihi.eq.n-1)ihi=n
c           ihi = min(i+inc, n)
            jlo = j
            jhi = min(j+inc, n)
            if(jhi.eq.n-1)jhi=n
c           jhi = min(j+inc-1, n)
            if (mod(ij,nproc) .eq. me) then
c               write(6,11) me, ilo, ihi, jlo, jhi
c 11            format(' node ',i2,' checking update ',4i4)
c               call ffflush(6)
               call ga_acc(g_a, ilo, ihi, jlo, jhi, b(ilo, jlo), n, x)
            endif
            ij = ij + 1
c     
c           Each process applies all updates to its local copy
c
            do jj = jlo, jhi
               do ii = ilo, ihi
                  a(ii,jj) = a(ii,jj) + x * b(ii,jj)
               enddo
            enddo
         enddo
      enddo
      call ga_sync()
c
c     All nodes check all of a
c
      call ga_get(g_a, 1, n, 1, n, b, n)
      do j = 1, n
         do i = 1, n
            if (abs(b(i,j) - a(i,j)) .gt. 1.0d-6) then
               write(6,*) ' acc ', me, i, j, a(i,j), b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_acc is OK'
         write(6,*)
      endif
      call ga_sync()
c
c     Check the ga_add function
c
      if (me .eq. 0) then
         write(6,91)
 91      format(/'> Checking add ...')
         call ffflush(6)
      endif
      if (.not. ga_create(MT_DBL, n, n, 'b', 1, 1, g_b)) then
         write(6,*) ' ga_create failed for second array'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
      crap = drand(12345)       ! Everyone has same seed
      do j = 1, n
         do i = 1, n
            b(i,j) = drand(0)
            a(i,j) = 0.1d0*a(i,j) + 0.9d0*b(i,j)
         enddo
      enddo
      if (me.eq.0) call ga_put(g_b, 1, n, 1, n, b, n)
      call ga_dadd(0.1d0, g_a, 0.9d0, g_b, g_b)
      call ga_get(g_b, 1, n, 1, n, b, n)
      do j = 1, n
         do i = 1, n
            if (abs(b(i,j) - a(i,j)) .gt. 1.0d-6) then
               write(6,*) ' add ', me, i, j, a(i,j), b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0) 
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' add is OK '
         write(6,*)
      endif
      call ga_sync()
c
c     Check the ddot function
c
      if (me .eq. 0) then
         write(6,19) 
 19      format(/'> Checking ddot ...')
         call ffflush(6)
      endif
      crap = drand(12345)       ! Everyone has same seed
      sum1 = 0.0d0
      do j = 1, n
         do i = 1, n
            b(i,j) = drand(0)
            sum1 = sum1 + a(i,j)*b(i,j)
         enddo
      enddo
      if (me.eq.0) then 
         call ga_put(g_b, 1, n, 1, n, b, n)
         call ga_put(g_a, 1, n, 1, n, a, n)
      endif
      call ga_sync()
      sum2 = ga_ddot(g_a,g_b)
      if (dabs(sum2 - sum1)/dabs(sum1) .gt. 1.0d-12) then
         write(6,*) ' ddot wrong ', sum1, sum2
         call ffflush(6)
         call ga_error('... exiting ',0)
      else if (me.eq.0) then
         write(6,*)
         write(6,*) ' ddot is OK '
         write(6,*)
      endif
c
c     Check the ga_dscal function
c
      if (me .eq. 0) then
         write(6,92)
 92      format(/'> Checking dscal ...')
         call ffflush(6)
      endif
c     call dscal(n*n, 0.123d0, a, 1)
      call ga_dscal(g_a, 0.123d0)
      call ga_get(g_a, 1, n, 1, n, b, n)
      do j = 1, n
         do i = 1, n
            a(i,j) = a(i,j)*0.123d0
            if (b(i,j) .ne. a(i,j)) then
               write(6,*) ' dscal ', me, i, j, a(i,j), b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' dscal is OK '
         write(6,*)
      endif
c
c     Check the ga_copy function
c
      if (me .eq. 0) then
         write(6,93)
 93      format(/'> Checking copy ...')
         call ffflush(6)
      endif
      call ga_copy(g_a, g_b)
      call ga_get(g_b, 1, n, 1, n, b, n)
      do j = 1, n
         do i = 1, n
            if (b(i,j) .ne. a(i,j)) then
               write(6,*) ' copy ', me, i, j, a(i,j), b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' copy is OK '
         write(6,*)
      endif
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*) '> Checking scatter/gather ... '
         call ffflush(6)
      endif
      call ga_sync()
c
      crap = drand(nodeid()*51 + 1) ! Different seed for each process
      do j = 1, 10 
       call ga_sync()
       if(me.eq.iran(ga_nnodes())-1) then
         do loop = 1,m 
           ilo = iran(n) 
           jlo = iran(n)
           iv(loop) = ilo
           jv(loop) = jlo
         enddo 
         call ga_dgather(g_a, v, iv, jv, m)
         do loop = 1,m
           ilo= iv(loop)
           jlo= jv(loop)
           call ga_get(g_a,ilo,ilo,jlo,jlo,v(loop),1)
           if(v(loop)  .ne. a(ilo,jlo))then
             write(6,*)me,' gather ', ilo,',',jlo,',', a(ilo,jlo)
     &             ,' ',v(loop)
             call ffflush(6)
             call ga_error('... exiting ',0)
           endif
         enddo
       endif
      enddo
c
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' gather is  OK'
         write(6,*)
         call ffflush(6)
      endif
c

      do j = 1,10 
       call ga_sync()
       if(me.eq.iran(nnodes())-1) then
         do loop = 1,m
           ilo = iran(n)
           jlo = iran(n)
           iv(loop) = ilo
           jv(loop) = jlo
c          v(loop) = DSIN(a(ilo,jlo)+b(ilo,jlo)) 
           v(loop) = 1d0 *(ilo+jlo)
         enddo 
         call ga_dscatter(g_a, v, iv, jv, m)
c        call ga_dgather(g_a, v, iv, jv, m)
         do loop = 1,m
           ilo= iv(loop)
           jlo= jv(loop) 
           call ga_get(g_a,ilo,ilo,jlo,jlo,w(loop),1)
c          if(v(loop)  .ne. w(loop))then 
           if(w(loop)  .ne. 1d0 *(ilo+jlo) )then 
             write(6,*)me,' scatter ', ilo,',',jlo,',',w(loop) 
     &             ,' ', 1d0 *(ilo+jlo) 
             call ffflush(6)
           endif
         enddo 
       endif
       call ga_sync()
      enddo
c
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' scatter is  OK'
         write(6,*)
      endif
c

c     
c     Delete the global arrays
c
      status = ga_destroy(g_b)
      status = ga_destroy(g_a)
c     
      end



      subroutine check_int()
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
c     
      integer n
      parameter (n = 128)
      integer a(n,n), b(n,n)
      logical status
      integer g_a, g_b
      integer iran, i, j, loop, nloop, ilo, ihi, jlo, jhi, itmp
      integer nproc, me, int, ij, inc, dimi,dimj,iproc, ii, jj
      double precision nwords
      parameter (nloop = 100)
      integer maxproc
      parameter (maxproc = 128)
      integer map(5,maxproc), found, np,k
      double precision drand, crap, sum1
      integer buf
      external drand
      intrinsic int
      iran(i) = int(drand(0)*dfloat(n)) + 1
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c     a() is a local copy of what the global array should start as
c
      do j = 1, n
         do i = 1, n
            a(i,j) = i-1 + (j-1)*1000
         enddo
      enddo
c     
c     Create a global array
c
      if (.not. ga_create(MT_INT, n, n, 'a', 1, 1, g_a)) then
         write(6,*) ' ga_create failed'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
c     
c     Zero the array
c
      if (me .eq. 0) then
         write(6,21) 
 21      format(/'> Checking zero ... ')
         call ffflush(6)
      endif
      call ga_zero(g_a)
c
c     Check that it is indeed zero
c
      call ga_get(g_a, 1, n, 1, n, b, n)
      do i = 1, n
         do j = 1, n
            if (b(i,j) .ne. 0) then
               write(6,*) ' zero ', me, i, j, b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_zero is OK'
         write(6,*)
      endif
      call ga_sync()
c
c     Each node fills in disjoint sections of the array
c
      if (me .eq. 0) then
         write(6,2) 
 2       format(/'> Checking disjoint put ... ')
         call ffflush(6)
      endif
      call ga_sync()
c
      inc = (n-1)/10 + 1
      ij = 0
      do j = 1, n, inc
         do i = 1, n, inc
            if (mod(ij,nproc) .eq. me) then
               ilo = i
               ihi = min(i+inc, n)
               jlo = j
               jhi = min(j+inc, n)
*               write(6,4) me, ilo, ihi, jlo, jhi
* 4             format(' node ',i2,' checking put ',4i4)
*               call ffflush(6)
               call ga_put(g_a, ilo, ihi, jlo, jhi, a(ilo, jlo), n)
            endif
            ij = ij + 1
         enddo
      enddo
      call ga_sync()
c
c     All nodes check all of a
c
      call ga_get(g_a, 1, n, 1, n, b, n)
      do i = 1, n
         do j = 1, n
            if (b(i,j) .ne. a(i,j)) then
               write(6,*) ' put ', me, i, j, a(i,j),b(i,j)
               call ffflush(6)
               call ga_error('... exiting ',0)
            endif
         enddo
      enddo
      call ga_sync()
c
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_put is OK'
         write(6,*)
      endif
c
c     Now check nloop random gets from each node
c
      if (me .eq. 0) then
         write(6,5) nloop
 5       format(/'> Checking random get (',i5,' calls)...')
         call ffflush(6)
      endif
      call ga_sync()
c
      nwords = 0
c
      crap = drand(nodeid()*51 + 1) ! Different seed for each process
      do loop = 1, nloop
         ilo = iran(loop)
         ihi = iran(loop)
         if (ihi.lt. ilo) then
            itmp = ihi
            ihi = ilo
            ilo = itmp
         endif
         jlo = iran(loop)
         jhi = iran(loop)
         if (jhi.lt. jlo) then
            itmp = jhi
            jhi = jlo
            jlo = itmp
         endif
c     
         nwords = nwords + (ihi-ilo+1)*(jhi-jlo+1)
c
         call ifill(n*n, 0.0d0, b, 1)
         call ga_get(g_a, ilo, ihi, jlo, jhi, b(ilo, jlo), n)
c
         if (me .eq. 0 .and. mod(loop-1, max(1,nloop/20)).eq.0) then
            write(6,1) loop, me, ilo, ihi, jlo, jhi, nwords
 1          format(' call ',i5, ' node ',i2,' checking get ',4i4,
     $           ' total ',d9.2)
            call ffflush(6)
         endif
c
         sum1 = 0.0d0
         do j = jlo, jhi
            do i = ilo, ihi
               sum1 = sum1 + b(i,j)
               if (b(i,j) .ne. a(i,j)) then
                  write(6,*) 'error ', i, j, b(i,j), a(i,j)
                  call ffflush(6)
                  call ga_error('... exiting ',0)
               endif
            enddo
         enddo
      enddo
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_get is OK'
         write(6,*)
      endif
c
      call ga_sync()
      if (me .eq. 0 .and. n.gt.7) then
         write(6,*)
         write(6,*) '> Checking ga_print_patch --- should see '
         write(6,*)' [2002 3002 4002 5002 6002]'  
         write(6,*)' [2003 3003 4003 5003 6003]'  
         write(6,*)' [2004 3004 4004 5004 6004]'  
         write(6,*)
         call ffflush(6)
      endif
      if(n.gt.5) call ga_print_patch(g_a,3,5,3,7,0)
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)
         write(6,*) '> Checking copy_patch ... '
         write(6,*)
         call ffflush(6)
      endif
      call ga_sync()
      if (.not. ga_create(MT_INT, n+1, n+1, 'b', 1, 1, g_b)) then
         write(6,*) ' ga_create failed for second array'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
      call ga_copy_patch(g_a,1,n/2,1,n/2, g_b,1,n/4,1,n) 
*     call ga_print(g_a,1,n/2,1,n/2)
*     call ga_print(g_b,1,n/4,1,n)
c
      do j = 1+me, n/2, ga_nnodes()
         call ga_get(g_b, 1, n/4, 2*j-1,2*j, b, n/4)
         do i = 1, n/2
            if(b(i,1) .ne. a(i,j))then
              print *,me,' error ', i, j, b(i,1), a(i,j)
              call ffflush(6)
            endif
         enddo
      enddo
c
c     copy back to the original shape
c
      call ga_copy_patch(g_b, 1,n/4,1,n, g_a,1,n/2,1,n/2) 
      do j = 1+me, n/2, ga_nnodes()
         call ga_get(g_a, 1, n/2, j,j, b, n/2)
         do i = 1, n/2
            if(b(i,1) .ne. a(i,j))then
              print *,me,' Error ', i, j, b(i,1), a(i,j)
              call ffflush(6)
            endif
         enddo
      enddo


      call ga_sync()
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_copy_patch is OK'
         write(6,*)
      endif
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)
         write(6,*) '> Checking read_inc ... '
         write(6,*)
         call ffflush(6)
      endif
      call ga_sync()
c
      crap = drand(nodeid()*51 + 1) ! Different seed for each process
      inc =5 
c     every processor will be operating on somebody elses data
c
      iproc = nnodes()-me-1
c
      call ga_distribution(g_a,iproc,ilo,ihi,jlo,jhi)
c
      dimi = ihi-ilo
      dimj = jhi-jlo
c     write(6,*) me,'..',ilo,ihi,jlo,jhi,'.',dimi,dimj
c     call ffflush(6)
      call ga_sync()
      if(ilo .gt.0 .and. jhi .gt. 0)then
       do loop = 1,nloop 
         ii= IABS(iran(dimi))
         jj= IABS(iran(dimj))
         i=ilo + Mod(ii,dimi)
         j=jlo + Mod(jj,dimj)
c
c        write(6,*) me,'..',ilo,ihi,jlo,jhi,'.',dimi,dimj,'..',i,j
c        call ffflush(6)
         buf = ga_read_inc(g_a,i,j,inc)
         if(a(i,j).ne. buf)then   
            write(6,*)me,'READ_inc ', i,',',j,',', a(i,j),' ',buf,me 
            call ffflush(6)
         endif
         call ga_get(g_a, i,i,j,j, buf,1) 
         a(i,j) = a(i,j)+inc 
         if(a(i,j).ne.  buf)then   
            write(6,*)me,'read_INC ', i,',',j,',', a(i,j),' ',buf,me
            call ffflush(6)
         endif
       enddo
      endif
      call ga_sync()
c
      if (me.eq.0) then
         write(6,*)
         write(6,*) ' read_inc is OK'
         write(6,*)
      endif
c
      if (me.eq.0) then
         write(6,*)
         write(6,*) '> checking ga_locate_region'
         write(6,*)
         call ffflush(6)
      endif

      call ga_locate_region(g_a, 1, n, 1,n, map,np)
      found = 0
      do j=1,n
         do i=1,n
            b(i,j)=-1
         enddo
      enddo
      if(me.eq.0)call ga_put(g_a,1,n,1,n,b,n)
      call ga_sync()
      do k = 1, np
         if(map(5,k).eq.me)then
                if(found.eq.1) then
                    write(6,*)'double entry in map for proc ',me
                    call ffflush(6)
                endif
                do j= map(3,k), map(4,k)
                        do i= map(1,k), map(2,k)
                        b(i,j)=1*me
                        enddo
                enddo
                call ga_put(g_a, map(1,k),map(2,k),map(3,k),map(4,k),
     &                 b(map(1,k),map(3,k)),n)
                found = 1
         endif
      enddo
      call ga_sync()
c
      do k = 1, np
         if(map(5,k).eq.me)then
                call ga_get(g_a, map(1,k),map(2,k),map(3,k),map(4,k),
     &                 a(map(1,k),map(3,k)),n)
                do j= map(3,k), map(4,k)
                      do i= map(1,k), map(2,k)
                        if(b(i,j).ne.a(i,j)) then
                           write(6,*)
     &                     'proc ',me, 'overlap with ',a(i,j)
                           call ffflush(6)
                        endif
                      enddo
                enddo
         endif
      enddo
      call ga_sync()
c
      if(me.eq.0)then

        call ga_get(g_a,1,n,1,n,a,n)
        do j=1,n
         do i=1,n
            if(a(i,j).eq.-1)then
               write(6,*)'i=',i,' j=',j, ' not assigned '
               call ga_error('... exiting ',0)
            endif
         enddo
        enddo
      endif

      if (me.eq.0) then
         write(6,*)
         write(6,*) ' ga_locate_region is OK'
         write(6,*)
         call ffflush(6)
      endif

c     
c     Delete the global array
c
      status = ga_destroy(g_a)
c     
      end

