      program test
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
      integer heap, stack, fudge
      logical status
      parameter (heap=200*200*2, fudge=10240, stack=0)
c
c     Intitialize the TCGMSG package
c
      call pbeginf
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
      call patch()
c
c     Tidy up the global array package
c
      call ga_terminate()
c
c
c     Tidy up after TCGMSG
c
      call pend()
      end

      subroutine patch()
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "tcgmsg.fh"
c
      integer n,m
      parameter (n = 128)
      parameter (m = n*n)
      double precision a(n,n), b(n,n), c(n,n), buf(m), drand, val
      double precision alpha, beta
      integer nproc, me 
      integer i, j, ailo, ajlo, bilo, bjlo, base, iran
      integer aihi, ajhi, bihi, bjhi
      integer g_a, g_b, g_c 
      integer rows, cols, loop
      logical status, dist_same 
      iran(i) = int(drand(0)*dfloat(i)) + 1
      dist_same = .false.
c
      me = ga_nodeid()
      nproc = ga_nnodes()
c
      do j = 1, n
         do i = 1, n
            a(i,j) = i-1 + (j-1)*n
            b(i,j) = i+j 
         enddo
      enddo
c
c***  Create a global array
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
c
#ifdef CRAY_T3D
      if(dist_same) then
#else
      if(dist_same.eq. .true.) then
#endif
         status = ga_duplicate(g_a, g_b, 'a_duplicated')
         if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0) 
         status = ga_duplicate(g_a, g_c, 'a_duplicated_again')
         if(.not.ga_compare_distr(g_a, g_c))
     $           call ga_error("g_c distribution different",0) 
      else
         status = ga_create(MT_DBL, n, n, 'b', 1, n, g_b)
         if (.not. status) call ga_error('ga_create failed:b',0) 
         status = ga_create(MT_DBL, n, n, 'c', n, 1, g_c)
         if (.not. status) call ga_error('ga_create failed:c',0) 
      endif
c
c***
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_dfill_patch ... '
         call ffflush(6)
      endif
c
      val = 1d0
      call ga_dfill_patch(g_a, 2,n/2, 2,n, val)
*     call ga_print(g_a, 1)
      do j = 2+me, n, nproc 
         call ga_get(g_a, 1,n/2, j,j, buf,n/2) 
         do i = 2, n/2
            if(buf(i) .ne.val ) then
               print *,me, ' error ',i,j, buf(i),val 
               call ga_error('exiting ...',0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)'   OK '
         call ffflush(6)
      endif
c  
      do j = 1+me, n, nproc 
         call ga_put(g_a,1,n,j,j,a(1,j),n) 
      enddo
c***
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_copy_patch ... '
         call ffflush(6)
      endif
c
      do loop =1, 10
              ailo = iran(n/2)
              ajlo = iran(n/2)
              aihi = min(n, -1+ailo+n/2)
              ajhi = min(n, -1+ajlo+n/4)
              rows = aihi -ailo+1
              cols = ajhi -ajlo +1
c
              bilo = iran(n/3)
              bjlo = iran(n/3)
              bihi = bilo + rows -1
              bjhi = bjlo + cols -1
              if (me .eq. 0) then
                write(6,'(2x,1h[,4i4,1h],5h --> ,1h[,4i4,1h])')
     $                ailo,aihi,ajlo,ajhi,
     $                bilo,bihi, bjlo, bjhi
 
                call ffflush(6)
              endif

c
              call ga_copy_patch('n', g_a, ailo, aihi, ajlo, ajhi,
     &                                g_b, bilo, bihi, bjlo, bjhi)
c             call ga_print(g_a,1)
c             call ga_print(g_b,1)
              call ga_get(g_b,bilo,bihi, bjlo, bjhi, buf, rows)
              base = 0
              do j = ajlo, ajhi
                 if(Mod(j,nproc).eq.me) then
                    do i = ailo, aihi
                       base = base+1
                       if(buf(base) .ne. a(i,j)) then
                          print *,me, ' error ',i,j, buf(base), a(i,j)
                          call ga_error('exiting ...',0)
                       endif
                    enddo
                 else
                    base = base + rows
                 endif
              enddo
      enddo
c
      ailo = iran(n/2)
      ajlo = iran(n/2)
      bilo = iran(n/2)
      bjlo = iran(n/2)
c
      if (me .eq. 0) then
                write(6,'(2x,1h[,4i4,1h],5h --> ,1h[,4i4,1h])')
     $                ailo,aihi,ajlo,ajhi,
     $                bilo,bihi, bjlo, bjhi
      endif

      call ga_copy_patch('n', g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                        g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print_patch(g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,1)
*     call ga_print(g_b, 1)
      call ga_get(g_b,bilo,bilo+n/3,bjlo, bjlo+n/2,buf,n/3+1) 
      base = 0
      do j = ajlo, ajlo+n/3 
         if(Mod(j,nproc).eq.me) then
            do i = ailo, ailo+n/2 
               base = base+1
               if(buf(base) .ne. a(i,j)) then
                  print *,me, ' error ',i,j, buf(base), a(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
      call ga_sync()
c
      if (me .eq. 0) then
         write(6,*)'  without transpose:   OK '
         call ffflush(6)
      endif
c 
      if (me .eq. 0) then
                write(6,'(2x,1h[,4i4,2h]~,5h --> ,1h[,4i4,1h])')
     $                ailo,aihi,ajlo,ajhi,
     $                bilo,bihi, bjlo, bjhi
      endif
      call ga_copy_patch('t', g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                        g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print_patch(g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,1)
*     call ga_print(g_b, 1)
      call ga_get(g_b,bilo,bilo+n/3,bjlo, bjlo+n/2,buf,n/3+1)
      base = 0
      do i = ailo, ailo+n/2
         if(Mod(j,nproc).eq.me) then
         do j = ajlo, ajlo+n/3
               base = base+1
               if(buf(base) .ne. a(i,j)) then
                  print *,me, ' error ',i,j, buf(base), a(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/3+1
         endif
      enddo
      call ga_sync()
c
      if (me .eq. 0) then
         write(6,*)'  transposed:   OK '
         call ffflush(6)
      endif
c
c***
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_dscal_patch ... '
         call ffflush(6)
      endif
      call ga_copy_patch('n', g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                        g_b, bilo,bilo+n/2, bjlo, bjlo+n/3)
      val = 1.d0
      call ga_dscal_patch(g_b, bilo,bilo+n/2,bjlo, bjlo+n/3,val)
      call ga_get(g_b,bilo,bilo+n/2,bjlo, bjlo+n/3,buf,n/2+1)
      base = 0
      do j = ajlo, ajlo+n/3
         if(Mod(j,nproc).eq.me) then
            do i = ailo, ailo+n/2
               base = base+1
               if(buf(base) .ne. a(i,j)*val) then
                  print *,me, ' error ',i,j, buf(base), a(i,j)*val
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
      call ga_sync()
c
      if (me .eq. 0) then
         write(6,*)'  OK '
         call ffflush(6)
      endif
c
c***
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_dadd_patch ... '
         call ffflush(6)
      endif
      alpha = .1d0
      beta = .2d0
      call ga_zero(g_c)
      call ga_dadd_patch(alpha, g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                    beta, g_b, bilo,bilo+n/2, bjlo, bjlo+n/3,
     $                          g_c, bilo,bilo+n/2, bjlo, bjlo+n/3)
      call ga_get(g_c,bilo,bilo+n/2,bjlo, bjlo+n/3,buf,n/2+1)
      base = 0
      val = val*beta + alpha
      do j = ajlo, ajlo+n/3
         if(Mod(j,nproc).eq.me) then
            do i = ailo, ailo+n/2
               base = base+1
               if(ABS(buf(base)- a(i,j)*val).gt.1d-5) then
                  print *,me, ' error ',i,j, buf(base), a(i,j)*val
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
      call ga_sync()
c
      if (me .eq. 0) then
         write(6,*)'  OK '
         call ffflush(6)
      endif
c
c***
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_ddot_patch ... '
         call ffflush(6)
      endif
      alpha= ga_ddot_patch(g_a,'n', ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                     g_c,'n', bilo,bilo+n/2, bjlo, bjlo+n/3)
      beta = 0d0
      do j = ajlo, ajlo+n/3
            do i = ailo, ailo+n/2
               beta = beta + a(i,j)**2
            enddo
      enddo
      if(ABS(beta*val- alpha).gt.1d-6*alpha) then
             print *,me, ' error ', beta*val, alpha
             call ga_error('exiting ...',0)
      endif
      call ga_sync()
c
      if (me .eq. 0) then
         write(6,*)'  OK '
         call ffflush(6)
      endif
c
c......................................................
      if (me .eq. 0) then
         print *, ' '
         write(6,*)'> Checking ga_matmul_patch ... '
         call ffflush(6)
      endif
      do j = 1+me, n, nproc
         call ga_put(g_b,1,n,j,j,b(1,j),n)
      enddo
      call ga_sync()
      call ga_matmul_patch('n','n', 1d0, 0d0, 
     $                      g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                      g_b, bilo,bilo+n/3, bjlo, bjlo+n/2,
     $                      g_c, bilo,bilo+n/2, bjlo, bjlo+n/2)
      call dgemm('n','n',n/2+1,n/2+1,n/3+1,1d0,a(ailo,ajlo), n,
     $            b(bilo,bjlo),n, 0d0, c, n)
*     call ga_print_patch(g_a, ailo,ailo+n/2, ajlo, ajlo+n/3)
*     call ga_print_patch(g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print(g_c, 1)
      call ga_get(g_c,bilo,bilo+n/2,bjlo, bjlo+n/2,buf,n/2+1)
      base = 0
      do j = 1, 1+n/2
         if(Mod(j,nproc).eq.me) then
            do i = 1, 1+n/2
               base = base+1
               if(ABS(buf(base)- c(i,j)).gt.1d-8) then
                  print *,me, ' error ',i,j, buf(base), c(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)'  a*b: OK '
         call ffflush(6)
      endif
c
      call ga_sync()
      call ga_matmul_patch('t','n', 1d0, 0d0,
     $                      g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                      g_b, bilo,bilo+n/3, bjlo, bjlo+n/2,
     $                      g_c, bilo,bilo+n/2, bjlo, bjlo+n/2)
      call dgemm('t','n',n/2+1,n/2+1,n/3+1,1d0,a(ajlo,ailo), n,
     $            b(bilo,bjlo),n, 0d0, c, n)
*     call ga_print(g_a,1) 
*     call ga_print_patch(g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print(g_c, 1)
      call ga_get(g_c,bilo,bilo+n/2,bjlo, bjlo+n/2,buf,n/2+1)
      base = 0
      do j = 1, 1+n/2
         if(Mod(j,nproc).eq.me) then
            do i = 1, 1+n/2
               base = base+1
               if(ABS(buf(base)- c(i,j)).gt.1d-8) then
                  print *,me, ' error ',i,j, buf(base), c(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)'  trans(a)*b: OK '
         call ffflush(6)
      endif
c
      call ga_sync()
      call ga_matmul_patch('n','t', 1d0, 0d0,
     $                      g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                      g_b, bilo,bilo+n/3, bjlo, bjlo+n/2,
     $                      g_c, bilo,bilo+n/2, bjlo, bjlo+n/2)
      call dgemm('n','t',n/2+1,n/2+1,n/3+1,1d0,a(ailo,ajlo), n,
     $            b(bjlo,bilo),n, 0d0, c, n)
*     call ga_print(g_a,1)
*     call ga_print_patch(g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print(g_c, 1)
      call ga_get(g_c,bilo,bilo+n/2,bjlo, bjlo+n/2,buf,n/2+1)
      base = 0
      do j = 1, 1+n/2
         if(Mod(j,nproc).eq.me) then
            do i = 1, 1+n/2
               base = base+1
               if(ABS(buf(base)- c(i,j)).gt.1d-8) then
                  print *,me, ' error ',i,j, buf(base), c(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)'  a*trans(b): OK '
         call ffflush(6)
      endif
c
      call ga_sync()
      call ga_matmul_patch('t','t', 1d0, 0d0,
     $                      g_a, ailo,ailo+n/2, ajlo, ajlo+n/3,
     $                      g_b, bilo,bilo+n/3, bjlo, bjlo+n/2,
     $                      g_c, bilo,bilo+n/2, bjlo, bjlo+n/2)
      call dgemm('t','t',n/2+1,n/2+1,n/3+1,1d0,a(ajlo,ailo), n,
     $            b(bjlo,bilo),n, 0d0, c, n)
*     call ga_print(g_a,1)
*     call ga_print_patch(g_b, bilo,bilo+n/3, bjlo, bjlo+n/2)
*     call ga_print(g_c, 1)
      call ga_get(g_c,bilo,bilo+n/2,bjlo, bjlo+n/2,buf,n/2+1)
      base = 0
      do j = 1, 1+n/2
         if(Mod(j,nproc).eq.me) then
            do i = 1, 1+n/2
               base = base+1
               if(ABS(buf(base)- c(i,j)).gt.1d-8) then
                  print *,me, ' error ',i,j, buf(base), c(i,j)
                  call ga_error('exiting ...',0)
               endif
            enddo
         else
            base = base +n/2+1
         endif
      enddo
c
      call ga_sync()
      if (me .eq. 0) then
         write(6,*)'  trans(a)*trans(b): OK '
         call ffflush(6)
      endif
c
      end
