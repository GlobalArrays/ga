c This program should run only in the network of MPPs (IWAY) environment 
c
c
      program test
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      logical status
      call pbeginf
      status = ma_init(MT_DBL, 50000, 50000/nnodes())
      if (.not. status) print *, 'ma_init failed'
      call ga_initialize()
      if(ga_nodeid().eq.0)then
         print *,' GA initialized'
         call ffflush(6)
      endif
 
      call merge_test()
 
      call ga_terminate()
      call pend
      end



      subroutine merge_test()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
      integer n,m
      parameter (n = 200)
      parameter (m = n*n)
      double precision a(n,n), buf(n)
      integer nproc, me 
      integer i, j, ilo, ihi, jlo, jhi
      integer g_a, g_b
      integer trial, max_trial, cluster, eps
      logical status 
c
      max_trial = 10
      me = ga_nodeid()
      nproc = ga_nnodes()
      if( me.eq.ga_net_nodeid())then
        cluster = 0
      else
        cluster = 1
      endif
      eps = n/max_trial
c
      do j = 1, n
         do i = 1, n
            a(i,j) = i-1 + (j-1)*n
         enddo
      enddo
c
c***  Create a global arrays
c
      if (me .eq. 0) then
         print *,ga_nodeid(), ' Creating Arrays'
         call ffflush(6)
      endif
      status = ga_create(MT_DBL, n, n, 'a', 1, 1, g_a)
      if (.not. status) then
         write(6,*) ' A: ga_create failed'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
      status = ga_create(MT_DBL, n, n, 'b', 1, 1, g_b)
      if (.not. status) then
         write(6,*) ' B: ga_create failed'
         call ffflush(6)
         call ga_error('... exiting ',0)
      endif
c
      call ga_zero(g_a)
c
c
c***
      do trial = 1, max_trial
        ilo = 1
        ihi = n
        jlo = 1
        jhi = n

c = 1   
        if(trial .eq.1) then
           do j = 1+ga_net_nodeid(), n, ga_net_nnodes()
             call ga_put(g_a,1,n,j,j,a(1,j),n)
           enddo
c
           if (me .eq. 0) then
             write(6,100)trial, ilo,ihi,jlo,jhi
100          format(i2,'> column merging [',i3,':',i3,',',
     $                 i3,':',i3,'] ')
             call ffflush(6)
           endif
        else
c > 2
          ilo = 1
          ihi = trial*eps 
          jlo = 1 
          jhi = trial*eps 
          if(cluster.eq.0)then
             call ga_dfill_patch(g_a, ilo,ihi,jlo,jhi,0d0)
             ilo = 1
             ihi = n
             jlo = 1
             jhi = n
          else
             call ga_zero(g_a)
             call ga_copy_patch('n', g_b,ilo,ihi,jlo,jhi,
     $                          g_a,ilo,ihi,jlo,jhi)
          endif
c
          if (me .eq. 0) then
             write(6,101)trial, ilo,ihi,jlo,jhi
101          format(i2,'>  patch merging [',i3,':',i3,',',
     $                 i3,':',i3,'] ')
            call ffflush(6)
          endif

        endif
c
        call ga_net_merge(g_a, ilo, ihi, jlo, jhi)
c       call ga_print(g_a,1)
c
c***    check if correct
        do j = 1+ga_nodeid(), n, ga_nnodes()
             call ga_get(g_a,1,n,j,j,buf,n)
             do i = 1, n
                if(a(i,j) .ne. buf(i)) then
                   print *, me, '(',i,j,')', a(i,j), buf(i)
                   call ga_error("merging failed",j)
                endif
             enddo
        enddo
        if (me .eq. 0) then
          write(6,*)'  OK '
          call ffflush(6)
        endif
c
        if(trial .eq.1) then
c         g_b keeps reference data for next trials
          call ga_copy(g_a,g_b)
        endif
      enddo
c
      status = ga_destroy(g_a)
c
      end
