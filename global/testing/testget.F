      program test
c***
c***  Last modification: Tue Oct  4 13:27:10 PDT 1994
c***
C$Id: testget.F,v 1.3 1995-02-02 23:14:28 d3g681 Exp $
      implicit none
#include "mafdecls.h"
#include "global.fh"
      integer heap
c     
c***  Intitialize the TCGMSG package
c
      call pbeginf
c
c     Initialize the MA package
c
      heap = 200000
      if (.not. ma_init(MT_DBL, heap,heap)) call parerr(1)
c
c***  Intitialize the GA package
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
c
      integer n, nn, num_chunks
      parameter (n = 710, nn = n*n, num_chunks=12)
      double precision buf(nn)
c
      integer g_a
      integer ilo, ihi, jlo, jhi
      integer nproc, me, loop
      integer chunk(num_chunks)
      data    chunk /1,3,4,9,16,30,64,91,128,171,256,353/
c     
      nproc = ga_nnodes()
      me = ga_nodeid()
c
c***  Create global array
      if (.not. ga_create(MT_DBL, n, n, 'a', 1, 1, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      if (me .eq. 0) then
         write(6,*)' preparing input data n=  ',n
         call flush(6)
      endif
      call ga_zero(g_a) 
c
      if (me .eq. 0) then
        print *,' '
        print *,'> Performance of ga_get, ga_put & ga_acc '
        print *,' '
        call flush(6)
      endif
c
      do loop=1,3
c
c***  local ops
c
c     everybody touches own data 
      call ga_dfill_patch(g_a, 1, n, 1, n , 1d0*me)
c
      call ga_distribution(g_a, me, ilo, ihi, jlo, jhi)
      call TestPutGetAcc
     &     (g_a, n, chunk, num_chunks, buf, ilo, ihi, jlo, jhi)
c
c***  remote ops
c
c     everybody touches own data
      call ga_dfill_patch(g_a, 1, n, 1, n , 1d0*me)
c
      call ga_distribution(g_a, nproc-1, ilo, ihi, jlo, jhi)
      call TestPutGetAcc
     &     (g_a, n, chunk, num_chunks, buf, ilo, ihi, jlo, jhi)

      enddo
      end


      subroutine TestPutGetAcc
     &          (g_a, n, chunk, num_chunks, buf, ilo, ihi, jlo,jhi)
      implicit none
#include "tcgmsg.fh"
#include "global.fh"
      integer num_chunks, chunk(num_chunks)
      integer n, ilo, ihi, jlo,jhi,g_a
      double precision buf(*), tg, tp, ta
      double precision time_acc, time_get, time_put
c
      integer me
      integer loop, jump, count, bytes
c
      me = nodeid()
      if (me .eq. 0) then
        write(*,*)' '
        write(*,*)'                    ACCESS ',
     &            '[',ilo,':',ihi,', ',jlo,':',jhi,']'
        write(*,*), 'bytes  loop         get                    put',
     &           '                 accumulate'
        call flush(6)
      endif
      call ga_sync()
c
      do loop = 1, num_chunks
c
        call ga_dfill_patch(g_a, 1, n, 1, n , 1d0*me)
        if (me .eq. 0) then
c
c         find how much data is accessed
          bytes = mdtob(1)*chunk(loop)*chunk(loop)
c
          jump  =  n/(40*loop)
          tg= time_get(g_a,ilo,ihi,jlo,jhi,buf,chunk(loop),jump,count)
          tp= time_put(g_a,ilo,ihi,jlo,jhi,buf,chunk(loop),jump,count)
          ta= time_acc(g_a,ilo,ihi,jlo,jhi,buf,chunk(loop),jump,count)
c
          write(6,77)bytes, count, tg, 1d-6*bytes/tg,
     &               tp, 1d-6*bytes/tp, ta, 1d-6*bytes/ta
          call flush(6)
        else
          call sleep(2)
        endif
        call ga_sync()
      enddo
c
77    format(i6, i5, 3(d10.3, d9.2,'MB/s'))
      end



      double precision function
     &       time_acc(g_a, is, ie, js, je, buf, chunk, jump, count)
c
      implicit none
      integer g_a, chunk, jump, count, is, js, ie, je
c
      integer ilo, ihi, jlo, jhi
      double precision tcgtime, seconds, buf 
c
      count = 0
      seconds = tcgtime()
        do ilo = is, ie -chunk-jump +1, chunk+jump
           ihi = ilo + chunk -1
           do jlo = js, je -chunk-jump +1, chunk+jump
              jhi = jlo + chunk -1
              count = count + 1
              call ga_acc(g_a, ilo, ihi, jlo, jhi, buf, chunk, 1d0)
           enddo
        enddo
      seconds = tcgtime() - seconds
c
      time_acc = seconds/count
      end


      double precision function
     &       time_get(g_a, is, ie, js, je, buf, chunk, jump, count)
c
      implicit none
      integer g_a, chunk, jump, count, is, js, ie, je
c
      integer ilo, ihi, jlo, jhi
      double precision tcgtime, seconds, buf
c
      count = 0
      seconds = tcgtime()
        do ilo = is, ie -chunk-jump +1, chunk+jump
           ihi = ilo + chunk -1
           do jlo = js, je -chunk-jump +1, chunk+jump
              jhi = jlo + chunk -1
              count = count + 1
              call ga_get(g_a, ilo, ihi, jlo, jhi, buf, chunk)
           enddo
        enddo
      seconds = tcgtime() - seconds
c
      time_get = seconds/count
      end



      double precision function
     &       time_put(g_a, is, ie, js, je, buf, chunk, jump, count)
c
      implicit none
      integer g_a, chunk, jump, count, is, js, ie, je
c
      integer ilo, ihi, jlo, jhi
      double precision tcgtime, seconds, buf
c
      count = 0
      seconds = tcgtime()
        do ilo = is, ie -chunk-jump +1, chunk+jump
           ihi = ilo + chunk -1
           do jlo = js, je -chunk-jump +1, chunk+jump
              jhi = jlo + chunk -1
              count = count + 1
              call ga_put(g_a, ilo, ihi, jlo, jhi, buf, chunk)
           enddo
        enddo
      seconds = tcgtime() - seconds
c
      time_put = seconds/count
      end

