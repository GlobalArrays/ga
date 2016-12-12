#if HAVE_CONFIG_H 
#   include "config.fh"
#endif
c $Id: nbtest.F,v 1.64.2.11 2007-04-06 22:37:35 d3g293 Exp $

#define BLOCK_DIM 200

      program main
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "testutil.fh"
      integer heap, stack, fudge, ma_heap, me, nproc
      integer ndim, dims(2), pdims(2), lo(2), hi(2), ld
      integer g_a, two, chunk(2)
      integer i, j, ii, jj, n, ibuf
      integer buf(BLOCK_DIM,BLOCK_DIM)
      integer nbbuf(BLOCK_DIM,BLOCK_DIM,2)
      integer nbhdl(2)
      GA_ACCESS_INDEX_TYPE g_ptr, ioff
      logical status, ok, ok2
      integer i_put, i_get, i_acc, i_nbput, i_nbget, i_nbacc
      integer i_putw, i_getw, i_accw
      double precision t_put, t_get, t_acc, t_nbput, t_nbget, t_nbacc
      double precision t_wput, t_wget, t_wacc, t_putw, t_getw, t_accw
      double precision tst
      parameter (heap=200*200*4, fudge=100, stack=200*200)
c     
c***  Intitialize a message passing library
c
#include "mp3.fh"
c
c***  Initialize GA
c
c     There are 2 choices: ga_initialize or ga_initialize_ltd.
c     In the first case, there is no explicit limit on memory usage.
c     In the second, user can set limit (per processor) in bytes.
c
      call ga_initialize()
      nproc = ga_nnodes()
      me = ga_nodeid()
c     we can also use GA_set_memory_limit BEFORE first ga_create call
c
      ma_heap = heap + fudge 
      ma_heap = 2*ma_heap
      if (me.eq.0) then
        print *, 'util_mdtob(', ma_heap, ')=', util_mdtob(ma_heap)
        print *
      endif
      call GA_set_memory_limit(util_mdtob(ma_heap))
c
      if(ga_nodeid().eq.0)then
         print *,'GA initialized on ',nproc,' processors'
         call ffflush(6)
      endif
c
c***  Initialize the MA package
c     MA must be initialized before any global array is allocated
c
      status = ma_init(MT_DCPL, stack, ma_heap)
      if (.not. status) call ga_error('ma_init failed',-1) 
c
c   Initialize all timing variables
c
      i_put = 0
      i_get = 0
      i_acc = 0
      i_nbput = 0
      i_nbget = 0
      i_nbacc = 0
      i_putw = 0
      i_getw = 0
      i_accw = 0
      t_put = 0.0d00
      t_get = 0.0d00
      t_acc = 0.0d00
      t_nbput = 0.0d00
      t_nbget = 0.0d00
      t_nbacc = 0.0d00
      t_wput = 0.0d00
      t_wget = 0.0d00
      t_wacc = 0.0d00
      t_putw = 0.0d00
      t_getw = 0.0d00
      t_accw = 0.0d00
c
c     Uncomment the below line to register external memory allocator
c     for dynamic arrays inside GA routines.
c      call register_ext_memory()
c
c   Factor processors into a roughly square grid
c
      call factor(nproc,pdims)
      if (me.eq.0) then
        write(6,*)
        write(6,'(a,i4,a,i4,a)') ' Processors configured on ',pdims(1),
     +              ' x ',pdims(2),' grid'
      endif
c
c   Create an array with each processor holding BLOCK_DIM*BLOCK_DIM
c   elements and initialize it to zero
c
      dims(1) = BLOCK_DIM*pdims(1)
      dims(2) = BLOCK_DIM*pdims(2)
      ld =  BLOCK_DIM
      g_a = ga_create_handle()
      two = 2
      call nga_set_data(g_a, two, dims, MT_INT)
      chunk(1) = BLOCK_DIM
      chunk(2) = BLOCK_DIM
      call nga_set_chunk(g_a, chunk)
      if (.not.ga_allocate(g_a)) then
        n = dims(1)*dims(2)
        write(6,'(a,i16)')
     +    'Failure to initialize global array of size ',n
        call ga_error('GA_Allocate failure',-1) 
      endif
      call ga_zero(g_a)
c
c   Initialize entire GA from process 0 using blocking puts
c
      if (me.eq.0) then
c
c   Loop over all processors
c
        do n = 0, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c
c   Fill buffer with correct values
c
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              buf(ii,jj) = (j-1)*dims(1)+(i-1)
            end do
          end do
c
c   Copy buffer to GA
c
          tst = ga_wtime()
          call nga_put(g_a,lo,hi,buf,ld)
          t_put = t_put + ga_wtime() - tst
          i_put = i_put + 1
        end do
      endif
      call ga_sync
c
c   Check values on each process for correctness
c
      ok = .true.
      call nga_distribution(g_a,me,lo,hi)
      call nga_access(g_a,lo,hi,g_ptr,ld)
      ioff = 0
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          n = int_mb(g_ptr+ioff)
          if (n.ne.(j-1)*dims(1)+(i-1)) ok = .false.
          ioff = ioff+1
        end do
      end do
      call nga_release(g_a, lo, hi)
c
      if (ok) then
        if (me.eq.0) then
          write(6,*)
          write(6,'(a)') 'OK for simple PUT test'
          write(6,*)
        endif
      else
        write(6,'(a,i4)') 'Error found in PUT test on process ',me
      endif
c
c  Collect all values on process 0 from GA
c
      ok = .true.
      if (me.eq.0) then
        do n = 0, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c          write(6,'(a,i4,a,i4,a,i4,a,i4,a,i4)') 'Accessing block (',
c     +      lo(1),',',hi(1),') (',lo(2),',',hi(2),') on process ',n
c
c   Copy contents of GA to local buffer
c
          tst = ga_wtime()
          call nga_get(g_a,lo,hi,buf,ld)
          t_get = t_get + ga_wtime() - tst
          i_get = i_get + 1
c
c   check buffer for correct values
c
          ok2 = .true.
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              if (buf(ii,jj).ne.(j-1)*dims(1)+(i-1)) ok2 = .false.
            end do
          end do
          if (.not.ok2) then
            write(6,'(a,i4)') 'Error found in GET test on process ',n
            ok = .false.
          endif
        end do
        if (ok) then
          if (me.eq.0) then
            write(6,*)
            write(6,'(a)') 'OK for simple GET test'
            write(6,*)
          endif
        endif
      endif
c
c   Double all values in the GA using accumulate
c
      if (me.eq.0) then
c
c   Loop over all processors
c
        do n = 0, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c
c   Fill buffer with correct values
c
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              buf(ii,jj) = (j-1)*dims(1)+(i-1)
            end do
          end do
c
c   Accumulate buffer to GA
c
          tst = ga_wtime()
          call nga_acc(g_a,lo,hi,buf,ld,1)
          t_acc = t_acc + ga_wtime() - tst
          i_acc = i_acc + 1
        end do
      endif
      call ga_sync
c
c   Check values on each process for correctness
c
      ok = .true.
      call nga_distribution(g_a,me,lo,hi)
      call nga_access(g_a,lo,hi,g_ptr,ld)
      ioff = 0
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          n = int_mb(g_ptr+ioff)
          if (n.ne.2*((j-1)*dims(1)+(i-1))) ok = .false.
          ioff = ioff+1
        end do
      end do
      call nga_release(g_a, lo, hi)
c
      if (ok) then
        if (me.eq.0) then
          write(6,*)
          write(6,'(a)') 'OK for simple ACC test'
          write(6,*)
        endif
      else
        write(6,'(a,i4)') 'Error found in ACC test on process ',me
      endif
c
c   Redo test using non-blocking puts, gets and accumulates
c
      call ga_zero(g_a)
      if (me.eq.0) then
c
c   Initialize puts by doing calculation for process 0
c
        tst = ga_wtime()
        lo(1) = 1
        lo(2) = 1
        hi(1) = BLOCK_DIM
        hi(2) = BLOCK_DIM
c
c   Fill first buffer with data
c
        do j = lo(2), hi(2)
          jj = j-lo(2)+1
          do i = lo(1), hi(1)
            ii = i-lo(1)+1
            nbbuf(ii,jj,1) = (j-1)*dims(1)+(i-1)
          end do
        end do
        t_putw = t_putw + ga_wtime() - tst
        i_putw = i_putw + 1
c
c   Start the first put
c
        tst = ga_wtime()
        call nga_nbput(g_a,lo,hi,nbbuf(1,1,1),ld,nbhdl(1))
        t_nbput = t_nbput + ga_wtime() - tst
        i_nbput = i_nbput + 1
c
c   Loop over all processors
c
        do n = 1, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          tst = ga_wtime()
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c
c   Fill buffer with correct values
c
          ibuf = mod(n,2)+1
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              nbbuf(ii,jj,ibuf) = (j-1)*dims(1)+(i-1)
            end do
          end do
          t_putw = t_putw + ga_wtime() - tst
          i_putw = i_putw + 1
c
c   Copy buffer to GA
c
          tst = ga_wtime()
          call nga_nbput(g_a,lo,hi,nbbuf(1,1,ibuf),ld,nbhdl(ibuf))
          t_nbput = t_nbput + ga_wtime() - tst
          i_nbput = i_nbput + 1
          tst = ga_wtime()
          call nga_nbwait(nbhdl(mod(n-1,2)+1))
          t_wput = t_wput + ga_wtime() - tst
        end do
        tst = ga_wtime()
        call nga_nbwait(nbhdl(mod(nproc-1,2)+1))
        t_wput = t_wput + ga_wtime() - tst
      endif
      call ga_sync
c
c   Check values on each process for correctness
c
      ok = .true.
      call nga_distribution(g_a,me,lo,hi)
      call nga_access(g_a,lo,hi,g_ptr,ld)
      ioff = 0
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          n = int_mb(g_ptr+ioff)
          if (n.ne.(j-1)*dims(1)+(i-1)) ok = .false.
          ioff = ioff+1
        end do
      end do
      call nga_release(g_a, lo, hi)
c
      if (ok) then
        if (me.eq.0) then
          write(6,*)
          write(6,'(a)') 'OK for Non-Blocking PUT test'
          write(6,*)
        endif
      else
        write(6,'(a,i4)')
     +      'Error found in Non-Blocking PUT test on process ',me
      endif
c
c  Collect all values on process 0 from GA using non-blocking calls
c
      ok = .true.
      if (me.eq.0) then
c
c  Start with get on process 0
c
        lo(1) = 1
        lo(2) = 1
        hi(1) = BLOCK_DIM
        hi(2) = BLOCK_DIM
c
c   Fill first buffer with data
c
        do j = lo(2), hi(2)
          jj = j-lo(2)+1
          do i = lo(1), hi(1)
            ii = i-lo(1)+1
            nbbuf(ii,jj,1) = (j-1)*dims(1)+(i-1)
          end do
        end do
c
c   Start the first get
c
        tst = ga_wtime()
        call nga_nbget(g_a,lo,hi,nbbuf(1,1,1),ld,nbhdl(1))
        t_nbget = t_nbget + ga_wtime() - tst
        i_nbget = i_nbget + 1
        do n = 1, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c          write(6,'(a,i4,a,i4,a,i4,a,i4,a,i4)') 'Accessing block (',
c     +      lo(1),',',hi(1),') (',lo(2),',',hi(2),') on process ',n
c
c   Copy contents of GA to local buffer
c
          ibuf = mod(n,2)+1
          tst = ga_wtime()
          call nga_nbget(g_a,lo,hi,nbbuf(1,1,ibuf),ld,nbhdl(ibuf))
          t_nbget = t_nbget + ga_wtime() - tst
          i_nbget = i_nbget + 1
          tst = ga_wtime()
          call nga_nbwait(nbhdl(mod(n-1,2)+1))
          t_wget = t_wget + ga_wtime() - tst
c
c   check buffer for correct values
c
          tst = ga_wtime()
          ok2 = .true.
          ibuf = mod(n-1,2)+1
          ii = mod(n-1,pdims(1))
          jj = (n-1-ii)/pdims(1)
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              if (nbbuf(ii,jj,ibuf).ne.(j-1)*dims(1)+(i-1))
     +             ok2 = .false.
            end do
          end do
          if (.not.ok2) then
            write(6,'(a,i4)') 'Error found in GET test on process ',n
            ok = .false.
          endif
          t_getw = t_getw + ga_wtime() - tst
          i_getw = i_getw + 1
        end do
        tst = ga_wtime()
        call nga_nbwait(nbhdl(mod(n-1,2)+1))
        t_wget = t_wget + ga_wtime() - tst
        tst = ga_wtime()
        ok2 = .true.
        ibuf = mod(nproc-1,2)+1
        ii = mod(n-1,pdims(1))
        jj = (n-1-ii)/pdims(1)
        lo(1) = ii*BLOCK_DIM + 1
        lo(2) = jj*BLOCK_DIM + 1
        hi(1) = (ii+1)*BLOCK_DIM
        hi(2) = (jj+1)*BLOCK_DIM
        do j = lo(2), hi(2)
          jj = j-lo(2)+1
          do i = lo(1), hi(1)
            ii = i-lo(1)+1
            if (nbbuf(ii,jj,ibuf).ne.(j-1)*dims(1)+(i-1)) ok2 = .false.
          end do
        end do
        if (.not.ok2) then
          write(6,'(a,i4)') 'Error found in GET test on process ',n
          ok = .false.
        endif
        if (ok) then
          if (me.eq.0) then
            write(6,*)
            write(6,'(a)') 'OK for Non-Blocking GET test'
            write(6,*)
          endif
        endif
        t_getw = t_getw + ga_wtime() - tst
        i_getw = i_getw + 1
      endif
c
c   Double all values in the GA using accumulate
c
      if (me.eq.0) then
c
c   Initialize puts by doing calculation for process 0
c
        tst = ga_wtime()
        lo(1) = 1
        lo(2) = 1
        hi(1) = BLOCK_DIM
        hi(2) = BLOCK_DIM
c
c   Fill first buffer with data
c
        do j = lo(2), hi(2)
          jj = j-lo(2)+1
          do i = lo(1), hi(1)
            ii = i-lo(1)+1
            nbbuf(ii,jj,1) = (j-1)*dims(1)+(i-1)
          end do
        end do
        t_accw = t_accw + ga_wtime() - tst
        i_accw = i_accw + 1
c
c   Start the first put
c
        tst = ga_wtime()
        call nga_nbacc(g_a,lo,hi,nbbuf(1,1,1),ld,1,nbhdl(1))
        t_nbacc = t_nbacc + ga_wtime() - tst
        i_nbacc = i_nbacc + 1
c
c   Loop over all processors
c
        do n = 1, nproc-1
c
c   Calculate process grid coordinates
c
          ii = mod(n,pdims(1))
          jj = (n-ii)/pdims(1)
c
c   Calculate lower and upper indices of the block held by process n
c
          tst = ga_wtime()
          lo(1) = ii*BLOCK_DIM + 1
          lo(2) = jj*BLOCK_DIM + 1
          hi(1) = (ii+1)*BLOCK_DIM
          hi(2) = (jj+1)*BLOCK_DIM
c
c   Fill buffer with correct values
c
          ibuf = mod(n,2)+1
          do j = lo(2), hi(2)
            jj = j-lo(2)+1
            do i = lo(1), hi(1)
              ii = i-lo(1)+1
              nbbuf(ii,jj,ibuf) = (j-1)*dims(1)+(i-1)
            end do
          end do
          t_accw = t_accw + ga_wtime() - tst
          i_accw = i_accw + 1
c
c   Copy buffer to GA
c
          tst = ga_wtime()
          call nga_nbacc(g_a,lo,hi,nbbuf(1,1,ibuf),ld,1,nbhdl(ibuf))
          t_nbacc = t_nbacc + ga_wtime() - tst
          i_nbacc = i_nbacc + 1
          tst = ga_wtime()
          call nga_nbwait(nbhdl(mod(n-1,2)+1))
          t_wacc = t_wacc + ga_wtime() - tst
        end do
        tst = ga_wtime()
        call nga_nbwait(nbhdl(mod(nproc-1,2)+1))
        t_wacc = t_wacc + ga_wtime() - tst
      endif
      call ga_sync
c
c   Check values on each process for correctness
c
      ok = .true.
      call nga_distribution(g_a,me,lo,hi)
      call nga_access(g_a,lo,hi,g_ptr,ld)
      ioff = 0
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          n = int_mb(g_ptr+ioff)
          if (n.ne.2*((j-1)*dims(1)+(i-1))) ok = .false.
          ioff = ioff+1
        end do
      end do
      call nga_release(g_a, lo, hi)
c
      if (ok) then
        if (me.eq.0) then
          write(6,*)
          write(6,'(a)') 'OK for Non-Blocking ACC test'
          write(6,*)
        endif
      else
        write(6,'(a,i4)')
     +      'Error found in Non-Blocking ACC test on process ',me
      endif
c
c   Print out timing numbers. Start by summing timers over all
c   processors
c
      call ga_igop(1,i_put,1,'+')
      call ga_igop(2,i_get,1,'+')
      call ga_igop(3,i_acc,1,'+')
      call ga_igop(4,i_nbput,1,'+')
      call ga_igop(5,i_nbget,1,'+')
      call ga_igop(6,i_nbacc,1,'+')
      call ga_igop(7,i_putw,1,'+')
      call ga_igop(8,i_getw,1,'+')
      call ga_igop(9,i_accw,1,'+')
      call ga_dgop(1,t_put,1,'+')
      call ga_dgop(2,t_get,1,'+')
      call ga_dgop(3,t_acc,1,'+')
      call ga_dgop(4,t_nbput,1,'+')
      call ga_dgop(5,t_nbget,1,'+')
      call ga_dgop(6,t_nbacc,1,'+')
      call ga_dgop(7,t_wput,1,'+')
      call ga_dgop(8,t_wget,1,'+')
      call ga_dgop(9,t_wacc,1,'+')
      call ga_dgop(1,t_putw,1,'+')
      call ga_dgop(2,t_getw,1,'+')
      call ga_dgop(3,t_accw,1,'+')
      if (me.eq.0) then
        write(6,'(a,f16.8)') 'Average time for blocking PUT:      ',
     +                     t_put/dble(i_put)
        write(6,'(a,f16.8)') 'Average time for blocking GET:      ',
     +                     t_get/dble(i_get)
        write(6,'(a,f16.8)') 'Average time for blocking ACC:      ',
     +                     t_acc/dble(i_acc)
        write(6,'(a,f16.8)') 'Average time for non-blocking PUT:  ',
     +                     t_nbput/dble(i_nbput)
        write(6,'(a,f16.8)') 'Average time for non-blocking GET:  ',
     +                     t_nbget/dble(i_nbget)
        write(6,'(a,f16.8)') 'Average time for non-blocking ACC:  ',
     +                     t_nbacc/dble(i_nbacc)
        write(6,'(a,f16.8)') 'Average time for wait on PUT:       ',
     +                     t_wput/dble(i_nbput)
        write(6,'(a,f16.8)') 'Average time for wait on GET:       ',
     +                     t_wget/dble(i_nbget)
        write(6,'(a,f16.8)') 'Average time for wait on ACC:       ',
     +                     t_wacc/dble(i_nbacc)
        write(6,'(a,f16.8)') 'Aggregate time on non-blocking PUT: ',
     +                     (t_nbput+t_wput)/dble(i_nbput)
        write(6,'(a,f16.8)') 'Aggregate time on non-blocking GET: ',
     +                     (t_nbget+t_wget)/dble(i_nbget)
        write(6,'(a,f16.8)') 'Aggregate time on non-blocking ACC: ',
     +                     (t_nbacc+t_wacc)/dble(i_nbacc)
        write(6,'(a,f16.8)') 'Average time in PUT work:           ',
     +                     t_putw/dble(i_putw)
        write(6,'(a,f16.8)') 'Average time in GET work:           ',
     +                     t_getw/dble(i_getw)
        write(6,'(a,f16.8)') 'Average time in ACC work:           ',
     +                     t_accw/dble(i_accw)
      endif
      
c
c***  Tidy up the GA package
c
      call ga_terminate()
c
c***  Tidy up after message-passing library
c
      call MP_FINALIZE()
c
      end
c
c  Obtain a 2x2 processor grid for p processors
c
      subroutine factor(p,dims)
      implicit none
      integer i,j,p,dims(2),imin,mdim
      integer ip,ifac,pmax,prime(1000)
      integer fac(1000)
c
      i = 1
      ip = p
      do i = 1, 2
        dims(i) = 1
      end do
c
c    factor p completely
c    first, find all prime numbers less than or equal to p
c
      pmax = 0
      do i = 2, p
        do j = 1, pmax
          if (mod(i,prime(j)).eq.0) go to 100
        end do
        pmax = pmax + 1
        prime(pmax) = i
  100   continue
      end do
c
c    find all prime factors of p
c
      ifac = 0
      do i = 1, pmax
  200   if (mod(ip,prime(i)).eq.0) then
          ifac = ifac + 1
          fac(ifac) = prime(i)
          ip = ip/prime(i)
          go to 200
        endif
      end do
c
c    determine dimensions of processor grid
c
      do i = ifac, 1, -1
c
c    find dimension with minimum value
c
        imin = dims(1)
        mdim = 1
        if (dims(2).lt.imin) then
          imin = dims(2)
          mdim = 2
        endif
        dims(mdim) = dims(mdim)*fac(i)
      end do
c
      return
      end

