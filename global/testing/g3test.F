c vector boxes lack arithmetic precision 
#ifdef CRAY_YMP
# define THRESH 1d-10
#elif defined(FUJITSU)
# define THRESH 1d-12
# define THRESHF 1e-7
#else
# define THRESH 1d-13
# define THRESHF 1e-5
#endif
#define MISMATCH(x,y) abs(x-y)/max(1d0,abs(x)).gt.THRESH
#define MISMATCHF(x,y) abs(x-y)/max(1.0,abs(x)).gt.THRESHF


      program main
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "testutil.fh"
#include "tcgmsg.fh"
      integer heap, stack, fudge, ma_heap, me
      integer nmax, DIM, nwidth, MAXPROC, nloop
      parameter (nmax = 20, DIM = 3, nwidth = 2, MAXPROC = 2000)
      parameter (nloop = 100)
      integer ndim, nproc, pdims(7), type, dcnt, g_a, maxval
      integer i, j, k, dims(7), width(7), map(2*nmax)
      integer lo(7), hi(7), ld(7)
      integer lo2(7), hi2(7), ld2(7)
      integer dims3(7), ld3(7), index3, chunk(7)
      integer a(nmax, nmax, nmax)
      integer b(nmax+2*nwidth,nmax+2*nwidth,nmax+2*nwidth)
      double precision start,finish,start1,finish1
      double precision t1,t2,t3,t4,t5,t6,t7,tmp
      logical status, safe_put, safe_get, has_data(0:MAXPROC-1)
      parameter (heap=30*30*30*4, fudge=100, stack=100*100)
c     
c***  Intitialize a message passing library
c
#ifdef MPI
      integer ierr
      call mpi_init(ierr)
#else
      call pbeginf
#endif
c
c***  Initialize GA
c
c     There are 2 choices: ga_initialize or ga_initialize_ltd.
c     In the first case, there is no explicit limit on memory usage.
c     In the second, user can set limit (per processor) in bytes.
c
      print*
      call ga_initialize()
      nproc = ga_nnodes()
      me = ga_nodeid()
c     we can also use GA_set_memory_limit BEFORE first ga_create call
c
      ma_heap = heap + fudge 
      call GA_set_memory_limit(util_mdtob(ma_heap))
c
      if(ga_nodeid().eq.0)then
         print *,' GA initialized '
         call ffflush(6)
      endif
c
c***  Initialize the MA package
c     MA must be initialized before any global array is allocated
c
      status = ma_init(MT_DCPL, stack, ma_heap)
      if (.not. status) call ga_error('ma_init failed',-1) 
c
      if(me.eq.0)then
        print *, 'using ', nproc, ' process(es)'
        call ffflush(6)
      endif
c
c   Test ghost distributions
c
      ndim = DIM
c
c   Create irregular distribution on all nodes
c
      call factor(nproc,ndim,pdims)
      dims(1) = pdims(1) * nmax
      dims(2) = pdims(2) * nmax
      dims(3) = pdims(3) * nmax
      maxval = 1
      do i = 1, ndim
        maxval = dims(i)*maxval
      end do
      maxval = maxval - 1
c
      dcnt = 1
      do i = 1, pdims(1)
        map(dcnt) = (i-1)*nmax + 1
        dcnt = dcnt + 1
      end do
      do i = 1, pdims(2)
        map(dcnt) = (i-1)*nmax + 1
        dcnt = dcnt + 1
      end do
      do i = 1, pdims(3)
        map(dcnt) = (i-1)*nmax + 1
        dcnt = dcnt + 1
      end do
c
      do i = 1, ndim
        width(i) = nwidth
        chunk(i) = 1
        if (pdims(i).gt.dims(i)) pdims(i) = dims(i)
        if (me.eq.0) then
          write(6,*) 'Value of pdims(',i,') is ',pdims(i)
        endif
        ld(i) = nmax
        call ffflush(6)
      end do
      if (me.eq.0) then
        do i = 1, dcnt - 1
          write(6,'("map(",i2,") = ",i5)') i,map(i)
          call ffflush(6)
        end do
      endif

      type = MT_INT
      status = nga_create_ghosts_irreg (type, ndim, dims, width,
     +        "test_array", map, pdims, g_a)
      if (status.and.me.eq.0) then
        write(6,*) '*'
        write(6,*) '* Global array creation was successful'
        write(6,*) '*'
      elseif (.not.status) then
        write(6,*) 'Global array creation failure on ',me
      endif
c
c   Find processors that have data
c
      call ga_sync
      do i = 0, nproc-1
        call nga_distribution(g_a, i, lo, hi)
        has_data(i) = .true.
        do j = 1, ndim
          if (lo(j).eq.0.and.hi(j).eq.-1) has_data(i) = .false.
        end do
        if (me.eq.i) then
          write(6,*) '*'
          write(6,*) '* Distribution on processor ',i
          write(6,*) '*'
          write(6,110) lo(1), hi(1)
          write(6,110) lo(2), hi(2)
          write(6,110) lo(3), hi(3)
  110     format(2i10)
        endif
        call ffflush(6)
        call ga_sync
      end do
c
c     initialize g_a
c
      call ga_sync
      call nga_distribution(g_a, me, lo, hi)
      do i = 1, hi(1) - lo(1) + 1
        do j = 1, hi(2) - lo(2) + 1
          do k = 1, hi(3) - lo(3) + 1
            a(i,j,k) = (i + lo(1) - 2)*dims(1)*dims(2)
     +               + (j + lo(2) - 2)*dims(1)
     +               + (k + lo(3) - 2)
          end do
        end do
      end do
      safe_put = .true.
      do i = 1, ndim
        if (hi(i).lt.lo(i)) safe_put = .false.
      end do
      if (has_data(me).and.safe_put) call nga_put(g_a, lo, hi, a, ld)
c
      t1 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        call ga_update_ghosts(g_a)
        finish = util_timer()
        t1 = t1 + finish - start
      end do
      t1 = t1/dble(nloop)

      t4 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        status = ga_update2_ghosts(g_a)
        finish = util_timer()
        t4 = t4 + finish - start
      end do
      t4 = t4/dble(nloop)

      t5 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        status = ga_update3_ghosts(g_a)
        finish = util_timer()
        t5 = t5 + finish - start
      end do
      t5 = t5/dble(nloop)

      t6 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        status = ga_update4_ghosts(g_a)
        finish = util_timer()
        t6 = t6 + finish - start
      end do
      t6 = t6/dble(nloop)

      t7 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        status = ga_update5_ghosts(g_a)
        finish = util_timer()
        t7 = t7 + finish - start
      end do
      t7 = t7/dble(nloop)

      if (me.eq.0) then
        write(6,*) '*'
        write(6,*) '*   Completed updates successfully'
        write(6,*) '*'
        call ffflush(6)
      endif
c
c     get patch with ghost cells
c
      do i = 1, ndim
        lo2(i) = lo(i) - width(i)
        hi2(i) = hi(i) + width(i)
        ld2(i) = ld(i) + 2*width(i)
      end do
      call ga_sync
      safe_get = .true.

      t2 = 0.0d00
      t3 = 0.0d00
      do i = 1, nloop
        start = util_timer()
        call ga_sync
        start1 = util_timer()
        if (has_data(me).and.safe_get)
     +    call nga_periodic_get(g_a, lo2, hi2, b, ld2)
        finish1 = util_timer()
        call ga_sync
        finish = util_timer()
        t2 = t2 + finish1 - start1
        t3 = t3 + finish - start
      end do
      t2 = t2/dble(nloop)
      t3 = t3/dble(nloop)

      if (me.eq.0) then
        write(6,*) '*'
        write(6,*) '*   Performing nga_access_ghosts'
        write(6,*) '*'
        call ffflush(6)
      endif
      if (has_data(me)) call nga_access_ghosts(g_a, dims3,
     +    index3, ld3)
      call ga_sync
      call atest(int_mb(index3),dims3,ld3(1),b,nmax+2*nwidth,has_data)
      call ga_sync
      tmp = t1
      call ga_dgop(1,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_update_ghosts ',tmp
      endif
      tmp = t1
      call ga_dgop(2,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_update_ghosts ',tmp
      endif
      tmp = t1
      call ga_dgop(3,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_update_ghosts ',tmp/dble(nproc)
      endif
      tmp = t4
      call ga_dgop(1,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_update2_ghosts ',tmp
      endif
      tmp = t4
      call ga_dgop(2,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_update2_ghosts ',tmp
      endif
      tmp = t4
      call ga_dgop(3,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_update2_ghosts ',
     +              tmp/dble(nproc)
      endif
      tmp = t5
      call ga_dgop(1,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_update3_ghosts ',tmp
      endif
      tmp = t5
      call ga_dgop(2,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_update3_ghosts ',tmp
      endif
      tmp = t5
      call ga_dgop(3,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_update3_ghosts ',
     +              tmp/dble(nproc)
      endif
      tmp = t6
      call ga_dgop(1,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_update4_ghosts ',tmp
      endif
      tmp = t6
      call ga_dgop(2,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_update4_ghosts ',tmp
      endif
      tmp = t6
      call ga_dgop(3,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_update4_ghosts ',
     +              tmp/dble(nproc)
      endif
      tmp = t7
      call ga_dgop(1,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_update5_ghosts ',tmp
      endif
      tmp = t7
      call ga_dgop(2,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_update5_ghosts ',tmp
      endif
      tmp = t7
      call ga_dgop(3,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_update5_ghosts ',
     +              tmp/dble(nproc)
      endif
      tmp = t2
      call ga_dgop(4,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for nga_periodic_get ',tmp
      endif
      tmp = t2
      call ga_dgop(5,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for nga_periodic_get ',tmp
      endif
      tmp = t2
      call ga_dgop(6,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for nga_periodic_get ',tmp/dble(nproc)
      endif
      tmp = t3
      call ga_dgop(4,tmp,1,'max')
      if (me.eq.0) then
        write(6,*) 'Maximum time for (sync)nga_periodic_get ',tmp
      endif
      tmp = t3
      call ga_dgop(5,tmp,1,'min')
      if (me.eq.0) then
        write(6,*) 'Minimum time for (sync)nga_periodic_get ',tmp
      endif
      tmp = t3
      call ga_dgop(6,tmp,1,'+')
      if (me.eq.0) then
        write(6,*) 'Average time for (sync)nga_periodic_get ',
     +              tmp/dble(nproc)
      endif
  127 continue
c
c***  Tidy up the GA package
c
      call ga_terminate()
c
c***  Tidy up after message-passing library
c
#ifdef MPI
      call mpi_finalize(ierr)
#else
      call pend()
#endif

c
      stop
      end
c
      subroutine atest(a,dim,ld,b,ld2,has_data)
#include "global.fh"
      integer ld,ld2
      integer dim(7)
      integer a(ld,ld,*), b(ld2,ld2,*)
      integer i, j, k, l, nproc
      logical has_data(0:1999), check_data

      nproc = ga_nnodes()
      check_data = .true.
      do l = 1, nproc
        call ga_sync
        if (l-1.eq.ga_nodeid().and.has_data(l-1)) then
          do i = 1, dim(1)
            do j = 1, dim(2)
              do k = 1, dim(3)
                if (a(i,j,k).ne.b(i,j,k)) check_data = .false.
              end do
            end do
          end do
          if (check_data) then
            write(6,*) '*'
            write(6,*) '*   Data from nga_access_ghosts and'
            write(6,*) '*   nga_periodic_get is the same on'
            write(6,*) '*   processor ',l-1
            write(6,*) '*'
          else
            write(6,*) '*'
            write(6,*) '*   Data from nga_access_ghosts and'
            write(6,*) '*   nga_periodic_get is NOT the same on'
            write(6,*) '*   processor ',l-1
            write(6,*) '*'
          endif
        endif
        call ffflush(6)
  101   format(10i5)
      enddo
c
      return
      end
c
      subroutine factor(p,ndim,dims)
      implicit none
      integer i,j,p,ndim,dims(7),imin,mdim
      integer ip,ifac,pmax,prime(1000)
      integer fac(1000)
c
      i = 1
      ip = p
      do i = 1, ndim
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
        do j = 2, ndim
          if (dims(j).lt.imin) then
            imin = dims(j)
            mdim = j
          endif
        end do
        dims(mdim) = dims(mdim)*fac(i)
      end do
c
      return
      end
