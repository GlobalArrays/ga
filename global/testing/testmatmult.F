      program ga_test
c $Id: testmatmult.F,v 1.1 2003-02-03 19:17:06 edo Exp $
c
c      test ga_dgemm performance with N 400,1600,3200
c      compile with make FLD_REN="optimized blas" testmatmult.x
c
      implicit none
      integer num,nummax,howmany
      parameter (nummax=3200,howmany=3)
      integer num1
      parameter(num1=100)
      integer i,ii
      real*8 h0(num1*num1)
      integer ihandle,ih_vec,ih_scr
      real*8 a
      real*8 t1,t2,t3
      integer itime,ntimes,nums(howmany)
      character*1 xn,xt
      data xn,xt/'n','t'/
      data nums/400,1600,3200/
#include "mafdecls.fh"
#include "global.fh"
#include "testutil.fh"
      call PBEGINF
      if (.not.ma_init(MT_DBL,1,20000000)) then
         call ga_error('failed: ma_init(MT_DBL,1,20000000)',10)
      endif
      call ga_initialize()
c
      do ii=1,howmany
      num=nums(ii)
      a = 0.5d0/(num*num)
      do i = 1, num1*num1
         h0(i) = a
      enddo
c
c
      if (.not.ga_create(MT_DBL,num,num,'ihandle',0,0,ihandle)) then
         call ga_error('failed: create ihandle',20)
      endif
      if (.not.ga_create(MT_DBL,num,num,'ih_vec',0,0,ih_vec)) then
         call ga_error('failed: create ih_vec',30)
      endif
      if (.not.ga_create(MT_DBL,num,num,'ih_scr',0,0,ih_scr)) then
         call ga_error('failed: create ih_scr',40)
      endif
c
c     Compute times assuming 500 mflops and 5 second target time
c
!      ntimes = max(5.0d0,5.0d0/(4.0d-9*num**3))
      ntimes = max(3.0d0,5.0d0/(4.0d-9*num**3))
c
         call load_ga_from_square(ih_vec,num,h0,num1)
         call load_ga_from_square(ihandle,num,h0,num1)
         t1 = util_timer()
         do itime = 1, ntimes
         call ga_dgemm(xt,xn,num,num,num,1.0d0,
     &        ih_vec, ihandle, 0.0d0, ih_scr)
         enddo
         t2 = util_timer()
         if (ga_nodeid().eq.0) then
            write(6,600)ntimes,t2-t1,
     .    2d0*ntimes*(num)**3d0/(t2-t1)*1d-6/ga_nnodes(),
     .    xt,xn,num
         endif
         t1 = util_timer()
         do itime = 1, ntimes
            call ga_dgemm(xn,xn,num,num,num,1.0d0,
     &        ih_scr, ih_vec, 0.0d0, ihandle)
         enddo
         t2 = util_timer()
         if (ga_nodeid().eq.0) then
            write(6,600)ntimes,t2-t1,
     .    2d0*ntimes*num**3d0/(t2-t1)*1d-6/ga_nnodes(),
     .    xn,xn,num
         endif
      if (.not.ga_destroy(ihandle)) then
         call ga_error('failed: destroy ihandle',20)
      endif
      if (.not.ga_destroy(ih_vec)) then
         call ga_error('failed: destroy ih_vec',30)
      endif
      if (.not.ga_destroy(ih_scr)) then
         call ga_error('failed: destroy ih_scr',40)
      endif
      enddo

600   format('*** ga_dgemm: ntimes ',i3,' time ',f8.2,
     . ' mflop/proc',f10.1,
     ,2a2,' N= ',i6)
c
c
      call ga_terminate
      call pend
c
      end
c
c-----------------------------------------------------------------------
c
      subroutine load_ga_from_square(handle,num,f,ndim)
      implicit none
      integer handle, memhandle
      integer num,ndim
      real*8 f(ndim,ndim)
      integer ilo, ihi, jlo, jhi, nx, ny, ibuff
      integer ga_nodeid, i1, i2, i, j, ix, jx
#include "mafdecls.fh"

      call ga_distribution(handle, ga_nodeid(), ilo, ihi, jlo, jhi)

      if(ihi.le.0)return
      if(jhi.le.0)return

c     nx = ihi - ilo + 1
c     ny = jhi - jlo + 1

      do i = ilo,ihi,ndim
         do j = jlo,jhi,ndim
            call ga_put(handle,i,min(ihi,i+ndim),j,min(jhi,j+ndim),
     &                    f,ndim)
         enddo
      enddo

      return
      end
c
c-----------------------------------------------------------------------
c
      subroutine load_ga_from_triangle(handle,f,ndim)
      implicit none
      integer handle, memhandle
      real*8 f(*)
      integer ndim
      integer ilo, ihi, jlo, jhi, nx, ny, ibuff
      integer ga_nodeid, i1, i2, i, j, ix, jx
#include "mafdecls.fh"

      call ga_distribution(handle, ga_nodeid(), ilo, ihi, jlo, jhi)

      if(ihi.le.0)return
      if(jhi.le.0)return

      nx = ihi - ilo + 1
      ny = jhi - jlo + 1

      if (.not.ma_alloc_get(MT_DBL,nx*ny,'flap',memhandle,ibuff)) then
         call ga_error('failed: allocate triangle',100)
      endif

      do i = 1,nx
         do j = 1,ny
            ix = i + ilo - 1
            jx = j + jlo - 1
            i1 = min(ix,jx)
            i2 = max(ix,jx)
            dbl_mb(ibuff + nx*(j-1) + (i-1) ) = f(i2*(i2-1)/2 + i1)
         enddo
      enddo

      call ga_put(handle,ilo,ihi,jlo,jhi,
     &              dbl_mb(ibuff),nx)

      if (.not.ma_free_heap(memhandle)) then
         call ga_error('failed: free triangle',100)
      endif

      return
      end
