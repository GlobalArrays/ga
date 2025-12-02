

#if HAVE_CONFIG_H
#   include "config.fh"
#endif
#define  MAXLOOP 100

      program test
      implicit none
#include "mafdecls.fh"
#include "global.fh"
      integer stack, heap
c
c***  Intitialize a message passing library
c
#include "mp3.fh"
c
c***  Intitialize the GA package
      call ga_initialize()
c     if(ga_nodeid().eq.0)print *,ga_nnodes(),' nodes'
c
      if(ga_uses_ma()) then
         stack = 200000
c         stack = 100000
      else
         stack = 60000
c         stack = 20000
      endif
      heap = stack
      if (.not. ma_init(MT_DBL, heap, stack))
     $    call ga_error("ma init failed",heap+stack)
c
c
c     test GA_FILL
      
c
c     test NGA_GET
      
c
c     test NGA_PUT
      
c
c     test NGA_ACC
      
c
c     test NGA_PERIODIC_GET
      
c
c     test NGA_PERIODIC_PUT
      
c
c     test NGA_PERIODIC_ACC
      
c
c     test NGA_FILL_PATCH
      
c
c     test NGA_COPY_PATCH
      
c
c     test NGA_SCALE_PATCH
      
c
c     test NGA_ADD_PATCH
      
c
c     test NGA_DOT_PATCH
      
c
c     test NGA_SCATTER
      
      call testit_NGA_SCATTER_int1()
	    
      call testit_NGA_SCATTER_dbl1()
	    
      call testit_NGA_SCATTER_dcpl1()
      
      call testit_NGA_SCATTER_int2()
	    
      call testit_NGA_SCATTER_dbl2()
	    
      call testit_NGA_SCATTER_dcpl2()
      
      call testit_NGA_SCATTER_int3()
	    
      call testit_NGA_SCATTER_dbl3()
	    
      call testit_NGA_SCATTER_dcpl3()
      
      call testit_NGA_SCATTER_int4()
	    
      call testit_NGA_SCATTER_dbl4()
	    
      call testit_NGA_SCATTER_dcpl4()
      
      call testit_NGA_SCATTER_int5()
	    
      call testit_NGA_SCATTER_dbl5()
	    
      call testit_NGA_SCATTER_dcpl5()
      
      call testit_NGA_SCATTER_int6()
	    
      call testit_NGA_SCATTER_dbl6()
	    
      call testit_NGA_SCATTER_dcpl6()
      
      call testit_NGA_SCATTER_int7()
	    
      call testit_NGA_SCATTER_dbl7()
	    
      call testit_NGA_SCATTER_dcpl7()
      
c
c     test NGA_SCATTER_ACC
      
      call testit_NGA_SCATTER_ACC_int1()
            
      call testit_NGA_SCATTER_ACC_dbl1()
            
      call testit_NGA_SCATTER_ACC_dcpl1()
      
      call testit_NGA_SCATTER_ACC_int2()
            
      call testit_NGA_SCATTER_ACC_dbl2()
            
      call testit_NGA_SCATTER_ACC_dcpl2()
      
      call testit_NGA_SCATTER_ACC_int3()
            
      call testit_NGA_SCATTER_ACC_dbl3()
            
      call testit_NGA_SCATTER_ACC_dcpl3()
      
      call testit_NGA_SCATTER_ACC_int4()
            
      call testit_NGA_SCATTER_ACC_dbl4()
            
      call testit_NGA_SCATTER_ACC_dcpl4()
      
      call testit_NGA_SCATTER_ACC_int5()
            
      call testit_NGA_SCATTER_ACC_dbl5()
            
      call testit_NGA_SCATTER_ACC_dcpl5()
      
      call testit_NGA_SCATTER_ACC_int6()
            
      call testit_NGA_SCATTER_ACC_dbl6()
            
      call testit_NGA_SCATTER_ACC_dcpl6()
      
      call testit_NGA_SCATTER_ACC_int7()
            
      call testit_NGA_SCATTER_ACC_dbl7()
            
      call testit_NGA_SCATTER_ACC_dcpl7()
      
c
c     test NGA_GATHER
      


c
      if(ga_nodeid().eq.0) print *, 'All tests successful'
c
      call ga_terminate()
      call MP_FINALIZE()
      end

c-----------------

      
         
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      integer a(n)
      integer b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(a(lo(1)) .ne.
     $           b(lo(1))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      integer a(n)
      integer b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
            a(d(1,i))=
     $           a(d(1,i))
     $           *alpha
     $           +b(lo(1))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(abs(a(lo(1)) -
     $           b(lo(1))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1)),
     $              'b=',
     $              b(lo(1)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      integer a(n,n)
      integer b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(a(lo(1),lo(2)) .ne.
     $           b(lo(1),lo(2))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      integer a(n,n)
      integer b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
            a(d(1,i),d(2,i))=
     $           a(d(1,i),d(2,i))
     $           *alpha
     $           +b(lo(1),lo(2))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2)) -
     $           b(lo(1),lo(2))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2)),
     $              'b=',
     $              b(lo(1),lo(2)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      integer a(n,n,n)
      integer b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3)) .ne.
     $           b(lo(1),lo(2),lo(3))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      integer a(n,n,n)
      integer b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
            a(d(1,i),d(2,i),d(3,i))=
     $           a(d(1,i),d(2,i),d(3,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3)) -
     $           b(lo(1),lo(2),lo(3))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      integer a(n,n,n,n)
      integer b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      integer a(n,n,n,n)
      integer b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4)) -
     $           b(lo(1),lo(2),lo(3),lo(4))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      integer a(n,n,n,n,n)
      integer b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      integer a(n,n,n,n,n)
      integer b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      integer a(n,n,n,n,n,n)
      integer b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      integer a(n,n,n,n,n,n)
      integer b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_int7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      integer a(n,n,n,n,n,n,n)
      integer b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_int7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      integer a(n,n,n,n,n,n,n)
      integer b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      integer v(m)
      integer d(ndim, m)
      double precision drand
      integer alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=int(drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = int(drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
      
         
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      double precision a(n)
      double precision b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(a(lo(1)) .ne.
     $           b(lo(1))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      double precision a(n)
      double precision b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
            a(d(1,i))=
     $           a(d(1,i))
     $           *alpha
     $           +b(lo(1))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(abs(a(lo(1)) -
     $           b(lo(1))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1)),
     $              'b=',
     $              b(lo(1)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      double precision a(n,n)
      double precision b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(a(lo(1),lo(2)) .ne.
     $           b(lo(1),lo(2))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      double precision a(n,n)
      double precision b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
            a(d(1,i),d(2,i))=
     $           a(d(1,i),d(2,i))
     $           *alpha
     $           +b(lo(1),lo(2))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2)) -
     $           b(lo(1),lo(2))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2)),
     $              'b=',
     $              b(lo(1),lo(2)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      double precision a(n,n,n)
      double precision b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3)) .ne.
     $           b(lo(1),lo(2),lo(3))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      double precision a(n,n,n)
      double precision b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
            a(d(1,i),d(2,i),d(3,i))=
     $           a(d(1,i),d(2,i),d(3,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3)) -
     $           b(lo(1),lo(2),lo(3))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      double precision a(n,n,n,n)
      double precision b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      double precision a(n,n,n,n)
      double precision b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4)) -
     $           b(lo(1),lo(2),lo(3),lo(4))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      double precision a(n,n,n,n,n)
      double precision b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      double precision a(n,n,n,n,n)
      double precision b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      double precision a(n,n,n,n,n,n)
      double precision b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      double precision a(n,n,n,n,n,n)
      double precision b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dbl7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      double precision a(n,n,n,n,n,n,n)
      double precision b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dbl7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      double precision a(n,n,n,n,n,n,n)
      double precision b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double precision v(m)
      integer d(ndim, m)
      double precision drand
      double precision alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=drand(0) * me*2+1 * 2
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = drand(0) *  * 2
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
      
         
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      double complex a(n)
      double complex b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(a(lo(1)) .ne.
     $           b(lo(1))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl1()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 20000)
      parameter (m = (20000**1)/100)
      parameter (ndim = 1)
      double complex a(n)
      double complex b(n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
            a(d(1,i))=
     $           a(d(1,i))
     $           *alpha
     $           +b(lo(1))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1)),ld)
c
c           compare the results
            if(abs(a(lo(1)) -
     $           b(lo(1))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1)),
     $              'b=',
     $              b(lo(1)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      double complex a(n,n)
      double complex b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(a(lo(1),lo(2)) .ne.
     $           b(lo(1),lo(2))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl2()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 140)
      parameter (m = (140**2)/100)
      parameter (ndim = 2)
      double complex a(n,n)
      double complex b(n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
            a(d(1,i),d(2,i))=
     $           a(d(1,i),d(2,i))
     $           *alpha
     $           +b(lo(1),lo(2))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2)) -
     $           b(lo(1),lo(2))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2)),
     $              'b=',
     $              b(lo(1),lo(2)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      double complex a(n,n,n)
      double complex b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3)) .ne.
     $           b(lo(1),lo(2),lo(3))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl3()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 27)
      parameter (m = (27**3)/100)
      parameter (ndim = 3)
      double complex a(n,n,n)
      double complex b(n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
            a(d(1,i),d(2,i),d(3,i))=
     $           a(d(1,i),d(2,i),d(3,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3)) -
     $           b(lo(1),lo(2),lo(3))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      double complex a(n,n,n,n)
      double complex b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl4()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 11)
      parameter (m = (11**4)/100)
      parameter (ndim = 4)
      double complex a(n,n,n,n)
      double complex b(n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4)) -
     $           b(lo(1),lo(2),lo(3),lo(4))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      double complex a(n,n,n,n,n)
      double complex b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl5()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 7)
      parameter (m = (7**5)/100)
      parameter (ndim = 5)
      double complex a(n,n,n,n,n)
      double complex b(n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      double complex a(n,n,n,n,n,n)
      double complex b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl6()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 5)
      parameter (m = (5**6)/100)
      parameter (ndim = 6)
      double complex a(n,n,n,n,n,n)
      double complex b(n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
            
	    
	    
	    
	    
      subroutine testit_NGA_SCATTER_dcpl7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      double complex a(n,n,n,n,n,n,n)
      double complex b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c------------------------------- NGA_SCATTER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
               do j=1,ndim
                  d(j,i) = int(drand(0)*real(n)) + 1
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        scatter the v to the global array
         call nga_scatter(g_a, v, d, m)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) .ne.
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) then
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
      subroutine testit_NGA_SCATTER_ACC_dcpl7()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c     
      integer n,m
      integer ndim
      parameter (n = 4)
      parameter (m = (4**7)/100)
      parameter (ndim = 7)
      double complex a(n,n,n,n,n,n,n)
      double complex b(n,n,n,n,n,n,n)
      integer lo(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop, idx
      double complex v(m)
      integer d(ndim, m)
      double precision drand
      double complex alpha
      integer unique,unique_index
      integer nproc, me
      logical status
c     
      nproc = ga_nnodes()
      me    = ga_nodeid()
c     
c---------------------- initialize the GA -----------------------
c     initialize the chunk, dims, ld, and calculate the number 
c     of elements
      total=1
      do i = 1,ndim
         chunk(i) = 0
         dims(i) = n
         ld(i) = n
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c-------------------------- NGA_SCATTER_ACC -----------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scatter_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      alpha=dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
c
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array and the array containing values
         do i = 1, m
c           generate indices
            unique_index = 0
            do while(unique_index.eq.0)
c              generate random number
               idx = int(drand(0)*(real(total/nproc)))
     $             +me*total/nproc
               if(idx .eq. 0) idx = idx+1
c
c              convert to ndim
               call conv_1ton(ndim,dims,idx,lo)
c
               do j=1,ndim
                  d(j,i) = lo(j)
               enddo
c              
               unique_index = unique(d,ndim,m,i)
            enddo
c
            v(i) = dcmplx(drand(0) *  * 2,
     $    -drand(0) *  * 2)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=v(i)
         enddo
c
c        the result should be
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
            a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))=
     $           a(d(1,i),d(2,i),d(3,i),d(4,i),d(5,i),d(6,i),d(7,i))
     $           *alpha
     $           +b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))
         enddo
c
         call ga_sync()
c     
c        scatter the v to the global array
         call nga_scatter_acc(g_a, v, d, m, alpha)
         call ga_sync()
c
c        collect each elements and compare
         do i = 1, m
            do j=1, ndim
               lo(j) = d(j,i)
            enddo
            call nga_get(g_a,lo,lo,
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
c           compare the results
            if(abs(a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)) -
     $           b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7))) 
     $           .gt. 1d-5) then
               print *,'a=',
     $              a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'b=',
     $              b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),
     $              'i=',i,'alpha=',alpha,'v=',v(i)
               call ga_error('bye', 0)
            endif
         enddo
      enddo
c
      call ga_sync()
      if(me.eq.0)then
         print *, 'OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
      status= ga_destroy(g_a)
      end

	    
	    
	    
	    
	    
	    
	    
	    
	    
c-----------------------
c 	Utility functions

      subroutine random_range(lo,hi,lop,hip,ndim)
      implicit none
      integer lo(1),hi(1),lop(1),hip(1),ndim
      integer i, range, swap, val,iran
      double precision drand
      iran(range) = int(drand(0)*dble(range)) + 1
c     iran(range) = range/2 
      do i = 1, ndim
         range = hi(i)-lo(i)+1
         val = iran(range)
         lop(i) = lo(i) + val 
         val = iran(range)
         hip(i) = hi(i) - val
         if(hip(i) .lt. lop(i))then
            swap =hip(i)
            hip(i)=lop(i)
            lop(i)=swap
         endif
         hip(i)=MIN(hip(i),hi(i))
         lop(i)=MAX(lop(i),lo(i))
      enddo
      end
c

c
      subroutine random_range_outbound(lo,hi,lop,hip,ndim)
      implicit none
      integer lo(1),hi(1),lop(1),hip(1),ndim
      integer i, range, val,iran
      double precision drand
      iran(range) = int(drand(0)*dble(range)) + 1
c     iran(range) = range/2
      do i = 1, ndim
         range = 2*(hi(i)-lo(i)+1)
         val = iran(range)
         lop(i) = lo(i) + val
         range = hi(i)-lo(i)+1
         val = iran(range)
         hip(i) = lop(i) + range - val
c
         hip(i) = hip(i)-hi(i)
         lop(i) = lop(i)-hi(i)
      enddo
      end
c
c
      integer function count_elems(lo,hi,ndim)
      implicit none
      integer lo(1),hi(1),ndim,elems,i
      elems=1
      do i=1,ndim
         elems = elems*(hi(i)-lo(i)+1)
      enddo
      count_elems = elems
      end
c

c     get the next available nindex in the range of lo and hi
      integer function next_index(ind,total,ndim,lo,hi,dims)
      implicit none
      integer ind,total,ndim,lo(ndim),hi(ndim),dims(ndim)
      integer i
      integer indx(8),nindex
c
      nindex = ind + 1
 200  call conv_1ton(ndim,dims,nindex,indx)
c
c     test if indx(i) is in the range of lo(i) and hi(i)
      do i=1,ndim
         if((indx(i).lt.lo(i)).or.(indx(i).gt.hi(i))) then
            nindex = nindex + 1
            if(nindex.gt.total) then
               next_index = 0
               goto 300
            else
               goto 200
            endif
         endif
      enddo
c
      next_index = nindex
 300  end

c     testing if the indices are unique
      integer function unique(ind,ndim,m,n)
      implicit none
      integer ndim,m,n
      integer ind(ndim,m)
      integer i,j,marker
c
      unique = 1
      do i = 1, n-1
         marker = 0
         do j = 1, ndim
            if(ind(j,n).eq.ind(j,i)) marker = marker + 1
         enddo
c     
         if(marker.eq.ndim) unique = 0
      enddo
c
      end

c
      subroutine prnt_rng(me,lo,hi,ndim)
      implicit none
      integer me,ndim
      integer lo(ndim),hi(ndim)
      integer i
c
      print *, me,': array section [',(lo(i),':',hi(i),i=1,ndim),']'
c
      end

c     divide the space into equal size patches according to nproc
c     and calculate my lo and hi
      subroutine my_space(me,nproc,ndim,total,dims,lo,hi)
      implicit none
      integer me,nproc,ndim,total
      integer dims(ndim),lo(ndim),hi(ndim)
      integer div,lop,hip,i
c
      div = total/nproc
c
      lop = div * me + 1
c
      if(me.eq.(nproc-1)) then
         hip = total
      else
         hip = div * (me+1)
      endif
c
      call conv_1ton(ndim,dims,lop,lo)
      call conv_1ton(ndim,dims,hip,hi)
c
c     swap the indices if the lo if larger thant hi
      do i = 1,ndim
         if(lo(i).gt.hi(i)) then
	    if(i.eq.ndim) call ga_error('bye',0)
            lo(i) = 1
            lo(i+1) = lo(i+1) + 1
         endif
      enddo
      end

c     convert the index from one dimension to n dimension
      subroutine conv_1ton(ndim,dims,ind1,indn)
      implicit none
      integer ndim
      integer dims(ndim)
      integer ind1,indn(ndim)
      integer range(8),remainder,i
c
      remainder = ind1
c     get the range of each dimension
      do i=1,ndim
         if(i.eq.1) then
            range(i) = dims(i)
         else
            range(i) = range(i-1) * dims(i)
         endif
      enddo
c
c     get the indices in each dimension
      do i = ndim,1,-1
         if(i.ne.1) then
            indn(i) = remainder/range(i-1)
            remainder = remainder - indn(i)*range(i-1)
            if(remainder.eq.0) then
               remainder = range(i-1)
            else
               indn(i) = indn(i) + 1
            endif
         else
            indn(i) = remainder
         endif
      enddo
c
      end


c     fill array with random numbers
      subroutine fill_array_int(a,n,val)
      implicit none
      integer n
      integer a(n),val
      integer i
      do i= 1, n
         a(i) = val
      enddo
      end       

c     initialize the array with random numbers
      subroutine init_array_int(a,n)
      implicit none
      integer n
      integer a(n)
      double precision drand
      integer i
      do i= 1, n
         a(i) = int(drand(0) * i * 2)
      enddo
      end       

c     if the elements do match, stop the program
      subroutine compare_patches_int(eps,
     $     total1,array1,lo1,hi1,ndim1,dims1,
     $     total2,array2,lo2,hi2,ndim2,dims2)
      implicit none
      double precision eps
      integer ndim1,ndim2,total1,total2
      integer array1(total1),array2(total2)
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      integer next_index
      integer index1,index2
      double precision diff,maxval
c     
c     initialize index1 and index2, searching from zeros
      index1 = 0
      index2 = 0
c     compare corresponding elements in each array
      index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
      index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      do while((index1.ne.0).and.(index2.ne.0))
         diff = abs(array1(index1) - array2(index2))
         maxval = max(abs(array1(index1)), abs(array2(index2)))
         if((maxval.eq.0).or.(maxval.lt.eps)) maxval = 1
         if(eps .lt. abs(diff)/maxval) then
            print *, 'Error: Comparison failed!'
            print *, array1(index1), array2(index2)
            call ga_error('bye',0)
         endif
         index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
         index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      enddo
c
c     at this point both index1 and index2 should be 0
      if((index1.ne.0).or.(index2.ne.0)) then
         print *, 'Error: # of elems dont match'
         call ga_error('bye',0)
      endif
c
      end

c     do patch = patch + buf * alpha
      subroutine scale_patch_int(total,
     $     alpha,arr1,lo1,hi1,ndim1,dims1,beta,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      integer arr1(total),arr2(total)
      integer alpha, beta
      integer next_index
      integer ind1, ind2
c
      ind1 = 0
      ind2 = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         arr1(ind1) = arr1(ind1)*alpha + arr2(ind2)*beta
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      end

c     transpose an array
      subroutine transpose_int(a1,a2,total,ndim,dims)
      implicit none
      integer ndim,total
      integer dims(ndim)
      integer a1(total),a2(total)
      integer i, j
      integer idx
      integer bv(8), bunit(8)
c
      bv(1)=0
      bunit(1)=1
      do i=2, ndim
         bv(i) = 0
         bunit(i) = bunit(i-1) * dims(i-1)
      enddo
c
      do i=1, total
         idx = 1
         do j=1, ndim
            idx = idx + bv(j) * bunit(ndim-j+1)
            if(mod(i,bunit(j)).eq.0) bv(j) = bv(j) + 1
            if(bv(j).ge.dims(j)) bv(j) = 0
         enddo
c         print *, 'i = ',i, 'idx = ',idx
         a2(idx) = a1(i)
      enddo
c
      do i=1, total
         a1(i) = a2(i)
      enddo
c
      end

c     do patch = patch + buf * alpha
      integer function dot_patch_int(
     $     total,arr1,lo1,hi1,ndim1,dims1,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      integer arr1(total),arr2(total)
      integer next_index
      integer ind1, ind2
      integer res
c
      ind1 = 0
      ind2 = 0
      res = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         res = res + arr1(ind1)*arr2(ind2)
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      dot_patch_int = res
c
      end


c     fill array with random numbers
      subroutine fill_array_dbl(a,n,val)
      implicit none
      integer n
      double precision a(n),val
      integer i
      do i= 1, n
         a(i) = val
      enddo
      end       

c     initialize the array with random numbers
      subroutine init_array_dbl(a,n)
      implicit none
      integer n
      double precision a(n)
      double precision drand
      integer i
      do i= 1, n
         a(i) = drand(0) * i * 2
      enddo
      end       

c     if the elements do match, stop the program
      subroutine compare_patches_dbl(eps,
     $     total1,array1,lo1,hi1,ndim1,dims1,
     $     total2,array2,lo2,hi2,ndim2,dims2)
      implicit none
      double precision eps
      integer ndim1,ndim2,total1,total2
      double precision array1(total1),array2(total2)
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      integer next_index
      integer index1,index2
      double precision diff,maxval
c     
c     initialize index1 and index2, searching from zeros
      index1 = 0
      index2 = 0
c     compare corresponding elements in each array
      index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
      index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      do while((index1.ne.0).and.(index2.ne.0))
         diff = abs(array1(index1) - array2(index2))
         maxval = max(abs(array1(index1)), abs(array2(index2)))
         if((maxval.eq.0).or.(maxval.lt.eps)) maxval = 1
         if(eps .lt. abs(diff)/maxval) then
            print *, 'Error: Comparison failed!'
            print *, array1(index1), array2(index2)
            call ga_error('bye',0)
         endif
         index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
         index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      enddo
c
c     at this point both index1 and index2 should be 0
      if((index1.ne.0).or.(index2.ne.0)) then
         print *, 'Error: # of elems dont match'
         call ga_error('bye',0)
      endif
c
      end

c     do patch = patch + buf * alpha
      subroutine scale_patch_dbl(total,
     $     alpha,arr1,lo1,hi1,ndim1,dims1,beta,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      double precision arr1(total),arr2(total)
      double precision alpha, beta
      integer next_index
      integer ind1, ind2
c
      ind1 = 0
      ind2 = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         arr1(ind1) = arr1(ind1)*alpha + arr2(ind2)*beta
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      end

c     transpose an array
      subroutine transpose_dbl(a1,a2,total,ndim,dims)
      implicit none
      integer ndim,total
      integer dims(ndim)
      double precision a1(total),a2(total)
      integer i, j
      integer idx
      integer bv(8), bunit(8)
c
      bv(1)=0
      bunit(1)=1
      do i=2, ndim
         bv(i) = 0
         bunit(i) = bunit(i-1) * dims(i-1)
      enddo
c
      do i=1, total
         idx = 1
         do j=1, ndim
            idx = idx + bv(j) * bunit(ndim-j+1)
            if(mod(i,bunit(j)).eq.0) bv(j) = bv(j) + 1
            if(bv(j).ge.dims(j)) bv(j) = 0
         enddo
c         print *, 'i = ',i, 'idx = ',idx
         a2(idx) = a1(i)
      enddo
c
      do i=1, total
         a1(i) = a2(i)
      enddo
c
      end

c     do patch = patch + buf * alpha
      double precision function dot_patch_dbl(
     $     total,arr1,lo1,hi1,ndim1,dims1,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      double precision arr1(total),arr2(total)
      integer next_index
      integer ind1, ind2
      double precision res
c
      ind1 = 0
      ind2 = 0
      res = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         res = res + arr1(ind1)*arr2(ind2)
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      dot_patch_dbl = res
c
      end


c     fill array with random numbers
      subroutine fill_array_dcpl(a,n,val)
      implicit none
      integer n
      double complex a(n),val
      integer i
      do i= 1, n
         a(i) = val
      enddo
      end       

c     initialize the array with random numbers
      subroutine init_array_dcpl(a,n)
      implicit none
      integer n
      double complex a(n)
      double precision drand
      integer i
      do i= 1, n
         a(i) = dcmplx(drand(0) * i * 2,
     $    -drand(0) * i * 2)
      enddo
      end       

c     if the elements do match, stop the program
      subroutine compare_patches_dcpl(eps,
     $     total1,array1,lo1,hi1,ndim1,dims1,
     $     total2,array2,lo2,hi2,ndim2,dims2)
      implicit none
      double precision eps
      integer ndim1,ndim2,total1,total2
      double complex array1(total1),array2(total2)
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      integer next_index
      integer index1,index2
      double precision diff,maxval
c     
c     initialize index1 and index2, searching from zeros
      index1 = 0
      index2 = 0
c     compare corresponding elements in each array
      index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
      index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      do while((index1.ne.0).and.(index2.ne.0))
         diff = abs(array1(index1) - array2(index2))
         maxval = max(abs(array1(index1)), abs(array2(index2)))
         if((maxval.eq.0).or.(maxval.lt.eps)) maxval = 1
         if(eps .lt. abs(diff)/maxval) then
            print *, 'Error: Comparison failed!'
            print *, array1(index1), array2(index2)
            call ga_error('bye',0)
         endif
         index1 = next_index(index1,total1,ndim1,lo1,hi1,dims1)
         index2 = next_index(index2,total2,ndim2,lo2,hi2,dims2)
      enddo
c
c     at this point both index1 and index2 should be 0
      if((index1.ne.0).or.(index2.ne.0)) then
         print *, 'Error: # of elems dont match'
         call ga_error('bye',0)
      endif
c
      end

c     do patch = patch + buf * alpha
      subroutine scale_patch_dcpl(total,
     $     alpha,arr1,lo1,hi1,ndim1,dims1,beta,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      double complex arr1(total),arr2(total)
      double complex alpha, beta
      integer next_index
      integer ind1, ind2
c
      ind1 = 0
      ind2 = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         arr1(ind1) = arr1(ind1)*alpha + arr2(ind2)*beta
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      end

c     transpose an array
      subroutine transpose_dcpl(a1,a2,total,ndim,dims)
      implicit none
      integer ndim,total
      integer dims(ndim)
      double complex a1(total),a2(total)
      integer i, j
      integer idx
      integer bv(8), bunit(8)
c
      bv(1)=0
      bunit(1)=1
      do i=2, ndim
         bv(i) = 0
         bunit(i) = bunit(i-1) * dims(i-1)
      enddo
c
      do i=1, total
         idx = 1
         do j=1, ndim
            idx = idx + bv(j) * bunit(ndim-j+1)
            if(mod(i,bunit(j)).eq.0) bv(j) = bv(j) + 1
            if(bv(j).ge.dims(j)) bv(j) = 0
         enddo
c         print *, 'i = ',i, 'idx = ',idx
         a2(idx) = a1(i)
      enddo
c
      do i=1, total
         a1(i) = a2(i)
      enddo
c
      end

c     do patch = patch + buf * alpha
      double complex function dot_patch_dcpl(
     $     total,arr1,lo1,hi1,ndim1,dims1,arr2,lo2,hi2,ndim2,dims2)
      implicit none
      integer ndim1,ndim2,total
      integer lo1(ndim1),hi1(ndim1),lo2(ndim2),hi2(ndim2)
      integer dims1(ndim1),dims2(ndim2)
      double complex arr1(total),arr2(total)
      integer next_index
      integer ind1, ind2
      double complex res
c
      ind1 = 0
      ind2 = 0
      res = 0
      ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
      ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
c
      do while(ind1.ne.0)
         res = res + arr1(ind1)*arr2(ind2)
         ind1 = next_index(ind1,total,ndim1,lo1,hi1,dims1)
         ind2 = next_index(ind2,total,ndim2,lo2,hi2,dims2)
      enddo
c
      dot_patch_dcpl = res
c
      end


