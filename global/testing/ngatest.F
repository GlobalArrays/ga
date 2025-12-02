

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
      
      call testit_GA_FILL_int1()
	     
      call testit_GA_FILL_dbl1()
	    
      call testit_GA_FILL_dcpl1()
      
      call testit_GA_FILL_int2()
	     
      call testit_GA_FILL_dbl2()
	    
      call testit_GA_FILL_dcpl2()
      
      call testit_GA_FILL_int3()
	     
      call testit_GA_FILL_dbl3()
	    
      call testit_GA_FILL_dcpl3()
      
      call testit_GA_FILL_int4()
	     
      call testit_GA_FILL_dbl4()
	    
      call testit_GA_FILL_dcpl4()
      
      call testit_GA_FILL_int5()
	     
      call testit_GA_FILL_dbl5()
	    
      call testit_GA_FILL_dcpl5()
      
      call testit_GA_FILL_int6()
	     
      call testit_GA_FILL_dbl6()
	    
      call testit_GA_FILL_dcpl6()
      
      call testit_GA_FILL_int7()
	     
      call testit_GA_FILL_dbl7()
	    
      call testit_GA_FILL_dcpl7()
      
c
c     test NGA_GET
      
      call testit_NGA_GET_int1()
	    
      call testit_NGA_GET_dbl1()
	    
      call testit_NGA_GET_dcpl1()
      
      call testit_NGA_GET_int2()
	    
      call testit_NGA_GET_dbl2()
	    
      call testit_NGA_GET_dcpl2()
      
      call testit_NGA_GET_int3()
	    
      call testit_NGA_GET_dbl3()
	    
      call testit_NGA_GET_dcpl3()
      
      call testit_NGA_GET_int4()
	    
      call testit_NGA_GET_dbl4()
	    
      call testit_NGA_GET_dcpl4()
      
      call testit_NGA_GET_int5()
	    
      call testit_NGA_GET_dbl5()
	    
      call testit_NGA_GET_dcpl5()
      
      call testit_NGA_GET_int6()
	    
      call testit_NGA_GET_dbl6()
	    
      call testit_NGA_GET_dcpl6()
      
      call testit_NGA_GET_int7()
	    
      call testit_NGA_GET_dbl7()
	    
      call testit_NGA_GET_dcpl7()
      
c
c     test NGA_PUT
      
      call testit_NGA_PUT_int1()
	    
      call testit_NGA_PUT_dbl1()
	    
      call testit_NGA_PUT_dcpl1()
      
      call testit_NGA_PUT_int2()
	    
      call testit_NGA_PUT_dbl2()
	    
      call testit_NGA_PUT_dcpl2()
      
      call testit_NGA_PUT_int3()
	    
      call testit_NGA_PUT_dbl3()
	    
      call testit_NGA_PUT_dcpl3()
      
      call testit_NGA_PUT_int4()
	    
      call testit_NGA_PUT_dbl4()
	    
      call testit_NGA_PUT_dcpl4()
      
      call testit_NGA_PUT_int5()
	    
      call testit_NGA_PUT_dbl5()
	    
      call testit_NGA_PUT_dcpl5()
      
      call testit_NGA_PUT_int6()
	    
      call testit_NGA_PUT_dbl6()
	    
      call testit_NGA_PUT_dcpl6()
      
      call testit_NGA_PUT_int7()
	    
      call testit_NGA_PUT_dbl7()
	    
      call testit_NGA_PUT_dcpl7()
      
c
c     test NGA_ACC
      
      call testit_NGA_ACC_int1()
	    
      call testit_NGA_ACC_dbl1()
	    
      call testit_NGA_ACC_dcpl1()
      
      call testit_NGA_ACC_int2()
	    
      call testit_NGA_ACC_dbl2()
	    
      call testit_NGA_ACC_dcpl2()
      
      call testit_NGA_ACC_int3()
	    
      call testit_NGA_ACC_dbl3()
	    
      call testit_NGA_ACC_dcpl3()
      
      call testit_NGA_ACC_int4()
	    
      call testit_NGA_ACC_dbl4()
	    
      call testit_NGA_ACC_dcpl4()
      
      call testit_NGA_ACC_int5()
	    
      call testit_NGA_ACC_dbl5()
	    
      call testit_NGA_ACC_dcpl5()
      
      call testit_NGA_ACC_int6()
	    
      call testit_NGA_ACC_dbl6()
	    
      call testit_NGA_ACC_dcpl6()
      
      call testit_NGA_ACC_int7()
	    
      call testit_NGA_ACC_dbl7()
	    
      call testit_NGA_ACC_dcpl7()
      
c
c     test NGA_PERIODIC_GET
      
c
c     test NGA_PERIODIC_PUT
      
c
c     test NGA_PERIODIC_ACC
      
c
c     test NGA_FILL_PATCH
      
      call testit_NGA_FILL_PATCH_int1()
	    
      call testit_NGA_FILL_PATCH_dbl1()
	    
      call testit_NGA_FILL_PATCH_dcpl1()
      
      call testit_NGA_FILL_PATCH_int2()
	    
      call testit_NGA_FILL_PATCH_dbl2()
	    
      call testit_NGA_FILL_PATCH_dcpl2()
      
      call testit_NGA_FILL_PATCH_int3()
	    
      call testit_NGA_FILL_PATCH_dbl3()
	    
      call testit_NGA_FILL_PATCH_dcpl3()
      
      call testit_NGA_FILL_PATCH_int4()
	    
      call testit_NGA_FILL_PATCH_dbl4()
	    
      call testit_NGA_FILL_PATCH_dcpl4()
      
      call testit_NGA_FILL_PATCH_int5()
	    
      call testit_NGA_FILL_PATCH_dbl5()
	    
      call testit_NGA_FILL_PATCH_dcpl5()
      
      call testit_NGA_FILL_PATCH_int6()
	    
      call testit_NGA_FILL_PATCH_dbl6()
	    
      call testit_NGA_FILL_PATCH_dcpl6()
      
      call testit_NGA_FILL_PATCH_int7()
	    
      call testit_NGA_FILL_PATCH_dbl7()
	    
      call testit_NGA_FILL_PATCH_dcpl7()
      
c
c     test NGA_COPY_PATCH
      
      call testit_NGA_COPY_PATCH_int1()
	    
      call testit_NGA_COPY_PATCH_dbl1()
	    
      call testit_NGA_COPY_PATCH_dcpl1()
      
      call testit_NGA_COPY_PATCH_int2()
	    
      call testit_NGA_COPY_PATCH_dbl2()
	    
      call testit_NGA_COPY_PATCH_dcpl2()
      
      call testit_NGA_COPY_PATCH_int3()
	    
      call testit_NGA_COPY_PATCH_dbl3()
	    
      call testit_NGA_COPY_PATCH_dcpl3()
      
      call testit_NGA_COPY_PATCH_int4()
	    
      call testit_NGA_COPY_PATCH_dbl4()
	    
      call testit_NGA_COPY_PATCH_dcpl4()
      
      call testit_NGA_COPY_PATCH_int5()
	    
      call testit_NGA_COPY_PATCH_dbl5()
	    
      call testit_NGA_COPY_PATCH_dcpl5()
      
      call testit_NGA_COPY_PATCH_int6()
	    
      call testit_NGA_COPY_PATCH_dbl6()
	    
      call testit_NGA_COPY_PATCH_dcpl6()
      
      call testit_NGA_COPY_PATCH_int7()
	    
      call testit_NGA_COPY_PATCH_dbl7()
	    
      call testit_NGA_COPY_PATCH_dcpl7()
      
c
c     test NGA_SCALE_PATCH
      
      call testit_NGA_SCALE_PATCH_int1()
	    
      call testit_NGA_SCALE_PATCH_dbl1()
	    
      call testit_NGA_SCALE_PATCH_dcpl1()
      
      call testit_NGA_SCALE_PATCH_int2()
	    
      call testit_NGA_SCALE_PATCH_dbl2()
	    
      call testit_NGA_SCALE_PATCH_dcpl2()
      
      call testit_NGA_SCALE_PATCH_int3()
	    
      call testit_NGA_SCALE_PATCH_dbl3()
	    
      call testit_NGA_SCALE_PATCH_dcpl3()
      
      call testit_NGA_SCALE_PATCH_int4()
	    
      call testit_NGA_SCALE_PATCH_dbl4()
	    
      call testit_NGA_SCALE_PATCH_dcpl4()
      
      call testit_NGA_SCALE_PATCH_int5()
	    
      call testit_NGA_SCALE_PATCH_dbl5()
	    
      call testit_NGA_SCALE_PATCH_dcpl5()
      
      call testit_NGA_SCALE_PATCH_int6()
	    
      call testit_NGA_SCALE_PATCH_dbl6()
	    
      call testit_NGA_SCALE_PATCH_dcpl6()
      
      call testit_NGA_SCALE_PATCH_int7()
	    
      call testit_NGA_SCALE_PATCH_dbl7()
	    
      call testit_NGA_SCALE_PATCH_dcpl7()
      
c
c     test NGA_ADD_PATCH
      
c
c     test NGA_DOT_PATCH
      
      call testit_NGA_DOT_PATCH_int1()
	    
      call testit_NGA_DOT_PATCH_dbl1()
	    
      call testit_NGA_DOT_PATCH_dcpl1()
      
      call testit_NGA_DOT_PATCH_int2()
	    
      call testit_NGA_DOT_PATCH_dbl2()
	    
      call testit_NGA_DOT_PATCH_dcpl2()
      
      call testit_NGA_DOT_PATCH_int3()
	    
      call testit_NGA_DOT_PATCH_dbl3()
	    
      call testit_NGA_DOT_PATCH_dcpl3()
      
      call testit_NGA_DOT_PATCH_int4()
	    
      call testit_NGA_DOT_PATCH_dbl4()
	    
      call testit_NGA_DOT_PATCH_dcpl4()
      
      call testit_NGA_DOT_PATCH_int5()
	    
      call testit_NGA_DOT_PATCH_dbl5()
	    
      call testit_NGA_DOT_PATCH_dcpl5()
      
      call testit_NGA_DOT_PATCH_int6()
	    
      call testit_NGA_DOT_PATCH_dbl6()
	    
      call testit_NGA_DOT_PATCH_dcpl6()
      
      call testit_NGA_DOT_PATCH_int7()
	    
      call testit_NGA_DOT_PATCH_dbl7()
	    
      call testit_NGA_DOT_PATCH_dcpl7()
      
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
      
c
c     test NGA_GATHER
      
      call testit_NGA_GATHER_int1()
	    
      call testit_NGA_GATHER_dbl1()
	    
      call testit_NGA_GATHER_dcpl1()
      
      call testit_NGA_GATHER_int2()
	    
      call testit_NGA_GATHER_dbl2()
	    
      call testit_NGA_GATHER_dcpl2()
      
      call testit_NGA_GATHER_int3()
	    
      call testit_NGA_GATHER_dbl3()
	    
      call testit_NGA_GATHER_dcpl3()
      
      call testit_NGA_GATHER_int4()
	    
      call testit_NGA_GATHER_dbl4()
	    
      call testit_NGA_GATHER_dcpl4()
      
      call testit_NGA_GATHER_int5()
	    
      call testit_NGA_GATHER_dbl5()
	    
      call testit_NGA_GATHER_dcpl5()
      
      call testit_NGA_GATHER_int6()
	    
      call testit_NGA_GATHER_dbl6()
	    
      call testit_NGA_GATHER_dcpl6()
      
      call testit_NGA_GATHER_int7()
	    
      call testit_NGA_GATHER_dbl7()
	    
      call testit_NGA_GATHER_dcpl7()
      


c
      if(ga_nodeid().eq.0) print *, 'All tests successful'
c
      call ga_terminate()
      call MP_FINALIZE()
      end

c-----------------

      
         
            
      subroutine testit_GA_FILL_int1()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int1()
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
      integer c(n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int1()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int1()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int1()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int1()
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
      integer c(n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int2()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int2()
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
      integer c(n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int2()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int2()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int2()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int2()
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
      integer c(n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int3()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int3()
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
      integer c(n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int3()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int3()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int3()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int3()
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
      integer c(n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int4()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int4()
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
      integer c(n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int4()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int4()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int4()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int4()
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
      integer c(n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int5()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int5()
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
      integer c(n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int5()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int5()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int5()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int5()
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
      integer c(n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int6()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int6()
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
      integer c(n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int6()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int6()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int6()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int6()
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
      integer c(n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_int7()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      integer val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = int(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_int7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_int(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_int7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      integer val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_int(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         call compare_patches_int(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_int7()
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
      integer c(n,n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      integer val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = int(drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_int(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_int(total,
     $        int(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
            call compare_patches_int(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_int7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_int7()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = int(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_int(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      call compare_patches_int(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_int7()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      integer d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_int(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c     
         call compare_patches_int(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_int7()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      integer val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = int(drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
c        check the result
         call scale_patch_int(total,
     $        val,a,lo,hi,ndim,dims,
     $        int(0),b,lo,hi,ndim,dims)
              
         call compare_patches_int(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_int7()
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
      integer c(n,n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      integer alpha, beta
      integer dot_patch_int
c     for different array dimensions
      
      integer d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_INT, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_INT, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_idot_patch'
         print *, '   - Data Type: integer'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_int(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_int(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_idot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_INT, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_int(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_int(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_idot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_int(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

      
         
            
      subroutine testit_GA_FILL_dbl1()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl1()
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
      double precision c(n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl1()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl1()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl1()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl1()
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
      double precision c(n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl2()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl2()
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
      double precision c(n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl2()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl2()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl2()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl2()
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
      double precision c(n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl3()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl3()
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
      double precision c(n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl3()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl3()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl3()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl3()
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
      double precision c(n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl4()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl4()
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
      double precision c(n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl4()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl4()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl4()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl4()
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
      double precision c(n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl5()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl5()
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
      double precision c(n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl5()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl5()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl5()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl5()
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
      double precision c(n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl6()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl6()
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
      double precision c(n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl6()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl6()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl6()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl6()
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
      double precision c(n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dbl7()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double precision val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dble(456)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dbl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dbl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dbl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double precision val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(234)
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dbl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         call compare_patches_dbl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dbl7()
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
      double precision c(n,n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double precision val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = drand(0) * me*2+1 * 2
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dbl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dbl(total,
     $        dble(1),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
            call compare_patches_dbl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dbl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dbl7()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dble(1234)
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dbl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      call compare_patches_dbl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dbl7()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double precision d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dbl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c     
         call compare_patches_dbl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dbl7()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double precision val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = drand(0) * 1 * 2
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
c        check the result
         call scale_patch_dbl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dble(0),b,lo,hi,ndim,dims)
              
         call compare_patches_dbl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dbl7()
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
      double precision c(n,n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double precision alpha, beta
      double precision dot_patch_dbl
c     for different array dimensions
      
      double precision d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DBL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DBL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_ddot_patch'
         print *, '   - Data Type: double precision'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_dbl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dbl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DBL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dbl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_dbl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_ddot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dbl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

      
         
            
      subroutine testit_GA_FILL_dcpl1()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl1()
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
      double complex c(n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl1()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl1()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl1()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl1()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl1()
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
      double complex c(n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 1'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl2()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl2()
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
      double complex c(n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl2()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl2()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl2()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl2()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl2()
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
      double complex c(n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n)
      integer dndim
      parameter (dndim = 2-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 2'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl3()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl3()
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
      double complex c(n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl3()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl3()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl3()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl3()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl3()
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
      double complex c(n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n,n)
      integer dndim
      parameter (dndim = 3-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 3'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl4()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl4()
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
      double complex c(n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl4()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl4()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl4()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl4()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl4()
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
      double complex c(n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n,n,n)
      integer dndim
      parameter (dndim = 4-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 4'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl5()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl5()
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
      double complex c(n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl5()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl5()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl5()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl5()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl5()
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
      double complex c(n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n,n,n,n)
      integer dndim
      parameter (dndim = 5-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 5'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl6()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl6()
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
      double complex c(n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl6()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl6()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl6()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl6()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl6()
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
      double complex c(n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n,n,n,n,n)
      integer dndim
      parameter (dndim = 6-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 6'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end

            
      subroutine testit_GA_FILL_dcpl7()
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
      integer lo(ndim),hi(ndim), dims(ndim), ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i,total
      integer elems, count_elems
      double complex val
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
c-------------------------------GA_FILL ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING ga_fill'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call nga_distribution(g_a, me, lo,hi)
      elems = count_elems(lo,hi,ndim)
c
      val = dcmplx(dble(456),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      if(elems.gt.0) then
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)	
      endif
      
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

	    
      subroutine testit_NGA_PUT_dcpl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim),hip(ndim)
      integer elems, count_elems
      integer nproc, me
      integer proc
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
c------------------------------- NGA_PUT ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_put'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a, proc, lo,hi)
      elems = count_elems(lo,hi,ndim)
      call init_array_dcpl(a,total)
c
      call ga_sync()
      if(elems.gt.0) then
         call nga_put(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         do loop = 1, MAXLOOP 
            call random_range(lo,hi,lop,hip,ndim)
            if(me.eq.0 .and. Mod(loop,10).eq.0)then
               call print_range(loop,lop,hip,ndim)
            endif
            call nga_put(g_a,lop,hip,
     $           a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),ld)
         enddo
         
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
      else
c     so that the random_range can be call the same number of times
c     in other words, drand can generate the same number for the
c     collective operations
         do loop=1, MAXLOOP
            call random_range(lo,hi,lop,hip,ndim)
         enddo
      endif
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

	    
      subroutine testit_NGA_GET_dcpl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total, loop
      integer lop(ndim), hip(ndim)
      double complex val
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
c------------------------------- NGA_GET ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_get'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(234),dble(0))
      call ga_fill(g_a,val)
      call ga_sync()
c     
      call fill_array_dcpl(a,total,val)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0 .and. Mod(loop,10).eq.0)then
            call print_range(loop,lo,hi,ndim)
         endif
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
         call compare_patches_dcpl(0d0,total,
     $        a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_ACC_dcpl7()
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
      double complex c(n,n,n,n,n,n,n)
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer lop(ndim),hip(ndim)
      integer chunk(ndim)
      integer i, total, loop
      double precision drand
      double complex val, alpha
      integer nproc, me, proc
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
c------------------------------- NGA_ACC ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_acc'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      call ga_sync()
      val = dcmplx(drand(0) * me*2+1 * 2,
     $    -drand(0) * me*2+1 * 2)
      call ga_fill(g_a,val)
      call ga_sync()
      proc =  nproc-1 -me       ! access other process memory 
      call nga_distribution(g_a,proc,lop,hip)
c      
      call init_array_dcpl(b,total)
c
      do loop = 1, MAXLOOP 
         call random_range(lop,hip,lo,hi,ndim)
         if(Mod(loop,10).eq.0)then
            if(Mod((loop/10),ndim).eq.me) then
               call print_range(loop,lo,hi,ndim)
            endif
         endif
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c        generate the alpha
         alpha = val
c
         call ga_sync()
c        keep a copy of the original patch
         if(hi(1).ne.-1) call nga_acc(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld,alpha)
c     
         call ga_sync()
c
         if(hi(1).ne.-1) then
c           scale the local copy of array            
            call scale_patch_dcpl(total,
     $        dcmplx(dble(1),dble(0)),a,lo,hi,ndim,dims,
     $        alpha,b,lo,hi,ndim,dims) 
c
c           get the patch from the global array
            call nga_get(g_a,lo,hi,
     $        c(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),ld)
c
            call compare_patches_dcpl(1d-2,total,
     $        a,lo,hi,ndim,dims,total,c,lo,hi,ndim,dims)
         endif
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

	    
	    
      subroutine testit_NGA_GATHER_dcpl7()
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
      integer lo(ndim),hi(ndim),dims(ndim),ld(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, j, total, loop
      integer elems, count_elems
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
c------------------------------- NGA_GATHER ----------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_gather'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lo, hi)
      elems = count_elems(lo, hi, ndim)
      if(elems.gt.0) call nga_put(g_a,lo,hi,
     $     a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      do loop = 1, MAXLOOP 
         call ga_sync()
c        initialize the index array
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
         enddo
c
c        gather from global array
         call nga_gather(g_a, v, d, m)
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
            if(v(i).ne.
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

	    
	    
	    
	    
      subroutine testit_NGA_FILL_PATCH_dcpl7()
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
      integer lo(ndim),hi(ndim),dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_FILL_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_fill_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
      val = dcmplx(dble(1234),dble(0))
c     initialize the lo and hi
      do i=1, ndim
         lo(i) = 2
         hi(i) = n - 1
      enddo
c
      call nga_fill_patch(g_a, lo, hi, val)
c
c     check the result
      call fill_array_dcpl(a,total,val)
      call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
      call compare_patches_dcpl(0d0,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
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

	    
      subroutine testit_NGA_COPY_PATCH_dcpl7()
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
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
c     for different array dimensions
      
      double complex d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)   
c     
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_COPY_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_copy_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c           
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c    
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
c        keep a copy of the origian array
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  standard copy patch: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping without transpose: OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(a,b,total,ndim,dims)
c     
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(ndim-i+1) + 1
            bhi(i) = ahi(ndim-i+1) + 1
         enddo
         if(me.eq.0)then
            call copy_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           '-->', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c     
         call nga_copy_patch('t', g_a, alo, ahi, g_b, blo, bhi)
c     
         call nga_get(g_b,blo,bhi,
     $        b(blo(1),blo(2),blo(3),blo(4),blo(5),blo(6),blo(7)),dims)
c     
c     adjust index of array a
         do i=1,ndim
            tlo(i) = alo(ndim-i+1)
            thi(i) = ahi(ndim-i+1)
         enddo
         call compare_patches_dcpl(0d0,total,
     $        a,tlo,thi,ndim,dims,total,b,blo,bhi,ndim,dims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  reshaping transposed: OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing copy patch on different dimensions'
c     
      call ga_sync()
c
c     initialize g_b
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c
      call ga_sync()
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call copy_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *, loop,': copy [',(alo(i),':',ahi(i),i=1,ndim),
c$$$     $           ']','-->','[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         call nga_copy_patch('n', g_a, alo, ahi, g_b, dlo, dhi)
c     
         call nga_get(g_b,dlo,dhi,
     $        d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
         call nga_get(g_a,alo,ahi,
     $        a(alo(1),alo(2),alo(3),alo(4),alo(5),alo(6),alo(7)),dims)
c     
         call compare_patches_dcpl(0d0,total,
     $        a,alo,ahi,ndim,dims,total,d,dlo,dhi,dndim,ddims)
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  copy patches on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
      end




	    
      subroutine testit_NGA_SCALE_PATCH_dcpl7()
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
      integer dims(ndim)
      integer g_a
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim)
      integer lo(ndim), hi(ndim)
      double precision drand
      double complex val
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c     
      call ga_sync()
c
c--------------------------- NGA_SCALE_PATCH -------------------------
      if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_scale_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
c
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
c     
      do loop=1, 10
         call random_range(lop,hip,lo,hi,ndim)
         if(me.eq.0)
     $        call print_range(loop,lo,hi,ndim)
c$$$     $        print *, loop,': scale [',(lo(i),':',hi(i), i=1,ndim),']'
c        the random number to scale
         val = dcmplx(drand(0) * 1 * 2,
     $    -drand(0) * 1 * 2)
c
c        keep a copy of the origian array
         call nga_get(g_a,lo,hi,
     $        a(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
         call nga_scale_patch(g_a,lo,hi,val)
c
         call nga_get(g_a,lo,hi,
     $        b(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6),lo(7)),dims)
c
c        check the result
         call scale_patch_dcpl(total,
     $        val,a,lo,hi,ndim,dims,
     $        dcmplx(dble(0),dble(0)),b,lo,hi,ndim,dims)
              
         call compare_patches_dcpl(1d-10,total,
     $     a,lo,hi,ndim,dims,total,b,lo,hi,ndim,dims)
c
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

	    
	    
      subroutine testit_NGA_DOT_PATCH_dcpl7()
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
      double complex c(n,n,n,n,n,n,n)
      integer dims(ndim)
      integer g_a, g_b
      integer chunk(ndim)
      integer i, total
      integer elems, count_elems
      integer loop
      integer lop(ndim), hip(ndim), hipl(ndim)
      integer alo(ndim), ahi(ndim)
      integer blo(ndim), bhi(ndim)
      integer tlo(ndim), thi(ndim)
      double complex alpha, beta
      double complex dot_patch_dcpl
c     for different array dimensions
      
      double complex d(n,n,n,n,n,n)
      integer dndim
      parameter (dndim = 7-1)
      integer ddims(dndim),dlo(dndim),dhi(dndim),dtotal
      
c      
      integer nproc, me
      logical status
      integer repeat
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
         total = total * dims(i)
      enddo
c
c***  Create global arrays
      if (.not. nga_create(MT_DCPL, ndim, dims, 'a', chunk, g_a))
     $     call ga_error(' ga_create failed ',1)
c
c     test the same distribution and different distribution seperately
      do repeat=1,2
         if(repeat.eq.1) then
            status = ga_duplicate(g_a, g_b, 'a_duplicated')
            if(.not.ga_compare_distr(g_a, g_b))
     $           call ga_error("g_b distribution different",0)
c
         else
            do i = 1,ndim
               if(mod(i,2).eq.0) chunk(i) = n
            enddo
            if (.not. nga_create(MT_DCPL, ndim, dims, 'b', chunk, g_b))
     $           call ga_error(' ga_create failed ',1)
         endif
c
         call ga_sync()
c     
c---------------------------NGA_DOT_PATCH -------------------------
c
      if(repeat.eq.1) then
         if(me.eq.0)then 
         print *, ' '
         print *, 'TESTING nga_zdot_patch'
         print *, '   - Data Type: double complex'
         print *, '   - Dimension: 7'
         print *, '   - Running on',nproc,'processes (processors)'
         call ffflush(6)
      endif
         if(me.eq.0) print *, 'Testing with the same distributions'
      else
         if(me.eq.0) print *, 'Testing with different distributions'
      endif
c     
c     initialize GA
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_dcpl(b,total)
      call nga_distribution(g_b, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_b,lop,hip,
     $     b(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
c     
      call ga_sync()
      do i = 1,ndim
         lop(i) = 1
         hipl(i) = n-1
         hip(i) = n
      enddo
c     
c---  
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'n',blo,bhi)
c
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,blo,bhi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' without transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---  
c     prepare array a, make it transposed
      call transpose_dcpl(b,c,total,ndim,dims)
c
      do loop=1, 10
         call random_range(lop,hipl,alo,ahi,ndim)
         do i=1, ndim
            blo(i) = alo(i) + 1
            bhi(i) = ahi(i) + 1
         enddo
         if(me.eq.0)then
            call dot_range(loop,alo,ahi,ndim,blo,bhi,ndim)
c$$$            print *, loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(blo(i),':',bhi(i), i=1,ndim),']'
         endif
c
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,g_b,'t',blo,bhi)
c
c        adjust index of array a
         do i=1,ndim
            tlo(i) = blo(ndim-i+1)
            thi(i) = bhi(ndim-i+1)
         enddo
c     
c        the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,b,tlo,thi,ndim,dims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, ' with transpose OK'
         print *, ' '
         call ffflush(6)
      endif
c---------------------------
c     
         status = ga_destroy(g_b)
      enddo
c
c-----------------------------------------------------------------
      
      
c     testing copy on differet dimensions
      dtotal = 1
      do i = 1,dndim
         ddims(i) = n
         dtotal = dtotal * ddims(i)
      enddo
c     
      if (.not. nga_create(MT_DCPL, dndim, ddims, 'd', chunk, g_b))
     $     call ga_error(' ga_create failed ',1)
c     
      if(me.eq.0) 
     $     print *, 'Testing dot patch on different dimensions'
c     
c     initialize GAs
      call init_array_dcpl(a,total)
      call nga_distribution(g_a, me, lop, hip)
      elems = count_elems(lop, hip, ndim)
      if(elems.gt.0) call nga_put(g_a,lop,hip,
     $     a(lop(1),lop(2),lop(3),lop(4),lop(5),lop(6),lop(7)),dims)
      call init_array_dcpl(d,dtotal)
      call nga_distribution(g_b, me, dlo, dhi)
      elems = count_elems(dlo, dhi, dndim)
      if(elems.gt.0) call nga_put(g_b,dlo,dhi,
     $     d(dlo(1),dlo(2),dlo(3),dlo(4),dlo(5),dlo(6)),ddims)
c     
      call ga_sync()
c     
c     calculate the maximum range of g_a that can fit into g_b
      do i = 1,ndim
         lop(i) = 1
         hip(i) = n
      enddo
      hip(dndim) = 1
c     
      do loop=1, 10
         call random_range(lop,hip,alo,ahi,ndim)
c     
         do i=1, dndim
            dlo(i) = alo(dndim-i+1)
            dhi(i) = ahi(dndim-i+1)
         enddo
         dlo(1) = alo(ndim)
         dhi(1) = ahi(ndim)
c     
         if(me.eq.0) then
            call dot_range(loop,alo,ahi,ndim,dlo,dhi,dndim)
c$$$            print *,loop,'dot: [',(alo(i),':',ahi(i), i=1,ndim),']', 
c$$$     $           ',', '[',(dlo(i),':',dhi(i), i=1,dndim),']'
         endif
c     
         alpha=nga_zdot_patch(g_a,'n',alo,ahi,
     $        g_b,'n',dlo,dhi)
c     
c     the result should be 
         beta = dot_patch_dcpl(total,
     $        a,alo,ahi,ndim,dims,d,dlo,dhi,dndim,ddims)
c     
         if(ABS(beta-alpha).gt.1d-6*ABS(alpha)) then
            print *,me, ' error ', beta, alpha
            call ga_error('exiting ...',0)
         endif
c     
      enddo
c     
      call ga_sync()
      if(me.eq.0)then
         print *, '  dot patch on different dimensions: OK'
         print *, ' '
         call ffflush(6)
      endif            
c     
      status = ga_destroy(g_b)
      
      
c---  
      status = ga_destroy(g_a)
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


