#if HAVE_CONFIG_H 
#   include "config.fh"
#endif
c $Id: test.F,v 1.64.2.11 2007-04-06 22:37:35 d3g293 Exp $
c vector boxes lack arithmetic precision 
# define THRESH 1d-13
# define THRESHF 2e-5

#define MISMATCH(x,y) abs(x-y)/max(1d0,abs(x)).gt.THRESH
#define MISMATCHF(x,y) abs(x-y)/max(1.0,abs(x)).gt.THRESHF 

      program main
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "testutil.fh"
      integer heap, stack, fudge, ma_heap, me, nproc, map(4096), block
      integer ndim, dim1, i
      integer version, subversion, patch
      logical status
      parameter (heap=800*800*4, fudge=100, stack=800*800)
      integer n, m, g_b
      parameter (n =100) 
      parameter (m=2*n)
      real a(n,n), b(n,n), v(m), w(m)
      real one, x
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
c
c     we can also use GA_set_memory_limit BEFORE first ga_create call
c
      ma_heap = heap/nproc + fudge 
      ma_heap = 2*ma_heap
#ifdef USE_RESTRICTED
      ma_heap = 2*ma_heap
#endif
      print *, 'util_mdtob(', ma_heap, ')=', util_mdtob(ma_heap)
      call GA_set_memory_limit(util_mdtob(ma_heap))
c
      call ga_version(version,subversion,patch)
      if(ga_nodeid().eq.0)then
         print *,' GA initialized '
         print *,' '
         write(6,'(a,i1,a,i1,a,i1)') ' Version ',version,'.',
     +      subversion,'.',patch
         call ffflush(6)
      endif
c
c***  Initialize the MA package
c     MA must be initialized before any global array is allocated
c
      status = ma_init(MT_DCPL, stack, ma_heap)
      if (.not. status) call ga_error('ma_init failed',-1) 
c
c     Uncomment the below line to register external memory allocator
c     for dynamic arrays inside GA routines.
c      call register_ext_memory()
c
      if(me.eq.(nproc-1))then
        print *, 'using ', nproc,' process(es) ', ga_cluster_nnodes(),
     $           ' cluster nodes'
        print *,'process ', me, ' is on node ',ga_cluster_nodeid(),
     $          ' with ',  ga_cluster_nprocs(-1), ' processes' 
        call ffflush(6)
      endif
c
c     overlapping accumulate
c
      call ga_sync()
      if (.not. ga_create(MT_REAL, n, n, 'b', 0, 0, g_b)) then
         call ga_error('ga_create failed for second array ',-1)
      endif
c
      call ga_zero(g_b)
      one = 1.0
      call ga_acc(g_b, n/2, n/2, n/2, n/2, one, 1, one)
      call ga_sync()
      if (me.eq.0) then
         call ga_get(g_b, n/2, n/2, n/2, n/2, b(1,1), 1)
         x = abs(b(1,1) -1*nproc)
         if(x.gt. 1e-10)then
            write(6,*)'val=',b(1,1),' expected=',nproc, x,' on proc ',me
            call ga_error('overlapping accumulate failed',-1)
         endif
         write(6,*)
         write(6,*) ' overlapping ga_acc is OK'
         write(6,*)
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
