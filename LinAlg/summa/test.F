#define GA_CREATE ga_create
      program test
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"

      call pbeginf
      if (.not. ma_init(MT_DBL, 1400000, 200000))
     $     call parerr(1)
c
      call ga_initialize()
      call testit()
      call ga_terminate()
c
      call pend()

      end


c-----------------

      subroutine testit()
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c     
      integer nmax, ntest
      parameter (nmax = 5000)

      parameter (ntest = 10)

      double precision a(nmax), b(nmax)
      integer          g_a, g_b, g_c, g_d
      integer          i, j
      integer          nproc, me
      double precision t0, t1, t2, t3
      logical          status

      integer          loop
      integer          ms(ntest), ns(ntest), ks(ntest)

      integer          m, n, k, dim1A, dim2A, dim1B, dim2B
      double precision alpha, beta, flops, err
      character        transa, transb
      logical          ltrana, ltranb


      logical          ga_create2
c     
      data   (ms(i), ns(i), ks(i), i = 1, ntest)
     $  /
     $     256, 256, 256, 
     $     256, 256, 120, 
     $     100, 256, 256, 
     $     512, 512, 512,
     $     1024, 1024, 1024,
     $     1024, 1024, 1024,
     $     1024, 1024, 1200,
     $     2024, 1024, 1024,
     $     2024, 1024, 1024,
     $     2048, 2048, 2048/

      nproc = ga_nnodes()
      me    = ga_nodeid()

c***  check summa 
c
      if (me .eq. 0) then
 	print *,' '
        print *,'>checking ga_summa P = ', nproc
 	print *,' '
        write(6,*)
     $    '  transA  transB  m   n   k   routine      time    MFLOPS'
        write(6,*)
     $    '  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        call ffflush(6)
      endif

        do loop = 1, ntest
   
          alpha = 1.
          beta  = 0.
          m     = ms( loop )
          n     = ns( loop )
          k     = ks( loop )

c
c***      transpose or not to transpose
c
          if(Mod(loop,2).eq.0)then
               transa = 't'
          else
               transa = 'n'
          endif
          if(Mod(loop,3).eq.0)then
               transb = 't'
          else
               transb = 'n'
          endif
          ltrana = transa .eq. 't' .or.  transa .eq. 'T'
          ltranb = transb .eq. 't' .or.  transb .eq. 'T'
c
c***      Create global arrays
c
          call ga_sync()

          if( .not. ltrana ) then
            dim1A = m
            dim2A = k
          else
            dim1A = k
            dim2A = m
          endif

          if( .not. ltranb ) then
            dim1B = k
            dim2B = n
          else
            dim1B = n
            dim2B = k
          endif
c
c***      create arrays
c
          if (.not. GA_CREATE(MT_DBL, dim1A, dim2A, 'A', 1, 1, g_a))
     $         call ga_error(' ga_create failed for A',0)
          if (.not. GA_CREATE(MT_DBL, dim1B, dim2B, 'B', 1, 1, g_b))
     $         call ga_error(' ga_create failed for B',0)
          if (.not. GA_CREATE(MT_DBL, m, n, 'C', 1, 1, g_c))
     $         call ga_error(' ga_create failed for C ',0)
          if (.not. GA_CREATE(MT_DBL, m, n, 'D', 1, 1, g_d))
     $         call ga_error(' ga_create failed for D ',0)
c
c***      initialize arrays
c
          t0 = tcgtime()
          do j = 1+me, dim2A, nproc
             do i = 1, dim1A
                a(i) = 1d0*(i+j)
             enddo
             call ga_put(g_a, 1, dim1A, j, j, a, dim1A)
          enddo
          do j = 1+me, dim2B, nproc
             do i = 1, dim1B
                b(i)= 1.1333d-1*j
             enddo
             call ga_put(g_b, 1, dim1B, j, j, b, dim1B)
          enddo
          call ga_sync()
          t1 = tcgtime()
*         if ( ga_nodeid() .eq. 0) then
*            write(6,*)'initialization time=',t1-t0
*            call ffflush(6)
*         endif

 
c     
c***      Do matrix-matrix multiply using GA default routine
c
          t0 = tcgtime()

          call ga_dgemm( transa, transb, m,n,k, alpha, g_a, g_b,
     $                    beta, g_c)
          t1 = tcgtime()
          flops = 1d-6*2*m*n*k/(t1-t0)
          if ( ga_nodeid() .eq. 0) then
             write(6,1000) transa,transb,m,n,k,'ga_dgemm',t1-t0,flops
             call ffflush(6)
          endif
c     
c***      Do summa matrix-matrix multiply
c
          t2 = tcgtime()

          call ga_summa( transa, transb, m,n,k, alpha, g_a, g_b,
     $                    beta, g_d)
          t3 = tcgtime()
          call ga_sync()
C
          flops = 1d-6*2*m*n*k/(t3-t2)
          if ( ga_nodeid() .eq. 0) then
             write(6,1000) transa,transb,m,n,k,'ga_summa',t3-t2,flops
             call ffflush(6)
          endif
c
c***      check if result is OK
c
          call ga_dadd(1d0, g_c, -1d0, g_d, g_c)
          err = ga_ddot(g_c, g_c)
          if(err.gt.1d-6. and. me.eq.0) then
             write(6,*) '>>> error= ',err
             call ffflush(6)
          endif
c
c..................................................................

          status =  ga_destroy(g_d)
          status =  ga_destroy(g_c)
          status =  ga_destroy(g_b)
          status =  ga_destroy(g_a)

      enddo

      if( me .eq. 0 ) then
 	 print *,' '
 	 print *,' '
         call ffflush(6)
      endif

 1000 format(2X, 3X, A1, 6X, A1, 2X, 3i5,2X, A, f9.2, f8.2)

      end
