      program test
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"

c     Intitialize the TCGMSG package
c
       call pbeginf
c
c     Initialize the MA package
c
      if (.not. ma_init(MT_DBL, 1500000, 1500000))
     $     call parerr(1)
c
c     Intitialize the GA package
c
      call ga_initialize()
c
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
      parameter (nmax = 20000)

      parameter (ntest = 7 )

      double precision a(nmax), b(nmax), c(nmax)
      integer          g_a, g_b, g_c
      integer          i, j, kk
      integer          nproc, me
      double precision dsin, sum, diffa, diffr
      double precision t0, t1
      logical          status

      integer          mnkmax, itrans, itest
      integer          ms(ntest), ns(ntest), ks(ntest)
      double precision alphas(ntest), betas(ntest)

      integer          m, n, k
      double precision alpha, beta
      character        transa, transb
      logical          ltrana, ltranb
      logical          ga_create2
c     
      DOUBLE PRECISION AVAIMEM
      data (alphas(i), betas(i),
     $      ms(i), ns(i), ks(i), i = 1, ntest)
     $  /
C     $     1.0d0, 0.0d0, 100, 100, 100,
C     $     1.0d0, 0.0d0, 200, 200, 200,
C     $     1.0d0, 0.0d0, 400, 400, 400,
C     $     1.0d0, 0.0d0, 800, 800, 800,
C     $     1.0d0, 0.0d0, 1600, 1600, 1600,
C     $     1.0d0, 0.0d0, 3200, 3200, 3200,
C     $     1.0d0, 0.0d0, 6400, 6400, 6400
     $     1.0d0, 0.0d0, 2424, 1227, 16160,
     $     1.0d0, 0.0d0, 16160, 1227, 2424,
     $     1.0d0, 0.0d0, 16160, 2424, 1227,
     $     1.0d0, 0.0d0, 2424, 16160, 1227,
     $     1.0d0, 0.0d0, 1227, 16160, 2424,
     $     1.0d0, 0.0d0, 2424, 1227, 16160,
C     $     1.0d0, 0.0d0, 1227, 2424, 16160,
     $     1.0d0, 0.0d0, 1227, 2424, 16160
     $  /

      nproc = ga_nnodes()
      me    = ga_nodeid()

c***  check SUMMA 
c
      if (me .eq. 0) then
 	print *,' '
        print *,'>checking ga_SUMMA'
 	print *,' '
 	print *,' nproc = ', nproc
        write(6,*)
     $    '  transA  transB   m    n    k    '
        write(6,*)
     $    '  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
        call ffflush(6)
      endif
c     
c***  a() is a local copy of what the global array should start as
c
      mnkmax = 0
      do j = 1, ntest
        mnkmax = max( mnkmax, ns(j), ms(j), ks(j) )
      enddo

      if( mnkmax .gt. nmax )
     $  call ga_error(' test_summa: Must increase PARAMETER nmax', 0)
        
C      do itrans = 1, 4
      do itrans = 1, 4

        transa = 'n'
        transb = 'n'

        if( itrans .eq. 2  .or.  itrans .eq. 4 ) transa = 't'
        if( itrans .eq. 3  .or.  itrans .eq. 4 ) transb = 't'

        ltrana = transa .eq. 't' .or.  transa .eq. 'T'
        ltranb = transb .eq. 't' .or.  transb .eq. 'T'

        do itest = 1, ntest
   
          alpha = alphas( itest )
          beta  = betas( itest )
          m     = ms( itest )
          n     = ns( itest )
          k     = ks( itest )

C          if( SQRT( MAX( M*N,K*N,M*K) / FLOAT(NPROC) ) .gt. 801.d0 )
C     $       goto 500
c
c***      Create global arrays
c
          call ga_sync()

          if( .not. ltrana ) then
            if (.not. ga_create2(MT_DBL, m, k, 'A', 1, 1, g_a))
     $         call ga_error(' ga_create failed ',2)
          else
            if (.not. ga_create2(MT_DBL, k, m, 'A', 1, 1, g_a))
     $         call ga_error(' ga_create failed ',2)
          endif

          if( .not. ltranb ) then
            if (.not. ga_create2(MT_DBL, k, n, 'B', 1, 1, g_b))
     $           call ga_error(' ga_create failed ',2)
          else
            if (.not. ga_create2(MT_DBL, n, k, 'B', 1, 1, g_b))
     $           call ga_error(' ga_create failed ',2)
          endif

          if (.not. ga_create2(MT_DBL, m, n, 'C', 1, 1, g_c))
     $         call ga_error(' ga_create failed ',2)
c     
c***      Fill in arrays A, B & C
c
          call ga_sync()

*          if (me .eq. 0) then

             if( .not. ltrana ) then

               do j = me+1, k, nproc
                  do i = 1, mnkmax
                    a(i) = 1d0 * (i+j)  
                  enddo

	          call ga_put(g_a, 1,m, j,j, a,nmax)
               enddo
             else
               do j = me+1, m, nproc
                  do i = 1, mnkmax
                    a(i) = 1d0 * (i+j)  
                  enddo

	          call ga_put(g_a, 1,k, j,j, a,nmax)
               enddo
             endif

             if( .not. ltranb ) then
               do j = me+1, n, nproc
                  do i = 1, mnkmax
*                     b(i) = 1.d0 + DSIN(1d0* (i+j))   
                     b(i) = 3.5d0 * i + 2.3d0 * j
                  enddo

	          call ga_put(g_b, 1,k, j,j, b,nmax)
               enddo
             else
               do j = me+1, k, nproc
                  do i = 1, mnkmax
*                     b(i) = 1.d0 + DSIN(1d0* (i+j))   
                     b(i) = 3.5d0 * i + 2.3d0 * j
                  enddo

	          call ga_put(g_b, 1,n, j,j, b,nmax)
               enddo
             endif

             do j = me+1, n, nproc
                do i = 1, mnkmax
                   c(i) = 2.d0 * (i+j)  
                enddo

	        call ga_put(g_c, 1,m, j,j, c,nmax)
             enddo

*          endif

C      if( me .eq. 0)
C     $   write(0,*) ' before ga_ga_dgemm                  AVAIMEM = ',
C     $                AVAIMEM()

c     
c***      Do matrix-matrix multiply
          call ga_sync()
          t0 = tcgtime()

          call ga_ga_dgemm( transa, transb, m,n,k, alpha, g_a, g_b,
     $                    beta, g_c)
          call ga_sync()
          t1 = tcgtime()

c     

          if (me .eq. 0) then

C             write(0,*) ' after ga_ga_dgemm / before ga_summa AVAIMEM = ',
C     $                  AVAIMEM()

             write(6,1000) 'ga_ga_dgemm', transa, transb, m, n, k, t1-t0
             call ffflush(6)

          endif
c
          call ga_sync()
          t0 = tcgtime()

          call ga_summa( transa, transb, m,n,k, alpha, g_a, g_b,
     $                    beta, g_c)
          call ga_sync()
          t1 = tcgtime()

          if (me .eq. 0) then

C             write(0,*) ' after  ga_summa                  AVAIMEM = ',
C     $                  AVAIMEM()


             write(6,1000) 'ga_summa', transa, transb, m, n, k, t1-t0
             call ffflush(6)

          endif


          call ga_sync()
c
c..................................................................

          status =  ga_destroy(g_c)
          status =  ga_destroy(g_b)
          status =  ga_destroy(g_a)

  500     continue
        enddo
      enddo

      if( me .eq. 0 ) then
 	 print *,' '
 	 print *,' '
         call ffflush(6)
      endif

 1000 format(2X, A10, 2X, 3X, A1, 6X, A1, 4X, 3i7, 2X, f10.2 )

      end
