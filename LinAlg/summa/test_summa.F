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
      if (.not. ma_init(MT_DBL, 200000, 500))
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
      parameter (nmax = 600)

C      parameter (ntest = 14 )
C      parameter (ntest = 5 )
      parameter (ntest = 34 )

      double precision a(nmax,nmax), b(nmax,nmax), c(nmax,nmax),
     $                 d(nmax,nmax)
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
      data (alphas(i), betas(i), ms(i), ns(i), ks(i), i = 1, 17 )
     $ /
     $     2.0d0, 0.0d0,   5,   5,   5,
     $     2.3d0, 4.7d0,   5,   7,   4,
     $     2.3d0, 4.7d0,   7,   5,   4,
     $     2.3d0, 4.7d0,   4,   5,   7,
     $     2.3d0, 4.7d0,   4,   7,   5,
     $     2.3d0, 4.7d0,   5,   4,   7,
     $     2.3d0, 4.7d0,   7,   4,   5,
     $     2.3d0, 4.7d0,   53,   73,   14, 
     $     2.3d0, 4.7d0,   73,   53,   14, 
     $     2.3d0, 4.7d0,   14,    53,   73,
     $     2.3d0, 4.7d0,   14,    73,   53,
     $     2.3d0, 4.7d0,   53,   14,    73,
     $     2.3d0, 4.7d0,   73,   14,    53,
     $     2.3d0, 4.7d0,  10,  10,  10,
     $     2.3d0, 4.7d0,   53,   73,   43,
     $     2.3d0, 4.7d0,   73,   53,   43,
     $     2.3d0, 4.7d0,   43,   53,   73 /

      data (alphas(i), betas(i),
     $      ms(i), ns(i), ks(i), i = 18, ntest)
     $  /
     $     2.3d0, 4.7d0,   43,   73,   53,
     $     2.3d0, 4.7d0,   53,   43,   73,
     $     2.3d0, 4.7d0,   73,   43,   53,
     $     2.3d0, 4.7d0, 50,   60,  50,
     $     2.3d0, 4.7d0,  50,  50,  50,
     $     2.3d0, 4.7d0, 110, 110, 110,
     $     2.3d0, 4.7d0, 210, 210, 210,
     $     2.3d0, 4.7d0,  100, 100, 100,
     $     2.3d0, 4.7d0,  90, 100, 110,
     $     2.3d0, 4.7d0,  90, 110, 100,
     $     2.3d0, 4.7d0, 100, 110,  90,
     $     2.3d0, 4.7d0, 110, 100,  90,
     $     2.3d0, 4.7d0, 110,  90, 100,
     $     2.3d0, 4.7d0, 100,  90, 110,
     $     2.3d0, 4.7d0, 200,  90, 110,
     $     2.3d0, 4.7d0, 200, 190, 110,
     $     2.3d0, 4.7d0,   300, 300, 250
C     $     2.3d0, 4.7d0,  50,  50,  50,
C     $     2.3d0, 4.7d0, 200, 200, 200,
CC     $     2.3d0, 4.7d0, 300, 300, 300,
C     $     2.3d0, 4.7d0, 512, 512, 512,
CC     $     2.3d0, 4.7d0, 200, 200, 200
     $  /

      nproc = ga_nnodes()
      me    = ga_nodeid()

c***  check summar
c
      if (me .eq. 0) then
 	print *,' '
        print *,'>checking ga_summa'
 	print *,' '
 	print *,' nproc = ', nproc
        write(6,*)
     $    '  transA  transB  m   n   k       diffr        diffa    time'
        write(6,*)
     $    '  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
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
        
      do j = 1, mnkmax
         do i = 1, mnkmax
            a(i,j) = 1d0 * (i+2.d0*j)  
C            b(i,j) = 1.d0 + DSIN(1d0* (i+j))   
            b(i,j) = 1.d0* (2.d0*i+j)
            c(i,j) = a(i,j) * 2.d0 + b(i,j)*3.d0
         enddo
      enddo

C      do itrans = 1, 4
      do itrans = 1, 4

        transa = 'n'
        transb = 'n'

        if( itrans .eq. 2  .or.  itrans .eq. 4 ) transa = 't'
        if( itrans .eq. 3  .or.  itrans .eq. 4 ) transb = 't'

        ltrana = transa .eq. 't' .or.  transa .eq. 'T'
        ltranb = transb .eq. 't' .or.  transb .eq. 'T'

C        do itest = 22, ntest
C        do itest = 8, ntest
        do itest = 1, ntest
   
          alpha = alphas( itest )
          beta  = betas( itest )
          m     = ms( itest )
          n     = ns( itest )
          k     = ks( itest )

c
c***      Create global arrays
c
          call ga_sync()

          if( .not. ltrana ) then
            if (.not. ga_create2(MT_DBL, m, k, 'A', 1, 1, g_a))
     $         call ga_error(' ga_create failed ',1)
          else
            if (.not. ga_create2(MT_DBL, k, m, 'A', 1, 1, g_a))
     $         call ga_error(' ga_create failed ',2)
          endif

          if( .not. ltranb ) then
            if (.not. ga_create2(MT_DBL, k, n, 'B', 1, 1, g_b))
     $           call ga_error(' ga_create failed ',3)
          else
            if (.not. ga_create2(MT_DBL, n, k, 'B', 1, 1, g_b))
     $           call ga_error(' ga_create failed ',4)
          endif

          if (.not. ga_create2(MT_DBL, m, n, 'C', 1, 1, g_c))
     $         call ga_error(' ga_create failed ',5)
c     
c***      Fill in arrays A, B & C
c
          call ga_sync()

          if (me .eq. 0) then

             if( .not. ltrana ) then
               do j = 1, k
	          call ga_put(g_a, 1,m, j,j, a(1,j),nmax)
               enddo
             else
               do j = 1, m
	          call ga_put(g_a, 1,k, j,j, a(1,j),nmax)
               enddo
             endif

             if( .not. ltranb ) then
               do j = 1, n
	          call ga_put(g_b, 1,k, j,j, b(1,j),nmax)
               enddo
             else
               do j = 1, k
	          call ga_put(g_b, 1,n, j,j, b(1,j),nmax)
               enddo
             endif

             do j = 1, n
	        call ga_put(g_c, 1,m, j,j, c(1,j),nmax)
             enddo

          endif

c     
c***      Do matrix-matrix multiply
c
          call ga_sync()
          t0 = tcgtime()

          call ga_summa( transa, transb, m,n,k, alpha, g_a, g_b,
     $                    beta, g_c)
          call ga_sync()
          t1 = tcgtime()

c     
c***      Check answer.
c
          if (me .eq. 0) then

             call GA_GET(g_c, 1, m, 1, n, d, nmax )

             diffa = 0.d0
             diffr = 0.d0

             do i = 1, m
               do j = 1, n

                 if( ltrana .AND.  ltranb ) THEN

                   sum = 0.d0
                   do kk = 1, k
                      sum = sum + a(kk,i) * b(j,kk)
                   enddo

                 else if( ltrana )  THEN

                   sum = 0.d0
                   do kk = 1, k
                      sum = sum + a(kk,i) * b(kk,j)
                   enddo

                 else if( ltranb ) THEN

                   sum = 0.d0
                   do kk = 1, k
                      sum = sum + a(i,kk) * b(j,kk)
                   enddo

                 else

                   sum = 0.d0
                   do kk = 1, k
                      sum = sum + a(i,kk) * b(kk,j)
                   enddo

                 endif

                 sum = alpha * sum + beta * c(i,j)

*                 write(6,*) i,j, sum, d(i,j)

                 if( sum .ne. 0.d0) then
                    diffr = max( diffr, abs((sum - d(i,j))/sum) )
                 else
                    diffa = max( diffa, abs(sum - d(i,j)) )
                 endif

               enddo
             enddo

             write(6,1000) transa, transb, m, n, k, diffr, diffa, t1-t0
             call ffflush(6)

          endif

          call ga_sync()
c
c..................................................................

          status =  ga_destroy(g_c)
          status =  ga_destroy(g_b)
          status =  ga_destroy(g_a)

        enddo
        if( me .eq. 0 )
     $    print *,' '
      enddo

      if( me .eq. 0 ) then
 	 print *,' '
 	 print *,' '
         call ffflush(6)
      endif

 1000 format(2X, 3X, A1, 6X, A1, 4X, 3i4,2X, 1P, 2e12.3, 2X, 0p, f10.2)

      end

