#if defined(CRAY_T3D) || defined(CRAY_T3E) || defined(KSR)
#   define xx_dgemm sgemm
#else
#   define xx_dgemm dgemm
#endif
c     ijk loops rearanged into ikj
c     Thu Aug 25 14:08:17 PDT 1994, JN
c
#ifdef USE_SUMMA
      subroutine ga_dgemm(transa, transb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)
C$Id: ga_dgemm.F,v 1.23 1997-03-04 06:16:55 d3e129 Exp $
      implicit none
      Character*1        transa, transb
      Integer            m, n, k
      Double precision   alpha, beta
      Integer            g_a, g_b, g_c
c
c     ga_summa calls ga_ga_dgemm to handle cases it does not cover
c
      call ga_summa(transa, transb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)
c
      end
#define GA_DGEMM ga_ga_dgemm
#endif
      subroutine GA_DGEMM(transa, transb, m, n, k, alpha, g_a, 
     $     g_b, beta, g_c)
C$Id: ga_dgemm.F,v 1.23 1997-03-04 06:16:55 d3e129 Exp $
      implicit none
      Character*1        transa, transb
      Integer            m, n, k
      Double precision   alpha, beta
      Integer            g_a, g_b, g_c
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c     
c     GA_DGEMM  performs one of the matrix-matrix operations:
c           C := alpha*op( A )*op( B ) + beta*C,
c     where  op( X ) is one of
c           op( X ) = X   or   op( X ) = X`,
c     
c     alpha and beta are scalars, and A, B and C are matrices, with op( A )
c     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
c     
c     On entry, TRANSA specifies the form of op( A ) to be used in
c     the matrix multiplication as follows:
c           transa = 'N' or 'n',  op( A ) = A.
c           transa = 'T' or 't',  op( A ) = A`.
c     
c     M      - On entry,  M  specifies  the number  of rows  of the  matrix
c              op( A )  and of the  matrix  C.  M  must  be at least  zero.
c     N      - On entry,  N  specifies the number  of columns of the matrix
c              op( B ) and the number of columns of the matrix C. N must be
c              at least zero.
c     K      - On entry,  K  specifies  the number of columns of the matrix
c              op( A ) and the number of rows of the matrix op( B ). K must
c              be at least  zero.
c
      integer ilo, ihi, jlo, jhi, klo, khi, ichunk, jchunk, kchunk
      integer idim, jdim, kdim, adim, bdim, cdim, ijk, me, nproc
      integer l_a, k_a, l_b, k_b, l_c, k_c
      logical status
C
      Logical Get_New_B ! Allow reuse of B patch when possible
C
      Double Precision Chunk_cube
      Integer Min_Tasks, Max_Chunk, Mem_Avail
      Parameter ( Min_Tasks = 10) ! Minimum acceptable tasks per node
c
C     Set defaults -- platform dependent
#ifdef DATA_SERVER
      ichunk = 256
      jchunk = 256
      kchunk = 256
#else
      ichunk = 128
      jchunk = 128
      kchunk = 128
#endif
C
      me = ga_nodeid()
      nproc = ga_nnodes()
C
C     Make an estimate of how large patches can be and still insure
C     enough tasks per processor that loads will be reasonably balanced.
C     
C     Patches per dimension are M/chunk, N/chunk, K/chunk so total tasks
C     is roughly (K*M*N)/(chunk**3).  Assume all chunk sizes are the 
C     same and solve for the one that provides the minimum acceptable 
C     number of tasks.
C
      chunk_cube = Dble(K)*Dble(M)*Dble(N) / (Min_tasks * NProc)
      Max_chunk = Int( Chunk_cube ** (1.0d0/3.0d0) )
c
c     RJH ... if chunks are too small then communication costs swap computation
c
      if (max_chunk .lt. 64) max_chunk = 64
C
C     If there is room to increase the chunk size without impacting
C     load balance, do it.
C
      If ( Max_Chunk .gt. Max(ichunk, jchunk, kchunk) ) then
C
C        Find out how much memory we can grab.  It will be used in
C        three chunks, and the result includes only the first one.
C
         Mem_Avail = MA_Inquire_Avail( MT_DBL )
     $      - 2 * MA_SizeOf_Overhead( MT_DBL )
	 Mem_Avail = 0.9 * Mem_Avail ! Do not use every last drop!
         Call GA_IGOp(42, Mem_Avail, 1, 'min')
C
         Max_Chunk = Min( Max_Chunk, Int(Sqrt(DBLE(Mem_Avail/3))) )
C
         IChunk = min(m,Max_Chunk)
         JChunk = min(n,Max_Chunk)
         KChunk = min(k,Max_Chunk)
      EndIf

      status = .true.
      status = ma_push_get(MT_DBL, ichunk*kchunk, 'ga_dgemm:a', l_a,k_a)
     $     .and. status
      status = ma_push_get(MT_DBL, kchunk*jchunk, 'ga_dgemm:b', l_b,k_b)
     $     .and. status
      status = ma_push_get(MT_DBL, ichunk*jchunk, 'ga_dgemm:c', l_c,k_c)
     $     .and. status
      if (.not. status) call ga_error('ga_dgemm: insufficent memory?',0)
c
      if (beta .eq. 0.0d0) then
         call ga_zero(g_c)
      else
         call ga_scale(g_c, beta)
      endif
c
      ijk = 0
      do jlo = 1, n, jchunk
         jhi = min(n, jlo+jchunk-1)
         jdim = jhi - jlo + 1
         do klo = 1, k, kchunk
            khi = min(k, klo+kchunk-1)
            kdim = khi - klo + 1
C
C           Each pass through the outer two loops means we need a
C           different patch of B.
C
            Get_New_B = .TRUE.
C
            do ilo = 1, m, ichunk
               if (mod(ijk,nproc) .eq. me) then
                  ihi = min(m, ilo+ichunk-1)
                  idim = ihi - ilo + 1
                  cdim = idim
                  call util_dfill(idim*jdim, 0.0d0, dbl_mb(k_c), 1)
                  if (transa.eq.'n' .or. transa.eq.'N') then
                     adim = idim
                     call ga_get(g_a, ilo, ihi, klo, khi,
     $                  dbl_mb(k_a), adim)
                  else
                     adim = kdim
                     call ga_get(g_a, klo, khi, ilo, ihi, 
     $                  dbl_mb(k_a), adim)
                  endif
C
C                 Avoid rereading B if it is the same patch as last time.
C
                  If ( Get_New_B ) then
                     if (transb.eq.'n' .or. transb.eq.'N') then
                        bdim = kdim
                        call ga_get(g_b, klo, khi, jlo, jhi, 
     $                     dbl_mb(k_b), bdim)
                     else
                        bdim = jdim
                        call ga_get(g_b, jlo, jhi, klo, khi, 
     $                     dbl_mb(k_b), bdim)
                     endif
                     Get_New_B = .FALSE. ! Until J or K change again
                  EndIf
C
                  call xx_dgemm(transa, transb, idim, jdim, kdim,
     $                 alpha, dbl_mb(k_a), adim, dbl_mb(k_b), bdim,
     $                 1.0d0, dbl_mb(k_c), cdim)
                  call ga_acc(g_c, ilo, ihi, jlo, jhi, dbl_mb(k_c), 
     $                 cdim, 1.0d0)
               endif
               ijk = ijk + 1
            enddo
         enddo
      enddo
      status = .true.
      status = status .and. ma_pop_stack(l_c)
      status = status .and. ma_pop_stack(l_b)
      status = status .and. ma_pop_stack(l_a)
      if (.not. status)call ga_error('ga_dgemm: pop of stack failed', 0)
      call ga_sync()
c
      end
