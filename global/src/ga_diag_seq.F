#if HAVE_CONFIG_H
#   include "config.fh"
#endif
#if (BLAS_SIZE ==4)
#define INTGR4 integer*4
#else
#define INTGR4 integer*8
#endif
c
c This file has been converted to use LAPACK circa 2011
c instead of EISPACK circa 1983 by Jeff Hammond circa 2014.
c
      subroutine gai_diag_seq(g_a, g_s, g_v, evals)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "galinalg.fh"
      integer g_a               ! Matrix to diagonalize
      integer g_s               ! Metric
      integer g_v               ! Global matrix to return evecs
      double precision evals(*) ! Local array to return evals
c
      integer n
#if ENABLE_EISPACK
      integer l_fv1, l_fv2, l_v
      MA_ACCESS_INDEX_TYPE k_fv1, k_fv2, k_v
#else
      integer l_wrk
      MA_ACCESS_INDEX_TYPE k_wrk
      integer n2
#endif
      integer l_a, l_s
      MA_ACCESS_INDEX_TYPE k_a, k_s
      integer dim1, dim2, type, me
      INTGR4 n4,ierr
      logical status
c
c
c     Solve the generalized eigen-value problem returning
c     all eigen-vectors and values in ascending order
c
c     The input matrices may be destroyed
c
      call ga_check_handle(g_a, 'ga_diag_seq a')
      call ga_check_handle(g_s, 'ga_diag_seq s')
      call ga_check_handle(g_v, 'ga_diag_seq v')
      call ga_sync()
c
c     Only process 0 does the diag
c
      call ga_inquire(g_a, type, dim1, dim2)
      if(dim1.ne.dim2)
     $  call ga_error('ga_diag_seq: nonsquare matrix ',0)
 
      n = dim1
      n4 = n
      me = ga_nodeid()
      if (me .eq. 0) then
c
c     allocate scratch space
c     
         status=ma_push_get(MT_DBL, n*n,
     $        'diag_seq:a', l_a, k_a)
         status=status.and.ma_push_get(MT_DBL, n*n,
     $        'diag_seq:s', l_s, k_s)
#if ENABLE_EISPACK
         status=status.and.ma_push_get(MT_DBL, n*n,
     $        'diag_seq:v', l_v, k_v)
         status=status.and.ma_push_get(MT_DBL, n,
     $        'diag_seq:fv1', l_fv1, k_fv1)
         status=status.and.ma_push_get(MT_DBL, n,
     $        'diag_seq:fv2', l_fv2, k_fv2)
#else
c LAPACK fails for n=1 without this
         n2 = max(n*n,3*n-1)
         status=status.and.ma_push_get(MT_DBL, n2,
     $        'diag_seq:wrk', l_wrk, k_wrk)
#endif
         if (.not. status)
     $        call ga_error('ga_diag_seq: insufficient memory', n)
c     
c     Fill local arrays from global arrays
c     
         call ga_get(g_a, 1, n, 1, n, dbl_mb(k_a), n)
         call ga_get(g_s, 1, n, 1, n, dbl_mb(k_s), n)
c
#ifdef DEBUG_MATH_LIBS
         write(6,*) ' fock matrix '
         call output(dbl_mb(k_a), 1, n, 1, n, n, n, 1)
         write(6,*) ' overlap matrix '
         call output(dbl_mb(k_s), 1, n, 1, n, n, n, 1)
#endif
#if ENABLE_EISPACK
         call rsg(n, n, dbl_mb(k_a), dbl_mb(k_s), evals, 1,
     $        dbl_mb(k_v), dbl_mb(k_fv1), dbl_mb(k_fv2), ierr)
#else
         call dsygv(1,'V','U',n4,dbl_mb(k_a),n,dbl_mb(k_s),n4,
     $              evals,dbl_mb(k_wrk),n2, ierr)
         if (ierr.ne.0)
     $       call ga_error('ga_diag_seq: dsygv failed',ierr)
c We used to copy to preserve code symmetry with EISPACK
c         call dcopy(n*n, dbl_mb(k_a), 1, dbl_mb(k_v), 1)
#endif
#ifdef DEBUG_MATH_LIBS
         write(6,*) 'eigen vectors '
         call output(dbl_mb(k_v), 1, n, 1, n, n, n, 1)
         write(6,*) 'eigen values '
         call output(evals, 1, n, 1, 1, n, 1, 1)
*         call forflush(6)
#endif
c     
c     Shove eigen-vectors back into global array
c     
#if ENABLE_EISPACK
         call ga_put(g_v, 1, n, 1, n, dbl_mb(k_v), n)
#else
         call ga_put(g_v, 1, n, 1, n, dbl_mb(k_a), n)
#endif
c     
c     Free scratch space
c     
#if ENABLE_EISPACK
         status = ma_pop_stack(l_fv2)
         status = ma_pop_stack(l_fv1)
         status = ma_pop_stack(l_v)
#else
         status = ma_pop_stack(l_wrk)
#endif
         status = ma_pop_stack(l_s)
         status = ma_pop_stack(l_a)
      endif
c     
c     Broadcast the eigenvalues to all processes
c
      call ga_brdcst(32500, evals, 
     $               ma_sizeof(MT_DBL,n,MT_BYTE), 0)
      call ga_sync()
c
      end
c
      subroutine gai_diag_std_seq(g_a,  g_v, evals)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
      integer g_a               ! Matrix to diagonalize
      integer g_v               ! Global matrix to return evecs
      double precision evals(*) ! Local array to return evals
c
      integer n
#if ENABLE_EISPACK
      integer l_fv1, l_fv2, l_v
      MA_ACCESS_INDEX_TYPE k_fv1, k_fv2, k_v
#else
      integer l_wrk
      MA_ACCESS_INDEX_TYPE k_wrk
      integer n2
#endif
      integer l_a
      MA_ACCESS_INDEX_TYPE k_a
      integer dim1, dim2, type, me
      INTGR4 n4,n2_i4,ierr
      logical status
c
c
c     Solve the standard eigen-value problem returning
c     all eigen-vectors and values in ascending order
c
c     The input matrices may be destroyed
c
      call ga_check_handle(g_a, 'ga_diag_std a')
      call ga_check_handle(g_v, 'ga_diag_std v')
      call ga_sync()
c
c     Only process 0 does the diag
c
      call ga_inquire(g_a, type, dim1, dim2)
      if(dim1.ne.dim2)
     $  call ga_error('ga_diag_std_seq: nonsquare matrix ',0)
 
      n = dim1
      n4 = n
      me = ga_nodeid()
      if (me .eq. 0) then
c
c     allocate scratch space
c     
         status=ma_push_get(MT_DBL, n*n,
     $        'diag_std_seq:a', l_a, k_a)
#if ENABLE_EISPACK
         status=status.and.ma_push_get(MT_DBL, n*n,
     $        'diag_std_seq:v', l_v, k_v)
         status=status.and.ma_push_get(MT_DBL, n,
     $        'diag_std_seq:fv1', l_fv1, k_fv1)
         status=status.and.ma_push_get(MT_DBL, n,
     $        'diag_std_seq:fv2', l_fv2, k_fv2)
#else
c LAPACK fails for n=1 without this
         n2 = max(n*n,3*n-1)
         n2_i4=n2
         status=status.and.ma_push_get(MT_DBL, n2,
     $        'diag_std_seq:wrk', l_wrk, k_wrk)
#endif
         if (.not. status)
     $      call ga_error('ga_diag_std_seq: insufficient memory', n)
c     
c     Fill local arrays from global arrays
c     
         call ga_get(g_a, 1, n, 1, n, dbl_mb(k_a), n)
c     
#ifdef DEBUG_MATH_LIBS
         write(6,*) ' input matrix '
         call output(dbl_mb(k_a), 1, n, 1, n, n, n, 1)
#endif
#if ENABLE_EISPACK
         call rs(n, n, dbl_mb(k_a),  evals, 1,
     $        dbl_mb(k_v), dbl_mb(k_fv1), dbl_mb(k_fv2), ierr)
#else
         call dsyev('V',  'L',  n4, dbl_mb(k_a), n4,   
     $              evals, dbl_mb(k_wrk), n2_i4,  ierr)
         if (ierr.ne.0)
     $       call ga_error('ga_diag_std_seq: dsyev failed',ierr)
c We used to copy to preserve code symmetry with EISPACK
c         call dcopy(n*n, dbl_mb(k_a), 1, dbl_mb(k_v), 1)
#endif
#ifdef DEBUG_MATH_LIBS
         write(6,*) 'eigen vectors '
         call output(dbl_mb(k_v), 1, n, 1, n, n, n, 1)
         write(6,*) 'eigen values '
         call output(evals, 1, n, 1, 1, n, 1, 1)
*         call forflush(6)
#endif
c     
c     Shove eigen-vectors back into global array
c     
#if ENABLE_EISPACK
         call ga_put(g_v, 1, n, 1, n, dbl_mb(k_v), n)
#else
         call ga_put(g_v, 1, n, 1, n, dbl_mb(k_a), n)
#endif
c     
c     Free scratch space
c     
#if ENABLE_EISPACK
         status = ma_pop_stack(l_fv2)
         status = ma_pop_stack(l_fv1)
         status = ma_pop_stack(l_v)
#else
         status = ma_pop_stack(l_wrk)
#endif
         status = ma_pop_stack(l_a)
      endif
c     
c     Broadcast the eigenvalues to all processes
c
      call ga_brdcst(32500, evals, 
     $               ma_sizeof(MT_DBL,n,MT_BYTE), 0)
      call ga_sync()
c
      end
