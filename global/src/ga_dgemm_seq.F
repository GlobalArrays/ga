      subroutine ga_dgemm_seq(transa, transb, m, n, k, alpha, g_a, g_b, 
     $     beta, g_c )
C$Id: ga_dgemm_seq.F,v 1.7 1995-02-02 23:13:13 d3g681 Exp $
      implicit none
#include "mafdecls.h"
#include "global.fh"
      character*1        transa, transb
      integer            m, n, k
      double precision   alpha, beta
      integer            g_a, g_b, g_c
c     
      integer l_a, k_a, l_b, k_b, l_c, k_c
      integer dim1a, dim2a,  dim1b, dim2b, dim1c, dim2c, type
      logical status
c
      call ga_sync()
      call ga_inquire(g_a, type, dim1a, dim2a)
      call ga_inquire(g_b, type, dim1b, dim2b)
      call ga_inquire(g_c, type, dim1c, dim2c)
c
      if (ga_nodeid() .eq. 0) then
         status = .true.
         status = ma_push_get(MT_DBL,dim1a*dim2a,
     $        'ga_dgemm_seq:a',l_a,k_a) .and. status
         status = ma_push_get(MT_DBL,dim1b*dim2b,
     $        'ga_dgemm_seq:b',l_b,k_b) .and. status
         status = ma_push_get(MT_DBL,dim1c*dim2c,
     $        'ga_dgemm_seq:c',l_c,k_c) .and. status
         if (.not. status) call ga_error('ga_dgemm_seq: memory?', 0)
c     
         call ga_get(g_a, 1, dim1a, 1, dim2a, dbl_mb(k_a), 
     $        dim1a)
         call ga_get(g_b, 1, dim1b, 1, dim2b, dbl_mb(k_b), 
     $        dim1b)
         call ga_get(g_c, 1, dim1c, 1, dim2c, dbl_mb(k_c), 
     $        dim1c)
c     
         call dgemm(transa, transb, m, n, k, alpha, dbl_mb(k_a), 
     $        dim1a,
     $        dbl_mb(k_b), dim1b, beta, dbl_mb(k_c), dim1c)
c     
         call ga_put(g_c, 1, dim1c, 1, dim2c, dbl_mb(k_c), 
     $        dim1c)
c     
         status = .true.
         status = status .and. ma_pop_stack(l_c)
         status = status .and. ma_pop_stack(l_b)
         status = status .and. ma_pop_stack(l_a)
         if (.not. status) call ga_error('ga_dgemm_seq: pop?', 0)
      endif
c
      call ga_sync
c
      end
