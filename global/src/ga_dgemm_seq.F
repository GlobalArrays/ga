      subroutine ga_dgemm_seq(transa, transb, m, n, k, alpha, g_a, g_b, 
     $     beta, g_c )
      implicit none
#include "mafdecls.h"
#include "global.fh"
      character*1        transa, transb
      integer            m, n, k
      double precision   alpha, beta
      integer            g_a, g_b, g_c
c     
      integer l_a, k_a, l_b, k_b, l_c, k_c
      logical status
c
      call ga_sync()
c
      if (me .eq. 0) then
         status = .true.
         status = ma_push_get(MT_DBL,dims(1,g_a)*dims(2,g_a),'a',
     $        l_a,k_a) .and. status
         status = ma_push_get(MT_DBL,dims(1,g_b)*dims(2,g_b),'b',
     $        l_b,k_b) .and. status
         status = ma_push_get(MT_DBL,dims(1,g_c)*dims(2,g_c),'c',
     $        l_c,k_c) .and. status
         if (.not. status) call errquit('ga_dgemm_seq: memory?, 0)
c     
         call ga_get(g_a, 1, dims(1,g_a), 1, dims(2,g_a), dbl_mb(k_a), 
     $        dims(1,g_a))
         call ga_get(g_b, 1, dims(1,g_b), 1, dims(2,g_b), dbl_mb(k_b), 
     $        dims(1,g_b))
         call ga_get(g_c, 1, dims(1,g_c), 1, dims(2,g_c), dbl_mb(k_c), 
     $        dims(1,g_c))
c     
         call dgemm(transa, transb, m, n, k, alpha, dbl_mb(k_a), 
     $        dims(1,g_a),
     $        dbl_mb(k_b), dims(1,g_b), beta, dbl_mb(k_c), dims(1,g_c))
c     
         call ga_put(g_c, 1, dims(1,g_c), 1, dims(2,g_c), dbl_mb(k_c), 
     $        dims(1,g_c))
c     
         status = .true.
         status = status .and. ma_pop_stack(l_c)
         status = status .and. ma_pop_stack(l_b)
         status = status .and. ma_pop_stack(l_a)
         if (.not. status) call errquit('ga_dgemm_seq: pop?, 0)
      endif
c
      call ga_sync
c
      end
