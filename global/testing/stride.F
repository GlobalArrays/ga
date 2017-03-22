#if HAVE_CONFIG_H
#   include "config.fh"
#endif
      program test_patch
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "testutil.fh"
      integer g_r
      integer g_2e3c
      integer nbfl, nbf2
      integer nsorb
      logical status
#include "mp3.fh"

      call ga_initialize

      nbfl = 200
      nsorb = 7200
      nbf2 = nbfl*nsorb

      status = ga_create(mt_dbl,nbfl,nbfl,'one',0,0,g_2e3c)
      status = ga_create(mt_dbl,nbf2,nbfl,'Rmat',-1,-1,g_r)

      call ga_zero(g_2e3c)

      call ga_fill(g_r,1.0d00)

      call ga_copy_patch('N',g_r,1,nbfl,1,nbfl,g_2e3c,1,nbfl,1,nbfl)
      call ga_terminate
      call MP_FINALIZE()
      stop
      end
