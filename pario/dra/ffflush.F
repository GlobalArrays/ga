#if HAVE_CONFIG_H
#   include "config.fh"
#endif
c $Id: ffflush.F,v 1.2 2002-10-07 16:51:51 d3h325 Exp $
      subroutine ffflush(unit)
      integer unit
c
#if HAVE_F77_FLUSH
      call F77_FLUSH(unit)
#endif
c
      end
      subroutine imp_finalize()
c     Centralized finalize used by MP_FINALIZE macro in mp3.fh
c     Calls MPI_Finalize only if MPI is initialized and not yet finalized.
      integer mp_i, mp_f, ierr
      include 'mpif.h'
      call MPI_Initialized(mp_i, ierr)
      if (mp_i .ne. 0) then
         call MPI_Finalized(mp_f, ierr)
         if (mp_f .eq. 0) then
            call mpi_finalize(ierr)
         end if
      end if
      return
      end

