      subroutine ga_summarize(verbose)
      implicit none
#include "global.fh"
#include "mafdecls.fh"
c
c     MIMD callable ... summarize all allocated arrays onto
c     FORTRAN unit 6 ... assumes no more than 100 arrays are
c     allocated and they are numbered 0, ...
c
      logical verbose            ! If true print distribution info
      character*80 name, form, head
      integer type, dim1, dim2, active
      integer ga_verify_handle
      external ga_verify_handle
      integer g_a, nproc, me, l_d, k_d, ii, lname, ldim1, ldim2, lp
      logical printed
c
      integer map, k, i
      map(k,i) = int_mb(k_d + i*4 + k - 1)
c
      printed = .false.
c
      me = ga_nodeid()
      nproc = ga_nnodes()
      if (.not. ma_push_get(mt_int, 4*nproc, 'ga_summarize', l_d, k_d))
     $     call ga_error('ga_summarize: insufficient mem', 4*nproc)
c
      write(6,1)
 1    format(/' Summary of allocated global arrays '/
     $        ' ----------------------------------'/)
c
      do g_a = 0, 99
         active = ga_verify_handle(g_a)
         if (active .eq. 1) then
c
c     Get info from standard GA interface
c
            printed = .true.
            call ga_inquire(g_a, type, dim1, dim2)
            call ga_inquire_name(g_a, name)
            do i = 0, nproc-1
               ii = k_d + 4*i - 1
               call ga_distribution(g_a, i, int_mb(ii+1), int_mb(ii+2),
     $              int_mb(ii+3), int_mb(ii+4))
            enddo
c
c     Determine length for format statements
c
            ldim1 = log10(dfloat(dim1)) + 1
            ldim2 = log10(dfloat(dim2)) + 1
            lp    = log10(dfloat(nproc)) + 1
            do lname = len(name),1,-1
               if (name(lname:lname).ne.' ') goto 10
            enddo
 10         continue
            write(form,3) ldim1, ldim1, ldim2, ldim2, lp
 3          format('(''     ('',i',i4,','':'',i',i4,','', '',i',i4,
     $           ','':'',i',i4,','') -> '',i',i4,')')
c
            if (type .eq. mt_int) then
               write(head,2) g_a, 'integer', lname, ldim1, ldim2
            else
               write(head,2) g_a, 'real', lname, ldim1, ldim2
            endif

 2          format('(''  array',i2,' => ', a ,' '',a', i2, 
     $           ',''('',i',i,','','',i',i,''')'')')
            
*     write(6,*) head
            write(6,head) name(1:lname), dim1, dim2
c
            if (verbose) then
               do i = 0, nproc-1
                  if (map(2,i).gt.0 .and. map(4,i).gt.0) then
                     write(6,form) (map(k,i),k=1,4), i
                  endif
               enddo
            endif
c
         endif
      enddo
c
      if (.not. ma_pop_stack(l_d))
     $     call ga_error('ga_summarize: pop_stack failed', 0)
c
      call ffflush(6)
c
      if (.not. printed) write(6,*) '  No active global arrays'
c
      write(6,*) ' '
c
      end




