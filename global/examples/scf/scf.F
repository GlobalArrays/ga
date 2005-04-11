      program scf
C$Id: scf.F,v 1.7 2005-04-11 15:47:56 d3g293 Exp $
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension orbs(maxnbfn*maxnbfn), dens(maxnnbfn), fock(maxnnbfn),
     $     work(maxnbfn*maxnbfn), evals(maxnbfn)
      integer heap, stack
      data tinit, tonel, ttwoel, tdiag, tdens, tprint /6*0.0d0/
      data eone, etwo, energy, deltad /4*0.0d0/
c     
c     initalize the parallel message passing environment
c     
#ifdef MPI
      integer ierr
      call mpi_init(ierr)
#else
      call pbeginf
#endif
      call ga_initialize()
c
c   Allocate memory
c
      heap = 2000000
      stack = 2000000
      if (.not.ma_init(MT_DBL, stack, heap))
     +   call ga_error("ma_init failed",-1)
      me = ga_nodeid()
      nproc = ga_nnodes()
c     
c     initialize a bunch of stuff and initial density matrix
c     
      rjunk = timer()
      call input
      call setarrays
      call ininrm
      call denges(dens, work)
      tinit = timer()
c
c     make initial orthogonal orbital set for jacobi diagonalizer
c
      call makeob(orbs, work)
c     
c     iterate
c     
      do 10 iter = 1, mxiter
c     
c     make info for sparsity test ... redone every iter to save space
c     
         call makesz(work, schwmax)
c     make the one particle contribution to the fock matrix (in fock)
c     and the partial contribution to the energy
c     
         call oneel(dens, work, schwmax, fock, eone)
c         call ga_dgop(1, eone, 1, '+')
         tonel = tonel + timer()
c     
c     compute the two particle contribution and then add up the
c     contributions from each process with ga_dgop
c     
         call twoel(dens, work, schwmax, fock, etwo)
c         call ga_dgop(2, etwo, 1, '+')
         call ga_dgop(3, fock, nnbfn, '+')
         ttwoel = ttwoel + timer()
c     
c     only process 0 diagonalizes and updates the density
c     
         if (me.eq.0) then
c     
c     diagonalize the fock matrix ...
c     
            call diagon(fock, orbs, evals, work, tester, iter)
         endif
c     
c     brodcast new density matrix and deltad to everyone
c     
         call ga_brdcst(4+MSGDBL, orbs, mdtob(nbfn*nbfn), 0)
         tdiag = tdiag + timer()
c
c     diagonalize the fock matrix using sequential code. Keep remaking
c     overlap matrix since ga_diag_seq does not guarantee that g_ident
c     is preserved.
c
         call makoverlap
         call ga_diag_seq(g_fock, g_ident, g_orbs, eigv)
c     
c     make the new density matrix in work from orbitals in orbs,
c     compute the norm of the change in the density matrix and
c     then update the density matrix in dens with damping.
c     
         call makden(orbs, work)
         deltad = dendif(dens, work)
         if (iter.eq.1) then
            scale = 0.0d0
         else if (iter .le. 5) then
            if (nbfn .gt. 60) then
               scale = 0.5d0
            else
               scale = 0.0d0
            endif
         else
            scale = 0.0d0
         endif
         call damp(scale, dens, work)
         tdens = tdens + timer()
c     
c     add up energy and print out convergence information
c     
         if (me.eq.0) then
           energy = enrep + eone + etwo
           call prnout(iter, energy, deltad, tester)
           tprint = tprint + timer()
         endif
c     
c     if converged then exit iteration loop
c     
         if (deltad .lt. tol) goto 20
 10   continue
      if(me.eq.0)
     $     write(6,*) ' SCF failed to converge in ', mxiter, ' iters'
c     
c     finished ... print out eigenvalues and occupied orbitals
c     
 20   continue
      call ga_igop(6, icut1, 1, '+')
      call ga_igop(7, icut2, 1, '+')
      call ga_igop(8, icut3, 1, '+')
      if (me.eq.0) then
c     
c     print out timing information
c     
         call prnfin(energy, evals, orbs)
         write(6,1) tinit, tonel, ttwoel, tdiag, tdens, tprint,
     $        nproc
 1       format(/'   init   onel  twoel   diag   dens   print  ncpu'/
     $        '  ------ ------ ------ ------ ------ ------ ------'/
     $        1x, 6f7.2, i7/)
c     
c     print out information on # integrals evaulated each iteration
c     
         nints = nnbfn*(nnbfn+1)/2
         nints = nbfn**4
         frac  = dble(icut3)/dble(nints)
         write(6,2) icut1, icut2, icut3, nints, frac
 2       format(/'       No. of integrals screened or computed '
     $        /'       -------------------------------------'/
     $        /1x,'   #ij test   #kl test   #compute     #total',
     $        '  fraction',
     $        /1x,'  ---------  ---------  ---------  ---------',
     $        '  --------',
     $        /1x,4(2x,i9),f9.3)
         call stats
      endif
c     
      call ga_terminate
#ifdef MPI
      call mpi_finalize
#else
      call pend
#endif
c     
      end
      subroutine makesz(schwarz, schwmax)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension schwarz(nnbfn)
      dimension work(maxnbfn,maxnbfn)
      integer lo(2),hi(2)
c
c     schwarz(ij) = (ij|ij) for sparsity test
c
      icut1 = 0
      icut2 = 0
      icut3 = 0
      ij = 0
      schwmax = 0.0d0
      call dfill(nnbfn, 0.0d0, schwarz, 1)
c
      me = ga_nodeid()
      nproc = ga_nnodes()
      do 10 i = 1, nbfn
         do 20 j = 1, i
            ij = ij + 1
            if (mod(ij,nproc).eq.me) then
               call g(gg, i, j, i, j)
               schwarz(ij) = sqrt(gg)
               schwmax = max(schwmax, schwarz(ij))
            endif
 20      continue
 10   continue
c
      call ga_dgop(101+MSGDBL, schwarz, nnbfn, '+')
c      call ga_dgop(102+MSGDBL, schwmax, 1, 'max')
c
      call ga_zero(g_schwarz)
      schwmax = 0.0d0
      call nga_distribution(g_schwarz,me,lo,hi)
      do i = lo(1), hi(1)
        do j = lo(2), hi(2)
          call g(gg,i,j,i,j)
          work(i,j) = sqrt(gg)
          schwmax = max(schwmax, work(i,j))
        end do
      end do
      call nga_put(g_schwarz,lo,hi,work(lo(1),lo(2)),maxnbfn)
      call ga_dgop(103+MSGDBL, schwmax, 1, 'max')
c
      end
      subroutine ininrm
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
c     write a little welcome message
c
      if (ga_nodeid().eq.0) write(6,1) natom, nocc, nbfn, tol
1     format(/' Example Direct Self Consistent Field Program '/
     $        ' -------------------------------------------- '//
     $        ' no. of atoms ............... ',i3/
     $        ' no. of occupied orbitals ... ',i3/
     $        ' no. of basis functions ..... ',i3/
     $        ' convergence threshold ...... ',d9.2//)
c
c     generate normalisation coefficients for the basis functions
c     and the index array iky
c
      do 10 i = 1, nbfn
         iky(i) = i*(i-1)/2
 10   continue
c
      do 20 i = 1, nbfn
         rnorm(i) = (expnt(i)*2.0d0/pi)**0.75d0
 20   continue
c
c     initialize common for computing f0
c
      call setfm
c
      end
      double precision function h(i,j)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
cvd$r novector
cvd$r noconcur
c
c     generate the one particle hamiltonian matrix element
c     over the normalized primitive 1s functions i and j
c
      f0val = 0.0d0
      sum = 0.0d0
      rab2 = (x(i)-x(j))**2 + (y(i)-y(j))**2 + (z(i)-z(j))**2
      facij = expnt(i)*expnt(j)/(expnt(i)+expnt(j))
      expij = exprjh(-facij*rab2)
      repij = (2.0d0*pi/(expnt(i)+expnt(j))) * expij
c
c     first do the nuclear attraction integrals
c
      do 10 iat = 1, natom
         xp = (x(i)*expnt(i) + x(j)*expnt(j))/(expnt(i)+expnt(j))
         yp = (y(i)*expnt(i) + y(j)*expnt(j))/(expnt(i)+expnt(j))
         zp = (z(i)*expnt(i) + z(j)*expnt(j))/(expnt(i)+expnt(j))
         rpc2 = (xp-ax(iat))**2 + (yp-ay(iat))**2 + (zp-az(iat))**2
c
         call f0(f0val, (expnt(i)+expnt(j))*rpc2)
         sum = sum - repij * q(iat) * f0val
 10   continue
c
c     add on the kinetic energy term
c
      sum = sum + facij*(3.0d0-2.0d0*facij*rab2) *
     $     (pi/(expnt(i)+expnt(j)))**1.5d0 * expij
c
c     finally multiply by the normalization constants
c
      h = sum * rnorm(i) * rnorm(j)
c
      end
      double precision function s(i,j)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
c     generate the overlap matrix element between the normalized
c     primitve gaussian 1s functions i and j
c
      rab2 = (x(i)-x(j))**2 + (y(i)-y(j))**2 + (z(i)-z(j))**2
      facij = expnt(i)*expnt(j)/(expnt(i)+expnt(j))
      s = (pi/(expnt(i)+expnt(j)))**1.5d0 * exprjh(-facij*rab2) *
     $     rnorm(i)*rnorm(j)
c
      end
      subroutine makden(orbs, dens)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension orbs(nbfn, nbfn), dens(nnbfn)
      dimension work(maxnbfn,maxnbfn), torbs(maxnbfn,maxnbfn)
      integer lo(2), hi(2), tlo(2), thi(2)
c
c     generate density matrix from orbitals in orbs. the first
c     nocc orbitals are doubly occupied. Note that the diagonal
c     elements are scaled by 0.5
c
      ij = 0
      do 10 i = 1, nbfn
         do 20 j = 1, i
            p = 0.0d0
            do 30 k = 1, nocc
               p = p + orbs(i,k)*orbs(j,k)
 30         continue
            ij = ij + 1
            dens(ij) = 2.0d0 * p
 20      continue
         dens(ij) = dens(ij)*0.5d0
 10   continue
      
      me = ga_nodeid()
      call nga_distribution(g_work,me,lo,hi)
      tlo(1) = lo(1)
      thi(1) = hi(1)
      tlo(2) = 1
      thi(2) = nocc
      ld = maxnbfn
      call nga_get(g_orbs,tlo,thi,torbs(tlo(1),tlo(2)),ld)
      if (lo(1).ne.lo(2).or.hi(1).ne.hi(2)) then
        tlo(1) = lo(2)
        thi(1) = hi(2)
        call nga_get(g_orbs,tlo,thi,torbs(tlo(1),tlo(2)),ld)
      endif

      pmax = 0.0d00
      do i = lo(1), hi(1)
        do j = lo(2), hi(2)
          p = 0.0d00
          do k = 1, nocc
            p = p + torbs(i,k)*torbs(j,k)
          end do
          work(i,j) = 2.0d00*p
        end do
      end do
      call nga_put(g_work,lo,hi,work(lo(1),lo(2)),ld)
c
      end
c
      subroutine oneel(dens, schwarz, schwmax, fock, eone)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer lo(2), hi(2), ld
      dimension dens(nnbfn), fock(nnbfn), schwarz(nnbfn)
      dimension work(maxnbfn,maxnbfn),tfock(maxnbfn,maxnbfn)
c
c     fill in the one-electron part of the fock matrix and
c     compute the one-electron energy contribution
c
c     simple structure to share out the work between processes
c
      me = ga_nodeid()
      nproc = ga_nnodes()
c
      call dfill(nnbfn, 0.0d0, fock, 1)
      do 10 i = me+1, nbfn, nproc
         do 20 j = 1,i
            ij = iky(i) + j
            if (schwarz(ij)*schwmax.gt.tol2e) fock(ij) = h(i,j)
 20      continue
 10   continue
      eone = scf_ddot(nnbfn, fock, 1, dens, 1)
c
      call nga_distribution(g_schwarz,me,lo,hi)
      call nga_get(g_schwarz,lo,hi,work(lo(1),lo(2)),maxnbfn)
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          tfock(i,j) = 0.0d00
          if (work(i,j)*schwmax.gt.tol2e) tfock(i,j) = h(i,j)
        end do
      end do
      call nga_put(g_fock,lo,hi,tfock(lo(1),lo(2)),maxnbfn)
      eone = 0.5d00*contract_matrices(g_fock,g_dens)
c
      end
      integer function nxtask(nproc)
      parameter (ichunk = 10)
      save icount, nleft
      data nleft, icount /0, 0/
c
c     wrapper round nxtval() to increase granularity
c     and thus reduce no. of requests to shared counter
c
      if(nproc.gt.0) then
         if(nleft.eq.0) then
            icount = nxtval(nproc) * ichunk
            nleft = ichunk
         endif
         nxtask = icount
         icount = icount + 1
         nleft = nleft -1
      else
          nleft = 0
          nxtask = 0
          junk = nxtval(nproc)
      endif
c
c     following does dumb static load balancing
c
c$$$      if(nproc.gt.0) then
c$$$         if (nleft .eq. 0) then
c$$$            icount = ga_nodeid()
c$$$            nleft = 1
c$$$         endif
c$$$         nxtask = icount
c$$$         icount = icount + ga_nnodes()
c$$$      else
c$$$          nleft = 0
c$$$          nxtask = 0
c$$$      endif
      end
c
      logical function next_chunk(lo,hi)
      include 'cscf.h'
      integer one
      parameter (one = 1)
      integer imax, lo(2), hi(2), ilo, jlo
      itask = nga_read_inc(g_counter,one,one)
      imax = nbfn/ichunk
      if (itask.lt.imax*imax) then
        if (nbfn - ichunk*imax.gt.0) imax = imax + 1
        ilo = mod(itask,imax)
        jlo = (itask-ilo)/imax
        lo(1) = ilo*ichunk + 1
        lo(2) = jlo*ichunk + 1
        hi(1) = min((ilo+1)*ichunk,nbfn)
        hi(2) = min((jlo+1)*ichunk,nbfn)
        next_chunk = .true.
      else
        next_chunk = .false.
      endif
      return
      end
c
      logical function next_4chunk(lo,hi,ilo,jlo,klo,llo)
      include 'cscf.h'
      integer one
      parameter (one = 1)
      integer imax, lo(4), hi(4), ilo, jlo, klo, llo, itmp
      itask = nga_read_inc(g_counter,one,one)
      imax = nbfn/ichunk
      if (nbfn - ichunk*imax.gt.0) imax = imax + 1
      if (itask.lt.imax**4) then
        ilo = mod(itask,imax)
        itmp = (itask - ilo)/imax
        jlo = mod(itmp,imax)
        itmp = (itmp - jlo)/imax
        klo = mod(itmp,imax)
        llo = (itmp - klo)/imax
        lo(1) = ilo*ichunk + 1
        lo(2) = jlo*ichunk + 1
        lo(3) = klo*ichunk + 1
        lo(4) = llo*ichunk + 1
        hi(1) = min((ilo+1)*ichunk,nbfn)
        hi(2) = min((jlo+1)*ichunk,nbfn)
        hi(3) = min((klo+1)*ichunk,nbfn)
        hi(4) = min((llo+1)*ichunk,nbfn)
        next_4chunk = .true.
      else
        next_4chunk = .false.
      endif
      return
      end
c
      subroutine clean_chunk(chunk)
      include 'cscf.h'
      double precision chunk(ichunk,ichunk)
      integer i,j
      do j = 1, ichunk
        do i = 1, ichunk
          chunk(i,j) = 0.0d00
        end do
      end do
      return
      end
c
      subroutine twoel(dens, schwarz, schwmax, fock, etwo)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension dens(nnbfn), fock(nnbfn), schwarz(nnbfn)
      dimension work(maxnbfn,maxnbfn)
      double precision f_ij(ichunk,ichunk),d_kl(ichunk,ichunk)
      double precision f_ik(ichunk,ichunk),d_jl(ichunk,ichunk)
      double precision s_ij(ichunk,ichunk),s_kl(ichunk,ichunk)
      double precision one
      integer nproc,lo(4),hi(4),ld,ich
      integer lo_ik(2),hi_ik(2),lo_jl(2),hi_jl(2)
      integer i,j,k,l,iloc,jloc,kloc,lloc,it,jt,kt,lt
      logical dotask, next_4chunk
c     
c     add in the two-electron contribution to the fock matrix
c     
      nproc = ga_nnodes()
c     
      gg = 0.0d0
c     
      next = nnbfn - nxtask(nproc)
      do 10 i = nbfn, 1, -1
        do 20 j = i, 1, -1
          ij = iky(i) + j
          if (ij .eq. next) then
            if (schwarz(ij)*schwmax .lt. tol2e) then
c              icut1 = icut1 + ij
            else
              do 30 k = 1, i
                lhi = k
                if (k.eq.i) lhi = j
                do 40 l = 1, lhi
                  kl = iky(k) + l
                  if (schwarz(ij)*schwarz(kl).lt.tol2e) then
c                    icut2 = icut2 + 1
                  else
c                    icut3 = icut3 + 1
c     
c     compute value of integral (ij|kl) and add into fock matrix
c     
                    call g(gg, i, j, k, l)
                    call addin(gg*0.5d0, i, j, k, l, fock,
     $                         dens, iky)
                  endif
 40             continue
 30           continue
            endif
            next = nnbfn - nxtask(nproc)
          endif
 20     continue
 10   continue
c
      etwo = scf_ddot(nnbfn, fock, 1, dens, 1)
      call ga_dgop(28,etwo,1,'+')
c
c     wait for all processes to finish work
c
      junk = nxtask(-nproc)
c
      one = 1.0d00
      call ga_zero(g_counter)
      ld = maxnbfn
      ich = ichunk
      dotask = next_4chunk(lo,hi,it,jt,kt,lt)
      itask = 0
      do while (dotask)
        lo_ik(1) = lo(1)
        lo_ik(2) = lo(3)
        hi_ik(1) = hi(1)
        hi_ik(2) = hi(3)
        lo_jl(1) = lo(2)
        lo_jl(2) = lo(4)
        hi_jl(1) = hi(2)
        hi_jl(2) = hi(4)
        call nga_get(g_schwarz,lo,hi,s_ij,ich)
        call nga_get(g_schwarz,lo(3),hi(3),s_kl,ich)
        call nga_get(g_dens,lo(3),hi(3),d_kl,ich)
        call nga_get(g_dens,lo_jl,hi_jl,d_jl,ich)
#if 0
        if (itask.eq.0) then
          write(6,*) ich
          write(6,*) lo(1),hi(1)
          write(6,*) lo(2),hi(2)
          write(6,*) lo(3),hi(3)
          write(6,*) lo(4),hi(4)
          write(6,*)
          call print_ga_block_ij(g_dens,lo(3))
          call print_chunk_matrix(d_kl,ich) 
        endif
#endif
        itask = itask + 1
        call clean_chunk(f_ij)
        call clean_chunk(f_ik)
        do i = lo(1), hi(1)
          iloc = i-lo(1) + 1
          do j = lo(2), hi(2)
            jloc = j-lo(2) + 1
            if (s_ij(iloc,jloc)*schwmax .lt. tol2e) then
              icut1 = icut1 + (hi(1)-lo(1)+1)*(hi(2)-lo(2)+1)
            else
              do k = lo(3), hi(3)
                kloc = k-lo(3) + 1
                do l = lo(4), hi(4)
                  lloc = l-lo(4) + 1
                  if (s_ij(iloc,jloc)*s_kl(kloc,lloc).lt.tol2e) then
                    icut2 = icut2 + 1
                  else
                    call g(gg, i, j, k, l)
                    f_ij(iloc,jloc) = f_ij(iloc,jloc)
     +                              + gg*d_kl(kloc,lloc)
                    f_ik(iloc,kloc) = f_ik(iloc,kloc)
     +                              - 0.5d00*gg*d_jl(jloc,lloc)
                    icut3 = icut3 + 1
                  endif
                end do
              end do
            endif
          end do
        end do
        call nga_acc(g_fock,lo,hi,f_ij,ich,one)
        call nga_acc(g_fock,lo_ik,hi_ik,f_ik,ich,one)
        dotask = next_4chunk(lo,hi,it,jt,kt,lt)
      end do
      etwo = 0.5d00*contract_matrices(g_fock,g_dens)
c
      end
      subroutine damp(fac, dens, work)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension dens(nnbfn), work(nnbfn)
c
      ofac = 1.0d0 - fac
      do 10 i = 1, nnbfn
         dens(i) = fac*dens(i) + ofac*work(i)
10    continue
      call ga_add(fac,g_dens,ofac,g_work,g_dens)
c
      end
      subroutine prnout(iter, energy, deltad, tester)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
c     printout results of each iteration
c
      if (ga_nodeid().ne.0) return
      write(6,1) iter, energy, deltad, tester
      call flush(6)
1     format(' iter=',i3,', energy=',f13.8,', deltad=',d9.2,
     $     ', deltaf=',d9.2)
c
      end
      double precision function dendif(dens, work)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension dens(nnbfn), work(nnbfn)
      integer me, lo(2), hi(2), ld1, ld2, d_idx, w_idx
c
c     compute largest change in density matrix elements
c
      denmax = 0.0d0
      do 10 i = 1, nnbfn
         denmax = max(denmax, abs(work(i)-dens(i)))
10    continue
      dendif = denmax
c
      me = ga_nodeid()
      call nga_distribution(g_dens, me, lo, hi)
      ld2 = hi(2) - lo(2) + 1
      call nga_access(g_dens, lo, hi, d_idx, ld1)
      call nga_access(g_work, lo, hi, w_idx, ld1)
      call find_max_diff(dbl_mb(d_idx),dbl_mb(w_idx),ld1,ld2,denmax)
      dendif = denmax
c
      end
      subroutine find_max_diff(a, b, ld1, ld2, maxdiff)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer ld1, ld2
      double precision a(ld1,ld2), b(ld1,ld2), maxdiff, xdiff
      maxdiff = 0.0d00
      do i = 1, ld1
        do j = 1, ld2
          xdiff = abs(a(i,j)-b(i,j))
          if (xdiff.gt.maxdiff) maxdiff = xdiff
        end do
      end do
      call ga_dgop(1,maxdiff,1,'max')
      return
      end
c
      subroutine prnfin(energy, evals, orbs)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension evals(nbfn), orbs(nbfn, nbfn)
      integer lo(2),hi(2),ld
c
c     printout final results
c
      if (ga_nodeid().ne.0) return
      write(6,1) energy
 1    format(//' final energy = ',f16.11//' eigenvalues')
      call output(eigv, 1, min(nbfn,nocc+5), 1, 1, nbfn, 1, 1)
      write(6,2)
2     format(//' eigenvectors ')
      lo(1) = 1
      lo(2) = 1
      hi(1) = nbfn
      hi(2) = nbfn
      ld = nbfn
      call nga_get(g_orbs,lo,hi,orbs,ld) 
      call output(orbs, 1, nbfn, 1, nocc, nbfn, nbfn, 1)
c
      return
      end
      subroutine g(value,i,j,k,l)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
c     compute the two electon integral (ij|kl) over normalized
c     primitive 1s gaussians
c
      f0val = 0.0d0
      rab2 = (x(i)-x(j))**2 + (y(i)-y(j))**2 + (z(i)-z(j))**2
      rcd2 = (x(k)-x(l))**2 + (y(k)-y(l))**2 + (z(k)-z(l))**2
      facij = expnt(i)*expnt(j)/(expnt(i)+expnt(j))
      fackl = expnt(k)*expnt(l)/(expnt(k)+expnt(l))
      exijkl = exprjh(- facij*rab2 - fackl*rcd2)
      denom = (expnt(i)+expnt(j))*(expnt(k)+expnt(l)) *
     $        sqrt(expnt(i)+expnt(j)+expnt(k)+expnt(l))
      fac = (expnt(i)+expnt(j))*(expnt(k)+expnt(l)) /
     $        (expnt(i)+expnt(j)+expnt(k)+expnt(l))
c
      xp = (x(i)*expnt(i) + x(j)*expnt(j))/(expnt(i)+expnt(j))
      yp = (y(i)*expnt(i) + y(j)*expnt(j))/(expnt(i)+expnt(j))
      zp = (z(i)*expnt(i) + z(j)*expnt(j))/(expnt(i)+expnt(j))
      xq = (x(k)*expnt(k) + x(l)*expnt(l))/(expnt(k)+expnt(l))
      yq = (y(k)*expnt(k) + y(l)*expnt(l))/(expnt(k)+expnt(l))
      zq = (z(k)*expnt(k) + z(l)*expnt(l))/(expnt(k)+expnt(l))
      rpq2 = (xp-xq)**2 + (yp-yq)**2 + (zp-zq)**2
c
      call f0(f0val, fac*rpq2)
      value = (2.0d0 * pi**2.5d0 / denom) * exijkl * f0val *
     $    rnorm(i)*rnorm(j)*rnorm(k)*rnorm(l)
c
      end
      subroutine diagon(fock, orbs, evals, work, tester, iter)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension fock(nnbfn), orbs(nbfn, nbfn), evals(nbfn),
     $     work(nbfn*nbfn)
      dimension ilifq(nbfn), work2(nbfn*nbfn)
c
      do 10 i = 1, nbfn
         ilifq(i) = (i-1)*nbfn
 10   continue
c
c     transform fock matrix from AO to MO basis
c
      call zsqua(nbfn, fock, work)
      call scf_dgemm('n', 'n', nbfn, nbfn, nbfn, 1.0d0, 
     $     work, nbfn, orbs, nbfn, 0.0d0, work2, nbfn)
      call scf_dgemm('t', 'n', nbfn, nbfn, nbfn, 1.0d0, 
     $     orbs, nbfn, work2, nbfn, 0.0d0, work, nbfn)
      call zfold(nbfn, fock, work)
c
      iop1 = 1
      iop2 = 2
      tester = 0.0d0
      do 20 i = 2, nbfn
         do 30 j = 1, i-1
            tester = max(tester, abs(fock(iky(i)+j)))
 30      continue
 20   continue
c
      if (tester.gt.0.3d0) then
        shift = 0.3d0
      else
        if (nbfn .gt. 60) then
           shift = 0.1d0
        else
           shift = 0.0d0
        endif
      endif	
      if (iter.ge.2) then
         do 40 i = nocc+1, nbfn
            fock(iky(i)+i) = fock(iky(i)+i) + shift
 40      continue
      endif
      thresh = min(0.0001d0,max(1.0d-12,tester*0.01d0))
      call jacobi(fock, iky, nbfn, orbs, ilifq, nbfn, evals, iop1,
     *     iop2,thresh)
      if (iter.ge.2) then
         do 50 i = nocc+1, nbfn
            evals(i) = evals(i) - shift
 50      continue
      endif
c
      end
      subroutine makeob(orbs, work)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension orbs(nbfn,nbfn), work(nbfn,nbfn)
      dimension tmp1(nbfn), tmp2(nbfn)
      integer lo(2),hi(2),ld
c
c     generate set of orthonormal vectors by orthoging twice
c     a set of random vectors ... don't do this at home!
c     ... should really diagonalize the overlap to get sym adaption
c
      if (ga_nodeid().eq.0) then
         call srand48(12345)
         do 10 i = 1, nbfn
            do 20 j = 1, nbfn
               work(j,i) = s(i,j)
               orbs(j,i) = drand48(0)
 20         continue
 10      continue
         call orthv2(nbfn, orbs, work, tmp1, tmp2)
         call orthv2(nbfn, orbs, work, tmp1, tmp2)
      endif
      call ga_brdcst(99+MSGDBL, orbs, mdtob(nbfn*nbfn), 0)
c
c  get transpose of matrix
c
      do i = 1, nbfn
        do j = 1, nbfn
          work(i,j) = orbs(j,i)
        end do
      end do
      if (ga_nodeid().eq.0) then
        lo(1) = 1
        lo(2) = 1
        hi(1) = nbfn
        hi(2) = nbfn
        ld = nbfn
        call nga_put(g_orbs,lo,hi,work,ld)
      endif
c
      end
      subroutine denges(dens, work)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      dimension dens(nnbfn), work(nbfn, nbfn)
c
c     Form guess density from superposition of atomic densities in the AO
c     basis set ... instead of doing the atomic SCF hardwire for this
c     small basis set for the Be atom.
c
      integer one, itask, lo(2), hi(2), ld
      dimension atdens(15,15)
      data atdens/
     $     0.000002,0.000027,0.000129,0.000428,0.000950,0.001180,
     $     0.000457,-0.000270,-0.000271,0.000004,0.000004,0.000004,
     $     0.000004,0.000004,0.000004,0.000027,0.000102,0.000987,
     $     0.003269,0.007254,0.009007,0.003492,-0.002099,-0.002108,
     $     0.000035,0.000035,0.000035,0.000035,0.000035,0.000035,
     $     0.000129,0.000987,0.002381,0.015766,0.034988,0.043433,
     $     0.016835,-0.010038,-0.010082,0.000166,0.000166,0.000166,
     $     0.000166,0.000166,0.000166,0.000428,0.003269,0.015766,
     $     0.026100,0.115858,0.144064,0.055967,-0.035878,-0.035990,
     $     0.000584,0.000584,0.000584,0.000584,0.000584,0.000584,
     $     0.000950,0.007254,0.034988,0.115858,0.128586,0.320120,
     $     0.124539,-0.083334,-0.083536,0.001346,0.001346,0.001346,
     $     0.001346,0.001346,0.001346,0.001180,0.009007,0.043433,
     $     0.144064,0.320120,0.201952,0.159935,-0.162762,-0.162267,
     $     0.002471,0.002471,0.002471,0.002471,0.002471,0.002471,
     $     0.000457,0.003492,0.016835,0.055967,0.124539,0.159935,
     $     0.032378,-0.093780,-0.093202,0.001372,0.001372,0.001372,
     $     0.001372,0.001372,0.001372,-0.000270,-0.002099,-0.010038,
     $     -0.035878,-0.083334,-0.162762,-0.093780,0.334488,0.660918,
     $     -0.009090,-0.009090,-0.009090,-0.009090,-0.009090,-0.009090,
     $     -0.000271,-0.002108,-0.010082,-0.035990,-0.083536,-0.162267,
     $     -0.093202,0.660918,0.326482,-0.008982,-0.008982,-0.008981,
     $     -0.008981,-0.008981,-0.008982,0.000004,0.000035,0.000166,
     $     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008982,
     $     0.000062,0.000124,0.000124,0.000124,0.000124,0.000124,
     $     0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     $     0.001372,-0.009090,-0.008982,0.000124,0.000062,0.000124,
     $     0.000124,0.000124,0.000124,0.000004,0.000035,0.000166,
     $     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008981,
     $     0.000124,0.000124,0.000062,0.000124,0.000124,0.000124,
     $     0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     $     0.001372,-0.009090,-0.008981,0.000124,0.000124,0.000124,
     $     0.000062,0.000124,0.000124,0.000004,0.000035,0.000166,
     $     0.000584,0.001346,0.002471,0.001372,-0.009090,-0.008981,
     $     0.000124,0.000124,0.000124,0.000124,0.000062,0.000124,
     $     0.000004,0.000035,0.000166,0.000584,0.001346,0.002471,
     $     0.001372,-0.009090,-0.008982,0.000124,0.000124,0.000124,
     $     0.000124,0.000124,0.000062/
c
      call dfill(nbfn*nbfn,0.0d0,work,1)
      do 10 iat = 1,natom
         ioff = (iat-1)*15
         do 20 i = 1,15
            do 30 j = 1,15
               work(ioff+j,ioff+i) = atdens(j,i)*0.5d0
 30         continue
 20      continue
 10   continue
      call zfold(nbfn,dens,work)
c
c   Create initial guess for density matrix in global array
c
      call ga_zero(g_dens)
      call ga_zero(g_counter)
      one = 1
      ld = 15
c
c   Correct for a factor of two along the diagonal
c
      do i = 1, ld
        atdens(i,i) = 2.0d00*atdens(i,i)
      end do
  100 itask = nga_read_inc(g_counter,one,one)
        if (itask.lt.natom) then
          ioff = itask*15
          lo(1) = ioff+1
          lo(2) = ioff+1
          hi(1) = ioff+15
          hi(2) = ioff+15
          call nga_put(g_dens,lo,hi,atdens,ld)
          go to 100
        endif
c
c
      end
      subroutine setarrays
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer one, two, dims(2)
      logical status
      one = 1
      two = 2
      g_counter = ga_create_handle()
      call ga_set_data(g_counter,one,one,MT_INT)
      status = ga_allocate(g_counter)
      call ga_zero(g_counter)

      dims(1) = nbfn
      dims(2) = nbfn
      g_dens = ga_create_handle()
      call ga_set_data(g_dens, two, dims, MT_DBL)
      status = ga_allocate(g_dens)
      call ga_zero(g_dens)

      g_schwarz = ga_create_handle()
      call ga_set_data(g_schwarz, two, dims, MT_DBL)
      status = ga_allocate(g_schwarz)
      call ga_zero(g_schwarz)

      g_fock = ga_create_handle()
      call ga_set_data(g_fock, two, dims, MT_DBL)
      status = ga_allocate(g_fock)
      call ga_zero(g_fock)

      g_work = ga_create_handle()
      call ga_set_data(g_work, two, dims, MT_DBL)
      status = ga_allocate(g_work)
      call ga_zero(g_work)

      g_ident = ga_create_handle()
      call ga_set_data(g_ident, two, dims, MT_DBL)
      status = ga_allocate(g_ident)
      call ga_zero(g_ident)

      g_orbs = ga_create_handle()
      call ga_set_data(g_orbs, two, dims, MT_DBL)
      status = ga_allocate(g_orbs)
      call ga_zero(g_orbs)

      return
      end
c
      subroutine makoverlap
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer me, lo(2), hi(2), ptr, ld(2)
      integer ld1, ld2
      me = ga_nodeid()
      call nga_distribution(g_ident, me, lo, hi)
      call nga_access(g_ident, lo, hi, ptr, ld)
      ld1 = hi(1) - lo(1) + 1
      ld2 = hi(2) - lo(2) + 1
      call setoverlap(dbl_mb(ptr),lo,hi,ld1,ld2)
      return
      end
c
      subroutine setoverlap(a,lo,hi,ld1,ld2)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer lo(2), hi(2)
      integer ld1, ld2, ii, jj
      double precision a(ld1,ld2)
      do i = 1, ld1
        ii = i + lo(1) - 1
        do j = 1, ld2
          jj = j + lo(2) - 1
          a(i,j) = s(ii,jj)
        end do
      end do
      return
      end
c
      subroutine print_ga_block(g_a)
      implicit double precision(a-h,o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer lo(2), hi(2), ptr, ld1, ld2
c
      me = ga_nodeid()
      call nga_distribution(g_a, me, lo, hi)
      ld1 = hi(1) - lo(1) + 1
      ld2 = hi(2) - lo(2) + 1
      call nga_access(g_a, lo, hi, ptr, ld)
      call dump_chunk(dbl_mb(ptr),ld1,ld2)
c
      return
      end
c
      subroutine print_ga_block_ij(g_a,tlo)
      implicit double precision(a-h,o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer lo(2), hi(2), ptr, ld1, ld2
c
      me = ga_nodeid()
      call nga_distribution(g_a, me, lo, hi)
      ld1 = hi(1) - lo(1) + 1
      ld2 = hi(2) - lo(2) + 1
      call nga_access(g_a, tlo, hi, ptr, ld)
      call dump_chunk(dbl_mb(ptr),ld1,ld2)
c
      return
      end
c
      subroutine dump_chunk(a,ld1,ld2)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer ld1, ld2
      double precision a(ld1, ld2)
      do i = 1, min(10,ld1)
        write(6,100) (a(i,j), j = 1, min(10,ld2))
      end do
      write(6,*)
  100 format(10f10.4)
      return
      end
c
      double precision function contract_matrices(g_a,g_b)
      implicit double precision(a-h,o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer lo(2), hi(2), ptr_a, ptr_b, ld, ld1, ld2
      double precision value
c
      me = ga_nodeid()
      call nga_distribution(g_a, me, lo, hi)
      ld1 = hi(1) - lo(1) + 1
      ld2 = hi(2) - lo(2) + 1
      call nga_access(g_a, lo, hi, ptr_a, ld)
      call nga_distribution(g_b, me, lo, hi)
      call nga_access(g_b, lo, hi, ptr_b, ld)
      call ga_sync
      call contract_chunk(dbl_mb(ptr_a),dbl_mb(ptr_b),ld1,ld2,value)
      contract_matrices=value
c
      return
      end
c
      subroutine contract_chunk(a,b,ld1,ld2,val)
      implicit double precision (a-h, o-z)
      include 'cscf.h'
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"
      integer ld1, ld2
      double precision a(ld1, ld2),b(ld1, ld2),val
      val = 0.0d00
      do i = 1, ld1
        do j = 1, ld2
          val = val + a(i,j)*b(i,j)
        end do
      end do
      call ga_dgop(7,val,1,'+')
      return
      end
c
      subroutine print_matrix(a)
      implicit double precision (a-h,o-z)
      include 'cscf.h'
      double precision a(*)
      integer i,j, ga_nodeid
      me = ga_nodeid()
      do i = 0, 9
        write(6,100) (me, a(nbfn*i+j),j=1,10)
  100   format(i4,10f10.4)
      end do
      write(6,*)
      return
      end
c
      subroutine print_chunk_matrix(a,ich)
      implicit double precision (a-h,o-z)
      include 'cscf.h'
      double precision a(*)
      integer i,j,ich
      do i = 0, 9
        write(6,100) (a(ich*i+j),j=1,10)
  100   format(10f10.4)
      end do
      write(6,*)
      return
      end
c
      subroutine print_triangle(a)
      implicit double precision (a-h,o-z)
      include 'cscf.h'
      double precision a(*)
      integer ij,i,j
      ij = 0
      do i = 1, 10
        write(6,100) (a(ij+j),j=1,i)
        ij = ij + i
  100   format(10f10.4)
      end do
      write(6,*)
      return
      end
