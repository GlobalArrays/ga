c $Id: scalapack.F,v 1.7 2004-11-30 17:44:54 edo Exp $
****************************************************************************
* GA to/from ScaLAPACK (square block scattered decomposition) interface    * 
*                                                                          *
* common variables:                                                        *
*     nnodes        - number of processors                                 *
*     iam           - my processor number                                  *
*     nprow/ npcol  - number of processor rows/cols in virtual proc. grid  *
*     myrow/mycol   - cordinates of my processor in virtual proc. grid     *
*                                                                          *
c***************************************************************************
c* 04/12/96  GVT  Changed the code to adapt to a new version of ScaLAPACK
c*           Giuseppe Vitillaro peppe@unipg.it
c***************************************************************************
      block data SLblock
      implicit none
c
#include "scalapack.fh"
      data init /.false./
      end


      subroutine SLinit
      implicit none
      integer  icontxt
c
#include "scalapack.fh"
      external blacs_pinfo, blacs_get, blacs_gridinit, blacs_gridinfo
c
      if(init)return
c
c**** call ga_sync before to enter in BLACS and after
      call ga_sync()
c     
      call blacs_pinfo(iam, nnodes)
c
      call FindGrid(nnodes, nprow, npcol)
c
      call blacs_get (0, 0, icontxt)
      igctxt = icontxt
c
      icontxt = igctxt
      call blacs_gridinit(icontxt, 'R', nprow, npcol)
      iSLctxt = icontxt
c
      call blacs_gridinfo(iSLctxt, nprow, npcol, myrow, mycol)
c
      init=.true.
c
      call ga_sync()
c    
      end



      subroutine FindGrid(nnodes, nprow, npcol)
c
c***  determine nprow, npcol from nnodes
c***  solution is searched in the neighborhood of the square grid 
c
      implicit none
      integer nnodes, nprow, npcol,i 
c
      npcol = int(sqrt(dble(nnodes)))
      do i = npcol, 1, -1 
         if(mod(nnodes,i).eq.0) goto 1
      enddo
1     continue
      npcol = i
      nprow = nnodes/npcol
      end



      subroutine blockpos(m, nb, idxb, pos)
c
c***  for m row/col find SL block num and its position within that block 
c
      implicit none
#include "scalapack.fh"
c
      integer m, nb, idxb, pos
      integer T
c
      T = nb * nnodes
      idxb = m/T
      pos = Mod(Mod(m,T),nb)
      end 



      subroutine blockplace(mypcord, npnum, dim, nb, lo,hi) 
c
c***  calculates range of indexes in the SBS array for data
c***  that the given processor holds (called for rows & cols)
c
      implicit none
c
      integer  dim, nb,lo, hi
      integer  np_or_q, npnum, mypcord 
      integer totblocks, minblocksproc, extrablocks, corr,
     *     BlocksBeforeMe 
      integer numroc
c
      np_or_q = numroc(dim, nb, mypcord, 0, npnum)
      if(np_or_q .lt. 1)then    ! this processor des not have any data 
         lo =  0
         hi = -1
         return
      endif
c
      totblocks    = (dim +nb-1)/nb
      minblocksproc = totblocks/npnum
      extrablocks  = Mod(totblocks,npnum) 
c
      if(extrablocks .le.  mypcord)then
         BlocksBeforeMe = minblocksproc * mypcord + extrablocks 
         if(extrablocks.gt.0)then       ! check if there are extra blocks
            corr = Mod(dim,nb)
            if(corr.ne.0)then           ! partial block might be among them
               BlocksBeforeMe = BlocksBeforeMe -1
            endif
         endif
      else
         BlocksBeforeMe = minblocksproc * mypcord + mypcord 
      endif
      lo = 1  + BlocksBeforeMe * nb + corr
      hi = lo + np_or_q -1
      end




      subroutine ga_to_SL(g_a, dim1, dim2, nbr, nbc, s_a, lda, np, nq)
c
c***  transforms a GA to SL format
c***  reference: Dongarra et al, 'A look at scalable dense lin. alg. libs'
c
      implicit none
#include "scalapack.fh"
      integer  nbr, nbc, lda, np, nq, dim1, dim2
      integer g_a
      double  precision  s_a(lda,nq)
      integer row, col, tcol, trow, rbase, cbase
      integer pcol, prow

c***  Synchronize at the beginning
c    
      rbase = 1
      cbase = 1
c
      Trow = nbr * nprow
      Tcol = nbc * npcol
c     
      do col  = 1, dim2, nbc
         pcol = mod(col,Tcol)/nbc        ! processor column that holds "col"
         if(mycol.eq.pcol) then
            do row  = 1, dim1, nbr
               prow = mod(row,Trow)/nbr  ! processor row that holds "row"
               if(myrow.eq.prow) then
                  if(rbase.gt.np .or. cbase .gt. nq) then
                     call ga_error(' ga_to_SL: rows/cols error ',0)
                  endif
                  call ga_get(g_a,row,min(row+nbr -1,dim1),col,
     &                    min(col+nbc -1,dim2), s_a(rbase,cbase), lda) 
                  rbase = rbase + nbr
               endif
            enddo
            rbase = 1
            cbase = cbase + nbc
         endif
      enddo

c**** ... and at the end
      end



      subroutine ga_from_SL(g_a, dim1, dim2, nbr, nbc, s_a, lda, np, nq)
c
c***  transforms a matrix from SL to GA format
c***  reference: Dongarra et al, 'A look at scalable dense lin. alg. libs'
c
      implicit none
#include "scalapack.fh"
      integer g_a, nbr, nbc, lda, np, nq, dim1, dim2
      double  precision s_a(lda,nq)
      integer row, col, tcol, trow, rbase, cbase
      integer pcol, prow
c
c**** Syncronize at the beginning
c   
      rbase = 1
      cbase = 1
c
      Trow = nbr * nprow
      Tcol = nbc * npcol
c
      do col  = 1, dim2, nbc
         pcol = mod(col,Tcol)/nbc        ! processor column that holds "col"
         if(mycol.eq.pcol) then
            do row  = 1, dim1, nbr
               prow = mod(row,Trow)/nbr  ! processor row that holds "row"
               if(myrow.eq.prow) then
                  if(rbase.gt.np .or. cbase .gt. nq)
     &               call ga_error(' ga_from_SL: rows/cols error ',0)
                  call ga_put(g_a,row,min(row+nbr-1,dim1),col,
     &                    min(col+nbc-1,dim2), s_a(rbase,cbase), lda) 
                  rbase = rbase + nbr
               endif
            enddo
            rbase = 1
            cbase = cbase + nbc
         endif
      enddo

c**** ... and at the end
      end


c**********************************************************************
c* Source : ga_lu_solve
c* Scope  : LU factorization
c*
c* 02/16/94  Jarek Nieplocha 
c* 04/12/96  GVT  Adapted the code to the new version of ScaLAPACK
c*           Giuseppe Vitillaro peppe@unipg.it
c**********************************************************************      
c
c length of scalapack array descriptor
# define DSCLEN 10
c
      subroutine ga_lu_solve(trans,g_a, g_b)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c
      character*1 trans         ! transpose or not  
      integer g_a               ! coefficient matrix A 
      integer g_b               ! rhs matrix, B, overwritten on exit by
                                ! the solution vector, X
c
c     solve the set of linear equations 
c
c           AX = B
c
c     with possibly multiple rhs stored as columns of matrix B
c
c     the matrix A is not destroyed
c
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      integer ha, adra          !A
      integer hb, adrb          !B
      integer hi, adri          !ipiv
c
c
      logical oactive           ! true iff this process participates
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
c
      integer mpA, nqA          ! number of rows/cols of A held by the processor
      integer mpB, nqB          ! number of rows/cols of B held by the processor
      integer info
      integer me
      integer n,lda, ldb, elemA, elemB
      integer numroc

      integer descA(DSCLEN), descB(DSCLEN) ! descriptors for distr. matrixes A,B
      integer nb                           ! block size for A and rhs
      data nb /64/
c
c***  check environment
c
      me     = ga_nodeid()
c
c***  check GA info for input arrays
c
      call ga_check_handle(g_a, 'ga_lu_solve: a')
      call ga_check_handle(g_b, 'ga_lu_solve: b')
      call ga_inquire(g_a, typeA, dimA1, dimA2)
      call ga_inquire(g_b, typeB, dimB1, dimB2)
c
      if (dimA1 .ne. dimA2) then
         call ga_error('ga_lu_solve: g_a must be square matrix ', 1)
      else if(dimA1 .ne. dimB1) then
         call ga_error('ga_lu_solve: dims of A and B do not match ',1)
      endif
c
      n = dimA1
c
c***  initialize SL interface
c 
      call SLinit()
c
c***  find SBS format parameters
c
      mpA = numroc(n, nb,myrow, 0, nprow)
      nqA = numroc(n, nb,mycol, 0, npcol)
      mpB = numroc(dimB1, nb,myrow, 0, nprow)
      nqB = numroc(dimB2, nb,mycol, 0, npcol) 
      lda = max(1,mpA)
      ldb = max(1,mpB)

      oactive = .true.
      if (oactive) then
c
c***     allocate A,B and ipiv arrays
c     
         elemA= mpA*nqA
         status = .true.
         if(elemA.ne.0)status = ma_push_get(MT_DBL,elemA,'a',ha,adra)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed A ', -1)
c
         elemB= mpB*nqB
         if(elemB.ne.0)status = ma_push_get(MT_DBL,elemB,'b',hb,adrb)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed B ', -1)
c
         status = ma_push_get(MT_INT,mpA+nb,'ipiv',hi,adri)
         if(.not.status) 
     &       call ga_error('ga_lu_solve: mem alloc failed ipiv ', -1)
c
c
c***     copy g_a to A and g_b to B using the SBS SL format 
c
         call ga_to_SL(g_a, n, n, nb, nb, dbl_mb(adrA), lda, mpA,nqA)
         call ga_to_SL(g_b, n,dimB2, nb,nb, dbl_mb(adrB), ldb, mpB, nqB)
c     
c***     fill SCALAPACK matrices descriptors
c
	 call descinit(descA, dimA1, dimA2, nb, nb, 0, 0, 
     &                 iSLctxt, lda, info)
         if(info.ne.0) call ga_error(' ga_lu_solve: descinit A failed ',
     &                               -info)
         call descinit(descB, dimB1, dimB2, nb, nb, 0, 0, 
     &                 iSLctxt, ldb, info)
         if(info.ne.0) call ga_error(' ga_lu_solve: descinit B failed ',
     &                               -info)
c
c***     LU factorization
c
         call ga_sync()
	 call pdgetrf(n, n, dbl_mb(adrA), 1, 1, descA, 
     &                int_mb(adrI), info)
         call ga_sync()
c
c
         if(info.eq.0)then
c
c***        solve: forward/backward substitution
c
            call pdgetrs(trans, n, dimB2, dbl_mb(adrA), 1, 1, descA,
     &                   int_mb(adrI), dbl_mb(adrB), 1, 1, descB, info)
            if(info.eq.0)then
c
c***           copy solution matrix back to g_b
c
               call ga_from_SL(g_b, dimB1, dimB2, nb, nb, dbl_mb(adrb),
     &                         mpB, ldb, nqB)
            else
               call ga_error(' ga_lu_solve: pdgetrs failed ', -info)
            endif
c
         else
            call ga_error(' ga_lu_solve: pdgetrf failed ', -info)
         endif
c
c***     deallocate work/SL arrays
c
         status = ma_pop_stack(hi)
         if(elemB.ne.0) status = status .and. ma_pop_stack(hb)
         if(elemA.ne.0) status = status .and. ma_pop_stack(ha)
         if(.not.status)
     &       call ga_error('ga_lu_solve:ma_pop_stack failed',0)
      endif
c
      call ga_sync()
      end

c     alt entry point for C that avoids char args
c
	subroutine ga_lu_solve_alt(tr, g_a, g_b)
	implicit none
        integer g_a, g_b, tr
        if(tr.eq.0) then
           call ga_lu_solve('n',g_a, g_b) 
        else
           call ga_lu_solve('y',g_a, g_b) 
        endif
        end


c**********************************************************************
c* Source : ga_llt
c* Scope  : interface between GA and LLT SCALAPACK routines
c*
c* 04/12/96  GVT  First Implementation
c*           Giuseppe Vitillaro peppe@unipg.it
c* 08/28/96  Jarek Nieplocha, some cleanup and optimizations 
c**********************************************************************

#define NB_ 64
#define DSCLEN 10
#define ga_dnormF(g_a) sqrt(ga_ddot(g_a, g_a)
      
c**********************************************************************
c* subroutine: ga_zeroUL
c*
c*             Set to zero the L/U triangle part of a NxN
c*             double precision global array A
c***********************************************************************
      subroutine ga_zeroUL(uplo, g_A)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
c****
      character*1 uplo         ! (input) 'U' or 'L'
      integer     g_A          ! (input/output)
      logical status
      integer dimA1, dimA2, typeA
      integer me, nproc
      integer n
      integer i, j, hBUF, adrBUF

c**** Check Environment
      nproc = ga_nnodes()
      me    = ga_nodeid()
      
c**** Check GA info for input array A
      call ga_check_handle(g_A, 'ga_zeroUL: A')
      call ga_inquire(g_A, typeA, dimA1, dimA2)
c****
      if (dimA1.ne.dimA2) then
         call ga_error('ga_zeroUL: g_A must be a square matrix ', 1)
      endif
      if (typeA.ne.MT_DBL) then
         call ga_error('ga_zeroUL: g_A must be double precision ', 1)
      endif
c****      
      n = dimA1
      
c**** Allocate BUF
      status = ma_push_get(MT_DBL, n, 'BUF', hBUF, adrBUF)
      if (.not.status)
     &     call ga_error(' ga_zeroUL: mem alloc failed BUF ', -1)
      
      call ga_sync()

      do i = me+1, n, nproc
         call ga_get(g_A, 1, n, i, i, dbl_mb(adrBUF), n)
         if (uplo.eq.'L') then
c****       case L: make zero the upper triangle            
            do j = 1,i-1
               dbl_mb(adrBUF + j - 1) = 0.d0
            end do !j
         elseif (uplo.eq.'U') then
c****       case U: make zero the lower triangle            
            do j = i+1, n
               dbl_mb(adrBUF + j - 1) = 0.d0
            end do !j
         else
            call ga_error('ga_symUL: uplo must be L or U ', 1)
         endif
         call ga_put(g_A, 1, n, i, i, dbl_mb(adrBUF), n)
      end do    !i
c
      status = ma_pop_stack(hBUF)
      call ga_sync()
      end


c**********************************************************************
c* subroutine: ga_symUL
c*
c*             Make a symmetric square matrix from
c*             double precision global array A in L/U triangle format
c***********************************************************************
      subroutine ga_symUL(uplo, g_A)

      implicit none
      
#include "mafdecls.fh"
#include "global.fh"
c****      
      character*1 uplo         ! (input) 'U' or 'L'
      integer     g_A          ! (input/output)
      logical status
      integer dimA1, dimA2, typeA
      integer me, nproc
      integer n
      integer i, j, hBUF, adrBUF, idx      

c**** Check Environment
      nproc = ga_nnodes()
      me    = ga_nodeid()
      
c**** Check GA info for input array A
      call ga_check_handle(g_A, 'ga_symUL: A')
      call ga_inquire(g_A, typeA, dimA1, dimA2)
c****
      if (dimA1.ne.dimA2) then
         call ga_error('ga_symUL: g_A must be a square matrix ', 1)
      endif
      if (typeA.ne.MT_DBL) then
         call ga_error('ga_symUL: g_A must be double precision ', 1)
      endif
c****      
      n = dimA1
c**** Allocate BUF
      status = .true.
      status = ma_push_get(MT_DBL, n, 'BUF', hBUF, adrBUF)
      if (.not.status)
     &     call ga_error(' ga_symUL: mem alloc failed BUF ', -1)
      
      call ga_sync()

      do i = me+1, n, nproc
         call ga_get(g_A, 1, n, i, i, dbl_mb(adrBUF), n)
         if (uplo.eq.'L') then
c****       case L : make sure upper triangle is zero
c****                and under the diag. is 2 times                 
            do j = 1, n
               idx = adrBUF + j - 1
               if (j.lt.i) then
                  dbl_mb(idx) = 0.d0
               elseif (j.gt.i) then
                  dbl_mb(idx) = 2.d0*dbl_mb(idx)
               endif
            end do ! j
         elseif (uplo.eq.'U') then
c****       case U : make sure lower triangle is zero
c****                and over the diag is 2 times               
            do j = 1, n
               idx = adrBUF + j - 1
               if (j.gt.i) then
                  dbl_mb(idx) = 0.d0
               elseif (j.lt.i) then
                  dbl_mb(idx) = 2.d0*dbl_mb(idx)
               endif
            end do ! j
         else
            call ga_error('ga_symUL: uplo must be L or U ', 1)
         endif
         call ga_put(g_A, 1, n, i, i, dbl_mb(adrBUF), n)
      end do    ! i
      
      status = ma_pop_stack(hBUF)
      call ga_symmetrize(g_A)
      end

      
c***********************************************************************
c* function :  ga_cholesky
c*
c*             Compute the Cholesky factorization of an NxN
c*             double precision symmetric positive definite matrix.
c*
c*             On succesful exit A will contain the L/U factor
c*             on the lower/upper triangular part of the matrix
c*                   
c*             It calls the PDPOTRF ScaLAPACK routine.
c*
c*             It returns
c*                   = 0 : successful exit
c*                   > 0 : the leading minor of this order
c*                         is not positive definite and the
c*                         factorization could not be completed
c*
c***********************************************************************
      integer function ga_cholesky(uplo, g_a)
      integer ga_llt_f
      ga_cholesky = ga_llt_f(uplo, g_A, -1)
      end



c***********************************************************************
c* function :  ga_llt_f
c*
c*             Internal function to compute the Cholesky factorization of 
c*             an NxN double precision symmetric positive definite GA.
c*
c*             Note: applications should use ga_cholesky
c*
c*             On succesful exit A will contain the L/U factor
c*             on the lower/upper triangular part of the matrix
c*
c*             This if (hsA.eq.-1), otherwise internal state
c*             is saved for future references. This is
c*             reserved for internal use. Users should
c*             not try to set hsA to something
c*             different from -1 if they do not
c*             know what they are doing! On exit hsA will
c*             contain the local A output array in SLS format.      
c*
c*             It calls the PDPOTRF ScaLAPACK routine.
c*      
c*             It returns
c*                   = 0 : successful exit
c*                   > 0 : the leading minor of this order
c*                         is not positive definite and the
c*                         factorization could not be completed      
c***********************************************************************
      integer function ga_llt_f(uplo, g_A, hsA)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c****
      character*1 uplo         ! (input) 'U' or 'L'
      integer     g_A          ! (input/output)
      integer     hsA          ! (input/output)
c****
      logical status
      integer hA, adrA
      integer dimA1, dimA2, typeA
      integer mpA, nqA
      integer info
      integer me
      integer n, ldA, elemA
      integer numroc
      integer nb
      integer descA(DSCLEN)
      data nb /NB_/

c**** Check Environment
      me = ga_nodeid()

c**** Check GA info for input array A
      call ga_check_handle(g_A, 'ga_llt_f: A')
      call ga_inquire(g_A, typeA, dimA1, dimA2)
c****
      if (dimA1.ne.dimA2) then
         call ga_error('ga_llt_f: g_A must be a square matrix ', 1)
      endif
c****
      n = dimA1
c**** Initialize SL Interface
      call SLinit()

c**** Find SBS format parameters
      mpA = numroc(n, nb, myrow, 0, nprow)
      nqA = numroc(n, nb, mycol, 0, npcol)
      ldA = max(1,mpA)

      call ga_sync()

c**** Allocate A
      elemA = mpA * nqA
      status = .true.
      if (elemA.ne.0)
     &     status = ma_push_get(MT_DBL, elemA, 'A', hA, adrA)
      if (.not.status)
     &     call ga_error('ga_llt_f: mem alloc failed A ', -1)

c**** Copy ga to A using SBS ScaLAPACK format      
      call ga_to_SL(g_a, n, n, nb, nb, dbl_mb(adrA), ldA, mpA, nqA)

c**** Fill ScaLAPACK matrix descriptor
      call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     &     iSLctxt, ldA, info)
      if(info.ne.0) call ga_error('ga_llt_f: descinit A failed',-info)

c**** CALL ScaLAPACK PDPOTRF LLT factorization routine *******
      call ga_sync()
      call PDPOTRF(uplo, n, dbl_mb(adrA), 1, 1, descA, info)
c*************************************************************
      
      if (info.eq.0) then
c****    Copy solution matrix back to A if hsA==-1
c****    and zero the L/U triangle part according to uplo         
         if (hsA.eq.-1) then
            call ga_from_SL(g_A, dimA1, dimA2, nb, nb, dbl_mb(adrA),
     &           mpA, ldA, nqA)
            
            call ga_zeroUL(uplo, g_A)
         endif
      elseif (info.lt.0) then
         call ga_error('ga_llt_f: PDPOTRF failed ', -info)
      endif
      ga_llt_f = info
c
c**** If the SL A array was allocated
      if (elemA.ne.0) then
c****    and hsA==-1 or info>0 (i.e. fact. cannot be completed)
c****    then deallocate the SL A MA array
         if ((hsA.eq.-1).or.(info.gt.0)) then
            status = ma_pop_stack(hA)
c****    otherwise just save the hA MA handle            
         else
            hsA = hA
         endif
      endif
c
      call ga_sync()
      end



c***********************************************************************
c* subroutine: ga_llt_s
c*
c*             Solves a system of linear equations
c*             
c*             A * X = B
c*             
c*             where A is the lower triangle L or the upper
c*             triangular Cholesky factor U of a NxN double precision
c*             symmetric positive definite global array (LL' or U'U)
c*             obtained from ga_llt_f routine.
c*
c*             If (hsA.eq.-1) then you "must" give in input
c*             the lower or upper triangular Cholesky factor.
c*             For internal use, if (hsA.ne.-1) then hsA is
c*             the MA handle of the Cholesky L/U factor already
c*             in SBS SL format.
c*
c*             On successful exit B will contain the solution X.      
c*      
c*             It calls the PDPOTRS ScaLAPACK routine.
c***********************************************************************
      subroutine ga_llt_s(uplo, g_A, g_B, hsA)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c****
      character*1 uplo         ! (input) 'U' or 'L'
      integer     g_A          ! (input)
      integer     g_B          ! (input/ouput)
      integer     hsA          ! (input)
c****
      logical status
      integer hA, adrA
      integer hB, adrB
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
      integer mpA, nqA
      integer mpB, nqB
      integer info
      integer me
      integer n, ldA, ldB, elemA, elemB
      integer numroc
      integer nb
      integer descA(DSCLEN), descB(DSCLEN)
c
      data nb /NB_/
c
c**** Check the Environment
      me = ga_nodeid()
      
c**** Check GA info for input array F, B
      call ga_check_handle(g_A, 'ga_llt_s: A')
      call ga_check_handle(g_B, 'ga_llt_s: B')
      call ga_inquire(g_A, typeA, dimA1, dimA2)
      call ga_inquire(g_B, typeB, dimB1, dimB2)
c****
      if (dimA1.ne.dimA2) then
         call ga_error('ga_llt_s: g_A must be a square matrix ', 1)
      else if (dimA1.ne.dimB1) then
         call ga_error('ga_llt_s: dims of A and B do not match ', 1)
      endif
c
      n = dimA1
      
c**** Initialize SL Interface
      call SLinit()

c**** Find SBS format parameters
      mpA = numroc(n, nb, myrow, 0, nprow)
      nqA = numroc(n, nb, mycol, 0, npcol)
      mpB = numroc(n, nb, myrow, 0, nprow)
      nqB = numroc(n, nb, mycol, 0, npcol)
      ldA = max(1,mpA)
      ldB = max(1,mpB)

      call ga_sync()

c**** Allocate A or retrieve it from the ga_llt_f calls
      elemA = mpA * nqA
      status = .true.
      if (elemA.ne.0) then
         if (hsA.eq.-1) then
            status = ma_push_get(MT_DBL, elemA, 'A', hA, adrA)
            if (.not.status)
     &           call ga_error('ga_llt_s: mem alloc failed A ', -1)
c****       copy g_A to A using SBS SL format            
            call ga_to_SL(g_A, n, n, nb, nb, dbl_mb(adrA),
     &           ldA, mpA, nqA)
         else
            hA = hsA
            status = ma_get_index(hA, adrA)
            if (.not.status)
     &           call ga_error('ga_llt_s: get index failed A ', -1)
c****       if hsA was given the SBS SL conversion is avoided            
         endif
      endif
      
c**** Allocate B and copy g_B to B in SBS SL format
      elemB = mpB * nqB
      if (elemB.ne.0) status = ma_push_get(MT_DBL, elemB, 'B', hB, adrB)
      if (.not.status)
     &     call ga_error('ga_llt_s: mem alloc failed B ', -1)
      call ga_to_SL(g_B, n, dimB2, nb, nb, dbl_mb(adrB), ldB, mpB, nqB)
c
c**** Fill ScaLAPACK matrix descriptors for A and B
      call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     &     iSLctxt, ldA, info)
      if(info.ne.0) call ga_error(' ga_llt_s: descinit A failed ',
     &     -info)
      
      call descinit(descB, dimB1, dimB2, nb, nb, 0,0,iSLctxt, ldB, info)
      if(info.ne.0) call ga_error('ga_llt_s: descinit B failed', -info)
     
c**** CALL ScaLAPACK PDPOTRS solver routine ***********************
      call ga_sync()
      call PDPOTRS(uplo, n, dimB2,
     &     dbl_mb(adrA), 1, 1, descA,
     &     dbl_mb(adrB), 1, 1, descB,
     &     info)
c******************************************************************

      if (info.eq.0) then
c****    copy solution matrix back to g_B
         call ga_from_SL(g_B, dimB1, dimB2, nb, nb, dbl_mb(adrB),
     &                   mpB, ldB, nqB)
      else
         call ga_error(' ga_llt_s: PDPOTRS failed: ', -info)
      endif

c**** deallocate work/SL arrays
c**** note that should not be others MA allocations
c**** between ga_llt_f and ga_llt_s beside A and B      
      if (elemB.ne.0) status = ma_pop_stack(hB)
      if (elemA.ne.0) status = ma_pop_stack(hA)
c
      call ga_sync()
      end


c***********************************************************************
c* function :  ga_llt_i
c*
c*             It computes the inverse of a global array
c*             that is the lower triangle L or the upper
c*             triangular Cholesky factor U of a NxN double precision
c*             symmetric positive definite global array (LL' or U'U)
c*             obtained from the ga_llt_f routine.
c*
c*             If (hsA.eq.-1) then you "must" give in input
c*             the lower or upper triangular Cholesky factor.
c*             For internal use, if (hsA.ne.-1) then hsA is
c*             the MA handle of the Cholesky L/U factor already
c*             in SBS SL format.
c*
c*             On successful exit A will contain the inverse.
c*      
c*             It calls the PDPOTRI ScaLAPACK routine.
c*      
c*             It returns
c*                   = 0 : successful exit
c*                   > 0 : it returns the index i of the (i,i)
c*                         element of the factor L/U that is zero and,
c*                         so, the inverse could not be computed        
c***********************************************************************
      integer function ga_llt_i(uplo, g_A, hsA)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c****
      character*1 uplo         ! (input) 'U' or 'L'
      integer     g_A          ! (input/output)
      integer     hsA          ! (input/output)
c****
      logical status
      integer hA, adrA
      integer dimA1, dimA2, typeA
      integer mpA, nqA
      integer info
      integer nproc, me
      integer n, ldA, elemA
      integer numroc
      integer nb
      integer descA(DSCLEN)
c****      
      data nb /NB_/
c
c**** Check Environment
      nproc = ga_nnodes()
      me = ga_nodeid()

c**** Check GA info for input array A
      call ga_check_handle(g_A, 'ga_llt_i: A')
      call ga_inquire(g_A, typeA, dimA1, dimA2)

c**** Check that is actually a square matrix
      if (dimA1.ne.dimA2) then
         call ga_error('ga_llt_i: g_A must be a square matrix ', 1)
      endif
c
      n = dimA1

c**** Initialize SL Interface
      call SLinit()

c**** Find SBS format parameters
      mpA = numroc(n, nb, myrow, 0, nprow)
      nqA = numroc(n, nb, mycol, 0, npcol)
      ldA = max(1, mpA)

      call ga_sync()

c**** Allocate A or retrieve it from ga_llt_f call
      elemA = mpA * nqA
      if (elemA.ne.0) then
         status = .true.
         if (hsA.eq.-1) then
            status = ma_push_get(MT_DBL, elemA, 'A', hA, adrA)
            if (.not.status)
     &           call ga_error(' ga_llt_i: mem alloc failed A ', -1)
c****       copy g_A to A using SBS SL format
            call ga_to_SL(g_A, n, n, nb, nb, dbl_mb(adrA),
     &           ldA, mpA, nqA)
         else
            hA = hsA
            status = ma_get_index(hA, adrA)
            if (.not.status)
     &           call ga_error(' ga_llt_i: get index failed A ', -1)
c****       if hsA was given the SBS SL conversion is avoided
         endif
      endif

      call ga_sync()

c**** Fill ScaLAPACK matrix descriptor for A
      call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     &     iSLctxt, ldA, info)
      if (info.ne.0) call ga_error(' ga_llt_i: descinit A failed ',
     &     -info)

c**** CALL ScaLAPACK PDPOTRI matrix inverter **********************
      call ga_sync()
      call PDPOTRI(uplo, n, dbl_mb(adrA), 1, 1, descA, info)
c******************************************************************

      if (info.eq.0) then
c****    Copy the inverse matrix back to A
c****    and symmetrize it         
         call ga_from_SL(g_A, dimA1, dimA2, nb, nb, dbl_mb(adrA),
     &        mpA, ldA, nqA)
         
         call ga_sync()
         
         call ga_symUL(uplo, g_A)
      elseif (info.lt.0) then
         call ga_error(' ga_llt_i: PDPOTRI failed ', -info)
      endif

      ga_llt_i = info

c**** deallocate work/SL arrays
c**** note that should not be others MA allocations
c**** between ga_llt_f and ga_llt_i
      if(elemA.ne.0) status = ma_pop_stack(hA)
c
      call ga_sync()
      end

      
c***********************************************************************
c* function :  ga_llt_solve
c*
c*             Solves a system of linear equations
c*             
c*             A * X = B
c*
c*             using the Cholesky factorization of an NxN
c*             double precision symmetric positive definite
c*             global array A.
c*      
c*             On successful exit B will contain the solution X.      
c*      
c*             It calls the ga_llt_f and ga_llt_s and so it
c*             actually refer to PDPORTF/PDPOTRS ScaLAPACK
c*             routines.
c*      
c*             It returns
c*                   = 0 : successful exit
c*                   > 0 : the leading minor of this order
c*                         is not positive definite and the
c*                         factorization could not be completed      
c***********************************************************************      
      integer function ga_llt_solve(g_A, g_B)
      implicit none
      character*1 uplo         ! (internal parameter) 'U' or 'L'
      integer     g_A          ! (input)
      integer     g_B          ! (input/output)
      integer     hsA
      integer     irc
      integer     ga_llt_f
c
c**** call the Cholesky factorization routine
      hsA  = 0
      uplo = 'L'
      irc  = ga_llt_f(uplo, g_A, hsA)

c**** check if the factorization is OK
      if (irc.eq.0) then
c****    if the factorization is OK then solve the system
c****    call the ga_llt_s internal interface
         call ga_llt_s(uplo, g_A, g_B, hsA)
         ga_llt_solve = 0
      else
c****    if the factorization is not OK just return the error
         ga_llt_solve = irc
      endif
      end


c***********************************************************************
c* function :  ga_spd_invert
c*
c*             It computes the inverse of a  double precision
c*             using the Cholesky factorization of a NxN double
c*             precision symmetric positive definite global array A.
c*
c*             On successful exit A will contain the inverse.
c*      
c*             It calls the ga_llt_f and ga_llt_i and so it
c*             actually refer to PDPORTF/PDPOTRI ScaLAPACK
c*             routines.      
c*      
c*             It returns
c*                   = 0 : successful exit
c*                   > 0 : the leading minor of this order
c*                         is not positive definite and the
c*                         factorization could not be completed            
c*                   < 0 : it returns the index i of the (i,i)
c*                         element of the factor L/U that is zero and,
c*                         so, the inverse could not be computed        
c***********************************************************************
      integer function ga_spd_invert(g_A)
      implicit none
      character*1 uplo         ! (internal parameter) 'U' or 'L'
      integer     g_A          ! (input)
      integer     hsA
      integer     ircF, ircI
      integer     ga_llt_f
      integer     ga_llt_i
c
c**** call the Cholesky factorization routine
      hsA  = 0
      uplo = 'L' 
      ircF = ga_llt_f(uplo, g_A, hsA)
c
c**** check if the factorization is OK
      if (ircF.eq.0) then
c****    if the factorization is OK then invert the matrix
c****    call the ga_llt_i internal interface
         ircI = ga_llt_i(uplo, g_A, hsA)
         if (ircI.eq.0) then
            ga_spd_invert = 0
         else
            ga_spd_invert = -ircI
         endif
      else
c****    if the factorization is not OK just return the error
         ga_spd_invert = ircF
      endif
      end

c***********************************************************************
c* function: ga_solve
c*
c*             Solves a system of linear equations
c*             
c*             A * X = B
c*
c*             It first will call the Cholesky factorization
c*             routine and, if sucessfully, will solve the system
c*             with the Cholesky solver. If Cholesky will be not
c*             be able to factorize A, then it will call the LU
c*             factorization routine and will solve the system
c*             with forward/backward substitution.      
c*      
c*             On B will contain the solution X.      
c*      
c*             It calls the ga_llt_f and ga_llt_s and ga_lu_solve
c*             and so it actually refer to PDGETRF/PDGETRS and
c*             PDPORTF/PDPOTRS ScaLAPACK routines.
c*
c*             It returns
c*                   = 0 : Cholesky factoriztion was succesful
c*                   > 0 : the leading minor of this order
c*                         is not positive definite and Cholesky 
c*                         factorization could not be completed
c*
c*    Note: High overhead due to multiple conversions to/from scalapack
c*          format can be avoided if ga_llt_solve and ga_lu_solve
c*          were not used as building blocks - possible optimization
c*          for the future (JN) 
c***********************************************************************
      integer function ga_solve(g_A, g_B)
      implicit none
      integer     g_A          ! (input)
      integer     g_B          ! (input/output)
      integer     irc
      integer     ga_llt_solve
c
c**** Call Cholesky solver as the first try      
      ga_solve = ga_llt_solve(g_A, g_B)
      if (ga_solve.eq.0) then
c****    if Cholesky was succcesully just return 
      else
c****    else if Cholesky failed
         call ga_lu_solve('N', g_A, g_B)
      endif
      end
c******************************************************************
c     
c     ga_pdsyev
c
c     interface into scalapack's Hermitian eigensolver
c     
c     ga interface to scalapack
c     
c     g_a .g_c(*,i) = eval(i).g_b(*,i)
c     
c     assume equal size blocks ...
c     
c******************************************************************
c     
      subroutine ga_pdsyev(g_a, g_b, eval)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c     
      integer g_a               ! matrix A
      integer g_b               ! matrix B
c
c all eigenvalues are real
c
      double precision eval(*)
c     
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      character*1 jobz, range, uplo
c
      integer ha, adra          !A
      integer hb, adrb          !B
c
c
      logical oactive           ! true iff this process participates
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
c
      integer mpA, nqA          ! rows/cols of A held by the processor
      integer mpB, nqB          ! rows/cols of B held by the processor
c
      integer me
      integer lda, ldb
      integer elemA,elemB
      integer numroc
      
      integer nb                ! block size
      integer descA(9), descB(9) !descriptor for scalapack
      
c     
      integer ngaps, hgap, adrgaps
      integer iclu, hclustr, adrclustr
      integer if, hfail, adrfail
      integer liwork,hiwork, adriwork
      integer lcwork,hcwork, adrcwork
c
      integer nn,mq0, np0
      double precision vl, vu, abstol, orfac
      integer il, iu
      integer m, nz
      integer n
      integer info

      integer iii
      double precision pdlamch
      external pdlamch
      integer iceil
!      intrinsic max
c     
c     processor dependent; machine dependent
c     
!      data nb/64/
       data nb/16/
c     
c***  check environment
c     
      me     = ga_nodeid()
c     
c***  check GA info for input arrays
c     
      call ga_check_handle(g_a, 'ga_pdsyev: a')
      call ga_check_handle(g_b, 'ga_pdsyev: b')
c     
      call ga_inquire(g_a, typeA, dimA1, dimA2)
      call ga_inquire(g_b, typeB, dimB1, dimB2)
      if(dimA1.ne.dima2) call ga_error(
     %     'ga_pdsyev: matrix A not square ',0)
      n=dima1
      if(dimb1.ne.dimb2) call ga_error(
     %     'ga_pdsyev: matrix B not square ',0)
      if(dimb1.ne.n) call ga_error(
     %     'ga_pdsyev: size matrix B and B differ ',0)
      

c     
c     
c***  initialize SL interface
c     
      call SLinit()
c     
c***  find SBS format parameters
c     
c     
      mpA = numroc(dimA1, nb, myrow, 0, nprow)
      nqA = numroc(dimA2, nb, mycol, 0, npcol)
c     
      mpB = numroc(dimB1, nb, myrow, 0, nprow)
      nqB = numroc(dimB2, nb, mycol, 0, npcol) 
c     
c     
      lda = max(1,mpA)
      ldb = max(1,mpB)
c     
c     
c     let scalapack check for errors
c     
c     should check to see if this is a compute node
c     check to see how this works in the new data server model
c     
      oactive = .true.
      if ( mpA*nqA .eq. 0 ) oactive = .false.
      if ( mpB*nqB .eq. 0 ) oactive = .false.
c     
      call ga_sync()
c     
!      if (oactive) then
      if (.true.) then
c     
         elemA= mpA*nqA
         status = .true.
         if(elemA.ne.0)status =
     $        ma_push_get(MT_DBL,elemA,'a',ha,adra)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed A ', -1)
c     
         elemB= mpB*nqB
c     
         if(elemB.ne.0)status =
     $        ma_push_get(MT_DBL,elemB,'b',hb,adrb)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed B ', -1)
c     
         ngaps = nprow*npcol
         if(ngaps.ne.0)status =
     $        ma_push_get(MT_DBL, ngaps ,'gap',hgap,adrgaps)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed gaps  ', -1)
c     
         iclu = 2*nprow*npcol
         iclu = 2*n
         if(iclu.ne.0)status =
     $        ma_push_get(MT_INT, iclu ,'iclustr',hclustr,adrclustr)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed iclustr  ', -1)
c     
         if = n
         if(if.ne.0) then
            status =
     $           ma_push_get(MT_INT, if ,'ifail',hfail,adrfail)
         endif
         if(.not.status)
     &        call ga_error('ga_pdsyev: mem alloc failed ifail  ', -1)
         
c     
c***  copy g_a to A and g_b to B using the block cyclic scalapack format 
c     
         
         call ga_to_SL(g_a, dimA1, dimA2, nb, nb,
     $        dbl_mb(adrA), lda, mpA, nqA)
c     
         call ga_to_SL(g_b, dimB1, dimB2, nb,nb,
     $        dbl_mb(adrB), ldb, mpB, nqB)
c     
c***  fill SCALAPACK matrix descriptors
c     
         call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     $        iSLctxt, lda, info)
         if(info.ne.0) call ga_error(' ga_pdsyev: descinit A failed ',
     .        -info)
c     
         call descinit(descB, dimB1, dimB2, nb, nb, 0, 0,
     .        iSLctxt, ldb, info)
         if(info.ne.0) call ga_error(' ga_pdsyev: descinit B failed ',
     .        -info)
c     
         jobz = 'V'
         range = 'A'
         uplo = 'L'
         vl = 0.d0
         vu = 0.d0
         il = 0
         iu = 0
         nz = 0
c     
c     ability to deal with orthonormality ; let's just
c     have the regular scalapack stuff for the moment
c     
         liwork = 6*max(n, nprow*npcol+1, 4) 
         if(liwork.ne.0)status =
     $        ma_push_get(MT_INT, liwork ,'iwork',hiwork,adriwork)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed iwork  ', -1)
c     
         nn = max(n, nb, 2)
         np0 = numroc(nn, nb, 0, 0, nprow)
         mq0 = numroc(nn, nb, 0, 0, npcol)
c     
c     
         orfac = 1.d-3
c     
c     
         lcwork = 5*n +MAX(5*NN,(NP0*MQ0 + 2*nb*nb))+
     +          ICEIL( N, NPROW*NPCOL)*NN+1

c     
         if(lcwork.ne.0)
     $        status = ma_push_get(MT_DBL, lcwork ,
     $        'cwork',hcwork,adrcwork)
         if(.not.status) 
     &        call ga_error('ga_pdsyev: mem alloc failed cwork  ', -1)
c     
c     
         abstol=pdlamch(iSLctxt, 'U')
c     
c     
         call pdsyevx(jobz, range, uplo,
     1        n, dbl_mb(adrA), 1, 1, descA,vl, 
     2        vu, il, iu, abstol, m, nz, eval, orfac, dbl_mb(adrB), 1, 
     3        1, descB, dbl_mb(adrcwork), lcwork,
     3        int_mb(adriwork), liwork, int_mb(adrfail),
     4        int_mb(adrclustr), dbl_mb(adrgaps), info)
c     

         if (nz .ne. n ) then
            if ( info .ne. 0 ) then
               if ( info .gt. 0 ) then
         call ga_error(' ga_pdsyev: argument is illegal ', info)
               else
         call ga_error(' ga_pdsyev: eigenvectors failed to converge ',
     $                 info)
               endif
            endif
         endif
c     
c     
c     
c***  copy solution matrix back to g_c
c     
         call ga_from_SL(g_b, dimA1, dimB2,
     $        nb, nb, dbl_mb(adrB),
     &        mpB, ldb, nqB)
c     
c     
c     
c***  deallocate work/SL arrays
c     
         if ( lcwork .ne. 0 ) status = ma_pop_stack(hcwork)
         if ( liwork .ne. 0 ) status = ma_pop_stack(hiwork)
         if ( if .ne. 0 ) status = ma_pop_stack(hfail)
         if ( iclu .ne. 0 ) status = ma_pop_stack(hclustr)
         if ( ngaps.ne.0 ) status = ma_pop_stack(hgap)
         if ( elemB .ne. 0 ) status = ma_pop_stack(hb)
         if ( elemA .ne. 0 ) status = ma_pop_stack(ha)
      endif
c     
      call ga_sync()
      return
      end
c******************************************************************
c     
c     ga_pdsygv
c
c     interface into scalapack's Hermitian eigensolver
c     
c     ga interface to scalapack
c     
c     g_a .g_c(*,i) = eval(i).g_s(*,i) g_b(*,i)
c     
c     assume equal size blocks ...
c     
c******************************************************************
      subroutine ga_pdsygv(g_a, g_s, g_b, eval)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c     
      integer g_a               ! matrix A
      integer g_b               ! matrix B
      integer g_s               ! metric
c
c all eigenvalues are real
c
      double precision eval(*)
c     
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      character*1 jobz, range, uplo
c
      integer ha, adra          !A
      integer hb, adrb          !B
      integer hs, adrs          !S
c
c
      logical oactive           ! true if this process participates
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
      integer dims1, dims2, types
c
      integer mpA, nqA          ! rows/cols of A held by the processor
      integer mpB, nqB          ! rows/cols of B held by the processor
      integer mps, nqs          ! rows/cols of B held by the processor
c
      integer me
      integer lda, ldb, lds, elemA, elemB, elems
      integer numroc
      
      integer nb                ! block size
      integer descA(9), descB(9) !descriptor for scalapack
      integer descs(9)
      
c     
      integer ngaps, hgap, adrgaps
      integer iclu, hclustr, adrclustr
      integer if, hfail, adrfail
      integer liwork, hiwork, adriwork
      integer lcwork, hcwork, adrcwork
c
      integer nn, mq0, np0
      double precision vl, vu, abstol, orfac
      integer il, iu
      integer m, nz
      integer n
      integer info

      integer iceil, iii
      double precision pdlamch
      external pdlamch
      external iceil
      intrinsic max
c     
c     processor dependent; machine dependent
c     
      data nb/64/
c     
c***  check environment
c     
      me     = ga_nodeid()
c     
c***  check GA info for input arrays
c     
      call ga_check_handle(g_a, 'ga_pdsygv: a')
      call ga_check_handle(g_b, 'ga_pdsygv: b')
      call ga_check_handle(g_s, 'ga_pdsygv: b')
c     
      call ga_inquire(g_a, typeA, dimA1, dimA2)
      call ga_inquire(g_b, typeB, dimB1, dimB2)
      call ga_inquire(g_s, types, dims1, dims2)
      if(dimA1.ne.dima2) call ga_error(
     %     'ga_pdsygv: matrix A not square ',0)
      n=dima1
      if(dimb1.ne.dimb2) call ga_error(
     %     'ga_pdsygv: matrix B not square ',0)
      if(dimb1.ne.n) call ga_error(
     %     'ga_pdsygv: size matrix B and A differ ',0)
      if(dims1.ne.dims2) call ga_error(
     %     'ga_pdsygv: matrix S not square ',0)
      if(dims1.ne.n) call ga_error(
     %     'ga_pdsygv: size matrix A and S differ ',0)
      

c     
c     
c***  initialize SL interface
c     
      call SLinit()
c     
c***  find SBS format parameters
c     
c     
      mpA = numroc(dimA1, nb, myrow, 0, nprow)
      nqA = numroc(dimA2, nb, mycol, 0, npcol)
c     
      mpB = numroc(dimB1, nb, myrow, 0, nprow)
      nqB = numroc(dimB2, nb, mycol, 0, npcol) 
c     
      mps = numroc(dims1, nb, myrow, 0, nprow)
      nqs = numroc(dims2, nb, mycol, 0, npcol) 
c     
c     
      lda = max(1,mpA)
      ldb = max(1,mpB)
      lds = max(1,mps)
c     
c     
c     let scalapack check for errors
c     
c     should check to see if this is a compute node
c     check to see how this works in the new data server model
c     
      oactive = .true.
      if ( mpA*nqA .eq. 0 ) oactive = .false.
      if ( mpB*nqB .eq. 0 ) oactive = .false.
      if ( mps*nqs .eq. 0 ) oactive = .false.
c     
      call ga_sync()
c     
!      if (oactive) then
      if (.true.) then
c     
         elemA= mpA*nqA
         status = .true.
         if(elemA.ne.0)status =
     $        ma_push_get(MT_DBL,elemA,'a',ha,adra)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed A ', -1)
c     
         elemB= mpB*nqB
         if(elemB.ne.0)status =
     $        ma_push_get(MT_DBL,elemB,'b',hb,adrb)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed B ', -1)
         elems= mps*nqs
         if(elems.ne.0)status =
     $        ma_push_get(MT_DBL,elems,'s',hs,adrs)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed S ', -1)
c     
         ngaps = nprow*npcol
         if(ngaps.ne.0)status =
     $        ma_push_get(MT_DBL, ngaps ,'gap',hgap,adrgaps)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed gaps  ', -1)
c     
         iclu = 2*nprow*npcol
         iclu = 2*n
         if(iclu.ne.0)status =
     $        ma_push_get(MT_INT, iclu ,'iclustr',hclustr,adrclustr)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed iclustr  ', -1)
c     
         if = n
         if(if.ne.0) then
            status =
     $           ma_push_get(MT_INT, if ,'ifail',hfail,adrfail)
         endif
         if(.not.status)
     &        call ga_error('ga_pdsygv: mem alloc failed ifail  ', -1)
         
c     
c***  copy g_a to A and g_b to B using the block cyclic scalapack format 
c     
         
         call ga_to_SL(g_a, dimA1, dimA2, nb, nb,
     $        dbl_mb(adrA), lda, mpA, nqA)
c     
         call ga_to_SL(g_b, dimB1, dimB2, nb,nb,
     $        dbl_mb(adrB), ldb, mpB, nqB)
         call ga_to_SL(g_s, dims1, dims2, nb,nb,
     $        dbl_mb(adrs), ldb, mps, nqs)
c     
c     
c***  fill SCALAPACK matrix descriptors
c     
         call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     $        iSLctxt, lda, info)
         if(info.ne.0) call ga_error(' ga_pdsygv: descinit A failed ',
     .        -info)
c     
         call descinit(descB, dimB1, dimB2, nb, nb, 0, 0,
     .        iSLctxt, ldb, info)
         if(info.ne.0) call ga_error(' ga_pdsygv: descinit B failed ',
     .        -info)
         call descinit(descs, dims1, dims2, nb, nb, 0, 0,
     .        iSLctxt, lds, info)
         if(info.ne.0) call ga_error(' ga_pdsygv: descinit S failed ',
     .        -info)
c     
c     call scalapack
c     
c     
         jobz = 'V'
         range = 'A'
         uplo = 'L'
         vl = 0.d0
         vu = 0.d0
         il = 0
         iu = 0
         nz = 0
c     
c     ability to deal with orthonormality ; let's just
c     have the regular scalapack stuff for the moment
c     
         liwork = 6*max(n, nprow*npcol+1, 4) 
         if(liwork.ne.0)status =
     $        ma_push_get(MT_INT, liwork ,'iwork',hiwork,adriwork)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed iwork  ', -1)
c     
         nn = max(n, nb, 2)
         np0 = numroc(nn, nb, 0, 0, nprow)
         mq0 = numroc(nn, nb, 0, 0, npcol)
c     
c     
         orfac = 1.d-3
c     
c     
         lcwork = 5*n +MAX(5*NN,(NP0*MQ0 + 2*nb*nb))+
     +          ICEIL( N, NPROW*NPCOL)*NN+1 

c     
         if(lcwork.ne.0)
     $        status = ma_push_get(MT_DBL, lcwork ,
     $        'cwork',hcwork,adrcwork)
         if(.not.status) 
     &        call ga_error('ga_pdsygv: mem alloc failed cwork  ', -1)
c     
         abstol=pdlamch(iSLctxt, 'U')
c     
c     
c     
         call pdsygvx(1,jobz, range, uplo,
     1        n, dbl_mb(adrA), 1, 1, 
     2        descA, dbl_mb(adrS), 1, 1,descs, vl,vu, il, iu, 
     3        abstol, m, nz, eval, orfac, dbl_mb(adrB), 1, 1, descB, 
     4        dbl_mb(adrcwork), lcwork,int_mb(adriwork), liwork, 
     4        int_mb(adrfail), int_mb(adrclustr), 
     5        dbl_mb(adrgaps), info)
c     
c     
c
         
c     
         if (nz .ne. n ) then
            if ( info .ne. 0 ) then
               if ( info .gt. 0 ) then
         call ga_error(' ga_pdsygv: argument is illegal ', info)
               else
         call ga_error(' ga_pdsygv: eigenvectors failed to converge ',
     $                 info)
               endif
            endif
         endif
c     
c     
c     
c***  copy solution matrix back to g_c
c     
c     
         call ga_from_SL(g_b, dimA1, dimB2,
     $        nb, nb, dbl_mb(adrB),
     &        mpB, ldb, nqB)
c     
     
c     
c***  deallocate work/SL arrays
c     
         if ( lcwork .ne. 0 ) status = ma_pop_stack(hcwork)
         if ( liwork .ne. 0 ) status = ma_pop_stack(hiwork)
         if ( if .ne. 0 ) status = ma_pop_stack(hfail)
         if ( iclu .ne. 0 ) status = ma_pop_stack(hclustr)
         if ( ngaps.ne.0 ) status = ma_pop_stack(hgap)
         if ( elemS .ne. 0 ) status = ma_pop_stack(hs)
         if ( elemB .ne. 0 ) status = ma_pop_stack(hb)
         if ( elemA .ne. 0 ) status = ma_pop_stack(ha)
      endif
c     
      call ga_sync()
      return
      end
c
c     interface to pdsyevd (divide&conquer)
c
      subroutine ga_pdsyevd(g_a, g_b, eval)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "scalapack.fh"
c     
      integer g_a               ! matrix A
      integer g_b               ! matrix B
c
c all eigenvalues are real
c
      double precision eval(*)
c     
c~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c
      logical status
c
      character*1 jobz, uplo
c
      integer ha, adra          !A
      integer hb, adrb          !B
c
c
      logical oactive           ! true iff this process participates
      integer dimA1, dimA2, typeA
      integer dimB1, dimB2, typeB
c
      integer mpA, nqA          ! rows/cols of A held by the processor
      integer mpB, nqB          ! rows/cols of B held by the processor
c
      integer me
      integer lda, ldb
      integer elemA,elemB
      integer numroc
      
      integer nb                ! block size
      integer descA(9), descB(9) !descriptor for scalapack
      
c     
      integer liwork,hiwork, adriwork
      integer lcwork,hcwork, adrcwork
c
      integer nn,nq, np,trilwmin,iarow,iacol
      integer n
      integer info

      double precision pdlamch
      integer iceil,indxg2p
      external pdlamch,iceil,indxg2p
c     
c     processor dependent; machine dependent
c     
      data nb/64/
c     
c***  check environment
c     
      me     = ga_nodeid()
c     
c***  check GA info for input arrays
c     
      call ga_check_handle(g_a, 'ga_pdsyevd: a')
      call ga_check_handle(g_b, 'ga_pdsyevd: b')
c     
      call ga_inquire(g_a, typeA, dimA1, dimA2)
      call ga_inquire(g_b, typeB, dimB1, dimB2)
      if(dimA1.ne.dima2) call ga_error(
     %     'ga_pdsyevd: matrix A not square ',0)
      n=dima1
      if(dimb1.ne.dimb2) call ga_error(
     %     'ga_pdsyevd: matrix B not square ',0)
      if(dimb1.ne.n) call ga_error(
     %     'ga_pdsyevd: size matrix B and B differ ',0)
      

c     
c     
c***  initialize SL interface
c     
      call SLinit()
c     
c***  find SBS format parameters
c     
c     
      mpA = numroc(dimA1, nb, myrow, 0, nprow)
      nqA = numroc(dimA2, nb, mycol, 0, npcol)
c     
      mpB = numroc(dimB1, nb, myrow, 0, nprow)
      nqB = numroc(dimB2, nb, mycol, 0, npcol) 
c     
c     
      lda = max(1,mpA)
      ldb = max(1,mpB)
c     
c     
c     let scalapack check for errors
c     
c     should check to see if this is a compute node
c     check to see how this works in the new data server model
c     
      oactive = .true.
      if ( mpA*nqA .eq. 0 ) oactive = .false.
      if ( mpB*nqB .eq. 0 ) oactive = .false.
c     
      call ga_sync()
c     
c     
      elemA= mpA*nqA
      status = .true.
      if(elemA.ne.0)status =
     $     ma_push_get(MT_DBL,elemA,'a',ha,adra)
      if(.not.status) 
     &     call ga_error('ga_pdsyevd: mem alloc failed A ', -1)
c     
      elemB= mpB*nqB
c     
      if(elemB.ne.0)status =
     $     ma_push_get(MT_DBL,elemB,'b',hb,adrb)
      if(.not.status) 
     &     call ga_error('ga_pdsyevd: mem alloc failed B ', -1)
c     
c     
c***  copy g_a to A and g_b to B using the block cyclic scalapack format 
c     
         
      call ga_to_SL(g_a, dimA1, dimA2, nb, nb,
     $     dbl_mb(adrA), lda, mpA, nqA)
c     
      call ga_to_SL(g_b, dimB1, dimB2, nb,nb,
     $     dbl_mb(adrB), ldb, mpB, nqB)
c     
c***  fill SCALAPACK matrix descriptors
c     
      call descinit(descA, dimA1, dimA2, nb, nb, 0, 0,
     $     iSLctxt, lda, info)
      if(info.ne.0) call ga_error(' ga_pdsyevd: descinit A failed ',
     .     -info)
c     
      call descinit(descB, dimB1, dimB2, nb, nb, 0, 0,
     .     iSLctxt, ldb, info)
      if(info.ne.0) call ga_error(' ga_pdsyevd: descinit B failed ',
     .     -info)
c     
      jobz = 'V'
      uplo = 'L'
c     
c     ability to deal with orthonormality ; let's just
c     have the regular scalapack stuff for the moment
c     
      liwork = 7*n + 8*npcol +2
      if(liwork.ne.0)status =
     $     ma_push_get(MT_INT, liwork ,'iwork',hiwork,adriwork)
      if(.not.status) 
     &     call ga_error('ga_pdsyevd: mem alloc failed iwork  ', -1)
c     
      nn = max(n, nb, 2)
      IAROW = INDXG2P( 1, nb, MYROW, DESCA( 7 ), NPROW )
      IACOL = INDXG2P( 1, nb, MYCOL, DESCA( 8 ), NPCOL )
            
      np = numroc(n, nb, myrow, iarow, nprow)
      nq = numroc(n, nb, mycol, iacol, npcol)
c     
c     
c     
          TRILWMIN = 3*N + MAX( NB*( NP+1 ), 3*NB )
          lcwork = MAX( 1+6*N+2*NP*NQ, TRILWMIN ) + 2*N
          lcwork=max(lcwork,8192)


c     
      if(lcwork.ne.0)
     $     status = ma_push_get(MT_DBL, lcwork ,
     $     'cwork',hcwork,adrcwork)
      if(.not.status) 
     &     call ga_error('ga_pdsyevd: mem alloc failed cwork  ', -1)
c     
c     
c     
      call pdsyevd(jobz, uplo,
     1     n, dbl_mb(adrA), 1, 1, descA,
     1     eval, dbl_mb(adrB), 1, 1, 
     2     descB, dbl_mb(adrcwork), lcwork,
     2     int_mb(adriwork), liwork, info)
c     
      if ( info .ne. 0 ) then
         if ( info .gt. 0 ) then
            call ga_error(' ga_pdsyevd: argument is illegal ', info)
         else
            call ga_error(
     $           ' ga_pdsyevd: eigenvectors failed to converge ',
     $           info)
         endif
      endif
c     
c     
c     
c***  copy solution matrix back to g_c
c     
      call ga_from_SL(g_b, dimA1, dimB2,
     $     nb, nb, dbl_mb(adrB),
     &     mpB, ldb, nqB)
c     
c     
c     
c***  deallocate work/SL arrays
c     
      if ( lcwork .ne. 0 ) status = ma_pop_stack(hcwork)
      if ( liwork .ne. 0 ) status = ma_pop_stack(hiwork)
      if ( elemB .ne. 0 ) status = ma_pop_stack(hb)
      if ( elemA .ne. 0 ) status = ma_pop_stack(ha)
c     
      call ga_sync()
      return
      end
