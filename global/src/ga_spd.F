#ifdef GA_USE_VAMPIR
#include "ga_vampir.fh"
#endif
c**********************************************************************
c* Source : ga_llt
c* Scope  : interface between GA and LLT SCALAPACK routines
c*
c* 04/12/96  GVT  First Implementation
c*           Giuseppe Vitillaro peppe@unipg.it
c* 08/28/96  Jarek Nieplocha, some cleanup and optimizations 
c**********************************************************************

#define NB 64
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_ZEROUL,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_ZEROUL,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_SYMUL,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_SYMUL,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_CHOLESKY,__FILE__,__LINE__)
#endif
      ga_cholesky = ga_llt_f(uplo, g_A, -1)
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_CHOLESKY,__FILE__,__LINE__)
#endif
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
#include "slface.fh"
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
      data nb /NB/

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
#include "slface.fh"
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
      data nb /NB/
c
c**** Check the Environment
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_LLT_S,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_LLT_S,__FILE__,__LINE__)
#endif
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
#include "slface.fh"
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
      data nb /NB/
c
c**** Check Environment
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_LLT_I,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_LLT_I,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_LLT_SOLVE,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_LLT_SOLVE,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_SPD_INVERT,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_SPD_INVERT,__FILE__,__LINE__)
#endif
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
#ifdef GA_USE_VAMPIR
      call vampir_begin_galinalg(GA_SOLVE,__FILE__,__LINE__)
#endif
      ga_solve = ga_llt_solve(g_A, g_B)
      if (ga_solve.eq.0) then
c****    if Cholesky was succcesully just return 
      else
c****    else if Cholesky failed
         call ga_lu_solve('N', g_A, g_B)
      endif
#ifdef GA_USE_VAMPIR
      call vampir_end_galinalg(GA_SOLVE,__FILE__,__LINE__)
#endif
      end
