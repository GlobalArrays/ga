c***********************************************************************
c* Source : testllt
c* Scope  : test LLT SCALAPACK routines
c*
c* 04/12/96  GVT  First Implementation
c*           Giuseppe Vitillaro peppe@unipg.it
c***********************************************************************

#define SIZE 100
#define UPLO 'L'      
#define TRESH 3.d0
#define ga_dnormF(g_a) sqrt(ga_ddot(g_a, g_a))

      
c***********************************************************************
c* Program: testllt
c***********************************************************************
      program testllt
      
      implicit none
      
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"

c****
      integer nproc
      integer hsize, ssize
      
c**** Initialize TCGMSG package
      call pbeginf

c**** get number of nodes and calculate memory allocation      
      nproc = nnodes()
      
      hsize = 6*SIZE*SIZE + 3*SIZE
      ssize = 3*SIZE*SIZE + 3*SIZE 
      
      hsize = (hsize/nproc) + 1
      ssize = (ssize/nproc) + 1 + 3*256*256
    
c**** Initialize MA package
      if (.not. ma_init(MT_DBL, ssize, hsize))
     &     call parerr(1)

c**** Initialize GA package
      call ga_initialize()
      
c**** Do Core Tests
      call ctests(UPLO)

c**** Exit from the GA package
      call ga_terminate()

c**** Exit from the TCGMSG package
      call pend()
      end

      
c***********************************************************************
c* subroutine: infos
c*             print informations about this run
c***********************************************************************      
      subroutine infos(me, nproc, uplo, eps)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"      
c****
      integer me
      integer nproc
      character*1 uplo
      double precision eps
c****      
      
      if (me.eq.0) then
         print *, ' '
         print *, 'Number of nodes  :  ', nproc
         print *, 'Problem size     :  ', SIZE
         print *, 'Uplo             :   ', uplo
         print *, 'Epsilon          :  ', eps
         call ffflush(6)
      endif

      return
      end

      
c***********************************************************************
c* subroutine: thead
c*             emit test header output
c***********************************************************************
      subroutine thead(header)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"            
c****
      character*(*) header
c****
      integer me

      me = ga_nodeid()
      if (me.eq.0) then
         print *, ' '
         print *, '> ', header, ' '
         call ffflush(6)
      endif

      return
      end

      
c***********************************************************************
c* subroutine: dthtest
c*             test a double precision against the THRESH parameter
c***********************************************************************
      subroutine dthtest(msg, dval)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"                  
c****      
      character*(*) msg
      double precision dval
c****
      integer me

      me = ga_nodeid()

      if (me.eq.0) then
         print *, '  ', msg, dval, ' '
         if (dval.lt.TRESH) then
            print *, '> success '
         else
            print *, '> failure '
         endif
         call ffflush(6)
      endif
      
      return
      end

      
c***********************************************************************
c* subrotine: stest
c*            test solver result
c***********************************************************************
      subroutine stest(irc, ierc)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"                  
c****      
      integer irc
      integer ierc
c****
      integer me

      me = ga_nodeid()

      if (me.eq.0) then
         
         if (irc.eq.0) then
            print *, '> LLT '
         elseif (irc.gt.0) then
            print *, '> LU  '
         endif

         if (irc.eq.ierc .or. (irc.gt.0 .and. ierc.gt.0)) then
            print *, '> success: expected factorization'
         else
            print *, '> failure: not expected factorization',irc,ierc
         endif
         
         call ffflush(6)
         
      endif
      
      return
      end

      
c***********************************************************************
c* subroutine: emsg
c*             error message
c***********************************************************************
      subroutine emsg(msg, ival)
      implicit none
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"                  
c****      
      character*(*) msg
      integer ival
c****
      integer me

      me = ga_nodeid()

      if (me.eq.0) then
         print *, '  >>> ', msg, ival, ' '
         print *, '> failure '
         call ffflush(6)
      endif
      
      return
      end

      

      
c***********************************************************************
c* subroutine: ctests
c*             do coretests for LLT Cholesky factorization, solver
c***********************************************************************
      subroutine ctests(uplo)
      
      implicit none
            
#include "mafdecls.fh"
#include "global.fh"
#include "tcgmsg.fh"

c****
      character*1 uplo
c****      
      integer n
      parameter (n=SIZE)

      external ga_llt_f, ga_llt_s, ga_llt_i
      integer ga_llt_f, ga_llt_i

      double precision A(n,n), X(n,n)
      
      integer g_A, g_B, g_X, g_Y
      integer g_A1, g_X1, g_Y1

      integer hsA

      integer i, j
      integer nproc, me
      double precision buf(n)
      
      double precision dnA, dnX, dnX1, dnY1, dnS, dnF, dnI
      integer irc
      
      double precision eps,  dlamch

      eps   =  dlamch('p')
      
      nproc = ga_nnodes()
      me    = ga_nodeid()

      call infos(me, nproc, uplo, eps)

      
c****************************************
c* Initialize tests variables
c* Generate a Lower triangula matrix
c* with large positive diagonal elements
c* so the LU decomposition will be
c* just this matrix      
c****************************************
c**** Initialize local arrays A, X
c**** they are local copies of the corrispondent
c**** global arrays
      do j = 1, n
         do i = 1, n
            X(i,j) = dsin(1.d0 * (i * j))
            A(i,j) = 0.d0
            if (i.gt.j) then
               A(i,j) = dsin(1.d0 * (i + j))
            endif
            if (i.eq.j) then
               A(i,j) = SIZE*dabs(dsin(10.d0 * i))
            endif
         end do ! i
      end do    ! j
      
c**** Create global arrays
      if (.not. ga_create(MT_DBL, n, n, 'A', 1, 1, g_A))
     &     call ga_error(' ga_create A failed ', 2)
      
      if (.not. ga_create(MT_DBL, n, n, 'B', 1, 1, g_B))
     &     call ga_error(' ga_create B failed ', 2)

      if (.not. ga_create(MT_DBL, n, n, 'A1', 1, 1, g_A1))
     &     call ga_error(' ga_create A1 failed ', 2)

      if (.not. ga_create(MT_DBL, n, n, 'X', 1, 1, g_X))
     &     call ga_error(' ga_create X failed ', 2)

      if (.not. ga_create(MT_DBL, n, n, 'X1', 1, 1, g_X1))
     &     call ga_error(' ga_create X1 failed ', 2)
      
      if (.not. ga_create(MT_DBL, n, 1, 'Y', 1, 1, g_Y))
     &     call ga_error(' ga_create Y failed ', 2)
      
      if (.not. ga_create(MT_DBL, n, 1, 'Y1', 1, 1, g_Y1))
     &     call ga_error(' ga_create Y1 failed ', 2)

c**** Fill in arrays A, X
      do j = me+1, n, nproc
         call ga_put(g_A, 1, n, j, j, A(1,j), n)
         call ga_put(g_X, 1, n, j, j, X(1,j), n)
      end do    ! j 

c**** Create A, B, Y
c**** It is granted that A, B, Y will not change
c**** during the execution of the test LLT driver      
      
c**** B = A * A'
      call ga_dgemm('N', 'T', n, n, n, 1.d0, g_A, g_A, 0.d0, g_B)

c**** A = B = A * A'
      call ga_copy(g_B, g_A)
      
c**** B = A * X      
      call ga_dgemm('N', 'N', n, n, n, 1.d0, g_A, g_X, 0.d0, g_B)

c**** Copy B(1:n,1:1) to Y(1:n) so Y will X column 1
      call ga_copy_patch('N', g_B, 1, n, 1, 1, g_Y, 1, n, 1, 1)

c**** dnA = ||A||
      dnA = ga_dnormF(g_A)

      
c****************************************************
c Test Cholesky factorization external interface
c****************************************************
      call thead("Test Cholesky factorization")

c**** copy A in X
      call ga_copy(g_A, g_X)

c**** call Cholesky factorization routine in ScaLAPACK PDPOTRF
c**** to obtain an LL'/U'U factorization (external interface)
      irc = ga_cholesky(uplo, g_X)
      
c**** if return code 'zero' is OK
      if (irc.eq.0) then

c****    A1 = X * X' or A1 = X' * X
         if (uplo.eq.'L') then
          call ga_dgemm('N', 'T', n, n, n, 1.d0, g_X, g_X, 0.d0, g_A1)
         else
          call ga_dgemm('T', 'N', n, n, n, 1.d0, g_X, g_X, 0.d0, g_A1)
         endif
         
c****    A1 = A - A1 = A - X * X' or A1 = A - A1 = A - X' * X
         call ga_add(1.0d0, g_A1, -1.0d0, g_A, g_A1)
      
c****    dnF = ||A - A1*A1'|| / (||A|| * N * eps) : SHAPE L
c****    dnF = ||A - A1'*A1|| / (||A|| * N * eps) : SHAPE U         
         dnF = ga_dnormF(g_A1) / (dnA * n * eps)

         if (uplo.eq.'L') then
            call dthtest("||LL' - A|| / (||A|| * N * eps) =", dnF)
         else   
            call dthtest("||U'U - A|| / (||A|| * N * eps) =", dnF)
         endif
         
c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &             irc)
      endif


c****************************************************
c Test Cholesky factorization and solver
c internal interfaces with a NxN RHS array
c****************************************************
      call thead("Test II Cholesky solver with a NxN RHS array")

c**** call Cholesky factorization routine in ScaLAPACK PDPOTRF
c**** to obtain an LL'/U'U factorization
c**** (internal interface: it will not destroy A)
      hsA = 0
      irc = ga_llt_f(uplo, g_A, hsA)
      
c**** if return code from ga_llt_f is zero
      if (irc.eq.0) then
         
c****    copy B in X
         call ga_copy(g_B, g_X)
         
c****    call Cholesky solver routine in ScaLAPACK PDPOTRS
c****    internal interface with an NxN RHS GA
         call ga_llt_s(uplo, g_A, g_X, hsA)

c****    A1 = A * X
         call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &        g_A, g_X, 0.d0, g_A1)

c****    A1 = A1 - B = A * X - B
         call ga_add(1.d0, g_A1, -1.d0, g_B, g_A1)
         
c****    dnX = ||X||
         dnX = ga_dnormF(g_X)
         
c****    dnS = ||AX - B|| / (||A|| * ||X|| * N * eps)
         dnS = ga_dnormF(G_A1) / (dnA * dnX * n * eps)

         call dthtest("||A*X - B||/(||A||*||X||*n*eps) =", dnS)

c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &        irc)
      endif


c****************************************************
c Test Cholesky factorization and solver
c internal interfaces with a single vector RHS
c****************************************************
      call thead("Test II Cholesky solver with a single vector RHS")

c**** call Cholesky factorization routine in ScaLAPACK PDPOTRF
c**** to obtain an LL'/U'U factorization
c**** (internal interface: it will not destroy A)
      hsA = 0
      irc = ga_llt_f(uplo, g_A, hsA)
      
c**** if return code from ga_llt_f is zero
      if (irc.eq.0) then
         
c****    copy Y in Y1
         call ga_copy(g_Y, g_Y1)
         
c****    call Cholesky solver routine in ScaLAPACK PDPOTRS
c****    internal interface with a single vector RHS
         call ga_llt_s(uplo, g_A, g_Y1, hsA)

c****    dnY1 = ||Y1||
         dnY1 = ga_dnormF(g_Y1)
                  
c****    Y1 = A * Y1
         call ga_dgemm('N', 'N', n, 1, n, 1.d0,
     &        g_A, g_Y1, 0.d0, g_Y1)

c****    Y1 = Y1 - Y = A * Y1 - Y
         call ga_add(1.d0, g_Y1, -1.d0, g_Y, g_Y1)

c****    dnS = ||AY1 - Y|| / (||A|| * ||Y1|| * N * eps)
         dnS = ga_dnormF(G_A1) / (dnA * dnY1 * n * eps)

         call dthtest("||A*X - V||/(||A||*||X||*n*eps) =", dnS)

c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &        irc)         
      endif

      
c****************************************************
c Test Cholesky solver with a NxN RHS array
c****************************************************
      call thead("Test EI Cholesky solver with a NxN RHS array")

c**** copy B in X
      call ga_copy(g_B, g_X)
         
c**** call Cholesky L/U solver with a NxN RHS
      irc = ga_llt_solve(g_A, g_X)
      
c**** if return code from ga_llt_solve is zero
      if (irc.eq.0) then
         
c****    A1 = A * X
         call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &        g_A, g_X, 0.d0, g_A1)

c****    A1 = A1 - B = A * X - B
         call ga_add(1.d0, g_A1, -1.d0, g_B, g_A1)

c****    dnX = ||X||
         dnX = ga_dnormF(g_X)
         
c****    dnS = ||AX - B|| / (||A|| * ||X|| * N * eps)
         dnS = ga_dnormF(G_A1) / (dnA * dnX * n * eps)

         call dthtest("||A*X - B||/(||A||*||X||*n*eps) =", dnS)
         
c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &        irc)                  
      endif

      
c****************************************************
c Test Cholesky solver with a single vector RHS
c****************************************************
      call thead("Test EI Cholesky solver with a single vector RHS")
      
c**** copy Y in Y1
      call ga_copy(g_Y, g_Y1)

c**** call Cholesky L/U solver with a single vector RHS
      irc = ga_llt_solve(g_A, g_Y1)
      
c**** if return code from ga_llt_solve is zero
      if (irc.eq.0) then

c****    dnY1 = ||Y1||
         dnY1 = ga_dnormF(g_Y1)
                  
c****    Y1 = A * Y1
         call ga_dgemm('N', 'N', n, 1, n, 1.d0,
     &        g_A, g_Y1, 0.d0, g_Y1)

c****    Y1 = Y1 - Y = A * Y1 - Y
         call ga_add(1.d0, g_Y1, -1.d0, g_Y, g_Y1)

c****    dnS = ||AY1 - Y|| / (||A|| * ||Y1|| * N * eps)
         dnS = ga_dnormF(G_A1) / (dnA * dnY1 * n * eps)
         
         call dthtest("||A*X - V||/(||A||*||X||*n*eps) =", dnS)
         
c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &        irc)                  
      endif

      
c****************************************************
c Test Cholesky factorization and inversion
c internal interfaces 
c****************************************************
      call thead("Test II inversion of an SPD matrix")

c**** copy A in X
      call ga_copy(g_A, g_X)

c**** call Cholesky factorization routine in ScaLAPACK PDPOTRF
c**** to obtain an LL'/U'U factorization
c**** (internal interface: it will not destroy X1)
      hsA = 0
      irc = ga_llt_f(uplo, g_X, hsA)
      
c**** if return code from ga_llt_f is zero
      if (irc.eq.0) then
         
c****    call Cholesky inversion routine in ScaLAPACK PDPOTRI
c****    internal interface
         irc = ga_llt_i(uplo, g_X, hsA)
         
c****    if the inversion could be done         
         if (irc.eq.0) then

c****       dnX = ||X|| = ||invA||
            dnX = ga_dnormF(g_X)

c****       A1 = A * X = A * invA
            call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &           g_A, g_X, 0.d0, g_A1)

c****       A1 = A1 - I = A * invA - I
            do j = me+1, n, nproc
               call ga_get(g_A1, j, j, j, j, buf, 1)
               buf(1) = buf(1) - 1.d0
               call ga_put(g_A1, j, j, j, j, buf, 1)
            end do ! j

c****       dnI = ||A*invA - I|| / (||A| * ||invA|| * N * eps)
            dnI = ga_dnormF(g_A1) / (dnA * dnX * n * eps)

            call dthtest('||A*invA - I||/(||A||*||invA||*n*eps) =',
     &           dnI)
         
         else
c****       otherwise if the ga_llt_i return code is > 0
            call emsg('there is a zero diagonal element:', irc)
         endif
      
c**** if return code is > 0
      else
         call emsg('It is not positive definite the minor of order:',
     &        irc)                  
      endif

      
c****************************************************
c Test Cholesky inversion
c****************************************************
      call thead("Test inversion of an SPD matrix")

c**** copy A in X
      call ga_copy(g_A, g_X)

c**** call Cholesky L/U inversion
      irc = ga_spd_invert(g_X)
      
c**** if return code from ga_spd_invert is zero
      if (irc.eq.0) then

c****    dnX = ||X|| = ||invA||
         dnX = ga_dnormF(g_X)
         
c****    A1 = A * X = A * invA
         call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &        g_A, g_X, 0.d0, g_A1)

c****    A1 = A1 - I = A * invA - I
         do j = me+1, n, nproc
            call ga_get(g_A1, j, j, j, j, buf, 1)
            buf(1) = buf(1) - 1.d0
            call ga_put(g_A1, j, j, j, j, buf, 1)
         end do ! j

c****    dnI = ||A*invA - I|| / (||A|| * ||invA|| * N * eps)
         dnI = ga_dnormF(g_A1) / (dnA * dnX * n * eps)

         call dthtest('||A*invA - I||/(||A||*||invA||*n*eps) =',
     &        dnI)
         
c**** if return code is < 0
      elseif (irc.lt.0) then
         call emsg('there is a zero diagonal element:', irc)
c**** if return code is > 0         
      elseif (irc.lt.0) then
         call emsg('It is not positive definite the minor of order:',
     &        irc)
      endif

      
c****************************************************
c* Test general solver with a NxN RSH
c* simmetric positive definite array       
c****************************************************
      call thead(
     & 'Test solver for a symmetric P.D. matrix and NxN RHS'
     &     )
      
c**** copy B in X
      call ga_copy(g_B, g_X)
         
c**** call general solver with a NxN RHS p.d. symmetric array
      irc = ga_solve(g_A, g_X)

c**** A1 = A * X
      call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &     g_A, g_X, 0.d0, g_A1)

c**** A1 = A1 - B = A * X - B
      call ga_add(1.d0, g_A1, -1.d0, g_B, g_A1)

c**** dnX = ||X||
      dnX = ga_dnormF(g_X)
      
c**** dnS = ||AX - B|| / (||A|| * ||X|| * N * eps)
      dnS = ga_dnormF(G_A1) / (dnA * dnX * n * eps)

      call dthtest("||A*X - B||/(||A||*||X||*n*eps) =", dnS)

      call stest(irc, 0)

      
c****************************************************
c* Test general solver with a NxN RSH
c* simmetric non positive definite array       
c****************************************************
      call thead(
     & 'Test solver for a symmetric NON P.D. matrix and NxN RHS')
      
c**** copy B on X1
      call ga_copy(g_B, g_X1)
      
c**** and now symmetrize X1
c**** so we obtain a symmetric matrix in X1
      call ga_symmetrize(g_X1)

c**** copy B in X
      call ga_copy(g_B, g_X)
      
c**** call general solver with a NxN RHS symmetric array
      irc = ga_solve(g_X1, g_X)
      
c**** dnX1 = ||X1||
      dnX1 = ga_dnormF(g_X1)

c**** dnX = ||X||
      dnX = ga_dnormF(g_X)

c**** A1 = X1 * X
      call ga_dgemm('N', 'N', n, n, n, 1.d0,
     &     g_X1, g_X, 0.d0, g_A1)

c**** A1 = A1 - B = X1 * X - B
      call ga_add(1.d0, g_A1, -1.d0, g_B, g_A1)

c**** dnS = ||X1*X - B|| / (||X1|| * ||X|| * N * eps)
      dnS = ga_dnormF(G_A1) / (dnX1 * dnX * n * eps)

      call dthtest("||A*X - B||/(||A||*||X||*n*eps) =", dnS)
     
      call stest(irc, 1)

      
c****************************************************
c CTESTS exit code
c****************************************************      
c**** just print a newline and return      
      if (me.eq.0) then
         print *, ' '
      endif

      return

      end
