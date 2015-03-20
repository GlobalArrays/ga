*> \brief \b DPOTRF
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download DPOTRF + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dpotrf.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dpotrf.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dpotrf.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DPOTRF( UPLO, N, A, LDA, INFO )
* 
*       .. Scalar Arguments ..
*       CHARACTER          UPLO
*       INTEGER            INFO, LDA, N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION   A( LDA, * )
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DPOTRF computes the Cholesky factorization of a real symmetric
*> positive definite matrix A.
*>
*> The factorization has the form
*>    A = U**T * U,  if UPLO = 'U', or
*>    A = L  * L**T,  if UPLO = 'L',
*> where U is an upper triangular matrix and L is lower triangular.
*>
*> This is the block version of the algorithm, calling Level 3 BLAS.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] UPLO
*> \verbatim
*>          UPLO is CHARACTER*1
*>          = 'U':  Upper triangle of A is stored;
*>          = 'L':  Lower triangle of A is stored.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The order of the matrix A.  N >= 0.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is DOUBLE PRECISION array, dimension (LDA,N)
*>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
*>          N-by-N upper triangular part of A contains the upper
*>          triangular part of the matrix A, and the strictly lower
*>          triangular part of A is not referenced.  If UPLO = 'L', the
*>          leading N-by-N lower triangular part of A contains the lower
*>          triangular part of the matrix A, and the strictly upper
*>          triangular part of A is not referenced.
*>
*>          On exit, if INFO = 0, the factor U or L from the Cholesky
*>          factorization A = U**T*U or A = L*L**T.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A.  LDA >= max(1,N).
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0:  successful exit
*>          < 0:  if INFO = -i, the i-th argument had an illegal value
*>          > 0:  if INFO = i, the leading minor of order i is not
*>                positive definite, and the factorization could not be
*>                completed.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date November 2011
*
*> \ingroup doublePOcomputational
*
*  =====================================================================
      SUBROUTINE GAL_DPOTRF(UPLO,N,A,LDA,INFO)
*
*  -- LAPACK computational routine (version 3.4.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      CHARACTERUPLO
      INTEGER INFO,LDA,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*)
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D+0)
*     ..
*     .. Local Scalars ..
      LOGICAL UPPER
      INTEGER J,JB,NB
*     ..
*     .. External Functions ..
      LOGICAL GAL_LSAME
      INTEGER GAL_ILAENV
      EXTERNAL GAL_LSAME,GAL_ILAENV
*     ..
*     .. External Subroutines ..
      EXTERNAL GAL_DGEMM,GAL_DPOTF2,GAL_DSYRK,GAL_DTRSM,GAL_XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MAX,MIN
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO=0
      UPPER=GAL_LSAME(UPLO,'U')
      IF(.NOT.UPPER.AND..NOT.GAL_LSAME(UPLO,'L'))THEN
      INFO=-1
      ELSE IF(N.LT.0)THEN
      INFO=-2
      ELSE IF(LDA.LT.MAX(1,N))THEN
      INFO=-4
      END IF
      IF(INFO.NE.0)THEN
      CALL GAL_XERBLA('GAL_DPOTRF',-INFO)
      RETURN
      END IF
*
*     Quick return if possible
*
      IF(N.EQ.0)
     $RETURN
*
*     Determine the block size for this environment.
*
      NB=GAL_ILAENV(1,'GAL_DPOTRF',UPLO,N,-1,-1,-1)
      IF(NB.LE.1.OR.NB.GE.N)THEN
*
*        Use unblocked code.
*
      CALL GAL_DPOTF2(UPLO,N,A,LDA,INFO)
      ELSE 
*
*        Use blocked code.
*
      IF(UPPER)THEN
*
*           Compute the Cholesky factorization A = U**T*U.
*
      DO 10J=1,N,NB
*
*              Update and factorize the current diagonal block and test
*              for non-positive-definiteness.
*
      JB=MIN(NB,N-J+1)
      CALL GAL_DSYRK('UPPER','TRANSPOSE',JB,J-1,-ONE,
     $A(1,J),LDA,ONE,A(J,J),LDA)
      CALL GAL_DPOTF2('UPPER',JB,A(J,J),LDA,INFO)
      IF(INFO.NE.0)
     $GO TO 30
      IF(J+JB.LE.N)THEN
*
*                 Compute the current block row.
*
      CALL GAL_DGEMM('TRANSPOSE','NOTRANSPOSE',JB,N-J-JB+1,
     $J-1,-ONE,A(1,J),LDA,A(1,J+JB),
     $LDA,ONE,A(J,J+JB),LDA)
      CALL GAL_DTRSM('LEFT','UPPER','TRANSPOSE','NON-UNIT',
     $JB,N-J-JB+1,ONE,A(J,J),LDA,
     $A(J,J+JB),LDA)
      END IF
   10 CONTINUE
*
      ELSE 
*
*           Compute the Cholesky factorization A = L*L**T.
*
      DO 20J=1,N,NB
*
*              Update and factorize the current diagonal block and test
*              for non-positive-definiteness.
*
      JB=MIN(NB,N-J+1)
      CALL GAL_DSYRK('LOWER','NOTRANSPOSE',JB,J-1,-ONE,
     $A(J,1),LDA,ONE,A(J,J),LDA)
      CALL GAL_DPOTF2('LOWER',JB,A(J,J),LDA,INFO)
      IF(INFO.NE.0)
     $GO TO 30
      IF(J+JB.LE.N)THEN
*
*                 Compute the current block column.
*
      CALL GAL_DGEMM('NOTRANSPOSE','TRANSPOSE',N-J-JB+1,JB,
     $J-1,-ONE,A(J+JB,1),LDA,A(J,1),
     $LDA,ONE,A(J+JB,J),LDA)
      CALL GAL_DTRSM('RIGHT','LOWER','TRANSPOSE','NON-UNIT',
     $N-J-JB+1,JB,ONE,A(J,J),LDA,
     $A(J+JB,J),LDA)
      END IF
   20 CONTINUE
      END IF
      END IF
      GO TO 40
*
   30 CONTINUE
      INFO=INFO+J-1
*
   40 CONTINUE
      RETURN
*
*     End of DPOTRF
*
      END 
