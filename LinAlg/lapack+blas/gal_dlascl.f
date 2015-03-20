*> \brief \b DLASCL multiplies a general rectangular matrix by a real scalar defined as cto/cfrom.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download DLASCL + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlascl.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlascl.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlascl.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DLASCL( TYPE, KL, KU, CFROM, CTO, M, N, A, LDA, INFO )
* 
*       .. Scalar Arguments ..
*       CHARACTER          TYPE
*       INTEGER            INFO, KL, KU, LDA, M, N
*       DOUBLE PRECISION   CFROM, CTO
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
*> DLASCL multiplies the M by N real matrix A by the real scalar
*> CTO/CFROM.  This is done without over/underflow as long as the final
*> result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
*> A may be full, upper triangular, lower triangular, upper Hessenberg,
*> or banded.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] TYPE
*> \verbatim
*>          TYPE is CHARACTER*1
*>          TYPE indices the storage type of the input matrix.
*>          = 'G':  A is a full matrix.
*>          = 'L':  A is a lower triangular matrix.
*>          = 'U':  A is an upper triangular matrix.
*>          = 'H':  A is an upper Hessenberg matrix.
*>          = 'B':  A is a symmetric band matrix with lower bandwidth KL
*>                  and upper bandwidth KU and with the only the lower
*>                  half stored.
*>          = 'Q':  A is a symmetric band matrix with lower bandwidth KL
*>                  and upper bandwidth KU and with the only the upper
*>                  half stored.
*>          = 'Z':  A is a band matrix with lower bandwidth KL and upper
*>                  bandwidth KU. See DGBTRF for storage details.
*> \endverbatim
*>
*> \param[in] KL
*> \verbatim
*>          KL is INTEGER
*>          The lower bandwidth of A.  Referenced only if TYPE = 'B',
*>          'Q' or 'Z'.
*> \endverbatim
*>
*> \param[in] KU
*> \verbatim
*>          KU is INTEGER
*>          The upper bandwidth of A.  Referenced only if TYPE = 'B',
*>          'Q' or 'Z'.
*> \endverbatim
*>
*> \param[in] CFROM
*> \verbatim
*>          CFROM is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[in] CTO
*> \verbatim
*>          CTO is DOUBLE PRECISION
*>
*>          The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
*>          without over/underflow if the final result CTO*A(I,J)/CFROM
*>          can be represented without over/underflow.  CFROM must be
*>          nonzero.
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix A.  M >= 0.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrix A.  N >= 0.
*> \endverbatim
*>
*> \param[in,out] A
*> \verbatim
*>          A is DOUBLE PRECISION array, dimension (LDA,N)
*>          The matrix to be multiplied by CTO/CFROM.  See TYPE for the
*>          storage type.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>          The leading dimension of the array A.  LDA >= max(1,M).
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          0  - successful exit
*>          <0 - if INFO = -i, the i-th argument had an illegal value.
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
*> \date September 2012
*
*> \ingroup auxOTHERauxiliary
*
*  =====================================================================
      SUBROUTINE GAL_DLASCL(TYPE,KL,KU,CFROM,CTO,M,N,A,LDA,INFO)
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      CHARACTERTYPE
      INTEGER INFO,KL,KU,LDA,M,N
      DOUBLE PRECISION CFROM,CTO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*)
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION ZERO,ONE
      PARAMETER (ZERO=0.0D0,ONE=1.0D0)
*     ..
*     .. Local Scalars ..
      LOGICAL DONE
      INTEGER I,ITYPE,J,K1,K2,K3,K4
      DOUBLE PRECISION BIGNUM,CFROM1,CFROMC,CTO1,CTOC,MUL,SMLNUM
*     ..
*     .. External Functions ..
      LOGICAL GAL_LSAME,GAL_DISNAN
      DOUBLE PRECISION GAL_DLAMCH
      EXTERNAL GAL_LSAME,GAL_DLAMCH,GAL_DISNAN
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS,MAX,MIN
*     ..
*     .. External Subroutines ..
      EXTERNAL GAL_XERBLA
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO=0
*
      IF(GAL_LSAME(TYPE,'G'))THEN
      ITYPE=0
      ELSE IF(GAL_LSAME(TYPE,'L'))THEN
      ITYPE=1
      ELSE IF(GAL_LSAME(TYPE,'U'))THEN
      ITYPE=2
      ELSE IF(GAL_LSAME(TYPE,'H'))THEN
      ITYPE=3
      ELSE IF(GAL_LSAME(TYPE,'B'))THEN
      ITYPE=4
      ELSE IF(GAL_LSAME(TYPE,'Q'))THEN
      ITYPE=5
      ELSE IF(GAL_LSAME(TYPE,'Z'))THEN
      ITYPE=6
      ELSE 
      ITYPE=-1
      END IF
*
      IF(ITYPE.EQ.-1)THEN
      INFO=-1
      ELSE IF(CFROM.EQ.ZERO.OR.GAL_DISNAN(CFROM))THEN
      INFO=-4
      ELSE IF(GAL_DISNAN(CTO))THEN
      INFO=-5
      ELSE IF(M.LT.0)THEN
      INFO=-6
      ELSE IF(N.LT.0.OR.(ITYPE.EQ.4.AND.N.NE.M).OR.
     $(ITYPE.EQ.5.AND.N.NE.M))THEN
      INFO=-7
      ELSE IF(ITYPE.LE.3.AND.LDA.LT.MAX(1,M))THEN
      INFO=-9
      ELSE IF(ITYPE.GE.4)THEN
      IF(KL.LT.0.OR.KL.GT.MAX(M-1,0))THEN
      INFO=-2
      ELSE IF(KU.LT.0.OR.KU.GT.MAX(N-1,0).OR.
     $((ITYPE.EQ.4.OR.ITYPE.EQ.5).AND.KL.NE.KU))
     $THEN
      INFO=-3
      ELSE IF((ITYPE.EQ.4.AND.LDA.LT.KL+1).OR.
     $(ITYPE.EQ.5.AND.LDA.LT.KU+1).OR.
     $(ITYPE.EQ.6.AND.LDA.LT.2*KL+KU+1))THEN
      INFO=-9
      END IF
      END IF
*
      IF(INFO.NE.0)THEN
      CALL GAL_XERBLA('GAL_DLASCL',-INFO)
      RETURN
      END IF
*
*     Quick return if possible
*
      IF(N.EQ.0.OR.M.EQ.0)
     $RETURN
*
*     Get machine parameters
*
      SMLNUM=GAL_DLAMCH('S')
      BIGNUM=ONE/SMLNUM
*
      CFROMC=CFROM
      CTOC=CTO
*
   10 CONTINUE
      CFROM1=CFROMC*SMLNUM
      IF(CFROM1.EQ.CFROMC)THEN
!     CFROMCISANINF.MULTIPLYBYACORRECTLYSIGNEDZEROFOR
!     FINITECTOC,ORANANIFCTOCISINFINITE.
      MUL=CTOC/CFROMC
      DONE=.TRUE.
      CTO1=CTOC
      ELSE 
      CTO1=CTOC/BIGNUM
      IF(CTO1.EQ.CTOC)THEN
!     CTOCISEITHER0ORANINF.INBOTHCASES,CTOCITSELF
!     SERVESASTHECORRECTMULTIPLICATIONFACTOR.
      MUL=CTOC
      DONE=.TRUE.
      CFROMC=ONE
      ELSE IF(ABS(CFROM1).GT.ABS(CTOC).AND.CTOC.NE.ZERO)THEN
      MUL=SMLNUM
      DONE=.FALSE.
      CFROMC=CFROM1
      ELSE IF(ABS(CTO1).GT.ABS(CFROMC))THEN
      MUL=BIGNUM
      DONE=.FALSE.
      CTOC=CTO1
      ELSE 
      MUL=CTOC/CFROMC
      DONE=.TRUE.
      END IF
      END IF
*
      IF(ITYPE.EQ.0)THEN
*
*        Full matrix
*
      DO 30J=1,N
      DO 20I=1,M
      A(I,J)=A(I,J)*MUL
   20 CONTINUE
   30 CONTINUE
*
      ELSE IF(ITYPE.EQ.1)THEN
*
*        Lower triangular matrix
*
      DO 50J=1,N
      DO 40I=J,M
      A(I,J)=A(I,J)*MUL
   40 CONTINUE
   50 CONTINUE
*
      ELSE IF(ITYPE.EQ.2)THEN
*
*        Upper triangular matrix
*
      DO 70J=1,N
      DO 60I=1,MIN(J,M)
      A(I,J)=A(I,J)*MUL
   60 CONTINUE
   70 CONTINUE
*
      ELSE IF(ITYPE.EQ.3)THEN
*
*        Upper Hessenberg matrix
*
      DO 90J=1,N
      DO 80I=1,MIN(J+1,M)
      A(I,J)=A(I,J)*MUL
   80 CONTINUE
   90 CONTINUE
*
      ELSE IF(ITYPE.EQ.4)THEN
*
*        Lower half of a symmetric band matrix
*
      K3=KL+1
      K4=N+1
      DO 110J=1,N
      DO 100I=1,MIN(K3,K4-J)
      A(I,J)=A(I,J)*MUL
  100 CONTINUE
  110 CONTINUE
*
      ELSE IF(ITYPE.EQ.5)THEN
*
*        Upper half of a symmetric band matrix
*
      K1=KU+2
      K3=KU+1
      DO 130J=1,N
      DO 120I=MAX(K1-J,1),K3
      A(I,J)=A(I,J)*MUL
  120 CONTINUE
  130 CONTINUE
*
      ELSE IF(ITYPE.EQ.6)THEN
*
*        Band matrix
*
      K1=KL+KU+2
      K2=KL+1
      K3=2*KL+KU+1
      K4=KL+KU+1+M
      DO 150J=1,N
      DO 140I=MAX(K1-J,K2),MIN(K3,K4-J)
      A(I,J)=A(I,J)*MUL
  140 CONTINUE
  150 CONTINUE
*
      END IF
*
      IF(.NOT.DONE)
     $GO TO 10
*
      RETURN
*
*     End of DLASCL
*
      END 
