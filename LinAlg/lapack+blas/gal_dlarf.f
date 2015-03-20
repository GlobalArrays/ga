*> \brief \b DLARF applies an elementary reflector to a general rectangular matrix.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*> \htmlonly
*> Download DLARF + dependencies 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarf.f"> 
*> [TGZ]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarf.f"> 
*> [ZIP]</a> 
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarf.f"> 
*> [TXT]</a>
*> \endhtmlonly 
*
*  Definition:
*  ===========
*
*       SUBROUTINE DLARF( SIDE, M, N, V, INCV, TAU, C, LDC, WORK )
* 
*       .. Scalar Arguments ..
*       CHARACTER          SIDE
*       INTEGER            INCV, LDC, M, N
*       DOUBLE PRECISION   TAU
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION   C( LDC, * ), V( * ), WORK( * )
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DLARF applies a real elementary reflector H to a real m by n matrix
*> C, from either the left or the right. H is represented in the form
*>
*>       H = I - tau * v * v**T
*>
*> where tau is a real scalar and v is a real vector.
*>
*> If tau = 0, then H is taken to be the unit matrix.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] SIDE
*> \verbatim
*>          SIDE is CHARACTER*1
*>          = 'L': form  H * C
*>          = 'R': form  C * H
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>          The number of rows of the matrix C.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of columns of the matrix C.
*> \endverbatim
*>
*> \param[in] V
*> \verbatim
*>          V is DOUBLE PRECISION array, dimension
*>                     (1 + (M-1)*abs(INCV)) if SIDE = 'L'
*>                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
*>          The vector v in the representation of H. V is not used if
*>          TAU = 0.
*> \endverbatim
*>
*> \param[in] INCV
*> \verbatim
*>          INCV is INTEGER
*>          The increment between elements of v. INCV <> 0.
*> \endverbatim
*>
*> \param[in] TAU
*> \verbatim
*>          TAU is DOUBLE PRECISION
*>          The value tau in the representation of H.
*> \endverbatim
*>
*> \param[in,out] C
*> \verbatim
*>          C is DOUBLE PRECISION array, dimension (LDC,N)
*>          On entry, the m by n matrix C.
*>          On exit, C is overwritten by the matrix H * C if SIDE = 'L',
*>          or C * H if SIDE = 'R'.
*> \endverbatim
*>
*> \param[in] LDC
*> \verbatim
*>          LDC is INTEGER
*>          The leading dimension of the array C. LDC >= max(1,M).
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is DOUBLE PRECISION array, dimension
*>                         (N) if SIDE = 'L'
*>                      or (M) if SIDE = 'R'
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
*> \ingroup doubleOTHERauxiliary
*
*  =====================================================================
      SUBROUTINE GAL_DLARF(SIDE,M,N,V,INCV,TAU,C,LDC,WORK)
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      CHARACTERSIDE
      INTEGER INCV,LDC,M,N
      DOUBLE PRECISION TAU
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION C(LDC,*),V(*),WORK(*)
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
*     ..
*     .. Local Scalars ..
      LOGICAL APPLYLEFT
      INTEGER I,LASTV,LASTC
*     ..
*     .. External Subroutines ..
      EXTERNAL GAL_DGEMV,GAL_DGER
*     ..
*     .. External Functions ..
      LOGICAL GAL_LSAME
      INTEGER GAL_ILADLR,GAL_ILADLC
      EXTERNAL GAL_LSAME,GAL_ILADLR,GAL_ILADLC
*     ..
*     .. Executable Statements ..
*
      APPLYLEFT=GAL_LSAME(SIDE,'L')
      LASTV=0
      LASTC=0
      IF(TAU.NE.ZERO)THEN
!     SETUPVARIABLESFORSCANNINGV.LASTVBEGINSPOINTINGTO THEEND 
!     OFV.
      IF(APPLYLEFT)THEN
      LASTV=M
      ELSE 
      LASTV=N
      END IF
      IF(INCV.GT.0)THEN
      I=1+(LASTV-1)*INCV
      ELSE 
      I=1
      END IF
!     LOOKFORTHELASTNON-ZEROROWINV.
      DO WHILE(LASTV.GT.0.AND.V(I).EQ.ZERO)
      LASTV=LASTV-1
      I=I-INCV
      END DO 
      IF(APPLYLEFT)THEN
!     SCANFORTHELASTNON-ZEROCOLUMNINC(1:LASTV,:).
      LASTC=GAL_ILADLC(LASTV,N,C,LDC)
      ELSE 
!     SCANFORTHELASTNON-ZEROROWINC(:,1:LASTV).
      LASTC=GAL_ILADLR(M,LASTV,C,LDC)
      END IF
      END IF
!     NOTETHATLASTC.EQ.0RENDERSTHEBLASOPERATIONSNULL;NOSPECIAL
!     CASEISNEEDEDATTHISLEVEL.
      IF(APPLYLEFT)THEN
*
*        Form  H * C
*
      IF(LASTV.GT.0)THEN
*
*           w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1)
*
      CALL GAL_DGEMV('TRANSPOSE',LASTV,LASTC,ONE,C,LDC,V,INCV,
     $ZERO,WORK,1)
*
*           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
*
      CALL GAL_DGER(LASTV,LASTC,-TAU,V,INCV,WORK,1,C,LDC)
      END IF
      ELSE 
*
*        Form  C * H
*
      IF(LASTV.GT.0)THEN
*
*           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
*
      CALL GAL_DGEMV('NOTRANSPOSE',LASTC,LASTV,ONE,C,LDC,
     $V,INCV,ZERO,WORK,1)
*
*           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
*
      CALL GAL_DGER(LASTC,LASTV,-TAU,WORK,1,V,INCV,C,LDC)
      END IF
      END IF
      RETURN
*
*     End of DLARF
*
      END 
