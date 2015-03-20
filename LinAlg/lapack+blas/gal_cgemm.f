*> \brief \b CGEMM
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       SUBROUTINE CGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
* 
*       .. Scalar Arguments ..
*       COMPLEX ALPHA,BETA
*       INTEGER K,LDA,LDB,LDC,M,N
*       CHARACTER TRANSA,TRANSB
*       ..
*       .. Array Arguments ..
*       COMPLEX A(LDA,*),B(LDB,*),C(LDC,*)
*       ..
*  
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> CGEMM  performs one of the matrix-matrix operations
*>
*>    C := alpha*op( A )*op( B ) + beta*C,
*>
*> where  op( X ) is one of
*>
*>    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
*>
*> alpha and beta are scalars, and A, B and C are matrices, with op( A )
*> an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] TRANSA
*> \verbatim
*>          TRANSA is CHARACTER*1
*>           On entry, TRANSA specifies the form of op( A ) to be used in
*>           the matrix multiplication as follows:
*>
*>              TRANSA = 'N' or 'n',  op( A ) = A.
*>
*>              TRANSA = 'T' or 't',  op( A ) = A**T.
*>
*>              TRANSA = 'C' or 'c',  op( A ) = A**H.
*> \endverbatim
*>
*> \param[in] TRANSB
*> \verbatim
*>          TRANSB is CHARACTER*1
*>           On entry, TRANSB specifies the form of op( B ) to be used in
*>           the matrix multiplication as follows:
*>
*>              TRANSB = 'N' or 'n',  op( B ) = B.
*>
*>              TRANSB = 'T' or 't',  op( B ) = B**T.
*>
*>              TRANSB = 'C' or 'c',  op( B ) = B**H.
*> \endverbatim
*>
*> \param[in] M
*> \verbatim
*>          M is INTEGER
*>           On entry,  M  specifies  the number  of rows  of the  matrix
*>           op( A )  and of the  matrix  C.  M  must  be at least  zero.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>           On entry,  N  specifies the number  of columns of the matrix
*>           op( B ) and the number of columns of the matrix C. N must be
*>           at least zero.
*> \endverbatim
*>
*> \param[in] K
*> \verbatim
*>          K is INTEGER
*>           On entry,  K  specifies  the number of columns of the matrix
*>           op( A ) and the number of rows of the matrix op( B ). K must
*>           be at least  zero.
*> \endverbatim
*>
*> \param[in] ALPHA
*> \verbatim
*>          ALPHA is COMPLEX
*>           On entry, ALPHA specifies the scalar alpha.
*> \endverbatim
*>
*> \param[in] A
*> \verbatim
*>          A is COMPLEX array of DIMENSION ( LDA, ka ), where ka is
*>           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
*>           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
*>           part of the array  A  must contain the matrix  A,  otherwise
*>           the leading  k by m  part of the array  A  must contain  the
*>           matrix A.
*> \endverbatim
*>
*> \param[in] LDA
*> \verbatim
*>          LDA is INTEGER
*>           On entry, LDA specifies the first dimension of A as declared
*>           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
*>           LDA must be at least  max( 1, m ), otherwise  LDA must be at
*>           least  max( 1, k ).
*> \endverbatim
*>
*> \param[in] B
*> \verbatim
*>          B is COMPLEX array of DIMENSION ( LDB, kb ), where kb is
*>           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
*>           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
*>           part of the array  B  must contain the matrix  B,  otherwise
*>           the leading  n by k  part of the array  B  must contain  the
*>           matrix B.
*> \endverbatim
*>
*> \param[in] LDB
*> \verbatim
*>          LDB is INTEGER
*>           On entry, LDB specifies the first dimension of B as declared
*>           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
*>           LDB must be at least  max( 1, k ), otherwise  LDB must be at
*>           least  max( 1, n ).
*> \endverbatim
*>
*> \param[in] BETA
*> \verbatim
*>          BETA is COMPLEX
*>           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*>           supplied as zero then C need not be set on input.
*> \endverbatim
*>
*> \param[in,out] C
*> \verbatim
*>          C is COMPLEX array of DIMENSION ( LDC, n ).
*>           Before entry, the leading  m by n  part of the array  C must
*>           contain the matrix  C,  except when  beta  is zero, in which
*>           case C need not be set on entry.
*>           On exit, the array  C  is overwritten by the  m by n  matrix
*>           ( alpha*op( A )*op( B ) + beta*C ).
*> \endverbatim
*>
*> \param[in] LDC
*> \verbatim
*>          LDC is INTEGER
*>           On entry, LDC specifies the first dimension of C as declared
*>           in  the  calling  (sub)  program.   LDC  must  be  at  least
*>           max( 1, m ).
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
*> \ingroup complex_blas_level3
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  Level 3 Blas routine.
*>
*>  -- Written on 8-February-1989.
*>     Jack Dongarra, Argonne National Laboratory.
*>     Iain Duff, AERE Harwell.
*>     Jeremy Du Croz, Numerical Algorithms Group Ltd.
*>     Sven Hammarling, Numerical Algorithms Group Ltd.
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE GAL_CGEMM(TRANSA,TRANSB,M,N,K,
     $ALPHA,A,LDA,B,LDB,BETA,C,LDC)
*
*  -- Reference BLAS level3 routine (version 3.4.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      COMPLEXALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTERTRANSA,TRANSB
*     ..
*     .. Array Arguments ..
      COMPLEXA(LDA,*),B(LDB,*),C(LDC,*)
*     ..
*
*  =====================================================================
*
*     .. External Functions ..
      LOGICAL GAL_LSAME
      EXTERNAL GAL_LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL GAL_XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC CONJG,MAX
*     ..
*     .. Local Scalars ..
      COMPLEXTEMP
      INTEGER I,INFO,J,L,NCOLA,NROWA,NROWB
      LOGICAL CONJA,CONJB,NOTA,NOTB
*     ..
*     .. Parameters ..
      COMPLEXONE
      PARAMETER (ONE=(1.0E+0,0.0E+0))
      COMPLEXZERO
      PARAMETER (ZERO=(0.0E+0,0.0E+0))
*     ..
*
*     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
*     conjugated or transposed, set  CONJA and CONJB  as true if  A  and
*     B  respectively are to be  transposed but  not conjugated  and set
*     NROWA, NCOLA and  NROWB  as the number of rows and  columns  of  A
*     and the number of rows of  B  respectively.
*
      NOTA=GAL_LSAME(TRANSA,'N')
      NOTB=GAL_LSAME(TRANSB,'N')
      CONJA=GAL_LSAME(TRANSA,'C')
      CONJB=GAL_LSAME(TRANSB,'C')
      IF(NOTA)THEN
      NROWA=M
      NCOLA=K
      ELSE 
      NROWA=K
      NCOLA=M
      END IF
      IF(NOTB)THEN
      NROWB=K
      ELSE 
      NROWB=N
      END IF
*
*     Test the input parameters.
*
      INFO=0
      IF((.NOT.NOTA).AND.(.NOT.CONJA).AND.
     +(.NOT.GAL_LSAME(TRANSA,'T')))THEN
      INFO=1
      ELSE IF((.NOT.NOTB).AND.(.NOT.CONJB).AND.
     +(.NOT.GAL_LSAME(TRANSB,'T')))THEN
      INFO=2
      ELSE IF(M.LT.0)THEN
      INFO=3
      ELSE IF(N.LT.0)THEN
      INFO=4
      ELSE IF(K.LT.0)THEN
      INFO=5
      ELSE IF(LDA.LT.MAX(1,NROWA))THEN
      INFO=8
      ELSE IF(LDB.LT.MAX(1,NROWB))THEN
      INFO=10
      ELSE IF(LDC.LT.MAX(1,M))THEN
      INFO=13
      END IF
      IF(INFO.NE.0)THEN
      CALL GAL_XERBLA('GAL_CGEMM',INFO)
      RETURN
      END IF
*
*     Quick return if possible.
*
      IF((M.EQ.0).OR.(N.EQ.0).OR.
     +(((ALPHA.EQ.ZERO).OR.(K.EQ.0)).AND.(BETA.EQ.ONE)))RETURN
*
*     And when  alpha.eq.zero.
*
      IF(ALPHA.EQ.ZERO)THEN
      IF(BETA.EQ.ZERO)THEN
      DO 20J=1,N
      DO 10I=1,M
      C(I,J)=ZERO
   10 CONTINUE
   20 CONTINUE
      ELSE 
      DO 40J=1,N
      DO 30I=1,M
      C(I,J)=BETA*C(I,J)
   30 CONTINUE
   40 CONTINUE
      END IF
      RETURN
      END IF
*
*     Start the operations.
*
      IF(NOTB)THEN
      IF(NOTA)THEN
*
*           Form  C := alpha*A*B + beta*C.
*
      DO 90J=1,N
      IF(BETA.EQ.ZERO)THEN
      DO 50I=1,M
      C(I,J)=ZERO
   50 CONTINUE
      ELSE IF(BETA.NE.ONE)THEN
      DO 60I=1,M
      C(I,J)=BETA*C(I,J)
   60 CONTINUE
      END IF
      DO 80L=1,K
      IF(B(L,J).NE.ZERO)THEN
      TEMP=ALPHA*B(L,J)
      DO 70I=1,M
      C(I,J)=C(I,J)+TEMP*A(I,L)
   70 CONTINUE
      END IF
   80 CONTINUE
   90 CONTINUE
      ELSE IF(CONJA)THEN
*
*           Form  C := alpha*A**H*B + beta*C.
*
      DO 120J=1,N
      DO 110I=1,M
      TEMP=ZERO
      DO 100L=1,K
      TEMP=TEMP+CONJG(A(L,I))*B(L,J)
  100 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  110 CONTINUE
  120 CONTINUE
      ELSE 
*
*           Form  C := alpha*A**T*B + beta*C
*
      DO 150J=1,N
      DO 140I=1,M
      TEMP=ZERO
      DO 130L=1,K
      TEMP=TEMP+A(L,I)*B(L,J)
  130 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  140 CONTINUE
  150 CONTINUE
      END IF
      ELSE IF(NOTA)THEN
      IF(CONJB)THEN
*
*           Form  C := alpha*A*B**H + beta*C.
*
      DO 200J=1,N
      IF(BETA.EQ.ZERO)THEN
      DO 160I=1,M
      C(I,J)=ZERO
  160 CONTINUE
      ELSE IF(BETA.NE.ONE)THEN
      DO 170I=1,M
      C(I,J)=BETA*C(I,J)
  170 CONTINUE
      END IF
      DO 190L=1,K
      IF(B(J,L).NE.ZERO)THEN
      TEMP=ALPHA*CONJG(B(J,L))
      DO 180I=1,M
      C(I,J)=C(I,J)+TEMP*A(I,L)
  180 CONTINUE
      END IF
  190 CONTINUE
  200 CONTINUE
      ELSE 
*
*           Form  C := alpha*A*B**T          + beta*C
*
      DO 250J=1,N
      IF(BETA.EQ.ZERO)THEN
      DO 210I=1,M
      C(I,J)=ZERO
  210 CONTINUE
      ELSE IF(BETA.NE.ONE)THEN
      DO 220I=1,M
      C(I,J)=BETA*C(I,J)
  220 CONTINUE
      END IF
      DO 240L=1,K
      IF(B(J,L).NE.ZERO)THEN
      TEMP=ALPHA*B(J,L)
      DO 230I=1,M
      C(I,J)=C(I,J)+TEMP*A(I,L)
  230 CONTINUE
      END IF
  240 CONTINUE
  250 CONTINUE
      END IF
      ELSE IF(CONJA)THEN
      IF(CONJB)THEN
*
*           Form  C := alpha*A**H*B**H + beta*C.
*
      DO 280J=1,N
      DO 270I=1,M
      TEMP=ZERO
      DO 260L=1,K
      TEMP=TEMP+CONJG(A(L,I))*CONJG(B(J,L))
  260 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  270 CONTINUE
  280 CONTINUE
      ELSE 
*
*           Form  C := alpha*A**H*B**T + beta*C
*
      DO 310J=1,N
      DO 300I=1,M
      TEMP=ZERO
      DO 290L=1,K
      TEMP=TEMP+CONJG(A(L,I))*B(J,L)
  290 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  300 CONTINUE
  310 CONTINUE
      END IF
      ELSE 
      IF(CONJB)THEN
*
*           Form  C := alpha*A**T*B**H + beta*C
*
      DO 340J=1,N
      DO 330I=1,M
      TEMP=ZERO
      DO 320L=1,K
      TEMP=TEMP+A(L,I)*CONJG(B(J,L))
  320 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  330 CONTINUE
  340 CONTINUE
      ELSE 
*
*           Form  C := alpha*A**T*B**T + beta*C
*
      DO 370J=1,N
      DO 360I=1,M
      TEMP=ZERO
      DO 350L=1,K
      TEMP=TEMP+A(L,I)*B(J,L)
  350 CONTINUE
      IF(BETA.EQ.ZERO)THEN
      C(I,J)=ALPHA*TEMP
      ELSE 
      C(I,J)=ALPHA*TEMP+BETA*C(I,J)
      END IF
  360 CONTINUE
  370 CONTINUE
      END IF
      END IF
*
      RETURN
*
*     End of CGEMM .
*
      END 
