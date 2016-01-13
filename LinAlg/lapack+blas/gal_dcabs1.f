      DOUBLE PRECISION FUNCTION GAL_DCABS1(Z)
*     .. Scalar Arguments ..
      DOUBLE COMPLEX Z
*     ..
*     ..
*  Purpose
*  =======
*
*  GAL_DCABS1 computes absolute value of a double complex number 
*
*  =====================================================================
*
*     .. Intrinsic Functions ..
      INTRINSIC ABS,DBLE,DIMAG
*
      GAL_DCABS1 = ABS(DBLE(Z)) + ABS(DIMAG(Z))
      RETURN
      END

