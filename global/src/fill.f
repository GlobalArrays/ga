      subroutine util_dfill(n,val,a,ia)
      implicit none
      double precision  a(*), val
      integer n, ia, i
c
c     initialise double precision array to scalar value
c
      if (ia.eq.1) then
         do 10 i = 1, n
            a(i) = val
 10      continue
      else
         do 20 i = 1,(n-1)*ia+1,ia
            a(i) = val
 20      continue
      endif
c
      end

      subroutine util_ifill(n,val,a,ia)
      implicit none
      integer n, ia, i, a(*),val
c
c     initialise integer array to scalar value
c
      if (ia.eq.1) then
         do 10 i = 1, n
            a(i) = val
 10      continue
      else
         do 20 i = 1,(n-1)*ia+1,ia
            a(i) = val
 20      continue
      endif
c
      end

      subroutine util_qfill(n,val,a,ia)
      implicit none
      double  complex  a(*), val
      integer n, ia, i
c
c     initialise double complex array to scalar value
c
      if (ia.eq.1) then
         do 10 i = 1, n
            a(i) = val
 10      continue
      else
         do 20 i = 1,(n-1)*ia+1,ia
            a(i) = val
 20      continue
      endif
c
      end

