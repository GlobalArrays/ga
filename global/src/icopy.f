      subroutine icopy(n,a,ia,b,ib)
      implicit none
      integer n, ia, ib
      integer a(ia,*), b(ib,*)
      integer i
c
c     copy a into b
c
      do 10 i = 1,n
         b(1,i) = a(1,i)
 10   continue
c
      end
