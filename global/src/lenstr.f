      integer function lenstr(a)
C$Id: lenstr.f,v 1.2 1995-02-02 23:13:43 d3g681 Exp $
      implicit none
      character*(*) a
      integer i
c
      integer len
      intrinsic len
c
c     return length of character string in a minus any
c     trailing blanks
c
      lenstr = 0
      do i = len(a), 1, -1
         if (a(i:i) .ne. ' ') then
            lenstr = i
            goto 10
         endif
      enddo
 10   continue
c
      end
