      subroutine ffflush(unit)
C$Id: ffflush.F,v 1.3 1995-02-02 23:13:09 d3g681 Exp $
      integer unit
c
#ifdef CRAY_T3D
*     if(unit.eq.6)then
*       call flush(101)
*     else
*       call flush(unit)
*     endif
#elif defined(NX)
      call forflush(unit)
#elif !defined(AIX)
      call flush(unit)
#endif
c
      end
