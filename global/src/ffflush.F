c $Id: ffflush.F,v 1.7 1999-07-28 00:25:29 d3h325 Exp $
      subroutine ffflush(unit)
      integer unit
c
#ifdef CRAY
*     if(unit.eq.6)then
*       call flush(101)
*     else
*       call flush(unit)
*     endif
#elif defined(NX)
      call forflush(unit)
#elif !defined(AIX) && !defined(HPUX)
      call flush(unit)
#endif
c
      end
