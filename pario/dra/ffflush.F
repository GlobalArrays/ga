c $Id: ffflush.F,v 1.2 2002-10-07 16:51:51 d3h325 Exp $
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
#elif !defined(AIX) && !defined(HPUX) && !defined(IFCLINUX)
      call flush(unit)
#endif
c
      end
