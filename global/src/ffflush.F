      subroutine ffflush(unit)
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
#elif !defined(AIX) && !defined(HPUX)
      call flush(unit)
#endif
c
      end
