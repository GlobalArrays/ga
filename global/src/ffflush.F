      subroutine ffflush(unit)
      integer unit
#ifdef NX
      call forflush(unit)
#elif !defined(AIX)
      call flush(unit)
#endif
c
      end
