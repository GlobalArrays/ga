c
c     $Header: /tmp/hpctools/ga/tcgmsg-mpi/hpuxargs.f,v 1.1 1995-10-12 00:06:29 d3h325 Exp $
c
      integer function hpargc()
      hpargc = iargc() + 1
      end
      integer function hpargv(index, arg, maxlen)
      character*256 arg
      hpargv = igetarg(index,arg,maxlen)
      end
