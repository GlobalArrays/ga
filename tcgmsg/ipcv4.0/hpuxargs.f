c
c     $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hpuxargs.f,v 1.1.1.1 1994-03-29 06:44:52 d3g681 Exp $
c
      integer function hpargc()
      hpargc = iargc() + 1
      end
      integer function hpargv(index, arg, maxlen)
      character*256 arg
      hpargv = igetarg(index,arg,maxlen)
      end
