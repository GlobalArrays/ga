c
c     $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hpuxargs.f,v 1.3 1995-02-24 02:13:55 d3h325 Exp $
c
      integer function hpargc()
      hpargc = iargc() + 1
      end
      integer function hpargv(index, arg, maxlen)
      character*256 arg
      hpargv = igetarg(index,arg,maxlen)
      end
