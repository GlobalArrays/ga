c
c     $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/hpuxargs.f,v 1.2 1995-02-02 23:25:08 d3g681 Exp $
c
      integer function hpargc()
C$Id: hpuxargs.f,v 1.2 1995-02-02 23:25:08 d3g681 Exp $
      hpargc = iargc() + 1
      end
      integer function hpargv(index, arg, maxlen)
      character*256 arg
      hpargv = igetarg(index,arg,maxlen)
      end
