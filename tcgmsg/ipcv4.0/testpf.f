c     $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/testpf.f,v 1.2 1995-02-02 23:26:04 d3g681 Exp $
      character*60 fname
C$Id: testpf.f,v 1.2 1995-02-02 23:26:04 d3g681 Exp $
      call pbeginf
      fname = ' '
      write(fname,'(a,i3.3)') '/tmp/pfcopy.test',nodeid()
      call pfcopy(5, 0, fname)
      call pend
      end
