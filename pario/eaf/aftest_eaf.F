c define filename for the tests
c
#define FNAME '/tmp/EAFfile'

	implicit none
cjqm#include "chemio.h"
#include "eaf.fh"


/*  This define just got added because the size declarations were removed from
    eaf.fh, but the test codes still use them.
*/    
#define EAF_SZ_INT 4

#define IO_TEST_SZ (1024*1024/128)
#define IO_NUM_FILES 4
#define MAX_ITER 4
#define MAP(x) (IO_TEST_SZ*((x-1)*EAF_SZ_INT))

	integer buf(IO_TEST_SZ, MAX_ITER), buf2(IO_TEST_SZ)
	character fname*200
	integer fnum,  iter
	integer i, sleep
	real    dummy
	integer fd(IO_NUM_FILES)
	integer stat, stat2
	integer buf_sz
	integer aio(IO_NUM_FILES, MAX_ITER)

	buf_sz = (IO_TEST_SZ*EAF_SZ_INT)


10        format (A,'.', I1)

	do fnum = 1, IO_NUM_FILES
	  write (fname, 10) FNAME, fnum

	  fd(fnum) = EAF_OpenPersist(fname, ELIO_RW)

          print *,'testing EAF_AWrite'
	  do iter = 1, MAX_ITER
	    do i = 1, IO_TEST_SZ
	      buf(i,iter) = i + fnum + (iter*10000)
	    enddo

	    stat = EAF_AWrite(fd(fnum), 
     $                         MAP(iter),
     $                         buf(1,iter), buf_sz, aio(fnum, iter))

	    if(stat .ne. 0) then 
	      write(0,*) "Bad Write status=", stat
	      stop 1
	    endif
	    write(0,*) "Submitted EAF_AWriteF  fnum=", fnum,
     $         "   iter=",iter,"  stat=",stat,
     $         "   aio=",aio(fnum, iter)
	  enddo
	enddo

	do fnum = 1, IO_NUM_FILES
	  do iter = 1, MAX_ITER
	    write(0,*) "Waiting to close  fnum=", fnum,
     $         "   iter=",iter,"  stat=",stat	
	    stat = EAF_Wait(aio(fnum, iter))
	  enddo
	  stat = EAF_Close(fd(fnum))
	  if(stat .ne. EAF_STAT_OK) then 
	    write(0,*) "Bad close1 status=", stat
	    stop 1
	  endif
	enddo


c.................................


        print *,'testing EAF_ARead'
	do fnum = 1, IO_NUM_FILES

	  write (fname, 10) FNAME, fnum

	  fd(fnum) = EAF_OpenPersist(fname, ELIO_R)
	  
	  do iter = 1, MAX_ITER
	    do i = 1, IO_TEST_SZ
	      buf(i,iter) = i + fnum + (iter*10000)
	    enddo

	    stat = EAF_ARead(fd(fnum), 
     $                         MAP(iter), 
     $                         buf2, buf_sz, 
     $                         aio(fnum,iter))
	    if(stat .ne. 0) then 
	      write(0,*) "Bad read status=", stat,
     $		"fnum=", fnum, "  iter=", iter
	      stop 1
	    endif

999         format(i5,$)
1000        stat = EAF_Probe(aio(fnum, iter), stat2)
	    write(0,999) aio(fnum, iter)
	    do sleep=1, 2000000
		dummy = 2.0**2.0+sleep
	    enddo
	    if(aio(fnum, iter) .ne. ELIO_DONE) goto 1000

	    write(0,*) "Finished reading file=",fnum, 
     $                     "  fd=",fd(fnum),
     $                     "  iter=",iter

	    do i = 1, IO_TEST_SZ
	      if(buf(i,iter) .ne. buf2(i)) then 
	        write(6,*) "Bad read of file=",fnum, 
     $                     "  fd=",fd(fnum),
     $                     "  iter=",iter,"  offset=",i,"   buf1=",
     $                     buf(i,iter),"  buf2=",buf2(i)
		stop 1
	      endif
            enddo
	  enddo
	enddo

	do fnum = 1, IO_NUM_FILES
	  stat = EAF_Close(fd(fnum))
	  if(stat .ne. EAF_STAT_OK) then 
	    write(0,*) "Bad close1 status=", stat
	    stop 1
	  endif
	enddo

cjqm	stat = EAF_TerminateF()
	if(stat .ne. EAF_STAT_OK) then 
	  write(0,*) "Bad terminate status = ", stat
	  stop 1
	endif

        print *,'All seems OK'
	end
