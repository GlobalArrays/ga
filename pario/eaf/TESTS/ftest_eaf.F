	implicit none

cjqm#include "chemio.h"
#include "eaf.fh"

/*  This define just got added because the size declarations were removed from
    eaf.fh, but the test codes still use them.
*/
#define EAF_SZ_INT 8

#define IO_TEST_SZ (1024*1024/128)
#define IO_NUM_FILES 4
#define MAX_ITER 4
#define MAP(x) (IO_TEST_SZ*((x-1)*EAF_SZ_INT))

	integer buf(IO_TEST_SZ), buf2(IO_TEST_SZ)
	character fname*200
	integer fnum, sz, iter
	integer i
	integer fd(IO_NUM_FILES)
	integer stat
	integer ret_sz
	integer buf_sz
	integer offset



	buf_sz = (IO_TEST_SZ*EAF_SZ_INT)
cjqm	stat = EAF_InitF()

cjqm	if(stat .ne. EAF_STAT_OK) then 
cjqm	  write(0,*) "Bad init status = ", stat
cjqm	  stop 1
cjqm	endif

	do fnum = 1, IO_NUM_FILES
#if 1
	  write (fname, 10) '/tmp/output', fnum
#else
	  write (fname, 10) '/pfs-temp/mogill/output',fnum
#endif

10        format (A,'.', I1)
	  fd(fnum) = EAF_OpenPersist(fname, ELIO_RW)

	  do iter = 1, MAX_ITER
	    do i = 1, IO_TEST_SZ
	      buf(i) = i + fnum + (iter*10000)
	    enddo

	    ret_sz = EAF_Write(fd(fnum), 
     $                          MAP(iter),
     $                          buf(1), buf_sz)

	    if(ret_sz .ne. buf_sz) then 
	      write(0,*) "Bad Write status=", stat
	      stop 1
	    endif
	  enddo
	enddo

	do fnum = 1, IO_NUM_FILES
	  stat = EAF_Close(fd(fnum))
	  if(stat .ne. EAF_STAT_OK) then 
	    write(0,*) "Bad close1 status=", stat
	    stop 1
	  endif
	enddo


c.................................


	do fnum = 1, IO_NUM_FILES
#if 1
	  write (fname, 10) '/tmp/output', fnum
#else
          write (fname, 10) '/pfs-temp/mogill/output',fnum
#endif

	  fd(fnum) = EAF_OpenScratch(fname, ELIO_R)
	  
	  do iter = 1, MAX_ITER
	    do i = 1, IO_TEST_SZ
	      buf(i) = i + fnum + (iter*10000)
	    enddo

	    ret_sz = EAF_Read(fd(fnum), 
     $                         MAP(iter), 
     $                         buf2(1), buf_sz)
	    if(ret_sz .ne. buf_sz) then 
	      write(0,*) "Bad read size=", ret_sz
	      write(0,*) "   fd(",fnum,")=",fd(fnum),
     $                   "   MAP(",iter,")=",
     $                   MAP(iter)
	      stop 1
	    endif
	    do i = 1, IO_TEST_SZ
	      if(buf(i) .ne. buf2(i)) then 
	        write(6,*) "Bad read of file=",fnum,
     $                     "  iter=",iter,"  offset=",i,"   buf1=",
     $                     buf(i),"  buf2=",buf2(i)
c		stop 1
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

	end
