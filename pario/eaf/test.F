#define FNAME '/scratch/EAFfile'
#define EAF_SZ_INT 4
#define IO_TEST_SZ (1024*1024/128)
#define IO_NUM_FILES 4
#define MAX_ITER 4
#define MAP(x) dfloat(IO_TEST_SZ*((x-1)*EAF_SZ_INT))

      program eaftest
c     
      call test1
      call test2
      call test3
c     
      end
      subroutine test1
      implicit none
#include "eaf.fh"
c     
      integer buf(IO_TEST_SZ, MAX_ITER), buf2(IO_TEST_SZ)
      character fname*200
      integer fnum,  iter
      integer i, sleep
      real    dummy
      integer fd(IO_NUM_FILES)
      integer stat, stat2
      integer buf_sz
      integer aio(IO_NUM_FILES, MAX_ITER)
c     
      buf_sz = (IO_TEST_SZ*EAF_SZ_INT)
c     
 10   format (A,'.', I1)
c     
      do fnum = 1, IO_NUM_FILES
         write (fname, 10) FNAME, fnum
c     
         call testerr(eaf_open(fname, eaf_rw, fd(fnum)))
c     
         print *,'testing EAF_AWrite'
         do iter = 1, MAX_ITER
            do i = 1, IO_TEST_SZ
               buf(i,iter) = i + fnum + (iter*10000)
            enddo

            call testerr(eaf_awrite(fd(fnum),
     $           MAP(iter),
     $           buf(1,iter), buf_sz, aio(fnum, iter)))

            write(0,*) 'Submitted EAF_AWriteF  fnum=', fnum,
     $           '   iter=',iter,'  stat=',stat,
     $           '   aio=',aio(fnum, iter)
         enddo
      enddo

      do fnum = 1, IO_NUM_FILES
         do iter = 1, MAX_ITER
            write(0,*) 'Waiting to close  fnum=', fnum,
     $           '   iter=',iter,'  stat=',stat       
            call testerr(eaf_wait(fd(fnum), aio(fnum, iter)))
         enddo
         call eaf_print_stats(fd(fnum))
         call testerr(eaf_close(fd(fnum)))
      enddo

c.................................

      print *,'testing EAF_ARead'
      do fnum = 1, IO_NUM_FILES

         write (fname, 10) FNAME, fnum

         call testerr(eaf_open(fname, eaf_r, fd(fnum)))
         
         do iter = 1, MAX_ITER
            do i = 1, IO_TEST_SZ
               buf(i,iter) = i + fnum + (iter*10000)
            enddo

            call testerr(eaf_aread(fd(fnum), 
     $           MAP(iter), 
     $           buf2, buf_sz, 
     $           aio(fnum,iter)))

 1000       call testerr(eaf_probe(aio(fnum, iter), stat2))
            write(0,999) stat2
 999        format(i5,$)
            if(stat2 .ne. 0) then
               do sleep=1, 2000000
                  dummy = 2.0**2.0+sleep
               enddo
               goto 1000
            endif

            write(0,*) 'Finished reading file=',fnum, 
     $           '  fd=',fd(fnum),
     $           '  iter=',iter

            do i = 1, IO_TEST_SZ
               if(buf(i,iter) .ne. buf2(i)) then 
                  write(6,*) 'Bad read of file=',fnum, 
     $                 '  fd=',fd(fnum),
     $                 '  iter=',iter,'  offset=',i,'   buf1=',
     $                 buf(i,iter),'  buf2=',buf2(i)
                  stop 1
               endif
            enddo
         enddo
      enddo
c     
      do fnum = 1, IO_NUM_FILES
         call eaf_print_stats(fd(fnum))
         call testerr(eaf_close(fd(fnum)))
      enddo
c     
      print *,'All seems OK'
      end
      subroutine test2
      implicit none
#include "eaf.fh"
      integer buf(IO_TEST_SZ), buf2(IO_TEST_SZ)
      character fname*200
      integer fnum,  iter
      integer i
      integer fd(IO_NUM_FILES)
      integer buf_sz
c     
      buf_sz = (IO_TEST_SZ*EAF_SZ_INT)
c     
      do fnum = 1, IO_NUM_FILES
         write (fname, 10) FNAME, fnum

 10      format (A,'.', I1)
         call testerr(eaf_open(fname, eaf_rw, fd(fnum)))

         print *, 'testing EAF_Write'
         do iter = 1, MAX_ITER
            do i = 1, IO_TEST_SZ
               buf(i) = i + fnum + (iter*10000)
            enddo
            call testerr(eaf_write(fd(fnum), 
     $           MAP(iter),
     $           buf(1), buf_sz))
         enddo
      enddo
c     
      do fnum = 1, IO_NUM_FILES
         call eaf_print_stats(fd(fnum))
         call testerr(eaf_close(fd(fnum)))
      enddo
c     
c.................................
c     
      do fnum = 1, IO_NUM_FILES
         write (fname, 10) FNAME, fnum
c     
         call testerr(eaf_open(fname, eaf_r, fd(fnum)))
c     
         print *, 'testing EAF_Read'
         do iter = 1, MAX_ITER
            do i = 1, IO_TEST_SZ
               buf(i) = i + fnum + (iter*10000)
            enddo
c     
            call testerr(eaf_read(fd(fnum), 
     $           MAP(iter), 
     $           buf2(1), buf_sz))
            do i = 1, IO_TEST_SZ
               if(buf(i) .ne. buf2(i)) then 
                  write(6,*) 'Bad read of file=',fnum,
     $                 '  iter=',iter,'  offset=',i,'   buf1=',
     $                 buf(i),'  buf2=',buf2(i)
                  stop 1
               endif
            enddo
         enddo
      enddo
c     
      do fnum = 1, IO_NUM_FILES
         call eaf_print_stats(fd(fnum))
         call testerr(eaf_close(fd(fnum)))
      enddo
c     
      print *, 'All seems OK'

      end
      subroutine testerr(code)
      implicit none
#include "eaf.fh"
      integer code
      character*80 msg
      integer fstrlen
      external fstrlen
c     
      call eaf_errmsg(code, msg)
      write(6,*) ' code ', code, ' = ', msg(1:fstrlen(msg))
***   stop 1
c
      end
      integer function fstrlen(a)
      implicit none
      character*(*) a
      integer i
      intrinsic len
c     
      do i = len(a),1,-1
         if (a(i:i) .ne. ' ') goto 10
      enddo
c     
 10   fstrlen = i
c     
      end
      subroutine test3
      implicit none
#include "eaf.fh"
      character fname*200
      character*8 fstype
      integer avail, fd
      double precision length
c
      write (fname, 10) FNAME, 1
 10   format (A,'.', I1)
c
      call testerr(eaf_stat(fname, avail, fstype))
      write(6,*) ' avail = ', avail, ' fstype = ', fstype
c
      call testerr(eaf_open(fname, eaf_rw, fd))
      call testerr(eaf_length(fd,length))
      write(6,*) ' file length = ', length
      call testerr(eaf_truncate(fd,129.0d0))
      call testerr(eaf_length(fd,length))
      write(6,*) ' file length (should be 129) = ', length
      call testerr(eaf_close(fd))
c
      write(6,*) ' This delete should succeed'
      call testerr(eaf_delete(fname))
c
      call testerr(eaf_open(fname, eaf_rw, fd))
      call testerr(eaf_length(fd,length))
      write(6,*) ' should be zero file length = ', length
      call testerr(eaf_close(fd))
c
      end
