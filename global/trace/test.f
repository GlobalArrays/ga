       real a(10000)
C$Id: test.f,v 1.2 1995-02-02 23:14:39 d3g681 Exp $
       integer i
       call trace_init(1000)
       do k = 1,10
         call trace_stime()
         do i = 1,10000
            a(i) = sin(real(i+k))
         enddo
         call trace_etime()
         call trace_genrec(k,k,i,k,i,999)
       enddo
       call trace_end(99)
       end
