*
* module: global.tcgmsg.F
* authors: Jarek Nieplocha and Robert Harrison
* last modification: Tue Jun 14 11:01:18 PDT 1994
*
* DISCLAIMER
* 
* This material was prepared as an account of work sponsored by an
* agency of the United States Government.  Neither the United States
* Government nor the United States Department of Energy, nor Battelle,
* nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
* ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
* COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
* SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
* INFRINGE PRIVATELY OWNED RIGHTS.
* 
* 
* ACKNOWLEDGMENT
* 
* This software and its documentation were produced with United States
* Government support under Contract Number DE-AC06-76RLO-1830 awarded by
* the United States Department of Energy.  The United States Government
* retains a paid-up non-exclusive, irrevocable worldwide license to
* reproduce, prepare derivative works, perform publicly and display
* publicly by or for the US Government, including the right to
* distribute to other US Government contractors.
*

      block data ga_block_data
      implicit none
#include "globalp.fh"
      data actv/max_global_array*.false./
      data tag_from/max_nproc*0/
      data tag_to/max_nproc*0/
      data in_handler /0/
#if   defined(PARAGON) 
      data req_sent, req_recv /2*0d0/
#endif
#if   defined(DATA_SERVER) 
      data access_handle /max_global_array* -1/
#endif
c
      end


      subroutine ga_snd_req(i1, i2, i3, i4, i5, i6, i7, i8, iproc,bytes)
      implicit none
#include "globalp.fh"
#include "tcgmsg.fh"
c
      integer i1, i2, i3, i4, i5, i6, i7, i8, iproc
      integer msg(9), ack, from, len, to, bytes
      EQUIVALENCE (msg, header_local)
c     
      to     = iproc
#if defined(disabledDATA_SERVER)
      to = to + nserver
#endif
      len = mdtob(header_size)+bytes
      tag_to(to) = tag_to(to) + 1
      req_sent = req_sent + 1 
      msg(1) = i1
      msg(2) = i2
      msg(3) = i3
      msg(4) = i4
      msg(5) = i5
      msg(6) = i6
      msg(7) = i7
      msg(8) = i8
      msg(9) = tag_to(to)
c
       
*     print *,me, GA_TYPE_REQ,' request sending to ',to, 'type=',i8
*     call ffflush(6)
#ifdef NX
c
c     on Intel machines we have to use csend because TCGMSG transforms
c     user type and we have interrupt receive posted for GA_TYPE_REQ message 
c
      call csend(GA_TYPE_REQ,header_local,len, to,0)
#else
      call ga_snd_msg(GA_TYPE_REQ, header_local, len, to,1)
#endif
c
#if   defined(DATA_SERVER) 
         call ga_rcv_msg(GA_TYPE_ACK, ack, mitob(1), len, to, from,1)
#endif
c
*     print *,me,' request done',i8
*     call ffflush(6)
      end


      subroutine ga_snd_msg(type, buf, buflen, to,sync)
      implicit none
      integer type, buf(*), buflen, to, sync
c
c     Wrapper around TCGMSG message passing routines
c
#if defined(TCGMSG)
      call snd(type, buf, buflen, to, sync)
#else
      call ga_error('ga_snd_msg: incorrect compilation flags', 0)
#endif
c
      end


      subroutine ga_rcv_msg(type, buf, buflen, msglen, from, 
     $                      whofrom, sync)
      implicit none
      integer type, buf(*), buflen, msglen, from, whofrom, sync
c
#if defined(TCGMSG)
      call rcv(type, buf, buflen, msglen, from, whofrom, sync)
#else
      call ga_error('ga_rcv_msg: incorrect compilation flags', 0)
#endif
      end



#ifdef DATA_SERVER
      subroutine ga_server()
#else
      subroutine ga_server(from)
#endif
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      integer msg(9)
      integer len
      integer from
      integer g_a
      integer ack
      integer value, Itmp_recv(1)
      integer  ga_local_read_inc, ga_map_myplace, ga_map_id 
      external ga_local_read_inc, ga_map_myplace, ga_map_id 
      EQUIVALENCE (msg, header_recv)
      EQUIVALENCE (Itmp_recv, tmp_recv)
      
c
c
#ifdef DATA_SERVER
 10   len = mdtob(tmpsize+header_size)
      call ga_rcv_msg(GA_TYPE_REQ, msg, len, len, -1, from,1)
      call ga_snd_msg(GA_TYPE_ACK, ack, mitob(1), from,1)
#else
      in_handler = 1
#endif
c
*     print *,me,' server received request ', msg(8)
*     call ffflush(6)
c
      tag_from(from) = msg(9)
c
      if (msg(8) .eq. GA_OP_GET) then
c
         len = (msg(3)-msg(2)+1)*(msg(5)-msg(4)+1)
         if (len .gt. tmpsize)
     $        call ga_error('ga_server: tmp too small', len)
         call ga_get_local(msg(1), msg(2), msg(3), msg(4),
     $        msg(5), tmp_recv, 1, msg(3)-msg(2)+1)
         if (msg(7) .eq. MT_DBL) then
            call ga_snd_msg(GA_TYPE_GET, tmp_recv, mdtob(len), from,1)
         else
            call ga_snd_msg(GA_TYPE_GET, tmp_recv, mitob(len), from,1)
         endif
c
      else if (msg(8) .eq. GA_OP_PUT) then
c
         len = (msg(3)-msg(2)+1)*(msg(5)-msg(4)+1)
         if (len .gt. tmpsize)
     $        call ga_error('ga_server: tmp too small', len)
         call ga_put_local(msg(1), msg(2), msg(3), msg(4),
     $        msg(5), tmp_recv, 1, msg(3)-msg(2)+1)
c
      else if (msg(8) .eq. GA_OP_ACC) then
c
         len = (msg(3)-msg(2)+1)*(msg(5)-msg(4)+1) + 1
         if (len .gt. tmpsize)
     $        call ga_error('ga_server: tmp too small', len)
         call ga_acc_local(msg(1), msg(2), msg(3), msg(4),
     $        msg(5), tmp_recv, msg(3)-msg(2)+1, tmp_recv(len))
c
      else if (msg(8) .eq. GA_OP_RDI) then
c
         value = ga_local_read_inc(msg(1), msg(2), msg(3), msg(4))
         call ga_snd_msg(GA_TYPE_RDI, value, mitob(1), from,1)
c
      else if (msg(8) .eq. GA_OP_DSC) then
c
         call ga_dscal(msg(1), tmp_recv(1))
c
      else if (msg(8) .eq. GA_OP_DST) then
c
         if (msg(2).gt.tmpsize/3 )
     $        call ga_error('ga_server: dscatter overflows buf',msg(2))
         call ga_handler_dscatter(msg(1), tmp_recv, tmp_recv(msg(2)+1),
     $        msg(2))
c
      else if (msg(8) .eq. GA_OP_DGT) then
c
         if (msg(2).gt.tmpsize/3 )
     $        call ga_error('ga_server: dgather overflows buf',msg(2))
         call ga_handler_dgather(msg(1), tmp_recv,  msg(2))
         call ga_snd_msg(GA_TYPE_DGT,tmp_recv, mdtob(msg(2)), from,1)
c
      else if (msg(8) .eq. GA_OP_CRE) then
c
         if (.not. ga_create_irreg(msg(7), msg(2), msg(3), 'dummy',
     $        Itmp_recv, msg(4), Itmp_recv(msg(4)+1), msg(5), g_a))
     $        call ga_error('ga_server: create failed', 0)
         if (g_a .ne. msg(1))
     $        call ga_error('ga_server: g_a does not match', g_a)
c
      else if (msg(8) .eq. GA_OP_DUP) then
c
         call ga_duplicate(msg(1),msg(2),'duplicate')
c
      else if (msg(8) .eq. GA_OP_DES) then
c
         if (.not. ga_destroy(msg(1)))
     $        call ga_error('ga_server: destroy failed', g_a)
c
      else if (msg(8) .eq. GA_OP_END) then
c
         call pend()
#ifdef AIX
         call exit(%val(0))
#else
         call exit(0)
#endif
c
      else
c
         call ga_error('ga_server: unknown request', 0)
      endif
c
*     print *,me,' server request serviced ', msg(8)
*     call ffflush(6)
#ifdef DATA_SERVER
      goto 10
#else
      in_handler = 0
      req_recv = req_recv + 1 
#endif
c
      end


      subroutine ga_sync()
      implicit none
#include "globalp.fh"
#include "tcgmsg.fh"
#ifdef PARAGON
      double precision req_outstanding 
#endif
c
c     integer ibuf, iproc, len, from
c
c     Synchronize the client/compute processes
c
#if defined(DATA_SERVER)
      integer  ga_nnodes, ga_nodeid, ga_same_id 
      external ga_nnodes, ga_nodeid, ga_same_id 
      if (me .ge. nserver) then
c
c     Just in case some code sharing between data/compute servers
c     causes a data server to call ga_sync
c
         return
      else
*        print *, me, 'calling group_sync'
*        call ffflush(6)
         call group_sync(GA_TYPE_SYN, ga_nnodes(), ga_nodeid,  
     $                   ga_same_id)
*        print *, me, 'after group_sync'
*        call ffflush(6)
      endif
c
#else
#ifdef PARAGON
      call gsync()
10    req_outstanding = req_sent-req_recv
       call ga_dgop(GA_TYPE_GOP,req_outstanding,1,'+')
      if(req_outstanding .ne. 0d0) goto 10
#else
#if defined(TCGMSG)
      call synch(GA_TYPE_SYN)     !tcgmsg
#else
      call ga_error('ga_sync: invalid compilation flags', 0)
#endif
#endif
#endif
c      
      end


      subroutine ga_terminate()
      implicit none
#include "globalp.fh"
#include "tcgmsg.fh"
c     
c     This routine called ONLY by compute processes
c
      call ga_sync()
c
#ifdef DATA_SERVER
      call ga_snd_req(0, 0, 0, 0, 0, 0, 0, GA_OP_END, me+nserver,0)
#endif
c
      end



      integer function ga_mask(newmask)
      integer newmask
#ifdef  SP1 
      integer oldmask
#endif
c
c     Set the mask for the receipt of remote accesses to local data
c     returning the old value of the mask
c
#ifdef NX
      integer masktrap, mynode
      external masktrap, mynode
#endif
c
c
      if(in_handler.ne.1)then
#ifdef   NX
           ga_mask = masktrap(newmask)
#elif    defined(SP1)
           call lockrnc(newmask,oldmask)
           ga_mask = oldmask
#else
           ga_mask = 0
#endif
      else
         ga_mask = 0
      endif
c
      end



      subroutine ga_initialize()
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c     
c     In the data server test implementation the highest numbered half
c     of the processes are assumed to be data servers in 1-1
c     correspondence with the application processes.
c     
c     integer i
      nproc = nnodes()
      me    = nodeid()
c
c     Pass adresses of int and DP base arrays to C
c
      call f2c_ma_address(dbl_mb, int_mb)
c 
#ifdef DATA_SERVER
      nserver = nproc/2
#else
      nserver = nproc
#endif
      if (nserver .gt. max_nproc)
     $     call ga_error('ga_initialize: too many processes', 0)
      oserver = me .ge. (nproc - nserver)
c
#if defined(DATA_SERVER)
      if (oserver) call ga_server()
#else
#if defined(NX) || defined(SP1)
      call ga_init_handler(header_recv, mdtob(tmpsize+header_size))
#else
      call ga_error('ga_initialize: incorrect compilation flags', 0)
#endif
#endif
c
#if defined(DATA_SERVER)
      if (me .eq. 0) then
         write(6,1) nproc-nserver, nproc-1
 1       format(/'Global array server processes are ',i3,' to ',i3/)
      endif
#endif
c
      call ga_sync()
c
      end


      subroutine  ga_reinit_handler()
      implicit none
#include "globalp.fh"
#include "tcgmsg.fh"
c
c     purpose of this routine is to pass address of receive buffer
c     to the C function that posts interrupt receive
c
#ifndef DATA_SERVER
      call ga_init_handler(header_recv, mdtob(tmpsize+header_size))
#endif
      end


      logical function ga_create(type, dim1, dim2, array_name,
     $     achunk1, achunk2, g_a)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      character*(*) array_name
      integer type, dim1, dim2, achunk1, achunk2, g_a
c
      integer  nchunk,  i, j
      logical ga_create_irreg
      integer lenstr, int, chunk1, chunk2
      integer map1(max_nproc), map2(max_nproc), nblock1, nblock2
      double precision dsqrt, dfloat
      intrinsic dsqrt, dfloat, int
      external lenstr, ga_create_irreg
c
c     Create a 2-d array with regular data distribution
c	
c      array_name    - a unique character string [input]
c
c      type          - MA type [input]
c
c      dim1/2        - array(dim1,dim2) as in FORTRAN [input]
c
c      chunk1/2      - minimum size that dimensions should
c                      be chunked up into [input]
c                      setting chunk1=dim1 gives distribution by 
c                              vertical strips (chunk2 columns)
c                      setting chunk2=dim2 gives distribution by
c                              horizontal strips (chunk1 rows)
c                      Actual chunks will be modified so that
c                      they are at least the size of the minimum and
c                      each process has either zero or one chunk.
c
c                      Specifying chunk1/2 as <= 1 will cause that
c                      dimension to be distributed evenly.
c
c      g_a           - integer handle for future references [output]
c
c
      ga_create = .false.
      chunk1 = min(achunk1, dim1)
      chunk2 = min(achunk2, dim2)
c
      if (.not. (type.eq.MT_DBL .or. type.eq.MT_INT)) then
         write(6,*) ' ga_create: type not yet supported ', array_name
         return
      else if (dim1.le.0 .or. dim2.le.0) then
         write(6,*) ' ga_create: array dimension invalid ',array_name
         write(6,*) ' dims: ', dim1, dim2
         return
      else if (chunk1.gt.dim1 .or. chunk2.gt.dim2) then
         write(6,*) ' ga_create: array chunk sizes invalid ',array_name,
     $        chunk1, chunk2
         return
      endif
c
c     Try to figure out an OK chunking ... want:
c        a) at most one chunk per processor
c        b) chunk dimensions should be >= user provided limits
c        c) chunks should be as square as possible (assumes starts square)
c        
      if (chunk1.le.1 .and. chunk2 .le. 1) then
         chunk1 = max(1, dim1/int(dsqrt(dfloat(nserver))))
         chunk2 = max(1, (dim1*dim2)/(nserver*chunk1))
      else if (chunk1.le.1) then
         chunk1 = max(1, (dim1*dim2)/(nserver*chunk2))
         chunk2 = chunk2
      else if (chunk2.le.1) then
         chunk1 = chunk1
         chunk2 = max(1, (dim1*dim2)/(nserver*chunk1))
      else
         chunk1 = max(1,chunk1)
         chunk2 = max(1,chunk2)
      endif
c
 10   continue
      nchunk = ((dim1-1)/chunk1 + 1)*((dim2-1)/chunk2 + 1)
      if (nchunk .gt. nserver) then
         if (chunk1.eq.dim1 .and. chunk2.eq.dim2)
     $        call ga_error('ga_create: chunking failed !!', 0)
         if (chunk1 .lt. dim1) then
            chunk1 = chunk1 + 1
         endif
         if (chunk2 .lt. dim2) then
            chunk2 = chunk2 + 1
         endif
         goto 10
      endif
c
c     Now build map arrays
c
      nblock1 = 0
      do i = 1, dim1, chunk1
         nblock1 = nblock1 + 1
         map1(nblock1) = i
      enddo
      nblock2 = 0
      do j = 1, dim2, chunk2
         nblock2 = nblock2 + 1
         map2(nblock2) = j
      enddo
c
      ga_create = ga_create_irreg(type, dim1, dim2, array_name,
     $     map1, nblock1, map2, nblock2, g_a)
c
      end


      subroutine ga_locate_region(g_a, ilo, ihi, jlo, jhi, map, np)
      implicit none
#include "globalp.fh"
      integer g_a, ilo, ihi, jlo, jhi, np, map(5,*)
c
c     Return the no. (np) of processors that the region maps onto
c     and in map return
c
c     map(1,p)  =  ilo on pth processor
c     map(2,p)  =  ihi on pth processor
c     map(3,p)  =  jlo
c     map(4,p)  =  jhi
c     map(5,p)  =  actual processor label
c
      integer ivp, ivp_lo, ivp_hi, jvp, jvp_lo, jvp_hi, ind, lo, hi
      integer iproc
c
c     Figure out range of virtual processor array indices
c
c     Use linear interpolation to approximate location, refine with
c     local linear search and also exploit ihi >= ilo, and optimize
c     for most references being small (i.e., in same virtual processor)
c
      ivp = (ilo-1)*pscl(1,g_a)
 10   lo = mapc(1,ivp,g_a)
      hi = mapc(2,ivp,g_a)
      if (lo.gt.ilo) then
        ivp = ivp - 1
        goto 10
      else if (hi.lt.ilo) then
         ivp = ivp + 1
         goto 10
      endif
      ivp_lo = ivp
 20   if (hi.lt.ihi) then
         ivp = ivp + 1
         lo = mapc(1,ivp,g_a)
         hi = mapc(2,ivp,g_a)
         goto 20
      endif
      ivp_hi = ivp
c
      jvp = (jlo-1)*pscl(2,g_a)
 30   ind = jvp*pdim(1,g_a)
      lo = mapc(3,ind,g_a)
      hi = mapc(4,ind,g_a)
      if (lo.gt.jlo) then
        jvp = jvp - 1
        goto 30
      else if (hi.lt.jlo) then
         jvp = jvp + 1
         goto 30
      endif
      jvp_lo = jvp
 40   if (hi.lt.jhi) then
         jvp = jvp + 1
         ind = jvp*pdim(1,g_a)
         lo = mapc(3,ind,g_a)
         hi = mapc(4,ind,g_a)
         goto 40
      endif
      jvp_hi = jvp
c
c     Figure out range for each processor
c
      np = 0
      do jvp = jvp_lo, jvp_hi
         do ivp = ivp_lo, ivp_hi
            np = np + 1
            iproc = ivp + jvp*pdim(1,g_a)
            map(1,np) = max(ilo, mapc(1, iproc, g_a))
            map(2,np) = min(ihi, mapc(2, iproc, g_a))
            map(3,np) = max(jlo, mapc(3, iproc, g_a))
            map(4,np) = min(jhi, mapc(4, iproc, g_a))
            map(5,np) = iproc
         enddo
      enddo
c
      end


      logical function ga_create_irreg(type, dim1, dim2, array_name,
     $     map1, nblock1, map2, nblock2, g_a)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      character*(*) array_name
      integer type, dim1, dim2, nblock1, nblock2, g_a
      integer map1(nblock1), map2(nblock2)
c
      integer iproc, ldim1, ldim2, lname, lga, lp, i, j
      character*256 form
      logical overbose, status, oterse
      integer lenstr, int
      double precision dsqrt, dfloat
      intrinsic dsqrt, dfloat, int
      external lenstr
      data overbose /.false./   ! Enables detailed printing
      data oterse   /.true./    ! Enables brief printing
c
c      array_name    - a unique character string [input]
c
c      type          - MA type [input]
c
c      dim1/2        - array(dim1,dim2) as in FORTRAN [input]
c
c      nblock1       - no. of blocks first dimension is divided into [input]
c
c      nblock2       - no. of blocks second dimension is divided into [input]
c
c      map1(b)       - first i in the bth block of first dimension [input]
c
c      map2(b)       - first j in the bth block of second dimension [input]
c
c      g_a           - integer handle for future references [output]
c
c     Synchronize all processes before messing with data structures
c
      call ga_sync()
c
      ga_create_irreg = .false.
c
      if (.not. (type.eq.MT_DBL .or. type.eq.MT_INT)) then
         write(6,*) ' ga_create: type not yet supported ', array_name
         return
      else if (dim1.le.0 .or. dim2.le.0) then
         write(6,*) ' ga_create: array dimension invalid ',array_name
         return
      else if (nblock1.le.0 .or. nblock2.le.0) then
         write(6,*) ' ga_create: nblock1/nblock2 <= 0 ?', array_name
         print *, 'nblocks',nblock1,nblock2
         return
      else if (nblock1*nblock2 .gt. nproc) then
         write(6,*) ' ga_create: too many blocks ', array_name
         return
      endif
c
c     Check map1 and map2
c
      status = map1(1) .eq. 1
      do i = 2, nblock1
         status = status .and.
     $        (map1(i).le.dim1) .and. (map1(i).ge.map1(i-1))
      enddo
      if (.not. status) then
         write(6,*) ' map1 invalid ', array_name, map2
         return
      endif
c
      status = map2(1) .eq. 1
      do i = 2, nblock2
         status = status .and.
     $        (map2(i).le.dim2) .and. (map2(i).ge.map2(i-1))
      enddo
      if (.not. status) then
         write(6,*) ' map2 invalid ', array_name, map2
         return
      endif
c
c     Get next free global_array handle
c
      do g_a = 1, max_global_array
         if (.not.actv(g_a)) goto 91
      enddo
      write(6,*) ' ga_create_irreg: too many arrays ', array_name
      return
 91   continue
c
#ifdef DATA_SERVER
c
c     In the data server implementation this info must be propagated to
c     the  data server process ... which basically just executes
c     this code all over again
c
      if (.not. oserver) then
         call int_block_move(map1,1,1, tmp_local,1,1,1, nblock1)
         call int_block_move(map2,1,1, tmp_local,nblock1+1,1,1, nblock2)
         call ga_snd_req(g_a, dim1, dim2, nblock1, nblock2, me, type,
     $        GA_OP_CRE, me+nserver, mitob(nblock1+nblock2))
      endif
#endif
c
c     Stash the info
c
      name(g_a)   = array_name
      dims(1,g_a) = dim1
      dims(2,g_a) = dim2
      ttyp(g_a)   = type
      indx(g_a)   = 0
c
      pdim(1,g_a) = nblock1
      pdim(2,g_a) = nblock2
c
c     Initialize distributions to 0:-1 => a null do loop range
c
      call ifill(nserver*2, 0, mapc(1, 0, g_a), 2)
      call ifill(nserver*2, -1, mapc(2, 0, g_a), 2)
c
c     Now figure out which process has what
c
      iproc = 0
      do j = 1, nblock2
         do i = 1, nblock1
            mapc(1,iproc,g_a) = map1(i)
            if (i .lt. nblock1) then
               mapc(2,iproc,g_a) = map1(i+1)-1
            else
               mapc(2,iproc,g_a) = dim1
            endif
            mapc(3,iproc,g_a) = map2(j)
            if (j.lt.nblock2) then
               mapc(4,iproc,g_a) = map2(j+1)-1
            else
               mapc(4,iproc,g_a) = dim2
            endif
            iproc = iproc + 1
         enddo
      enddo
c
#ifdef DATA_SERVER
      if (oserver) then
         iproc = me - nserver
      else
         iproc = me
      endif
#else
      iproc = me
#endif
c
c     Figure out exact size of chunk I (or my server) is holding
c
      ldim(1, g_a) = mapc(2,iproc,g_a) - mapc(1,iproc,g_a) + 1
      ldim(2, g_a) = mapc(4,iproc,g_a) - mapc(3,iproc,g_a) + 1
      pscl(1, g_a) = dfloat(pdim(1,g_a)) / dfloat(dims(1,g_a))
      pscl(2, g_a) = dfloat(pdim(2,g_a)) / dfloat(dims(2,g_a))
c
c     OK ... now each (server) process merely has to allocate its own
c     memory and we are in business
c     
      if (.not. oserver) then
         continue
      else if (ldim(1,g_a)*ldim(2,g_a) .eq. 0) then
         continue
      else if (.not. ma_alloc_get(type, ldim(1,g_a)*ldim(2,g_a),
     $     array_name, hand(g_a), indx(g_a))) then
         write(6,*) ' ga_create_irreg: failed to allocate memory',
     $        array_name, ldim(1,g_a), ldim(2,g_a)
         call ga_error('exiting ',0)
      endif 
c
c     Initialize the array to garbage 
c     
      if ( oserver) then
         if (type .eq. MT_DBL) then
            call dfill(ldim(1,g_a)*ldim(2,g_a), -1.0d+300, 
     $           dbl_mb(indx(g_a)), 1)
         else
            call ifill(ldim(1,g_a)*ldim(2,g_a), -2147483647, 
     $           int_mb(indx(g_a)), 1)
         endif
      endif
c
c$$$      if (me .eq. 0) then
c$$$         ldim1 = log10(dfloat(dim1)) + 1
c$$$         ldim2 = log10(dfloat(dim2)) + 1
c$$$         lga = log10(dfloat(g_a)) + 1
c$$$         lp = log10(dfloat(nproc)) + 1
c$$$         lname = lenstr(array_name)
c$$$         if (type .eq. MT_DBL) then
c$$$            write(form, 1) lname, ldim1, ldim2, lga
c$$$         else
c$$$            write(form, 2) lname, ldim1, ldim2, lga
c$$$         endif
c$$$ 1       format('('' ga_create: double precision "'',a',i4,',''"('',i',
c$$$     $        i4,','','',i',i4,','') has handle '',i',i4,')')
c$$$ 2       format('('' ga_create: integer "'',a',i4,',''"('',i',i4,
c$$$     $        ','','',i',i4,','') has handle '',i',i4,')')
c$$$         write(6,*)
c$$$         write(6,form) array_name, dim1, dim2, g_a
c$$$         write(form,3) ldim1, ldim1, ldim2, ldim2, lp
c$$$ 3       format('(''     ('',i',i4,','':'',i',i4,','', '',i',i4,
c$$$     $        ','':'',i',i4,','') -> '',i',i4,')')
c$$$         if (overbose) then
c$$$            do iproc = 0,nserver-1
c$$$               write(6,form) (mapc(i,iproc,g_a),i=1,4),iproc
c$$$            enddo
c$$$         else if (oterse) then
c$$$            write(6,4) ' 1st dimension ', (map1(i), i=1, nblock1)
c$$$            write(6,4) ' 2nd dimension ', (map2(i), i=1, nblock2)
c$$$ 4          format(a,10(12i5/14x))
c$$$         endif
c$$$         write(6,*)
c$$$      endif
c
      ga_create_irreg = .true.
      actv(g_a) = .true.
c
c     Synchronize right at the end to ensure no use of array until
c     all is complete
c
      call ga_sync()
c
      end



      subroutine ga_duplicate(g_a, g_b, array_name)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      character*(*) array_name
      integer g_a, g_b
c
      integer iproc, i
      logical overbose, oterse
      integer lenstr, int
      double precision dsqrt, dfloat
      intrinsic dsqrt, dfloat, int
      external lenstr
      data overbose /.false./   ! Enables detailed printing
      data oterse   /.true./    ! Enables brief printing
c
c      array_name    - a unique character string [input]
c      g_a           - integer handle for reference array [input]
c      g_b           - integer handle for new array [output]
c
c     Synchronize all processes before messing with data structures
c
      call ga_sync()
c
      call ga_check_handle(g_a, 'ga_duplicate')
c
c     Get next free global_array handle for g_b
c
      do g_b = 1, max_global_array
         if (.not.actv(g_b)) goto 91
      enddo
      call ga_error( ' ga_duplicate: too many arrays ', 0) 
 91   continue
c
#ifdef DATA_SERVER
c
      if (.not. oserver) then
         call ga_snd_req(g_a, g_b, 0,0,0,0,0, GA_OP_DUP, me+nserver,0)
      endif
#endif
c
      name(g_b)   = array_name
      dims(1,g_b) = dims(1,g_a)
      dims(2,g_b) = dims(2,g_a)
      ttyp(g_b)   = ttyp(g_a)
      indx(g_b)   = 0
c
      pdim(1,g_b) = pdim(1,g_a)
      pdim(2,g_b) = pdim(2,g_a)
c
#ifdef DATA_SERVER
      if (oserver) then
         iproc = me - nserver
      else
         iproc = me
      endif
#else
      iproc = me
#endif
c
c
      ldim(1, g_b) = ldim(1, g_a)
      ldim(2, g_b) = ldim(2, g_a)
      pscl(1, g_b) = pscl(1, g_a)
      pscl(2, g_b) = pscl(2, g_a)
c
      do iproc = 0, nproc-1
         do i = 1,4
            mapc(i,iproc,g_b) = mapc(i,iproc,g_a)
         enddo
      enddo
c
c     OK ... now each (server) process merely has to allocate its own
c     memory and we are in business
c     
      if (.not. oserver) then
         continue
      else if (ldim(1,g_b)*ldim(2,g_b) .eq. 0) then
         continue
      else if (.not.ma_alloc_get(ttyp(g_b), ldim(1,g_b)*ldim(2,g_b),
     $     array_name, hand(g_b), indx(g_b))) then
         write(6,*) ' ga_create_irreg: failed to allocate memory',
     $        array_name, ldim(1,g_b), ldim(2,g_b)
         call ga_error('exiting ',0)
      endif 
c
c     Initialize the array to garbage 
c     
      if ( oserver) then
         if (ttyp(g_b)  .eq. MT_DBL) then
            call dfill(ldim(1,g_b)*ldim(2,g_b), -1.0d+300, 
     $           dbl_mb(indx(g_b)), 1)
         else
            call ifill(ldim(1,g_b)*ldim(2,g_b), -2147483647, 
     $           int_mb(indx(g_b)), 1)
         endif
      endif
c
      actv(g_b) = .true.
      call ga_sync()
c
      end


      logical function ga_destroy(g_a)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      integer g_a
c
      call ga_sync()
      ga_destroy = .false.
c
      call ga_check_handle(g_a, 'ga_destroy')
c
      actv(g_a) = .false.
#ifdef DATA_SERVER
      access_handle(g_a) = -1
#endif
c
      if (.not. oserver) then
         call ga_snd_req(g_a, 0, 0, 0, 0, 0, 0, GA_OP_DES, me+nserver,0)
         ga_destroy = .true.
      else if (ldim(1,g_a)*ldim(2,g_a) .eq. 0) then
         ga_destroy = .true.
      else if (.not. ma_free_heap(hand(g_a))) then
         write(6,*) ' ga_destroy: failed to free memory?', g_a
      else
         ga_destroy = .true.
      endif
c
      end


      subroutine ga_inquire(g_a, type, dim1, dim2)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, dim1, dim2, type
c      
c      g_a        - array handle [input]
c      array_name - name given to create [output]
c      type       - type given to create [output]
c      dim1/2     - global array dimensions [output]
c
      call ga_check_handle(g_a, 'ga_inquire')
      type       = ttyp(g_a)
      dim1       = dims(1,g_a)
      dim2       = dims(2,g_a)
c
      end

      subroutine ga_inquire_name(g_a, array_name)
      implicit none
      integer g_a
      character*(*) array_name
#include "globalp.fh"
c
      call ga_check_handle(g_a, 'ga_inquire_name')
      array_name = name(g_a)
      end

      subroutine ga_get(g_a, ilo, ihi, jlo, jhi, buf, ld)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ld
      double precision buf(*)
c
      integer ip, iproc, oldmask, ga_mask
      integer ilo_get, ihi_get, jlo_get, jhi_get
      integer ilo_ugh, ihi_ugh, jlo_ugh, jhi_ugh, ilimit, jlimit
      integer map(5, max_nproc), np
c
c
#ifdef GA_TRACE
      call trace_stime
#endif
      call ga_check_handle(g_a, 'ga_get')
      if (ilo.le.0 .or. ihi.gt.dims(1,g_a) .or.
     $         jlo.le.0 .or. jhi.gt.dims(2,g_a)) then
         call ga_error(' ga_get: indices out of range ', g_a)
      endif
c
c     Map index ranges onto processors
c
      call ga_locate_region(g_a, ilo, ihi, jlo, jhi, map, np)
c
c     Loop thru pieces
c
      do ip = 1, np
         ilo_get = map(1,ip)
         ihi_get = map(2,ip)
         jlo_get = map(3,ip)
         jhi_get = map(4,ip)
         iproc   = map(5,ip)
c
         if (iproc.eq.me .and. oserver) then
            oldmask = ga_mask(1)
            call ga_get_local(g_a, ilo_get, ihi_get, jlo_get,
     $           jhi_get, buf,
     $           ilo_get-ilo+1+(jlo_get-jlo)*ld, ld)
            oldmask = ga_mask(oldmask)
c     
         else
c     
c     Process data in chunks defined by the available buffer space
c     for incoming/outgoing messages. Ugh, loops within loops within ...
c     
            ilimit = min(tmpsize, ihi_get-ilo_get+1)
            jlimit = min(tmpsize/ilimit, jhi_get-jlo_get+1)
            do jlo_ugh = jlo_get, jhi_get, jlimit
               jhi_ugh = min(jhi_get, jlo_ugh+jlimit-1)
               do ilo_ugh = ilo_get, ihi_get, ilimit
                  ihi_ugh = min(ihi_get, ilo_ugh+ilimit-1)
c
                  call ga_get_remote(g_a, ilo_ugh, ihi_ugh,
     $                 jlo_ugh, jhi_ugh, buf,
     $                 ilo_ugh-ilo+1+(jlo_ugh-jlo)*ld, ld, iproc)
               enddo
            enddo
         endif
      enddo
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, ilo, ihi, jlo, jhi, GA_OP_GET)
#endif
c
      end


      subroutine ga_get_local(g_a, ilo, ihi, jlo, jhi, buf, bbase, ld)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, bbase, ld
      double precision buf(*)
c
      integer gbase, iproc
c
c     First do usual paranoia check on arguments and data structure
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
c
      call ga_check_handle(g_a, 'ga_get_local')
      if (ilo.lt.mapc(1,iproc,g_a) .or.
     $     ihi.gt.mapc(2,iproc,g_a) .or.
     $     jlo.lt.mapc(3,iproc,g_a) .or.
     $     jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_get_local: local indices out of range',g_a)
      else
c
c     All seems OK ... inside a critical section get the data
c
         gbase = indx(g_a) + ilo - mapc(1,iproc,g_a) +
     $        (jlo-mapc(3,iproc,g_a))*ldim(1,g_a) 
         if (ttyp(g_a) .eq. MT_DBL) then
           call dbl_block_move_p(dbl_mb, gbase, ldim(1,g_a), buf, bbase,
     $           ld, ihi-ilo+1, jhi-jlo+1)
         else
           call int_block_move_p(int_mb, gbase, ldim(1,g_a), buf, bbase,
     $           ld, ihi-ilo+1, jhi-jlo+1)
         endif
      endif
c
      end



      subroutine ga_get_remote(g_a, ilo, ihi, jlo, jhi, buf, bbase,
     $     ld, iproc)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, bbase, ld, iproc
      double precision buf(*)
c
      integer len, from, ldtmp, to
c
c     First do usual paranoia check on arguments and data structure
c
      call ga_check_handle(g_a, 'ga_get_remote')
      if (iproc.lt.0 .or. iproc.ge.nserver) then
         call ga_error(' ga_get_remote: iproc out of range ', iproc)
      else if (ilo.lt.mapc(1,iproc,g_a) .or.
     $         ihi.gt.mapc(2,iproc,g_a) .or.
     $         jlo.lt.mapc(3,iproc,g_a) .or.
     $         jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_get_remote: mapc violated ', g_a)
      else if ((ihi-ilo+1)*(jhi-jlo+1).gt.tmpsize) then
         call ga_error(' ga_get_remote: internal buffer too small',
     $        (ihi-ilo+1)*(jhi-jlo+1))
      else
c
c     All seems OK ... do a remote request 
c
         to =iproc
#ifndef DATA_SERVER
c
c        Post asynchronous receive for the data
c
         call ga_rcv_msg(GA_TYPE_GET, tmp_local, mdtob(tmpsize),
     $           len, to, from, 0)
#else
         to = to + nserver
#endif
         call ga_snd_req(g_a, ilo, ihi, jlo, jhi, me, ttyp(g_a),
     $        GA_OP_GET, to,0)
c
c
c        Receive data and copy from tmp_local into final location
c     
         call waitcom(to)
         ldtmp = ihi-ilo+1
c
         if (ttyp(g_a) .eq. MT_DBL) then
#ifdef DATA_SERVER
            call ga_rcv_msg(GA_TYPE_GET, tmp_local, mdtob(tmpsize),
     $           len, to, from,1)
c
c           cannot get reliable message lengths from NX with interrupts
c
            if (len .ne. mdtob((ihi-ilo+1)*(jhi-jlo+1)))
     $          call ga_error('ga_get_remote: response wrong size', len)
#endif
            call dbl_block_move(tmp_local, 1, ldtmp, buf, bbase, ld,
     $           ihi-ilo+1, jhi-jlo+1)
         else
#ifdef DATA_SERVER
            call ga_rcv_msg(GA_TYPE_GET, tmp_local, mdtob(tmpsize), len,
     $           to, from,1)
c
c           cannot get reliable message lengths from NX with interrupts
c
            if (len .ne. mitob((ihi-ilo+1)*(jhi-jlo+1)))
     $          call ga_error('ga_get_remote: response wrong size', len)
#endif
            call int_block_move(tmp_local, 1, ldtmp, buf, bbase, ld,
     $           ihi-ilo+1, jhi-jlo+1)
         endif
c
      endif
c     
      end



      subroutine ga_put(g_a, ilo, ihi, jlo, jhi, buf, ld)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ld
      double precision buf(*)
c
      integer ip, iproc, oldmask, ga_mask
      integer ilo_put, ihi_put, jlo_put, jhi_put
      integer ilo_ugh, ihi_ugh, jlo_ugh, jhi_ugh, ilimit, jlimit
      integer map(5,max_nproc), np
c
#ifdef GA_TRACE
      call trace_stime
#endif
c
      call ga_check_handle(g_a, 'ga_put')
      if (ilo.le.0 .or. ihi.gt.dims(1,g_a) .or.
     $     jlo.le.0 .or. jhi.gt.dims(2,g_a)) then
         call ga_error(' ga_put: indices out of range ', g_a)
      endif
c
c     Map index ranges onto processes
c
      call ga_locate_region(g_a, ilo, ihi, jlo, jhi, map, np)
c
c     Loop thru pieces
c
      do ip = 1, np
         ilo_put = map(1,ip)
         ihi_put = map(2,ip)
         jlo_put = map(3,ip)
         jhi_put = map(4,ip)
         iproc   = map(5,ip)
c
         if (iproc .eq. me .and. oserver) then
            oldmask = ga_mask(1)
            call ga_put_local(g_a, ilo_put, ihi_put, jlo_put,
     $           jhi_put, buf, 
     $           ilo_put-ilo+1+(jlo_put-jlo)*ld, ld)
            oldmask = ga_mask(oldmask)
         else
c     
c           Process data in chunks defined by the available buffer space
c           for incoming/outgoing messages. Ugh, loops within loops within ...
c     
            ilimit = min(tmpsize, ihi_put-ilo_put+1)
            jlimit = min(tmpsize/ilimit, jhi_put-jlo_put+1)
            do jlo_ugh = jlo_put, jhi_put, jlimit
               jhi_ugh = min(jhi_put, jlo_ugh+jlimit-1)
               do ilo_ugh = ilo_put, ihi_put, ilimit
                  ihi_ugh = min(ihi_put, ilo_ugh+ilimit-1)
                  call ga_put_remote(g_a, ilo_ugh, ihi_ugh,
     $                 jlo_ugh, jhi_ugh, buf,
     $                 ilo_ugh-ilo+1+(jlo_ugh-jlo)*ld, ld, iproc)
               enddo
            enddo
         endif
      enddo
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, ilo, ihi, jlo, jhi, GA_OP_PUT)
#endif
c
      end


      subroutine ga_put_local(g_a, ilo, ihi, jlo, jhi, buf, bbase, ld)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, bbase, ld
      double precision buf(*)
c
      integer gbase, iproc
c
c     First do usual paranoia check on arguments and data structure
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
c
      call ga_check_handle(g_a, 'ga_put_local')
      if (ilo.lt.mapc(1,iproc,g_a) .or.
     $     ihi.gt.mapc(2,iproc,g_a) .or.
     $     jlo.lt.mapc(3,iproc,g_a) .or.
     $     jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_put_local: local indices out of range', g_a)
      else if (ldim(1,g_a)*ldim(2,g_a).ne.0 .and. indx(g_a).eq.0) then
         call ga_error(' ga_put_local: null pointer? ', g_a)
      else
c
c     All seems OK ... inside a critical section put the data
c
         gbase = indx(g_a) + ilo - mapc(1,iproc,g_a) +
     $        (jlo-mapc(3,iproc,g_a))*ldim(1,g_a) 
c
         if (ttyp(g_a) .eq. MT_DBL) then
            call dbl_block_move_p(buf, bbase, ld, dbl_mb, gbase,
     $           ldim(1,g_a), ihi-ilo+1, jhi-jlo+1)
         else
            call int_block_move_p(buf, bbase, ld, int_mb, gbase,
     $           ldim(1,g_a), ihi-ilo+1, jhi-jlo+1)
         endif
      endif
c
      end


      subroutine ga_put_remote(g_a, ilo, ihi, jlo, jhi, buf, bbase,
     $     ld, iproc)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, bbase, ld, iproc, bytes
      double precision buf(*)
c
      integer ldtmp, len, to
c
c     First do usual paranoia check on arguments and data structure
c
      call ga_check_handle(g_a, 'ga_put_remote')
      if (iproc.lt.0 .or. iproc.ge.nserver) then
         call ga_error(' ga_put_remote: iproc out of range ', iproc)
      else if (ilo.lt.mapc(1,iproc,g_a) .or.
     $         ihi.gt.mapc(2,iproc,g_a) .or.
     $         jlo.lt.mapc(3,iproc,g_a) .or.
     $         jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_put_remote: invalid index range ', g_a)
      else if ((ihi-ilo+1)*(jhi-jlo+1).gt.tmpsize) then
         call ga_error(' ga_put_remote: internal buffer too small',
     $        (ihi-ilo+1)*(jhi-jlo+1))
      else
c
c     All seems OK ... do a remote request
c
         ldtmp = ihi-ilo+1
         len = (ihi-ilo+1)*(jhi-jlo+1)
         if (ttyp(g_a) .eq. MT_DBL) then
            call dbl_block_move(buf, bbase, ld, tmp_local, 1, ldtmp,
     $           ldtmp, jhi-jlo+1)
            bytes = mdtob(len)
c           write(*,'(5e11.2)')(tmp_local(to),to=1,len)
c           call ffflush(6)
         else
            call int_block_move(buf, bbase, ld, tmp_local, 1, ldtmp,
     $           ldtmp, jhi-jlo+1)
            bytes = mitob(len)
         endif
c
         to = iproc
#ifdef DATA_SERVER
         to = to+nserver
#endif
         call ga_snd_req(g_a, ilo, ihi, jlo, jhi, me, ttyp(g_a),
     $        GA_OP_PUT, to, bytes)
      endif
c     
      end


      subroutine ga_acc(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ld
      double precision buf(1:ld, 1:*), alpha
c
      integer ip, iproc, oldmask, ga_mask
      integer ilo_acc, ihi_acc, jlo_acc, jhi_acc
      integer ilo_ugh, ihi_ugh, jlo_ugh, jhi_ugh, ilimit, jlimit
      integer map(5, max_nproc), np
c
#ifdef GA_TRACE
      call trace_stime
#endif
c
      call ga_check_handle(g_a, 'ga_acc')
      if (ttyp(g_a) .ne. MT_DBL) then
         call ga_error(' ga_acc: only supported for doubles', g_a)
      else if (ilo.le.0 .or. ihi.gt.dims(1,g_a) .or.
     $         jlo.le.0 .or. jhi.gt.dims(2,g_a)) then
         call ga_error(' ga_acc: indices out of range ', g_a)
      endif
c
c     Map index ranges onto processes
c
      call ga_locate_region(g_a, ilo, ihi, jlo, jhi, map, np)
c
c     Loop thru pieces
c
      do ip = 1, np
         ilo_acc = map(1,ip)
         ihi_acc = map(2,ip)
         jlo_acc = map(3,ip)
         jhi_acc = map(4,ip)
         iproc   = map(5,ip)
c
         if (iproc .eq. me .and. oserver) then
            oldmask = ga_mask(1)
            call ga_acc_local(g_a, ilo_acc, ihi_acc, jlo_acc,
     $           jhi_acc, buf(ilo_acc-ilo+1,jlo_acc-jlo+1), ld,
     $           alpha)
            oldmask = ga_mask(oldmask)
         else
c     
c           Process data in chunks defined by the available buffer space
c           for incoming/outgoing messages. Ugh, loops within loops within ...
c     
            ilimit = min(tmpsize-1, ihi_acc-ilo_acc+1)
            jlimit = min((tmpsize-1)/ilimit, jhi_acc-jlo_acc+1)
            do jlo_ugh = jlo_acc, jhi_acc, jlimit
               jhi_ugh = min(jhi_acc, jlo_ugh+jlimit-1)
               do ilo_ugh = ilo_acc, ihi_acc, ilimit
                  ihi_ugh = min(ihi_acc, ilo_ugh+ilimit-1)
                  call ga_acc_remote(g_a, ilo_ugh, ihi_ugh, jlo_ugh,
     $                 jhi_ugh, buf(ilo_ugh-ilo+1,jlo_ugh-jlo+1), ld,
     $                 alpha, iproc)
               enddo
            enddo
         endif
      enddo
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, ilo, ihi, jlo, jhi, GA_OP_ACC)
#endif
c
      end


      subroutine ga_acc_local(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha)
      implicit none
#include "mafdecls.h"
#include "global.fh"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ld
      double precision buf(1:ld, 1:*), alpha
c
      integer i, j, gbase, iproc
c
c     First do usual paranoia check on arguments and data structure
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
      call ga_check_handle(g_a, 'ga_acc_local')
      if (ilo.lt.mapc(1,iproc,g_a) .or.
     $         ihi.gt.mapc(2,iproc,g_a) .or.
     $         jlo.lt.mapc(3,iproc,g_a) .or.
     $         jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_acc_local: local indices out of range', g_a)
      else if (ldim(1,g_a)*ldim(2,g_a).ne.0 .and. indx(g_a).eq.0) then
         call ga_error(' ga_acc_local: null pointer? ', g_a)
      else
c
c     All seems OK ... inside a critical section acc the data
c
         gbase = indx(g_a) - mapc(1,iproc,g_a) +
     $        (jlo-mapc(3,iproc,g_a))*ldim(1,g_a) 
         do j = jlo, jhi
            do i = ilo, ihi
               dbl_mb(gbase + i) = dbl_mb(gbase + i) +
     $              alpha*buf(i-ilo+1,j-jlo+1)
            enddo
            gbase = gbase + ldim(1,g_a)
         enddo
      endif
      end


      subroutine ga_acc_remote(g_a, ilo, ihi, jlo, jhi, buf, ld, alpha,
     $     iproc)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
#include "tcgmsg.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ld, iproc
      double precision buf(1:ld, 1:*), alpha
c
      integer ldtmp, len, to
c
c     First do usual paranoia check on arguments and data structure
c
      call ga_check_handle(g_a, 'ga_acc_remote')
      if (iproc.lt.0 .or. iproc.ge.nserver) then
         call ga_error(' ga_acc_remote: iproc out of range ', iproc)
      else if (ilo.lt.mapc(1,iproc,g_a) .or.
     $         ihi.gt.mapc(2,iproc,g_a) .or.
     $         jlo.lt.mapc(3,iproc,g_a) .or.
     $         jhi.gt.mapc(4,iproc,g_a)) then
         call ga_error(' ga_acc_remote: invalid index range ', g_a)
      else if ((ihi-ilo+1)*(jhi-jlo+1)+1.gt.tmpsize) then
         call ga_error(' ga_acc_remote: internal buffer too small',
     $        (ihi-ilo+1)*(jhi-jlo+1))
      else
c
c     All seems OK ... do a remote request
c
c        Copy data to temporary buffer (dont forget alpha)
c     
         ldtmp = ihi-ilo+1
         call dbl_block_move(buf,1,ld,tmp_local,1,ldtmp,ldtmp,jhi-jlo+1)
         len = (ihi-ilo+1)*(jhi-jlo+1) + 1
         tmp_local(len) = alpha
c
         to = iproc
#ifdef DATA_SERVER
         to = to+nserver
#endif
         call ga_snd_req(g_a, ilo, ihi, jlo, jhi, me, ttyp(g_a),
     $        GA_OP_ACC, to, mdtob(len))
      endif
c     
      end


      subroutine dbl_block_move(a, abase, ald, b, bbase, bld, ni, nj)
      implicit none
      double precision a(*), b(*)
      integer abase, bbase, ald, bld, ni, nj
c
c     a(1:ni,1:nj) -> b(1:ni,1:nj) where ald, bld are the leading
c     dimensions of a, b respectively and abase, bbase are the offsets
c     in the respective arrays (as linear entities) to the first word
c     to be copied.
c
c     This routine is necessary as one cannot cast pointers in FORTRAN
c     so all address arithmetic must be deferred until U are in a routine
c     with the arrays correctly typed (of course it violates the FORTRAN
c     standard to not match the type of formal and actual arguments
c     but it does work ... at least so far)
c
      integer i, j, aoff, boff
c
      aoff = abase
      boff = bbase
      do j = 0, nj-1
         do i = 0, ni-1
            b(boff+i) = a(aoff+i)
         enddo
         aoff = aoff + ald
         boff = boff + bld
      enddo
c
      end


      subroutine int_block_move(a, abase, ald, b, bbase, bld, ni, nj)
      implicit none
      integer a(*), b(*)
      integer abase, bbase, ald, bld, ni, nj
c
c     a(1:ni,1:nj) -> b(1:ni,1:nj) where ald, bld are the leading
c     dimensions of a, b respectively and abase, bbase are the offsets
c     in the respective arrays (as linear entities) to the first word
c     to be copied.
c
c     This routine is necessary as one cannot cast pointers in FORTRAN
c     so all address arithmetic must be deferred until U are in a routine
c     with the arrays correctly typed (of course it violates the FORTRAN
c     standard to not match the type of formal and actual arguments
c     but it does work ... at least so far)
c
      integer i, j, aoff, boff
c
      aoff = abase
      boff = bbase
      do j = 0, nj-1
         do i = 0, ni-1
            b(boff+i) = a(aoff+i)
         enddo
         aoff = aoff + ald
         boff = boff + bld
      enddo
c
      end


      subroutine dbl_block_move_p(a, abase, ald, b, bbase, bld, ni, nj)
      implicit none
      double precision a(*), b(*)
      integer abase, bbase, ald, bld, ni, nj
c
c     private copy of dbl_block_move, used only in ga_xxx_local
c     to avoid reentrancy problems on the SP1
c
      integer i, j, aoff, boff
c
      aoff = abase
      boff = bbase
      do j = 0, nj-1
         do i = 0, ni-1
            b(boff+i) = a(aoff+i)
         enddo
         aoff = aoff + ald
         boff = boff + bld
      enddo
      end


      subroutine int_block_move_p(a, abase, ald, b, bbase, bld, ni, nj)
      implicit none
      integer a(*), b(*)
      integer abase, bbase, ald, bld, ni, nj
c
      integer i, j, aoff, boff
c     private copy of int_block_move, used only in ga_xxx_local
c     to avoid reentrancy problems on the SP1
c
      aoff = abase
      boff = bbase
      do j = 0, nj-1
         do i = 0, ni-1
            b(boff+i) = a(aoff+i)
         enddo
         aoff = aoff + ald
         boff = boff + bld
      enddo
c
      end


      subroutine ga_access(g_a, ilo, ihi, jlo, jhi, ma_index,ld)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi, ma_index, ld
      integer node
c
c      g_a             - array handle [input]
c      iproc           - process number [input]
c      ilo/ihi/jlo/jhi - range held by process iproc [output]
c      ma_index        - if (iproc == this process) returns
c                        MA index in appropriately typed base array
c      ld              - leading dimension
c
#ifdef DATA_SERVER
      integer elem
#endif
c
      node =  me
      call ga_check_handle(g_a, 'ga_access')
      if (ilo .ge. mapc(1, node, g_a) .and.
     &    ihi .le. mapc(2, node, g_a) .and.
     &    jlo .ge. mapc(3, node, g_a) .and.
     &    jhi .le. mapc(4, node, g_a) .and.
     &    ilo .le. ihi .and. jlo .le. jhi) then 
c
#ifdef DATA_SERVER
      if(access_handle(g_a).ne.-1)
     & call ga_error('ga_access: only one access per array allowed',g_a)
c
c           The least efficient approach used: compute nodes copy the 
c           data to a local temporary buffer and then provide access
c           to this buffer -- must be deallocated in ga_release 
c
c           First, allocate temporary memory for the patch
c
            elem = (ihi-ilo+1)*(jhi-jlo+1)
            if(ttyp(g_a).eq.MT_DBL)then
               if(.not.ma_alloc_get(MT_DBL,elem,'tm',access_handle(g_a),   
     $                 ma_index)) 
     $                 call ga_error('ga_access: mem aloc failed',g_a)
            else if(ttyp(g_a).eq.MT_INT)then 
               if(.not.ma_alloc_get(MT_INT,elem,'tm',access_handle(g_a),   
     $                 ma_index)) 
     $                 call ga_error('ga_access: mem aloc failed',g_a)
            else
               call ga_error('ga_access: type not supported',g_a)
            endif
c
c           Now, get the data from the server
c
            ld = ihi-ilo+1
            if(ttyp(g_a).eq.MT_DBL)then
               call ga_get(g_a, ilo,ihi, jlo,jhi, dbl_mb(ma_index),ld)
            else
               call ga_get(g_a, ilo,ihi, jlo,jhi, int_mb(ma_index),ld)
            endif
            access_index(g_a) = ma_index
#else
c
c           On SP-1 and Intel machines just provide pointer to the data
c
            ma_index = indx(g_a) + ilo - mapc(1,node,g_a) +
     $                (jlo-mapc(3,node,g_a))*ldim(1,g_a)
            ld = ldim(1,g_a)
#endif
      else
          print *,me,' ilo,ihi,jlo,jhi:',ilo,ihi,jlo,jhi
          print *,'mapc:',(mapc(ld, node, g_a),ld=1,4)
          call ga_error(' ga_access: indices out of range ', g_a) 
      endif
c
      end


      subroutine ga_release(g_a, ilo, ihi, jlo, jhi)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi
c
c     called if the data in the PATCH has NOT been CHANGED
c
#ifdef DATA_SERVER
      if(access_handle(g_a).eq.-1) return
      if(.not. ma_free_heap(access_handle(g_a)))
     $   call ga_error('ga_release: wrong  handle ? ',g_a)  
      access_handle(g_a) = -1
#endif
c
      end


      subroutine ga_release_update(g_a, ilo, ihi, jlo, jhi)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, ilo, ihi, jlo, jhi
c
c     called if the data in the PATCH has been CHANGED
c
#ifdef DATA_SERVER
      integer ld 
c
      if(access_handle(g_a).eq.-1) return
c
c     update the server's copy of the data
      ld = ihi-ilo+1
      if(ttyp(g_a).eq.MT_DBL)then
        call ga_put(g_a, ilo,ihi, jlo,jhi, dbl_mb(access_index(g_a)),
     $              ld)
      else if(ttyp(g_a).eq.MT_INT)then
        call ga_put(g_a, ilo,ihi, jlo,jhi, int_mb(access_index(g_a)),
     $              ld)
      else
        call ga_error('ga_release_u: type not supported',g_a)
      endif
c
      if(.not. ma_free_heap(access_handle(g_a)))
     $   call ga_error('ga_release_u: wrong  handle ? ',g_a)  
      access_handle(g_a) = -1
#endif
c
      end

 

      subroutine ga_distribution(g_a, iproc, ilo, ihi, jlo, jhi)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, iproc, ilo, ihi, jlo, jhi
      integer node
c
c      g_a             - array handle [input]
c      iproc           - process number [input]
c      ilo/ihi/jlo/jhi - range held by process iproc [output]
c
      node = iproc
#ifdef DATA_SERVER
      if (node .ge. nserver) node = node - nserver
#endif
      call ga_check_handle(g_a, 'ga_distribution')
      if (iproc.lt. 0 .or. iproc.ge.nproc) then
         call ga_error(' ga_distribution: iproc invalid ', iproc)
      else
         ilo = mapc(1, node, g_a)
         ihi = mapc(2, node, g_a)
         jlo = mapc(3, node, g_a)
         jhi = mapc(4, node, g_a)
      endif
c
      end
      integer function ga_verify_handle(g_a)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
      integer g_a
      logical status
c
c     Return 1/0 if array is active/inactive
c
      status = g_a.gt.0 .and. g_a.le.max_global_array
      if (status) status = actv(g_a)
      if (status) then
         ga_verify_handle = 1
      else
         ga_verify_handle = 0
      endif
c
      end
      subroutine ga_check_handle(g_a, string)
#include "mafdecls.h"
#include "globalp.fh"
#include "tcgmsg.fh"
c
      integer g_a
      character *(*) string
c
c     Check that the global array handle g_a is valid ... if not
c     call ga_error with the string provided
c
      if (g_a.le.0 .or. g_a.gt.max_global_array) then
         write(6,*) ' ga_check_handle: ', string
         call ga_error(' invalid global array handle ', g_a)
      else if (.not. actv(g_a)) then
         write(6,*) ' ga_check_handle: ', string
         call ga_error(' global array is inactive ',  g_a)
      endif
c
      end


      integer function ga_nnodes()
      implicit none
c
c     Return no. of compute processes
c
      integer nproc
#ifdef TCGMSG
      integer nnodes
      external nnodes
      nproc = nnodes()
#else
      call ga_error('ga_nnodes: invalid compiliation flags',0)
#endif
c
#ifdef DATA_SERVER
      if (mod(nproc,2).ne.0) 
     $    call ga_error('ga_nnodes: must have even no. of processes',0)
      ga_nnodes = nproc/2
#else
      ga_nnodes = nproc
#endif
c
      end

      integer function ga_map_myplace()
      implicit none
#include "globalp.fh"
#include "tcgmsg.fh"
c
c     maps ga_nodeid() to 0 ... ga_nnodes()-1 
c
#if   defined(DATA_SERVER) && defined(TCGMSG)
      integer ga_nodeid
c
      ga_map_myplace = ga_nodeid() - nserver
#else
      call ga_error('ga_map_myplace: WRONG COMPILATION FLAGS',0)
#endif
c
      end


      integer function ga_map_id(node)
      implicit none
      integer  node
#include "globalp.fh"
#include "tcgmsg.fh"
c
c     maps node in (0 .. ga_nnodes()-1) to (ga_nnodes .. nnodes()-1) 
c
#if   defined(DATA_SERVER) && defined(TCGMSG)
c
      ga_map_id = node + nserver 
#else
      call ga_error('ga_map_id: WRONG COMPILATION FLAGS',0) 
#endif
c
      end


      integer function ga_same_id(node)
      implicit none
      integer  node
      ga_same_id = node
c
      end


      integer function ga_nodeid()
      implicit none
c
c     Return id of compute process =  0, ..., nnodes()-1
c     Can be also called by data_server nodes
c
#ifdef TCGMSG
      integer nodeid
      external nodeid
      ga_nodeid = nodeid()
#else
      call ga_error('ga_nodeid: invalid compilation flags',0)
#endif
c
      end

      integer function ga_id_to_msg_id(node)
      integer node
c
c     Map from GA compute process ids to message passing process ids
c
      ga_id_to_msg_id = node
c
      end


      logical function ga_locate(g_a, i, j, owner)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a, i, j, owner
      integer ivp, jvp, lo, hi, ind
c
c     Return the logical compute process id that 'owns' i,j
c
      if (i.le.0 .or. i.gt.dims(1,g_a) .or.
     $    j.le.0 .or. j.gt.dims(2,g_a)) then
         ga_locate = .false.
         owner = -1
      else
         ivp = (i-1)*pscl(1,g_a)
 10      lo = mapc(1,ivp,g_a)
         hi = mapc(2,ivp,g_a)
         if (lo.gt.i) then
            ivp = ivp - 1
            goto 10
         else if (hi.lt.i) then
            ivp = ivp + 1
            goto 10
         endif
c     
         jvp = (j-1)*pscl(2,g_a)
 20      ind = jvp*pdim(1,g_a)
         lo = mapc(3,ind,g_a)
         hi = mapc(4,ind,g_a)
         if (lo.gt.j) then
            jvp = jvp - 1
            goto 20
         else if (hi.lt.j) then
            jvp = jvp + 1
            goto 20
         endif
c
         owner = ivp + jvp*pdim(1,g_a)
         ga_locate = .true.
      endif
c
      end


      subroutine ga_dscatter(g_a, v, i, j, nv)
      implicit none

#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
#include "global.fh"
c     
      integer g_a
      integer nv                 ! No. of elements to scatter
      double precision v(nv)     ! Values to scatter
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
c
      integer l_p, k_p           ! MA handle/index for process list
      integer loop, nelem, p, first, limit,  oldmask
c
c     Scatter elements of v into the global array
c     NOTE: the arrays v, i, j, are returned with values intact
c           but with their order shuffled
c
      if (nv .le. 0) return
#ifdef GA_TRACE
      call trace_stime
#endif
c
c     Error checking
c
      call ga_check_handle(g_a, 'ga_dscatter')
      if (ttyp(g_a) .ne. MT_DBL) 
     $     call ga_error('ga_dscatter: must be double', g_a)
c
c     Allocate and generate list of owning processors
c
      if (.not. ma_push_get(MT_INT, nv, 'ga_dscatter', l_p, k_p))
     $     call ga_error('ga_dscatter: failed to allocate p', nv)
c
      do loop = 1,nv
         if (.not. ga_locate(g_a, i(loop), j(loop), int_mb(k_p+loop-1)))
     $        call ga_error('ga_dscatter: invalid i/j',
     $        i(loop)*100000 + j(loop))
      enddo
c
c     Sort the entries by processor
c
      call ga_sort_scat2(nv, v, i, j, int_mb(k_p))	
c
c     Limit messages to available buffer size
c
      limit = tmpsize/3   ! 3 arrays will travel in the same message
c     limit = min(itmpsize, tmpsize)
c
c     Loop thru array sending elements to each processor in turn
c
      nelem = 0
      first = 0
      p = -1
      do loop = 1, nv
         if (int_mb(k_p+loop-1).ne.p .or. nelem.eq.limit) then
c
c           New processor encountered, send last batch
c
            if (nelem.gt.0 .and. p.eq.me .and. oserver) then
               oldmask = ga_mask(1)
               call ga_local_dscatter(
     $              g_a, v(first), i(first), j(first), nelem)
               oldmask = ga_mask(oldmask)
            else if (nelem.gt.0) then
               call ga_remote_dscatter(
     $              g_a, v(first), i(first), j(first), nelem, p)
            endif
            first = loop
            nelem = 1
            p = int_mb(k_p+loop-1)
         else
            nelem = nelem + 1
         endif
      enddo
      if (nelem.gt.0 .and. p.eq.me .and. oserver) then
         call ga_local_dscatter(
     $        g_a, v(first), i(first), j(first), nelem)
      else if (nelem.gt.0) then
         call ga_remote_dscatter(
     $        g_a, v(first), i(first), j(first), nelem, p)
      endif
c
      if(.not. ma_pop_stack(l_p))
     $     call ga_error('ga_dscatter: pop stack failed!', g_a)
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, 0, 0, 0, 0, GA_OP_DST)
#endif
c
      end

      subroutine ga_remote_dscatter(g_a, v, i, j, nv, p)
      implicit none
#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a
      integer nv                 ! No. of elements to scatter
      double precision v(nv)     ! Values to scatter
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
      integer p                  ! Compute process that 'owns' data
      integer to
c
c
c
      call dbl_block_move(v,1,1, tmp_local,1,1,1, nv)
      call int_block_move(i,1,1, tmp_local(nv+1),1,1,1, nv)
      call int_block_move(j,1,1, tmp_local(nv+1),nv+1,1,1, nv)
      to = p
#ifdef DATA_SERVER
         to = to+nserver
#endif
      call ga_snd_req(g_a, nv, 0, 0, 0, 0, 0, GA_OP_DST, to,
     $                mdtob(nv)+2*mitob(nv))
c
      end


      subroutine ga_local_dscatter(g_a, v, i, j, nv)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a
      integer nv                 ! No. of elements to scatter
      double precision v(nv)     ! Values to scatter
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
c
c     Scatter elements into local piece of g_a
c
      integer k, ij, base, ilo, jlo, idim, iproc
#include "global.fh"
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
      base = indx(g_a)
      ilo  = mapc(1,iproc,g_a)
      jlo  = mapc(3,iproc,g_a)
      idim = ldim(1,g_a)
c
c     oldmask = ga_mask(1)
      do k = 1, nv
         ij = base + (i(k)-ilo) + (j(k)-jlo)*idim
         dbl_mb(ij) = v(k)
      enddo
c     oldmask = ga_mask(oldmask)
c
      end


      subroutine ga_handler_dscatter(g_a, v, ij, nv)
      implicit none
      integer g_a
      integer nv                 ! No. of elements to scatter
      double precision v(*)  
      integer ij(*)
c
c     this routine is present solely to pass correct
c     address of array j (residing in tmp_recv) to ga_local_dscatter
c     because of no pointer arithmetic in fortran
c
      call ga_local_dscatter(g_a, v, ij, ij(nv+1), nv)
      end



      subroutine ga_dgather(g_a, v, i, j, nv)
      implicit none

#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
#include "global.fh"
c     
      integer g_a
      integer nv                 ! No. of elements to gather
      double precision v(nv)     ! Values to gather
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
c
      integer l_p, k_p           ! MA handle/index for process list
      integer loop, nelem, p, first, limit,  oldmask
c
c     Gather elements of v into the global array
c     NOTE: the arrays v, i, j, are returned with values intact
c           but with their order shuffled
c
      if (nv .le. 0) return
#ifdef GA_TRACE
      call trace_stime
#endif
c
c     Error checking
c
      call ga_check_handle(g_a, 'ga_dgather')
      if (ttyp(g_a) .ne. MT_DBL) 
     $     call ga_error('ga_dgather: must be double', g_a)
c
c     Allocate and generate list of owning processors
c
      if (.not. ma_push_get(MT_INT, nv, 'ga_dgather', l_p, k_p))
     $     call ga_error('ga_dgather: failed to allocate p', nv)
c
      do loop = 1,nv
         if (.not. ga_locate(g_a, i(loop), j(loop), int_mb(k_p+loop-1)))
     $        call ga_error('ga_dgather: invalid i/j',
     $        i(loop)*100000 + j(loop))
      enddo
c
c     Sort the entries by processor
c     OK, it does not make sense to sort v since it is empty but we
c     use what we have
c
      call ga_sort_scat2(nv, v, i, j, int_mb(k_p))	
c
c     Limit messages to available buffer size
c
      limit = tmpsize/2   ! 2 int arrays will travel in the same message
c
c     Loop thru array sending elements to each processor in turn
c
      nelem = 0
      first = 0
      p = -1
      do loop = 1, nv
         if (int_mb(k_p+loop-1).ne.p .or. nelem.eq.limit) then
c
c           New processor encountered, send last batch
c
            if (nelem.gt.0 .and. p.eq.me .and. oserver) then
               oldmask = ga_mask(1)
               call ga_local_dgather(
     $              g_a, v(first), i(first), j(first), nelem)
               oldmask = ga_mask(oldmask)
            else if (nelem.gt.0) then
               call ga_remote_dgather(
     $              g_a, v(first), i(first), j(first), nelem, p)
            endif
            first = loop
            nelem = 1
            p = int_mb(k_p+loop-1)
         else
            nelem = nelem + 1
         endif
      enddo
      if (nelem.gt.0 .and. p.eq.me .and. oserver) then
         call ga_local_dgather(
     $        g_a, v(first), i(first), j(first), nelem)
      else if (nelem.gt.0) then
         call ga_remote_dgather(
     $        g_a, v(first), i(first), j(first), nelem, p)
      endif
c
      if(.not. ma_pop_stack(l_p))
     $     call ga_error('ga_dgather: pop stack failed!', g_a)
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, 0, 0, 0, 0, GA_OP_DGT)
#endif
c
      end


      subroutine ga_remote_dgather(g_a, v, i, j, nv, p)
      implicit none
#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a
      integer nv                 ! No. of elements to gather
      double precision v(nv)     ! Values to gather
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
      integer p                  ! Compute process that 'owns' data
c
      integer node               ! Actual id that owns the data
      integer len, from
c
      node = p
#ifndef DATA_SERVER
c
c***  Post asynchronous receive for the data
c
c     Let's prey that MP system won't try to use tmp_loc as workspace
c     - buffer for both outgoing message - request with i & j and 
c     incoming message - values of v
c
      call ga_rcv_msg(GA_TYPE_DGT, tmp_local, mdtob(tmpsize),
     $           len, node, from, 0)
#else
      node = node+nserver 
#endif
c
c
      call int_block_move(i,1,1, tmp_local,1,1,1, nv)
      call int_block_move(j,1,1, tmp_local,nv+1,1,1, nv)
      call ga_snd_req(g_a, nv, 0, 0, 0, 0, 0, GA_OP_DGT, node,
     $                mitob(2*nv))
c
c***  Receive data and copy from tmp_local into final location
c    
      call waitcom(node)
#ifdef DATA_SERVER
      call ga_rcv_msg(GA_TYPE_DGT, tmp_local, mdtob(tmpsize),
     $           len, node, from,1)
c
c     cannot get reliable message lengths from NX with interrupts
c
      if (len .ne. mdtob(nv))then
          print *,'should be: ',mdtob(nv)
          call ga_error('ga_gather_remote: response wrong size', len)
      endif
#endif
      call dbl_block_move(tmp_local,1,1, v,1,1,1, nv)
c
      end


      subroutine ga_local_dgather(g_a, v, i, j, nv)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
c
      integer g_a
      integer nv                 ! No. of elements to gather
      double precision v(nv)     ! Values to gather
      integer i(nv), j(nv)       ! v(k) -> g_a(i(k),j(k))
c
c     Gather elements into local piece of g_a
c
      integer k, ij, base, ilo, jlo, idim, iproc
#include "global.fh"
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
      base = indx(g_a)
      ilo  = mapc(1,iproc,g_a)
      jlo  = mapc(3,iproc,g_a)
      idim = ldim(1,g_a)
c
      do k = 1, nv
         ij = base + (i(k)-ilo) + (j(k)-jlo)*idim
         v(k) = dbl_mb(ij)
      enddo
c
      end


      subroutine ga_handler_dgather(g_a, tmp, nv)
      implicit none
#include "globalp.fh"
      integer g_a
      integer nv                 ! No. of elements to gather
      double precision tmp(*)  
c
c     this routine is present to copy i & j from receive buffer to ijtmp_recv 
c     we need to avoid overwritting tmp_recv bufer by response (v)
c
      call int_block_move(tmp,1,1, ijtmp_recv,1,1,1, 2*nv)        ! i & j
      call ga_local_dgather(g_a, tmp, ijtmp_recv, ijtmp_recv(nv+1), nv)
      end


      subroutine ga_sort_scat(nv, v, i, j, p)	
      implicit none
      integer nv, i(nv), j(nv), p(nv)
      double precision v(nv)
c     
      integer k, l
c
c     Sort entries, grouping them by processor label
c
c     This routine is superceded by ga_sort_scat2 (hsort.scat.c)
c     but this will still function.
c
      do k = 1, nv
         do l = k+1, nv
            if (p(l) .lt. p(k))
     $           call ga_swap(v, i, j, p, k, l)
         enddo
      enddo
c
      end

      subroutine ga_swap(v, i, j, p, k, l)
      implicit none
      integer i(*), j(*), p(*), k, l, itmp
      double precision v(*), dtmp
c
c     Only used by ga_sort_scat ... ga_sort_scat2 has 
c     this function manually inlined.
c
      dtmp = v(k)
      v(k) = v(l)
      v(l) = dtmp
      itmp = i(k)
      i(k) = i(l)
      i(l) = itmp
      itmp = j(k)
      j(k) = j(l)
      j(l) = itmp
      itmp = p(k)
      p(k) = p(l)
      p(l) = itmp
c
      end


      integer function ga_read_inc(g_a, i, j, inc)
      implicit none
#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
c     
      integer g_a               ! GA handle
      integer i, j              ! Indices of array element
      integer inc               ! Increment
      integer p                 ! Owning process
      logical ga_locate
      external ga_locate
      integer ga_local_read_inc, ga_remote_read_inc, ga_mask, oldmask
      external ga_local_read_inc, ga_remote_read_inc
c
c     Atomically read and increment the integer array element g_a(i,j)
c
c     *BEGIN CRITICAL SECTION*
c     return value = a(i,j)
c     a(i,j) += inc
c     *END CRITICAL SECTION*
c
#ifdef GA_TRACE
      call trace_stime
#endif
c
c     Error checking
c
      call ga_check_handle(g_a, 'ga_read_inc')
      if (ttyp(g_a) .ne. MT_INT) 
     $     call ga_error('ga_read_inc: must be integer', g_a)
c
c     Figure out who owns the element and act accordingly
c
      if (.not. ga_locate(g_a, i, j, p))
     $     call ga_error('ga_read_inc: i/j out of bounds', i*10000+j)
      if (p .eq. me .and. oserver) then
         oldmask = ga_mask(1)
         ga_read_inc = ga_local_read_inc(g_a, i, j, inc)
         oldmask = ga_mask(oldmask)
      else
         ga_read_inc = ga_remote_read_inc(g_a, i, j, inc, p)
      endif
c
#ifdef GA_TRACE
      call trace_etime
      call trace_genrec(g_a, 0, 0, 0, 0, GA_OP_RDI)
#endif
c
      end


      integer function ga_remote_read_inc(g_a, i, j, inc, p)
      implicit none
#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a
      integer i, j, inc, p
      integer node,  value, len, from
c
      node = p
c
#ifdef DATA_SERVER 
      node = node + nserver 
      call ga_snd_req(g_a, i, j, inc, 0, 0, 0, GA_OP_RDI, node,0)
      call ga_rcv_msg(GA_TYPE_RDI, value, mitob(1), len, node, from,1)
#else
      call ga_rcv_msg(GA_TYPE_RDI, value, mitob(1), len, node, from,0)
      call ga_snd_req(g_a, i, j, inc, 0, 0, 0, GA_OP_RDI, node,0)
      call waitcom(node)
#endif
      ga_remote_read_inc = value
c
      end


      integer function ga_local_read_inc(g_a, i, j, inc)
      implicit none
#include "mafdecls.h"
#include "globalp.fh"
#include "global.fh"
c
      integer g_a
      integer i, j, inc, iproc, base, ilo, jlo, idim, ij
c
#ifdef DATA_SERVER
      iproc = me - nserver
#else
      iproc = me
#endif
      base = indx(g_a)
      ilo  = mapc(1,iproc,g_a)
      jlo  = mapc(3,iproc,g_a)
      idim = ldim(1,g_a)
      ij   = base + (i-ilo) + (j-jlo)*idim
c
      ga_local_read_inc = int_mb(ij)
      int_mb(ij) = int_mb(ij) + inc
c
      end


      subroutine  ga_copy_private(g_a, g_b)
      implicit none
#include "mafdecls.h"
#include "tcgmsg.fh"
#include "globalp.fh"
c
      integer g_a, g_b
c
      integer ldim1, ldim2, lda, ldb
      integer atype, ailo, aihi, ajlo, ajhi, adim1, adim2, abase
      integer btype, bilo, bihi, bjlo, bjhi, bdim1, bdim2, bbase
c
c
      call ga_check_handle(g_a, 'ga_copy')
      call ga_check_handle(g_b, 'ga_copy')
c
      if(g_a .eq. g_b) call ga_error('ga_copy: g_a = g_b',0) 
c
      call ga_sync()
c
      call ga_inquire(g_a,  atype, adim1, adim2)
      call ga_distribution(g_a, me, ailo, aihi, ajlo, ajhi)
c
      call ga_inquire(g_b,  btype, bdim1, bdim2)
      call ga_distribution(g_b, me, bilo, bihi, bjlo, bjhi)
c
      if (atype.ne.btype)
     $        call ga_error('ga_copy: types must match', 0)
      if (adim1.ne.bdim1 .or. adim2.ne.bdim2)
     $        call ga_error('ga_copy: arrays not conformant', 0)
      if (ailo.ne.bilo .or. aihi.ne.bihi .or.
     $        ajlo.ne.bjlo .or. ajhi.ne.bjhi)
     $        call ga_error('ga_copy: distribution not identical',0)
c
      if (aihi.gt.adim1 .or. ajhi.gt.adim2)
     $        call ga_error('ga_copy: strange dimensions ', 0)
      if (ailo .gt. 0 .and. ajhi .gt. 0) then
            call ga_access(g_a,  ailo, aihi, ajlo, ajhi, abase, lda)
            call ga_access(g_b, bilo, bihi, bjlo, bjhi, bbase, ldb)
c
            ldim1 = (aihi-ailo+1)
            ldim2 = (ajhi-ajlo+1)
            if (abase.eq.0 .or. bbase.eq.0)
     $           call ga_error(' ga_copy: bases are strange ', 0)
            if (ldim1.le.0 .or. ldim2.le.0)
     $           call ga_error('ga_copy: strange ldims ',0)
            if (ldim1.ne.ldim(1,g_a) .or. ldim2.ne.ldim(2,g_a))
     $           call ga_error('ga_copy: ldims are wrong ', ldim1)
            if (atype .eq. MT_DBL) then
               call rdcopy(ldim1*ldim2, dbl_mb(abase), 1,
     $              dbl_mb(bbase), 1)
            else
c              write(6,*) ' copying integer array ? '
               call icopy(ldim1*ldim2, int_mb(abase), 1,
     $              int_mb(bbase), 1)
            endif
            call ga_release(g_a,  ailo, aihi, ajlo, ajhi) 
            call ga_release_update(g_b,  bilo, bihi, bjlo, bjhi) 
      endif
c     endif
      call ga_sync()
      end

