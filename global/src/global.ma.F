c..................................................................
c      Two auxiliary routines to provide MA base addresses and    :
c                                      sizes of the data sizes    :
c      C routines called: ma__get_ptr() and ma_diff()             :
c.................................................................. 
c
       subroutine ma__base_address(type, address)
       implicit none
#include "mafdecls.h"
       Integer type
       Double Precision address
       External ma__get_ptr
       Integer fout
c      print *,'in ma__base_address -- type ', type, '>',MT_DBL
c      call ffflush(fout)
       if(type .eq. MT_DBL)then
c          print *,'in ma__base_address -- calling ma__get_ptr 1'
           call ffflush(fout)
          call ma__get_ptr(address, dbl_mb(1))
       else if(type .eq. MT_INT) then
c          print *,'in ma__base_address -- calling ma__get_ptr 2'
c          call ffflush(fout)
          call ma__get_ptr(address, int_mb(1))
       else if(type .eq. MT_BYTE)then
c          print *,'in ma__base_address -- calling ma__get_ptr 3'
c          call ffflush(fout)
          call ma__get_ptr(address, byte_mb(1))
       else if(type .eq. MT_REAL) then
c          print *,'in ma__base_address -- calling ma__get_ptr 4'
c          call ffflush(fout)
          call ma__get_ptr(address, real_mb(1))
       else if(type .eq. MT_LOG) then
c          print *,'in ma__base_address -- calling ma__get_ptr 5'
c          call ffflush(fout)
          call ma__get_ptr(address, log_mb(1))
       else if(type .eq. MT_SCPL) then
c          print *,'in ma__base_address -- calling ma__get_ptr 6'
c          call ffflush(fout)
          call ma__get_ptr(address, scpl_mb(1))
       else if(type .eq. MT_DCPL) then
c          print *,'in ma__base_address -- calling ma__get_ptr 7'
c          call ffflush(fout)
          call ma__get_ptr(address, dcpl_mb(1))
       else
           print *,'in ma__base_address -- wrong type ',type
           call ffflush(fout)
       endif
c      print *,' in ma__base_address ', address
c      call ffflush(fout)
       end 


       integer function ma__sizeof(type)
       implicit none
#include "mafdecls.h"
       integer ma__diff
       Integer type
       if(type .eq. MT_DBL)then
          ma__sizeof = ma__diff(dbl_mb(1), dbl_mb(2))
       else if(type .eq. MT_INT) then
          ma__sizeof = ma__diff(int_mb(1), int_mb(2))
       else if(type .eq. MT_BYTE)then
          ma__sizeof = ma__diff(byte_mb(1), byte_mb(2))
       else if(type .eq. MT_REAL) then
          ma__sizeof = ma__diff(real_mb(1), real_mb(2))
       else if(type .eq. MT_LOG) then
          ma__sizeof = ma__diff(log_mb(1), log_mb(2))
       else if(type .eq. MT_SCPL) then
          ma__sizeof = ma__diff(scpl_mb(1), scpl_mb(2))
       else if(type .eq. MT_DCPL) then
          ma__sizeof = ma__diff(dcpl_mb(1), dcpl_mb(2))
       endif
       end

       

