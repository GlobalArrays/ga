c Thu Sep 29 09:39:57 PDT 1994
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
       if(type .eq. MT_DBL)then
          call ma__get_ptr(address, dbl_mb(1))
       else if(type .eq. MT_INT) then
          call ma__get_ptr(address, int_mb(1))
       else if(type .eq. MT_BYTE)then
          call ma__get_ptr(address, byte_mb(1))
       else if(type .eq. MT_REAL) then
          call ma__get_ptr(address, real_mb(1))
       else if(type .eq. MT_LOG) then
          call ma__get_ptr(address, log_mb(1))
       else if(type .eq. MT_SCPL) then
          call ma__get_ptr(address, scpl_mb(1))
       else if(type .eq. MT_DCPL) then
          call ma__get_ptr(address, dcpl_mb(1))
       else
          call ga_error('ma__base_address: wrong type',type)
       endif
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
