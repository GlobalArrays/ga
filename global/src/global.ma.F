c Thu Sep 29 09:39:57 PDT 1994
c..................................................................
c      Two auxiliary routines to provide MA base addresses and    :
c                                      sizes of the data sizes    :
c      C routines called: ga_ma_get_ptr() and ma_diff()             :
c.................................................................. 
c
       subroutine ga_ma_base_address(type, address)
       implicit none
#include "mafdecls.fh"
       Integer type
       Double Precision address
       External ga_ma_get_ptr
       if(type .eq. MT_DBL)then
          call ga_ma_get_ptr(address, dbl_mb(1))
       else if(type .eq. MT_INT) then
          call ga_ma_get_ptr(address, int_mb(1))
       else if(type .eq. MT_DCPL) then
          call ga_ma_get_ptr(address, dcpl_mb(1))
c      else if(type .eq. MT_BYTE)then
c         call ga_ma_get_ptr(address, byte_mb(1))
       else if(type .eq. MT_REAL) then
          call ga_ma_get_ptr(address, real_mb(1))
c      else if(type .eq. MT_LOG) then
c         call ga_ma_get_ptr(address, log_mb(1))
       else if(type .eq. MT_SCPL) then
          call ga_ma_get_ptr(address, scpl_mb(1))
       else
          call ga_error('ga_ma_base_address: wrong type',type)
       endif
       end 


       integer function ga_ma_sizeof(type)
       implicit none
#include "mafdecls.fh"
       integer ga_ma_diff
       Integer type
       if(type .eq. MT_DBL)then
          ga_ma_sizeof = ga_ma_diff(dbl_mb(1), dbl_mb(2))
       else if(type .eq. MT_INT) then
          ga_ma_sizeof = ga_ma_diff(int_mb(1), int_mb(2))
       else if(type .eq. MT_DCPL) then
          ga_ma_sizeof = ga_ma_diff(dcpl_mb(1), dcpl_mb(2))
c      else if(type .eq. MT_BYTE)then
c         ga_ma_sizeof = ga_ma_diff(byte_mb(1), byte_mb(2))
       else if(type .eq. MT_REAL) then
          ga_ma_sizeof = ga_ma_diff(real_mb(1), real_mb(2))
c      else if(type .eq. MT_LOG) then
c         ga_ma_sizeof = ga_ma_diff(log_mb(1), log_mb(2))
       else if(type .eq. MT_SCPL) then
          ga_ma_sizeof = ga_ma_diff(scpl_mb(1), scpl_mb(2))
       endif
       end
