c Thu Sep 29 09:39:57 PDT 1994
c..................................................................
c      Two auxiliary routines to provide MA base addresses and    :
c                                      sizes of the data sizes    :
c      C routines called: ma_ga_get_ptr() and ma_diff()             :
c.................................................................. 
c
       subroutine ma_ga_base_address(type, address)
       implicit none
#include "mafdecls.fh"
       Integer type
       Double Precision address
       External ma_ga_get_ptr
       if(type .eq. MT_DBL)then
          call ma_ga_get_ptr(address, dbl_mb(1))
       else if(type .eq. MT_INT) then
          call ma_ga_get_ptr(address, int_mb(1))
       else if(type .eq. MT_BYTE)then
          call ma_ga_get_ptr(address, byte_mb(1))
       else if(type .eq. MT_REAL) then
          call ma_ga_get_ptr(address, real_mb(1))
       else if(type .eq. MT_LOG) then
          call ma_ga_get_ptr(address, log_mb(1))
       else if(type .eq. MT_SCPL) then
          call ma_ga_get_ptr(address, scpl_mb(1))
       else if(type .eq. MT_DCPL) then
          call ma_ga_get_ptr(address, dcpl_mb(1))
       else
          call ga_error('ma_ga_base_address: wrong type',type)
       endif
       end 


       integer function ma_ga_sizeof(type)
       implicit none
#include "mafdecls.fh"
       integer ma_ga_diff
       Integer type
       if(type .eq. MT_DBL)then
          ma_ga_sizeof = ma_ga_diff(dbl_mb(1), dbl_mb(2))
       else if(type .eq. MT_INT) then
          ma_ga_sizeof = ma_ga_diff(int_mb(1), int_mb(2))
       else if(type .eq. MT_BYTE)then
          ma_ga_sizeof = ma_ga_diff(byte_mb(1), byte_mb(2))
       else if(type .eq. MT_REAL) then
          ma_ga_sizeof = ma_ga_diff(real_mb(1), real_mb(2))
       else if(type .eq. MT_LOG) then
          ma_ga_sizeof = ma_ga_diff(log_mb(1), log_mb(2))
       else if(type .eq. MT_SCPL) then
          ma_ga_sizeof = ma_ga_diff(scpl_mb(1), scpl_mb(2))
       else if(type .eq. MT_DCPL) then
          ma_ga_sizeof = ma_ga_diff(dcpl_mb(1), dcpl_mb(2))
       endif
       end
