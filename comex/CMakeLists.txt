#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: implements a primative CMake build that can be used to build
#              GA on Windows-based systems. Only MPI-based runtimes are
#              supported.
# 
# DISCLAIMER
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
# ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
# COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
# SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
# INFRINGE PRIVATELY OWNED RIGHTS.
#
#
# ACKNOWLEDGMENT
#
# This software and its documentation were produced with United States
# Government support under Contract Number DE-AC06-76RLO-1830 awarded by
# the United States Department of Energy.  The United States Government
# retains a paid-up non-exclusive, irrevocable worldwide license to
# reproduce, prepare derivative works, perform publicly and display
# publicly by or for the US Government, including the right to
# distribute to other US Government contractors.
#
# -*- mode: cmake -*-
# -------------------------------------------------------------
# file: CMakeLists.txt
# -------------------------------------------------------------

option (COMEX_NETWORK_MPI_TS "use MPI 2-sided protocol for communication" OFF)
option (COMEX_NETWORK_MPI_PR "use MPI progress ranks protocol for communication" OFF)
option (COMEX_NETWORK_MPI3 "use MPI RMA protocols for communication" OFF)
include_directories(AFTER src-common)
if (MPI_TS)
  set(COMEX_DEVICE
    src-mpi/comex.c
    src-mpi/groups.c
  )
  set (COMEX_NETWORK_MPI_TS ON)
  include_directories(AFTER src-mpi)
elseif (MPI_PR)
  set(COMEX_DEVICE
    src-mpi-pr/comex.c
    src-mpi-pr/groups.c
    src-mpi-pr/reg_cache.c
  )
  set (COMEX_NETWORK_MPI_PR ON)
  include_directories(AFTER src-mpi-pr)
elseif (MPI3)
  set(COMEX_DEVICE
    src-mpi3/comex.c
    src-mpi3/groups.c
    src-mpi3/reg_win.c
  )
  set (COMEX_NETWORK_MPI3 ON)
  include_directories(AFTER src-mpi3)
endif()

# -------------------------------------------------------------
# figure out what BLAS library looks like
# -------------------------------------------------------------
if (NOT DEFINED ENABLE_BLAS)
option (ENABLE_BLAS "Include external BLAS" ON)
endif()
INCLUDE( CheckCSourceCompiles )
if (ENABLE_BLAS)
set (CMAKE_REQUIRED_LIBRARIES blas)
CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = caxpy (n,c1,ca,1,cb,1);
char daxpy_result = daxpy (n,d1,da,1,db,1);
char saxpy_result = saxpy (n,s1,sa,1,sb,1);
char zaxpy_result = zaxpy (n,z1,za,1,zb,1);
char ccopy_result = ccopy (n,c1,ca,1,cb,1);
char dcopy_result = dcopy (n,d1,da,1,db,1);
char scopy_result = scopy (n,s1,sa,1,sb,1);
char zcopy_result = zcopy (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_1_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = caxpy_ (n,c1,ca,1,cb,1);
char daxpy_result = daxpy_ (n,d1,da,1,db,1);
char saxpy_result = saxpy_ (n,s1,sa,1,sb,1);
char zaxpy_result = zaxpy_ (n,z1,za,1,zb,1);
char ccopy_result = ccopy_ (n,c1,ca,1,cb,1);
char dcopy_result = dcopy_ (n,d1,da,1,db,1);
char scopy_result = scopy_ (n,s1,sa,1,sb,1);
char zcopy_result = zcopy_ (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_2_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = caxpy__ (n,c1,ca,1,cb,1);
char daxpy_result = daxpy__ (n,d1,da,1,db,1);
char saxpy_result = saxpy__ (n,s1,sa,1,sb,1);
char zaxpy_result = zaxpy__ (n,z1,za,1,zb,1);
char ccopy_result = ccopy__ (n,c1,ca,1,cb,1);
char dcopy_result = dcopy__ (n,d1,da,1,db,1);
char scopy_result = scopy__ (n,s1,sa,1,sb,1);
char zcopy_result = zcopy__ (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_3_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = CAXPY (n,c1,ca,1,cb,1);
char daxpy_result = DAXPY (n,d1,da,1,db,1);
char saxpy_result = SAXPY (n,s1,sa,1,sb,1);
char zaxpy_result = ZAXPY (n,z1,za,1,zb,1);
char ccopy_result = CCOPY (n,c1,ca,1,cb,1);
char dcopy_result = DCOPY (n,d1,da,1,db,1);
char scopy_result = SCOPY (n,s1,sa,1,sb,1);
char zcopy_result = ZCOPY (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_4_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = CAXPY_ (n,c1,ca,1,cb,1);
char daxpy_result = DAXPY_ (n,d1,da,1,db,1);
char saxpy_result = SAXPY_ (n,s1,sa,1,sb,1);
char zaxpy_result = ZAXPY_ (n,z1,za,1,zb,1);
char ccopy_result = CCOPY_ (n,c1,ca,1,cb,1);
char dcopy_result = DCOPY_ (n,d1,da,1,db,1);
char scopy_result = SCOPY_ (n,s1,sa,1,sb,1);
char zcopy_result = ZCOPY_ (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_5_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
typedef struct {float dummy[2];} SingleComplex;
typedef struct {double dummy[2];} DoubleComplex;
int n = 100;
SingleComplex c1,ca[100],cb[100];
double d1,da[100],db[100];
float s1,sa[100],sb[100];
DoubleComplex z1,za[100],zb[100];
char caxpy_result = CAXPY__ (n,c1,ca,1,cb,1);
char daxpy_result = DAXPY__ (n,d1,da,1,db,1);
char saxpy_result = SAXPY__ (n,s1,sa,1,sb,1);
char zaxpy_result = ZAXPY__ (n,z1,za,1,zb,1);
char ccopy_result = CCOPY__ (n,c1,ca,1,cb,1);
char dcopy_result = DCOPY__ (n,d1,da,1,db,1);
char scopy_result = SCOPY__ (n,s1,sa,1,sb,1);
char zcopy_result = ZCOPY__ (n,z1,za,1,zb,1);
return 0;
}
"
    BLAS_6_SIGNATURE )
endif()

# set blas symbols
if (BLAS_1_SIGNATURE)
  set(CM_BLAS_CAXPY caxpy)
  set(CM_BLAS_DAXPY daxpy)
  set(CM_BLAS_SAXPY saxpy)
  set(CM_BLAS_ZAXPY zaxpy)
  set(CM_BLAS_CCOPY ccopy)
  set(CM_BLAS_DCOPY dcopy)
  set(CM_BLAS_SCOPY scopy)
  set(CM_BLAS_ZCOPY zcopy)
elseif (BLAS_2_SIGNATURE)
  set(CM_BLAS_CAXPY _caxpy)
  set(CM_BLAS_DAXPY _daxpy)
  set(CM_BLAS_SAXPY _saxpy)
  set(CM_BLAS_ZAXPY _zaxpy)
  set(CM_BLAS_CCOPY _ccopy)
  set(CM_BLAS_DCOPY _dcopy)
  set(CM_BLAS_SCOPY _scopy)
  set(CM_BLAS_ZCOPY _zcopy)
elseif (BLAS_3_SIGNATURE)
  set(CM_BLAS_CAXPY __caxpy)
  set(CM_BLAS_DAXPY __daxpy)
  set(CM_BLAS_SAXPY __saxpy)
  set(CM_BLAS_ZAXPY __zaxpy)
  set(CM_BLAS_CCOPY __ccopy)
  set(CM_BLAS_DCOPY __dcopy)
  set(CM_BLAS_SCOPY __scopy)
  set(CM_BLAS_ZCOPY __zcopy)
elseif (BLAS_4_SIGNATURE)
  set(CM_BLAS_CAXPY CAXPY)
  set(CM_BLAS_DAXPY DAXPY)
  set(CM_BLAS_SAXPY SAXPY)
  set(CM_BLAS_ZAXPY ZAXPY)
  set(CM_BLAS_CCOPY CCOPY)
  set(CM_BLAS_DCOPY DCOPY)
  set(CM_BLAS_SCOPY SCOPY)
  set(CM_BLAS_ZCOPY ZCOPY)
elseif (BLAS_5_SIGNATURE)
  set(CM_BLAS_CAXPY _CAXPY)
  set(CM_BLAS_DAXPY _DAXPY)
  set(CM_BLAS_SAXPY _SAXPY)
  set(CM_BLAS_ZAXPY _ZAXPY)
  set(CM_BLAS_CCOPY _CCOPY)
  set(CM_BLAS_DCOPY _DCOPY)
  set(CM_BLAS_SCOPY _SCOPY)
  set(CM_BLAS_ZCOPY _ZCOPY)
elseif (BLAS_6_SIGNATURE)
  set(CM_BLAS_CAXPY __CAXPY)
  set(CM_BLAS_DAXPY __DAXPY)
  set(CM_BLAS_SAXPY __SAXPY)
  set(CM_BLAS_ZAXPY __ZAXPY)
  set(CM_BLAS_CCOPY __CCOPY)
  set(CM_BLAS_DCOPY __DCOPY)
  set(CM_BLAS_SCOPY __SCOPY)
  set(CM_BLAS_ZCOPY __ZCOPY)
else()
  set(CM_BLAS_CAXPY _caxpy)
  set(CM_BLAS_DAXPY _daxpy)
  set(CM_BLAS_SAXPY _saxpy)
  set(CM_BLAS_ZAXPY _zaxpy)
  set(CM_BLAS_CCOPY _ccopy)
  set(CM_BLAS_DCOPY _dcopy)
  set(CM_BLAS_SCOPY _scopy)
  set(CM_BLAS_ZCOPY _zcopy)
endif()

# check for weak alias pragma

CHECK_C_SOURCE_COMPILES(
"
extern void weakf(int c);
#pragma weak weakf = __weakf
void __weakf(int c) {}
int main(int argc, char **argv) {
  weakf(0);
  return(0);
}
"
   HAVE_SYS_WEAK_ALIAS_PRAGMA )

# check for availability of some functions
include(CheckFunctionExists)
CHECK_FUNCTION_EXISTS(sched_setaffinity
     HAVE_SCHED_SETAFFINITY)
CHECK_FUNCTION_EXISTS(pthread_setaffinity_np
     HAVE_PTHREAD_SETAFFINITY_NP)

# generate config.h header file
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_CURRENT_SOURCE_DIR}/config.h )

# -------------------------------------------------------------
# ARMCI and COMEX header installation
# -------------------------------------------------------------

set(ARMCI_HEADERS
  src-armci/armci.h
  src-armci/message.h
  src-armci/parmci.h
)

set(COMEX_DEVICE_HEADERS
  src-common/comex.h
)

install (FILES
  ${ARMCI_HEADERS}
  ${COMEX_DEVICE_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# ARMCI and COMEX library installation
# -------------------------------------------------------------

set(ARMCI_FILES
  src-armci/armci.c
  src-armci/capi.c
  src-armci/groups.c
  src-armci/iterator.c
  src-armci/message.c
)

add_library(armci_comex OBJECT
  ${ARMCI_FILES}
  ${COMEX_DEVICE}
)

add_library(armci
  ${ARMCI_FILES}
  ${COMEX_DEVICE}
)

target_link_libraries(armci)

install (TARGETS
  armci
  DESTINATION lib
)

add_library(comex
 ${COMEX_DEVICE}
)

target_link_libraries(comex)

install (TARGETS
  comex
  DESTINATION lib
)
