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
option (COMEX_NETWORK_MPI_MT "use MPI multi-threading protocol for communication" OFF)
option (COMEX_NETWORK_MPI_PT "use MPI progress threads protocol for communication" OFF)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR} src-common)
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
elseif (MPI_MT)
  set(COMEX_DEVICE
    src-mpi-mt/comex.c
    src-mpi-mt/groups.c
  )
  set (COMEX_NETWORK_MPI_MT ON)
  include_directories(AFTER src-mpi-mt)
elseif (MPI_PT)
  set(COMEX_DEVICE
    src-mpi-pt/comex.c
    src-mpi-pt/groups.c
    src-mpi-pt/reg_cache.c
  )
  set (COMEX_NETWORK_MPI_PT ON)
  include_directories(AFTER src-mpi-pt)
endif()

find_package(MPI)

add_definitions (-DHAVE_CONFIG_H=1)

# -------------------------------------------------------------
# Check for variable sizes, include files and functions
# -------------------------------------------------------------

# check size of different variables
include(CheckTypeSize)
check_type_size("int" CM_SIZEOF_INT)
check_type_size("double" CM_SIZEOF_DOUBLE)
check_type_size("float" CM_SIZEOF_FLOAT)
check_type_size("long" CM_SIZEOF_LONG)
check_type_size("long double" CM_SIZEOF_LONG_DOUBLE)
check_type_size("long long" CM_SIZEOF_LONG_LONG)
check_type_size("short" CM_SIZEOF_SHORT)

# check for standard C/C++ include files
include(CheckIncludeFiles)
check_include_files("assert.h" HAVE_ASSERT_H)
check_include_files("limits.h" HAVE_LIMITS_H)
check_include_files("linux/limits.h" HAVE_LINUX_LIMITS_H)
check_include_files("malloc.h" HAVE_MALLOC_H)
check_include_files("math.h" HAVE_MATH_H)
check_include_files("stddef.h" HAVE_STDDEF_H)
check_include_files("stdint.h" HAVE_STDINT_H)
check_include_files("stdio.h" HAVE_STDIO_H)
check_include_files("stdlib.h" HAVE_STDLIB_H)
check_include_files("strings.h" HAVE_STRINGS_H)
check_include_files("string.h" HAVE_STRING_H)
check_include_files("sys/types.h" HAVE_SYS_TYPES_H)
check_include_files("unistd.h" HAVE_UNISTD_H)
check_include_files("windows.h" HAVE_WINDOWS_H)

# check for certain functions
include(CheckFunctionExists)
check_function_exists("bzero" HAVE_BZERO)

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
  set(CM_BLAS_CAXPY caxpy_)
  set(CM_BLAS_DAXPY daxpy_)
  set(CM_BLAS_SAXPY saxpy_)
  set(CM_BLAS_ZAXPY zaxpy_)
  set(CM_BLAS_CCOPY ccopy_)
  set(CM_BLAS_DCOPY dcopy_)
  set(CM_BLAS_SCOPY scopy_)
  set(CM_BLAS_ZCOPY zcopy_)
elseif (BLAS_3_SIGNATURE)
  set(CM_BLAS_CAXPY caxpy__)
  set(CM_BLAS_DAXPY daxpy__)
  set(CM_BLAS_SAXPY saxpy__)
  set(CM_BLAS_ZAXPY zaxpy__)
  set(CM_BLAS_CCOPY ccopy__)
  set(CM_BLAS_DCOPY dcopy__)
  set(CM_BLAS_SCOPY scopy__)
  set(CM_BLAS_ZCOPY zcopy__)
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
  set(CM_BLAS_CAXPY CAXPY_)
  set(CM_BLAS_DAXPY DAXPY_)
  set(CM_BLAS_SAXPY SAXPY_)
  set(CM_BLAS_ZAXPY ZAXPY_)
  set(CM_BLAS_CCOPY CCOPY_)
  set(CM_BLAS_DCOPY DCOPY_)
  set(CM_BLAS_SCOPY SCOPY_)
  set(CM_BLAS_ZCOPY ZCOPY_)
elseif (BLAS_6_SIGNATURE)
  set(CM_BLAS_CAXPY CAXPY__)
  set(CM_BLAS_DAXPY DAXPY__)
  set(CM_BLAS_SAXPY SAXPY__)
  set(CM_BLAS_ZAXPY ZAXPY__)
  set(CM_BLAS_CCOPY CCOPY__)
  set(CM_BLAS_DCOPY DCOPY__)
  set(CM_BLAS_SCOPY SCOPY__)
  set(CM_BLAS_ZCOPY ZCOPY__)
else()
  set(CM_BLAS_CAXPY caxpy_)
  set(CM_BLAS_DAXPY daxpy_)
  set(CM_BLAS_SAXPY saxpy_)
  set(CM_BLAS_ZAXPY zaxpy_)
  set(CM_BLAS_CCOPY ccopy_)
  set(CM_BLAS_DCOPY dcopy_)
  set(CM_BLAS_SCOPY scopy_)
  set(CM_BLAS_ZCOPY zcopy_)
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

# Check for  FUNCTION_NAME

CHECK_C_SOURCE_COMPILES(
"
#include <stdio.h>
int sub(int i) {
  printf(\"%s\", __func__);
}
int main(int argc, char *argv[])
{
  sub(0);
  return 0;
}
"
    FUNCTION_1_SIGNATURE )

CHECK_C_SOURCE_COMPILES(
"
#include <stdio.h>
int sub(int i) {
  printf(\"%s\", __FUNCTION__);
}
int main(int argc, char *argv[])
{
  sub(0);
  return 0;
}
"
    FUNCTION_2_SIGNATURE )
if (FUNCTION_1_SIGNATURE)
  set(CM_FUNCTION_NAME __func__)
elseif(FUNCTION_2_SIGNATURE)
  set(CM_FUNCTION_NAME __FUNCTION__)
else()
  set(CM_FUNCTION_NAME __func__)
endif()

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
