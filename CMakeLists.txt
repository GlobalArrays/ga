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

cmake_minimum_required (VERSION 2.6.4)
project (GlobalArrays)

#set (GlobalArrays_VERSION_MAJOR 5)
#set (GlobalArrays_VERSION_MINOR 6)

# This project uses C,C++ and Fortran
#enable_language(CC)
#enable_language(CXX)
enable_language(C CXX Fortran)

add_definitions (-DHAVE_CONFIG_H=1)

# add user configuration options
option (GA_RUNTIME "MPI runtime used to build GA" 'MPI_2SIDED')
option (MPI_TS "use MPI 2-sided protocol for communication" ON)
option (MPI_PR "use MPI progress ranks protocol for communication" OFF)
option (MPI3 "use MPI RMA protocols for communication" OFF)
option (ENABLE_CXX "Build C++ interface" OFF)
option (ENABLE_FORTRAN "Build Fortran interface" ON)
if (GA_RUNTIME STREQUAL 'MPI_PROGRESS')
  set (MPI_TS OFF)
  set (MPI_PR ON)
elseif (GA_RUNTIME STREQUAL 'MPI_RMA')
  set (MPI_TS OFF)
  set (MPI3 ON)
endif()
option (MSG_COMMS_MPI "Using MPI runtime for communication" ON)

if (ENABLE_FORTRAN)
  add_custom_command(
    OUTPUT config.fh
    DEPENDS config.h
    COMMAND ${PROJECT_SOURCE_DIR}/tools/config_fh_from_h.pl < config.h
  )
endif()

INCLUDE( CheckCSourceCompiles )
# Check for restrict keyword
FOREACH( ac_kw __restrict __restrict__ _Restrict restrict )
    CHECK_C_SOURCE_COMPILES(
"
typedef int * int_ptr;
int foo (int_ptr ${ac_kw} ip) {
    return ip[0];
}
int main() {
    int s[1];
    int * ${ac_kw} t = s;
    t[0] = 0;
    return foo(t); 
}   
"
    HAVE_RESTRICT )
    IF( HAVE_RESTRICT )
        SET( ac_cv_c_restrict ${ac_kw} )
        BREAK( )
    ENDIF( )
ENDFOREACH( )
IF( HAVE_RESTRICT )
    SET( restrict ${ac_cv_c_restrict} )
ELSE( )
    SET( restrict " " )
ENDIF( )

# Check for inline keyword
CHECK_C_SOURCE_COMPILES(
"
typedef int foo_t;
static inline foo_t static_foo(){return 0;}
foo_t foo(){return 0;}
int main(int argc, char *argv[]){return 0;}
"
    HAVE_INLINE_NATIVE )
IF( HAVE_INLINE_NATIVE )
ELSE ( )
    FOREACH( ac_kw __inline__ __inline )
        CHECK_C_SOURCE_COMPILES(
"
typedef int foo_t;
static ${ac_kw} foo_t static_foo(){return 0;}
foo_t foo(){return 0;}
int main(int argc, char *argv[]){return 0;}
"
        HAVE_INLINE )
        IF( HAVE_INLINE )
            SET( ac_cv_c_inline ${ac_kw} )
            BREAK( )
        ENDIF( )
    ENDFOREACH( )
    IF( HAVE_INLINE )
        SET( inline ${ac_cv_c_inline} )
    ELSE( )
        SET( inline " " )
    ENDIF( )
ENDIF( )

option (F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS "Set F77 macros" ON)
set (CM_SIZEOF_F77_DOUBLE 8)
set (CM_SIZEOF_F77_REAL 4)
set (CM_SIZEOF_F77_INTEGER 4)

set (CM_BLAS_SIZE ${CM_SIZEOF_F77_INTEGER})
# check size of different variables
include(CheckTypeSize)
check_type_size("int" CM_SIZEOF_INT)
check_type_size("double" CM_SIZEOF_DOUBLE)
#check_type_size("double precision" CM_SIZEOF_F77_DOUBLE LANGUAGE Fortran)
#check_type_size("integer" CM_SIZEOF_F77_INTEGER LANGUAGE Fortran)
#check_type_size("real" CM_SIZEOF_F77_REAL LANGUAGE Fortran)
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

# check for numerical libraries
option (HAVE_BLAS "BLAS libraries detected" OFF)
find_package(BLAS)
if (BLAS_FOUND)
set (HAVE_BLAS ON)
endif()

option (HAVE_LAPACK "LAPACK libraries detected" OFF)
find_package(LAPACK)
if (LAPACK_FOUND)
set (HAVE_LAPACK ON)
endif()

CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
pause();
return 0;
}
"
    HAVE_PAUSE )

# hardwire memcpy and strchr since these have been standard for years
set (HAVE_MEMCPY 1)
set (HAVE_STRCHR 1)

# hardwire ARMCI configuration options
set (HAVE_ARMCI_GROUP_COMM 1)
set (HAVE_ARMCI_GROUP_COMM_MEMBER 0)
set (HAVE_ARMCI_INITIALIZED 1)

CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_SOURCE_DIR}/config.h )

# suppress any checks to see if test codes run. Only check for compilation.
# use for cross-compilation situations
option (CHECK_COMPILATION_ONLY "Check compilation only" OFF)

# -------------------------------------------------------------
# MPI compiler
# -------------------------------------------------------------
message(STATUS "Checking MPI ...")
find_package (MPI REQUIRED)
include_directories(AFTER ${MPI_INCLUDE_PATH})

#if (ENABLE_CRAY_BUILD)
#  set (CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#endif()

# This is here for cmake version 2.6 
if (NOT MPI_LIBRARY OR NOT MPI_EXTRA_LIBRARY)
# Punt if MPI_LIBRARY or MPI_EXTRA_LIBRARY not found
  set(MPI_CXX_LIBRARIES "")
else()
  if (NOT MPI_CXX_LIBRARIES)
   set(MPI_CXX_LIBRARIES 
     ${MPI_LIBRARY} 
     ${MPI_EXTRA_LIBRARY}
   )
  endif()
endif()
message(STATUS "MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

#set(GRIDPACK_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENABLE_CRAY_BUILD)
  if (NOT ${MPI_LIBRARY} OR NOT ${MPI_EXTRA_LIBRARY})
    set(MPI_LIBRARY "")
    set(MPI_EXTRA_LIBRARY "")
  endif()
endif()

# -------------------------------------------------------------
#  Configure GA library
# -------------------------------------------------------------
if (ENABLE_FORTRAN)
  set(GA_FORTRAN_INTERFACE_C_FILES
#fapi.c
  )
  set(GA_FORTRAN_INTERFACE_H_FILES
#cnames.h
  )
endif()

set(WAPI_FILES
  ${PROJECT_SOURCE_DIR}/global/src/ga-wapi.h
  ${PROJECT_SOURCE_DIR}/global/src/ga-wapidefs.h
)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/global/src
                    ${PROJECT_SOURCE_DIR}/gaf2c
                    ${PROJECT_SOURCE_DIR}/ma
                    ${PROJECT_SOURCE_DIR}/comex/src-armci
                    ${PROJECT_SOURCE_DIR}/tcgmsg
                    ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
)

set(GA_FILES
  ${PROJECT_SOURCE_DIR}/global/src/base.c
  ${PROJECT_SOURCE_DIR}/global/src/onesided.c
  ${PROJECT_SOURCE_DIR}/global/src/collect.c
  ${PROJECT_SOURCE_DIR}/global/src/capi.c
  ${PROJECT_SOURCE_DIR}/global/src/datatypes.c
  ${PROJECT_SOURCE_DIR}/global/src/decomp.c
  ${PROJECT_SOURCE_DIR}/global/src/DP.c
  ${PROJECT_SOURCE_DIR}/global/src/elem_alg.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_diag_seqc.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_malloc.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_profile.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_solve_seq.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_symmetr.c
  ${PROJECT_SOURCE_DIR}/global/src/ga_trace.c
  ${PROJECT_SOURCE_DIR}/global/src/global.nalg.c
  ${PROJECT_SOURCE_DIR}/global/src/global.npatch.c
  ${PROJECT_SOURCE_DIR}/global/src/global.periodic.c
  ${PROJECT_SOURCE_DIR}/global/src/global.util.c
  ${PROJECT_SOURCE_DIR}/global/src/hsort.scat.c
  ${PROJECT_SOURCE_DIR}/global/src/matmul.c
  ${PROJECT_SOURCE_DIR}/global/src/matrix.c
  ${PROJECT_SOURCE_DIR}/global/src/nbutil.c
  ${PROJECT_SOURCE_DIR}/global/src/peigstubs.c
  ${PROJECT_SOURCE_DIR}/global/src/sclstubs.c
  ${PROJECT_SOURCE_DIR}/global/src/select.c
  ${PROJECT_SOURCE_DIR}/global/src/sparse.c
  ${GA_FORTRAN_INTERFACE_C_FILES}
)

add_custom_target(machine_files
  DEPENDS
  ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h
)

add_custom_command(
  TARGET machine_files
  DEPENDS ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h
  COMMAND ${PROJECT_SOURCE_DIR}/tools/ga_papi_to_wapi.pl <
          ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h >
          ${PROJECT_SOURCE_DIR}/global/src/ga-wapi.h
)

add_custom_command(
  TARGET machine_files
  DEPENDS ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h
  COMMAND ${PROJECT_SOURCE_DIR}/tools/ga_papi_to_wapidefs.pl <
          ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h >
          ${PROJECT_SOURCE_DIR}/global/src/ga-wapidefs.h
)

# -------------------------------------------------------------
# Global Arrays header installation
# -------------------------------------------------------------
install(FILES
#  ${PROJECT_SOURCE_DIR}/global/src/abstract_ops.h
#  ${PROJECT_SOURCE_DIR}/global/src/base.h
#  ${PROJECT_SOURCE_DIR}/global/src/cnames.h
  ${PROJECT_SOURCE_DIR}/global/src/gacommon.h
#  ${PROJECT_SOURCE_DIR}/global/src/gaconfig.h
  ${PROJECT_SOURCE_DIR}/global/src/ga.h
  ${PROJECT_SOURCE_DIR}/global/src/ga-mpi.h
  ${PROJECT_SOURCE_DIR}/global/src/ga-papi.h
#  ${PROJECT_SOURCE_DIR}/global/src/ga_profile.h
#  ${PROJECT_SOURCE_DIR}/global/src/ga-wapidefs.h
  ${PROJECT_SOURCE_DIR}/global/src/ga-wapi.h
#  ${PROJECT_SOURCE_DIR}/global/src/global.h
#  ${PROJECT_SOURCE_DIR}/global/src/globalp.h
#  ${PROJECT_SOURCE_DIR}/global/src/matmul.h
  ${GA_FORTRAN_INTERFACE_H_FILES}
  DESTINATION include
)

# -------------------------------------------------------------
#  Configure COMEX and ARMCI libraries
# -------------------------------------------------------------

include_directories(AFTER ${PROJECT_SOURCE_DIR}/comex/src-common)
if (MPI_TS)
  set(COMEX_DEVICE
    ${PROJECT_SOURCE_DIR}/comex/src-mpi/comex.c
    ${PROJECT_SOURCE_DIR}/comex/src-mpi/groups.c
  )
  include_directories(AFTER ${PROJECT_SOURCE_DIR}/comex/src-mpi)
elseif (MPI_PR)
  set(COMEX_DEVICE
    ${PROJECT_SOURCE_DIR}/comex/src-mpi-pr/comex.c
    ${PROJECT_SOURCE_DIR}/comex/src-mpi-pr/groups.c
    ${PROJECT_SOURCE_DIR}/comex/src-mpi-pr/reg_cache.c
  )
  include_directories(AFTER ${PROJECT_SOURCE_DIR}/comex/src-mpi-pr)
elseif (MPI3)
  set(COMEX_DEVICE
    ${PROJECT_SOURCE_DIR}/comex/src-mpi3/comex.c
    ${PROJECT_SOURCE_DIR}/comex/src-mpi3/groups.c
    ${PROJECT_SOURCE_DIR}/comex/src-mpi3/reg_win.c
  )
  include_directories(AFTER ${PROJECT_SOURCE_DIR}/comex/src-mpi3)
endif()
include_directories(AFTER ${PROJECT_SOURCE_DIR}/comex/src-armci)

# -------------------------------------------------------------
# ARMCI and COMEX header installation
# -------------------------------------------------------------

set(ARMCI_HEADERS
  ${PROJECT_SOURCE_DIR}/comex/src-armci/armci.h
  ${PROJECT_SOURCE_DIR}/comex/src-armci/message.h
  ${PROJECT_SOURCE_DIR}/comex/src-armci/parmci.h
)

set(COMEX_DEVICE_HEADERS
  ${PROJECT_SOURCE_DIR}/comex/src-common/comex.h
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
  ${PROJECT_SOURCE_DIR}/comex/src-armci/armci.c
  ${PROJECT_SOURCE_DIR}/comex/src-armci/capi.c
  ${PROJECT_SOURCE_DIR}/comex/src-armci/groups.c
  ${PROJECT_SOURCE_DIR}/comex/src-armci/iterator.c
  ${PROJECT_SOURCE_DIR}/comex/src-armci/message.c
)

add_library(armci
  ${ARMCI_FILES}
)

install (TARGETS
  armci
  DESTINATION lib
)

add_library(comex
 ${COMEX_DEVICE}
)

install (TARGETS
  comex
  DESTINATION lib
)

# -------------------------------------------------------------
#  Configure MA libraries
# -------------------------------------------------------------

# -------------------------------------------------------------
# MA header installation
# -------------------------------------------------------------

set(MA_HEADERS
  ${PROJECT_SOURCE_DIR}/ma/ma.h
#  ${PROJECT_SOURCE_DIR}/ma/error.h
  ${PROJECT_SOURCE_DIR}/ma/macdecls.h
  ${PROJECT_SOURCE_DIR}/ma/macommon.h
  ${PROJECT_SOURCE_DIR}/ma/matypes.h
#  memcpy.h
  ${PROJECT_SOURCE_DIR}/ma/scope.h
  ${PROJECT_SOURCE_DIR}/ma/string-util.h
  ${PROJECT_SOURCE_DIR}/ma/table.h
)

install (FILES
  ${MA_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# MA library installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(MA_FORTRAN_FILES ${PROJECT_SOURCE_DIR}/ma/f2c.c)
endif()

set(MA_FILES
  ${PROJECT_SOURCE_DIR}/ma/ma.c
  ${PROJECT_SOURCE_DIR}/ma/error.c
  ${PROJECT_SOURCE_DIR}/ma/string-util.c
  ${PROJECT_SOURCE_DIR}/ma/table.c
  ${MA_FORTRAN_FILES}
)

# -------------------------------------------------------------
#  Configure PARIO libraries
# -------------------------------------------------------------
include_directories(AFTER ${PROJECT_SOURCE_DIR}/pario/dra
                           ${PROJECT_SOURCE_DIR}/pario/eaf
                           ${PROJECT_SOURCE_DIR}/pario/elio
                           ${PROJECT_SOURCE_DIR}/pario/sf
)

# -------------------------------------------------------------
# PARIO header installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(DRA_FORTRAN_HEADERS
      dra.fh
      draf2c.h)
  set(DRA_FORTRAN_FILES
      ${PROJECT_SOURCE_DIR}/pario/dra/fortran.c)
  set(EAF_FORTRAN_HEADERS
      eaf.fh)
  set(EAF_FORTRAN_FILES
      ${PROJECT_SOURCE_DIR}/pario/eaf/eaf_f2c.c)
  set(SF_FORTRAN_HEADERS
      sf.fh
      sff2c.h)
  set(SF_FORTRAN_FILES
      ${PROJECT_SOURCE_DIR}/pario/sf/sf_fortran.c)
endif()

set(DRA_HEADERS
  ${PROJECT_SOURCE_DIR}/pario/dra/dra.h
#  ${PROJECT_SOURCE_DIR}/pario/dra/buffers.h
#  ${PROJECT_SOURCE_DIR}/pario/dra/drap.h
  ${DRA_FORTRAN_HEADERS}
)

set(EAF_HEADERS
  ${PROJECT_SOURCE_DIR}/pario/eaf/eaf.h
#  ${PROJECT_SOURCE_DIR}/pario/eaf/eafP.h
  ${EAF_FORTRAN_HEADERS}
)

set(ELIO_HEADERS
   ${PROJECT_SOURCE_DIR}/pario/elio/elio.h
   ${PROJECT_SOURCE_DIR}/pario/elio/chemio.h
#   ${PROJECT_SOURCE_DIR}/pario/elio/eliop.h
#   ${PROJECT_SOURCE_DIR}/pario/elio/pablo.h
)

set(SF_HEADERS
    ${PROJECT_SOURCE_DIR}/pario/sf/sf.h
    ${PROJECT_SOURCE_DIR}/pario/sf/coms.h
    ${SF_FORTRAN_HEADERS}
)

install (FILES
  ${DRA_HEADERS}
  ${EAF_HEADERS}
  ${ELIO_HEADERS}
  ${SF_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
#  PARIO executable files
# -------------------------------------------------------------

set(DRA_FILES
    ${PROJECT_SOURCE_DIR}/pario/dra/disk.arrays.c
    ${PROJECT_SOURCE_DIR}/pario/dra/buffers.c
    ${PROJECT_SOURCE_DIR}/pario/dra/capi.c
    ${PROJECT_SOURCE_DIR}/pario/dra/disk.param.c
    ${PROJECT_SOURCE_DIR}/pario/dra/env.c
    ${PROJECT_SOURCE_DIR}/pario/dra/global.unsup.c
    ${PROJECT_SOURCE_DIR}/pario/dra/patch.util.c
    ${PROJECT_SOURCE_DIR}/pario/dra/util.c
    ${DRA_FORTRAN_FILES}
)

set(EAF_FILES
    ${PROJECT_SOURCE_DIR}/pario/eaf/eaf.c
    ${EAF_FORTRAN_FILES}
)

set(ELIO_FILES
    ${PROJECT_SOURCE_DIR}/pario/elio/elio.c
    ${PROJECT_SOURCE_DIR}/pario/elio/stat.c
)

set(SF_FILES
     ${PROJECT_SOURCE_DIR}/pario/sf/shared.files.c
     ${PROJECT_SOURCE_DIR}/pario/sf/sf_capi.c
     ${SF_FORTRAN_FILES}
)

add_library(ga
  ${GA_FILES}
  ${ARMCI_FILES}
  ${COMEX_FILES}
  ${MA_FILES}
  ${DRA_FILES}
  ${EAF_FILES}
  ${ELIO_FILES}
  ${SF_FILES}
)

add_dependencies(ga machine_files)

# -------------------------------------------------------------
# Global Arrays library installation
# -------------------------------------------------------------

target_link_libraries(ga)

install (TARGETS
  ga
  DESTINATION lib
)

# -------------------------------------------------------------
# Subdirectories
# -------------------------------------------------------------
if (ENABLE_CXX)
  add_subdirectory(ga++/src)
endif()
