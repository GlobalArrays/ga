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
option (ENABLE_FORTRAN "Build Fortran interface" OFF)

if (ENABLE_FORTRAN)
  enable_language(C CXX Fortran)
else()
  enable_language(C CXX)
endif()

add_definitions (-DHAVE_CONFIG_H=1)

# add user configuration options
option (GA_RUNTIME "MPI runtime used to build GA" 'MPI_2SIDED')
option (MPI_TS "use MPI 2-sided protocol for communication" ON)
option (MPI_PR "use MPI progress ranks protocol for communication" OFF)
option (MPI3 "use MPI RMA protocols for communication" OFF)
option (ENABLE_CXX "Build C++ interface" OFF)
if (GA_RUNTIME STREQUAL MPI_PROGRESS)
  set (MPI_TS OFF)
  set (MPI_PR ON)
elseif (GA_RUNTIME STREQUAL MPI_RMA)
  set (MPI_TS OFF)
  set (MPI3 ON)
endif()
option (MSG_COMMS_MPI "Using MPI runtime for communication" ON)
option (NDEBUG "Debug option used by developers" OFF)
option (ENABLE_ARMCI_MEM_OPTION "User option for managing memory" ON)

option (ENABLE_I8 "Use 8 byte Fortran integers" OFF)
if (ENABLE_FORTRAN)
  add_custom_command(
    OUTPUT config.fh
    DEPENDS config.h
    COMMAND ${PROJECT_SOURCE_DIR}/tools/config_fh_from_h.pl < config.h
  )
endif()
option (F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS "Fortran/C interface property" ON)

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

if (ENABLE_FORTRAN)
  include( FortranCInterface )
  FortranCInterface_HEADER(${CMAKE_SOURCE_DIR}/f2c_cmake.h MACRO_NAMESPACE F77_FUNC_)
endif()

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

# hardwire various GA configuration parameters. Use convention that parameters
# are defined and set to 0 if not used
set (CYGWIN 0)
set (DECOSF 0)

# check for numerical libraries. These should set variables BLAS_FOUND and
# LAPACK_FOUND
find_package(BLAS)
if (BLAS_FOUND)
  set(HAVE_BLAS 1)
endif()
find_package(LAPACK)
if (LAPACK_FOUND)
  set(HAVE_LAPACK 1)
endif()

# check for availability of functions by seeing if small programs compile
CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
pause();
return 0;
}
"
    HAVE_PAUSE )

# check for availability of long double C-type
CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
long double x;
return 0;
}
"
    HAVE_LONG_DOUBLE )

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

if (HAVE_LONG_DOUBLE)
  set(MA_LONG_DOUBLE "long double")
else()
  set(MA_LONG_DOUBLE "struct {double dummy[2];}")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
  set(LINUX 1)
endif()

# hardwire memcpy and strchr since these have been standard for years
set (HAVE_MEMCPY 1)
set (HAVE_STRCHR 1)

# miscellaneous hardwired parameters (mostly not used)
set (ENABLE_CHECKPOINT 0)
set (ENABLE_PROFILING 0)
set (ENABLE_TRACE 0)
set (STATS 1)
set (USE_MALLOC 0)

# hardwire ARMCI configuration options
set (HAVE_ARMCI_GROUP_COMM 1)
set (HAVE_ARMCI_GROUP_COMM_MEMBER 0)
set (HAVE_ARMCI_INITIALIZED 1)

# suppress any checks to see if test codes run. Only check for compilation.
# use for cross-compilation situations
option (CHECK_COMPILATION_ONLY "Check compilation only" OFF)

if (ENABLE_FORTRAN)
  if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
     set (F77_GETARG GETARG)
     set (F77_GETARG_ARGS "i,s")
     set (F77_GETARG_DECLS "intrinsic GETARG")
     set (F77_IARGC IARGC)
  elseif (Fortran_COMPILER_NAME MATCHES "ifort.*")
     set (F77_GETARG GETARG)
     set (F77_GETARG_ARGS "i,s")
     set (F77_GETARG_DECLS "GETARG")
     set (F77_IARGC IARGC)
  elseif (Fortran_COMPILER_NAME MATCHES "g77.*")
     set (F77_GETARG GETARG)
     set (F77_GETARG_ARGS "i,s")
     set (F77_GETARG_DECLS "intrinsic GETARG")
     set (F77_IARGC IARGC)
  else()
     set (F77_GETARG GETARG)
     set (F77_GETARG_ARGS "i,s")
     set (F77_GETARG_DECLS "external GETARG")
     set (F77_IARGC IARGC)
  endif()
else()
# need to set these variable even if only compiling C/C++
   set (F77_GETARG GETARG)
   set (F77_GETARG_ARGS "i,s")
   set (F77_GETARG_DECLS "external GETARG")
   set (F77_IARGC IARGC)
endif() 

# Hardwire these settings. No way to check for Fortran integer size in CMake
# (that we can find)
set (CM_SIZEOF_F77_DOUBLE 8)
set (CM_SIZEOF_F77_REAL 4)
set (CM_SIZEOF_F77_INTEGER 4)
set (F2C_INTEGER_C_TYPE int)
set (F2C_REAL_C_TYPE float)
set (F2C_DOUBLE_PRECISION_C_TYPE double)
if (ENABLE_I8)
  set (CM_SIZEOF_F77_INTEGER 8)
  set (F2C_INTEGER_C_TYPE long)
endif()
set (C_POINTER_AS_INTEGER int)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set (C_POINTER_AS_INTEGER long)
endif()
set (CM_BLAS_SIZE ${CM_SIZEOF_F77_INTEGER})

# -------------------------------------------------------------
# Create include files from templates
# -------------------------------------------------------------
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_SOURCE_DIR}/config.h )
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/farg.h.in
                ${CMAKE_SOURCE_DIR}/gaf2c/farg.h )
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/typesf2c.h.in
                ${CMAKE_SOURCE_DIR}/gaf2c/typesf2c.h )
CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/cmake/matypes.h.in
                ${CMAKE_SOURCE_DIR}/ma/matypes.h )

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

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENABLE_CRAY_BUILD)
  if (NOT ${MPI_LIBRARY} OR NOT ${MPI_EXTRA_LIBRARY})
    set(MPI_LIBRARY "")
    set(MPI_EXTRA_LIBRARY "")
  endif()
endif()

add_library(ga
  $<TARGET_OBJECTS:ma>
  $<TARGET_OBJECTS:dra>
  $<TARGET_OBJECTS:eaf>
  $<TARGET_OBJECTS:elio>
  $<TARGET_OBJECTS:sf>
  $<TARGET_OBJECTS:armci_comex>
  $<TARGET_OBJECTS:ga_src>
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
add_subdirectory(global/src)
add_subdirectory(comex)
if (ENABLE_CXX)
  add_subdirectory(ga++/src)
endif()
if (ENABLE_FORTRAN)
  add_subdirectory(ma)
endif()
add_subdirectory(pario)
