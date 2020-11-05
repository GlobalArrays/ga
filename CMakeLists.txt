#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: CMake build for GA. Only MPI-based runtimes are supported.
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

cmake_minimum_required (VERSION 3.17.0 FATAL_ERROR)
project (GlobalArrays VERSION 5.8.0 LANGUAGES C CXX)

# Append local CMake directory
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
include(ga-utils)

ga_option(ENABLE_FORTRAN ON)
ga_option(CMAKE_CXX_EXTENSIONS OFF)

set(NOFORT 1)
set(ENABLE_F77 0)
if (ENABLE_FORTRAN)
  enable_language(Fortran)
  set(NOFORT 0)
  set(ENABLE_F77 1)
endif()

ga_option(CMAKE_BUILD_TYPE Release)
ga_option(BLAS_VENDOR ReferenceBLAS)
ga_option(ENABLE_TESTS ON)
ga_option(ENABLE_PROFILING OFF)
#Options for user provided LinAlg libraries
ga_option(ENABLE_BLAS OFF)
ga_option(ENABLE_SCALAPACK OFF)
ga_option(ENABLE_EISPACK OFF)
ga_option(ENABLE_DPCPP OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

include(${PROJECT_SOURCE_DIR}/cmake/ga-compiler-options.cmake)

if(ENABLE_BLAS)
  if(NOT ENABLE_FORTRAN)
    message(FATAL_ERROR "ENABLE_BLAS=ON needs ENABLE_FORTRAN=ON")
  endif()  
endif()

if(MSVC AND ENABLE_FORTRAN)
  message(FATAL_ERROR "MSVC build needs ENABLE_FORTRAN=OFF")
endif()

if(ENABLE_PROFILING)
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR "GA profiling is currently not supported with Clang compilers. Please set ENABLE_PROFILING=OFF")
  endif()
endif()

# GA Runtime options: MPI runtime used to build GA
ga_option(GA_RUNTIME MPI_2SIDED)

if (GA_RUNTIME STREQUAL MPI_PROGRESS_RANK)
  set (MPI_PR TRUE CACHE BOOL "use MPI progress ranks protocol for communication")
elseif (GA_RUNTIME STREQUAL MPI_RMA)
  set (MPI3 TRUE CACHE BOOL "use MPI RMA protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_MULTITHREADED)
  set (MPI_MT TRUE CACHE BOOL "use MPI multi-threading protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_PROGRESS_THREAD)
  set (MPI_PT TRUE CACHE BOOL "use MPI progress thread protocols for communication")
elseif (GA_RUNTIME STREQUAL MPI_2SIDED)
  set (MPI_TS TRUE CACHE BOOL "use MPI 2-sided protocol for communication")
else()
  message(STATUS "GA_RUNTIME provided \"${GA_RUNTIME}\" is not supported")
endif()

option (MSG_COMMS_MPI "Using MPI runtime for communication" ON)
option (ENABLE_ARMCI_MEM_OPTION "User option for managing memory" ON)
option (F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS "Fortran/C interface property" ON)

message(STATUS "Checking MPI ...")
find_package (MPI REQUIRED)
set(GA_EXTRA_LIBS ${GA_EXTRA_LIBS} MPI::MPI_C MPI::MPI_CXX)
if(ENABLE_FORTRAN)
  list(APPEND GA_EXTRA_LIBS MPI::MPI_Fortran)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/ga-checks.cmake)

if (ENABLE_FORTRAN)
  include( FortranCInterface )
  FortranCInterface_HEADER(${CMAKE_CURRENT_BINARY_DIR}/f2c_cmake.h MACRO_NAMESPACE F77_FUNC_)
else()
  CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/f2c_dummy.h.in
                  ${CMAKE_CURRENT_BINARY_DIR}/f2c_cmake.h )
endif()

# set some flags for MS Visual C Compiler
if (MSVC)
  add_definitions(/D NOMINMAX /D _CRT_SECURE_NO_WARNINGS /D _CRT_NONSTDC_NO_WARNINGS)
endif()
   
# hardwire various GA configuration parameters. Use convention that parameters
# are defined and set to 0 if not used
set (CYGWIN 0)
set (DECOSF 0)

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
  set(LINUX 1)
  set(LINUX64 1)
endif()

set(USE_I8 OFF)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(USE_I8 ON)
  set(BLAS_SIZE 8)
endif()

option (ENABLE_I8 "Use 8 byte Fortran integers" ${USE_I8})
message(STATUS "void size: ${CMAKE_SIZEOF_VOID_P}, USE_I8: ${USE_I8}, ENABLE_I8: ${ENABLE_I8}")

include(${PROJECT_SOURCE_DIR}/cmake/ga-linalg.cmake)

# hardwire memcpy and strchr since these have been standard for years
set (HAVE_MEMCPY 1)
set (HAVE_STRCHR 1)

# miscellaneous hardwired parameters (mostly not used)
set (ENABLE_CHECKPOINT 0)
set (GA_PROFILING 0)
set (ENABLE_TRACE 0)
set (STATS 1)
set (USE_MALLOC 0)
if(ENABLE_PROFILING)
  set (GA_PROFILING 1)
endif()


# hardwire ARMCI configuration options
set (HAVE_ARMCI_GROUP_COMM 1)
set (HAVE_ARMCI_GROUP_COMM_MEMBER 0)
set (HAVE_ARMCI_INITIALIZED 1)

# suppress any checks to see if test codes run. Only check for compilation.
# use for cross-compilation situations
option (CHECK_COMPILATION_ONLY "Check compilation only" OFF)

if (ENABLE_FORTRAN)
  message(STATUS "CMAKE_Fortran_COMPILER: ${CMAKE_Fortran_COMPILER}")
  message(STATUS "CMAKE_Fortran_COMPILER_ID: ${CMAKE_Fortran_COMPILER_ID}")
  set (F90_MODULE )
  set (F77_GETARG GETARG)
  set (F77_GETARG_ARGS "i,s")
  set (F77_GETARG_DECLS "")
  set (F77_IARGC IARGC)
  set (F77_FLUSH flush)
  set (HAVE_F77_FLUSH 1)

  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
     message(STATUS "Using GNU Fortran compiler settings")
     set (F77_GETARG_DECLS "intrinsic GETARG")
     set (FORTRAN_I8_FLAG -fdefault-integer-8)
     set (FORTRAN_I4_FLAG "")
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
     message(STATUS "Using Intel Fortran compiler settings")
     set (FORTRAN_I8_FLAG -i8)
     set (FORTRAN_I4_FLAG -i4)
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
     message(STATUS "Using PGI Fortran compiler settings")
     set (FORTRAN_I8_FLAG -i8)
     set (FORTRAN_I4_FLAG -i4)
  elseif (CMAKE_Fortran_COMPILER_ID MATCHES "XL")
     message(STATUS "Using IBM XL Fortran compiler settings")
     set (FORTRAN_I8_FLAG -qintsize=8)
     set (FORTRAN_I4_FLAG -qintsize=4)
  endif()
else()
# need to set these variable even if only compiling C/C++
   set (F77_GETARG GETARG)
   set (F77_GETARG_ARGS "i,s")
   set (F77_GETARG_DECLS "external GETARG")
   set (F77_IARGC IARGC)
   set (FORTRAN_I8_FLAG -i8)
   set (FORTRAN_I4_FLAG -i4)
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
  set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FORTRAN_I8_FLAG}")
  message(STATUS "Fortran flag ${CMAKE_Fortran_FLAGS} i8 ${FORTRAN_I8_FLAG}")
endif()
set (C_POINTER_AS_INTEGER int)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set (C_POINTER_AS_INTEGER long)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NOUSE_MMAP 1)
  endif()
  set(GA_FORTRAN_ACCESS_INDEX integer*8)
else()
  set(GA_FORTRAN_ACCESS_INDEX integer*4)
endif()
# set (CM_BLAS_SIZE ${CM_SIZEOF_F77_INTEGER})

# -------------------------------------------------------------
# Create include files from templates
# -------------------------------------------------------------
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/config.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/farg.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/gaf2c/farg.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/typesf2c.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/gaf2c/typesf2c.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/matypes.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/ma/matypes.h )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mafdecls.fh.in
                ${CMAKE_CURRENT_BINARY_DIR}/ma/mafdecls.fh )
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/global.fh.in
                ${CMAKE_CURRENT_BINARY_DIR}/global/src/global.fh )

if (ENABLE_FORTRAN)
message(STATUS "CMAKE_COMMAND:${CMAKE_COMMAND}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/config.fh
    COMMAND ${CMAKE_COMMAND} -D INPUT:PATH="${CMAKE_CURRENT_BINARY_DIR}/config.h" -D OUTPUT:PATH="${CMAKE_CURRENT_BINARY_DIR}/config.fh" -P ${PROJECT_SOURCE_DIR}/tools/config_fh_from_h.cmake DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/config.h
  )
  add_custom_target(
    GenerateConfigFH ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/config.fh
  )
endif()

include_directories(AFTER ${MPI_INCLUDE_PATH})
add_definitions (-DHAVE_CONFIG_H)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND GA_EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})

if(NOT ENABLE_FORTRAN OR BUILD_SHARED_LIBS)
  if(NOT MSVC)
    list(APPEND GA_EXTRA_LIBS m)
  endif()
endif()
if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
  if(NOT MSVC)
    list(APPEND GA_EXTRA_LIBS rt)
  endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fno-aggressive-loop-optimizations)
endif()

if(ENABLE_BLAS)
  list(APPEND GA_EXTRA_LIBS ${linalg_lib})
  add_compile_options(${BLAS_COMPILE_OPTIONS})
  #include_directories(${BLAS_INCLUDE_DIRS})
endif()

if(ENABLE_PROFILING)
  set(ga_profiler $<TARGET_OBJECTS:ga_tools>)
endif()

add_library(ga
  $<TARGET_OBJECTS:ma>
  # $<TARGET_OBJECTS:dra>
  # $<TARGET_OBJECTS:eaf>
  # $<TARGET_OBJECTS:elio>
  # $<TARGET_OBJECTS:sf>
  $<TARGET_OBJECTS:gaf2c>
  $<TARGET_OBJECTS:armci_comex>
  $<TARGET_OBJECTS:ga_src>
  $<TARGET_OBJECTS:linalg>
  ${ga_profiler}
)
add_library(GlobalArrays::ga ALIAS ga)

set (CMAKE_REQUIRED_LIBRARIES lapack blas)
target_link_libraries(ga PUBLIC ${GA_EXTRA_LIBS})
target_include_directories(ga
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/ga>
)
if(ENABLE_BLAS)
  target_include_directories(ga
    INTERFACE
      $<INSTALL_INTERFACE:${BLAS_INCLUDE_DIRS}>
  )
  target_compile_options(ga
    INTERFACE
      $<INSTALL_INTERFACE:${BLAS_COMPILE_OPTIONS}>
  )
endif()

if(DEFINED GA_GCC_TOOLCHAIN_FLAG)
  target_compile_options(ga PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:${GA_GCC_TOOLCHAIN_FLAG}>)
endif()

# -------------------------------------------------------------
# Subdirectories
# -------------------------------------------------------------
add_subdirectory(global/src)
add_subdirectory(comex)
add_subdirectory(ga++/src)
add_subdirectory(ma)
# add_subdirectory(pario)
add_subdirectory(gaf2c)
add_subdirectory(LinAlg/lapack+blas)

if(ENABLE_PROFILING)
  add_subdirectory(tools)
endif()

if(ENABLE_TESTS)
  # turn testing on
  enable_testing()
  add_subdirectory(global/testing)
endif()


# --------------------------------------
# Global Arrays installation
# --------------------------------------

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/GlobalArrays)

#dra eaf elio sf
install(TARGETS ma ga_src linalg
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS ga
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

#Export the targets to a script
install(EXPORT globalarrays-targets
    FILE
        globalarrays-targets.cmake
    NAMESPACE
        GlobalArrays::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/globalarrays-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the config, configversion files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

if(ENABLE_BLAS)
  install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindBLAS.cmake
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindLAPACK.cmake
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindStandardFortran.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
  )

  install(DIRECTORY
    ${CMAKE_CURRENT_LIST_DIR}/cmake/linalg-modules
    DESTINATION ${INSTALL_CONFIGDIR}
  )

  if(ENABLE_DPCPP)
    install(FILES
      ${CMAKE_CURRENT_LIST_DIR}/cmake/FindIntelSYCL.cmake
      DESTINATION ${INSTALL_CONFIGDIR}
    )
  endif()
endif()

export(EXPORT globalarrays-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/globalarrays-targets.cmake
    NAMESPACE GlobalArrays::)

