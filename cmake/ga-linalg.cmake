#
# module: GlobalArrays.cmake
# author: Bruce Palmer
# description: Define utility functions.
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


function(ga_set_blasroot __blasvendor __blasvar)
  if("${LINALG_VENDOR}" STREQUAL "${__blasvendor}")
    set(__ebv_exists FALSE)
    ga_path_exists(${__blasvar} __bv_exists)
    if (DEFINED ENV{${__blasvar}})
      set(__eblasvar $ENV{${__blasvar}})
      ga_path_exists(__eblasvar __ebv_exists)
      if(__ebv_exists)
        set(${__blasvar} ${__eblasvar} PARENT_SCOPE)
      endif()
    endif()
    if(NOT __bv_exists AND NOT __ebv_exists)
      message(FATAL_ERROR "Could not find the following ${__blasvar} path: ${__eblasvar} ${${__blasvar}}")
    endif()
  endif()
endfunction()

if( "sycl" IN_LIST LINALG_OPTIONAL_COMPONENTS )
  set(ENABLE_DPCPP ON)
elseif(ENABLE_DPCPP)
  list(APPEND LINALG_OPTIONAL_COMPONENTS "sycl")
endif()

function(check_ga_blas_options)
  # ga_is_valid(${LINALG_VENDOR}    _lav_set)
  ga_is_valid(LINALG_PREFIX     _lap_set)
  ga_is_valid(BLAS_PREFIX       _lbp_set)
  ga_is_valid(LAPACK_PREFIX     _llp_set)
  ga_is_valid(ScaLAPACK_PREFIX  _lsp_set)
  if(NOT (_lap_set OR _lbp_set OR _llp_set OR _lsp_set) )
    message(FATAL_ERROR "ENABLE_BLAS=ON but the options \
    to specify the root of the LinAlg libraries installation \
    are not set. Please refer to README.md")
  endif()
endfunction()

#Check if provided paths are valid and export
if (ENABLE_BLAS)
  check_ga_blas_options()
  ga_path_exists(LINALG_PREFIX __la_exists)
  if(NOT __la_exists)
    message(FATAL_ERROR "Could not find the following ${LINALG_VENDOR} installation path at: ${LINALG_PREFIX}")
  endif()
endif()

if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64" AND "${LINALG_VENDOR}" STREQUAL "IntelMKL")
  message( FATAL_ERROR "IntelMKL is not supported for ARM architectures" )
endif()

# check for numerical libraries. These should set variables BLAS_FOUND and
# LAPACK_FOUND
set(GA_BLAS_ILP64 OFF)
if (ENABLE_BLAS)
    set(BLAS_PREFERENCE_LIST      ${LINALG_VENDOR})
    set(LAPACK_PREFERENCE_LIST    ${LINALG_VENDOR})
    set(ScaLAPACK_PREFERENCE_LIST ${LINALG_VENDOR})

    set(LINALG_PREFER_STATIC ON)
    if(BUILD_SHARED_LIBS)
      set(LINALG_PREFER_STATIC OFF)
    endif()
    
    if(ENABLE_DPCPP)
      set(LINALG_THREAD_LAYER "sequential")
    endif()
    
    set(${LINALG_VENDOR}_PREFERS_STATIC    ${LINALG_PREFER_STATIC})
    set(ReferenceLAPACK_PREFERS_STATIC     ${LINALG_PREFER_STATIC})
    set(ReferenceScaLAPACK_PREFERS_STATIC  ${LINALG_PREFER_STATIC})

    set(BLAS_SIZE 4)
    set(${LINALG_VENDOR}_THREAD_LAYER  ${LINALG_THREAD_LAYER})
    set(BLAS_REQUIRED_COMPONENTS       ${LINALG_REQUIRED_COMPONENTS})
    set(LAPACK_REQUIRED_COMPONENTS     ${LINALG_REQUIRED_COMPONENTS})
    set(ScaLAPACK_REQUIRED_COMPONENTS  ${LINALG_REQUIRED_COMPONENTS})
    set(BLAS_OPTIONAL_COMPONENTS       ${LINALG_OPTIONAL_COMPONENTS})
    set(LAPACK_OPTIONAL_COMPONENTS     ${LINALG_OPTIONAL_COMPONENTS})
    set(ScaLAPACK_OPTIONAL_COMPONENTS  ${LINALG_OPTIONAL_COMPONENTS})    

    set(use_openmp ON)
    set(_blis_essl_set OFF)
    
    if(${LINALG_VENDOR} MATCHES "BLIS" OR ${LINALG_VENDOR} MATCHES "IBMESSL")
      set(_blis_essl_set ON)
    endif()

    if("sequential" IN_LIST LINALG_THREAD_LAYER OR _blis_essl_set)
      set(use_openmp OFF)
    endif()

    if(_blis_essl_set OR ${LINALG_VENDOR} MATCHES "OpenBLAS")
      set(use_openmp OFF)
      # if(_blis_essl_set) #Assume openblas does not have lapack
        set(LAPACK_PREFERENCE_LIST ReferenceLAPACK)
      # endif()
      if(ENABLE_SCALAPACK)
        set(ScaLAPACK_PREFERENCE_LIST ReferenceScaLAPACK)
      endif()
    endif()

    if( "ilp64" IN_LIST LINALG_REQUIRED_COMPONENTS )
      set(BLAS_SIZE 8)
      set(GA_BLAS_ILP64 ON)
    endif()

    if(NOT BLAS_PREFIX)
      set(BLAS_PREFIX ${LINALG_PREFIX})
    endif()
    if(NOT LAPACK_PREFIX)
      set(LAPACK_PREFIX ${LINALG_PREFIX})
    endif()
    if(NOT ScaLAPACK_PREFIX)
      set(ScaLAPACK_PREFIX ${LINALG_PREFIX})
    endif()

    if(ENABLE_SCALAPACK)
      find_package(ScaLAPACK)
      if (ScaLAPACK_FOUND)
        set(HAVE_SCALAPACK 1)
      else()
        message(FATAL_ERROR "ENABLE_SCALAPACK=ON, but a ScaLAPACK library was not found")
      endif()
    endif()

    find_package(LAPACK)
    if (LAPACK_FOUND)
      set(HAVE_LAPACK 1)
    else()
      message(FATAL_ERROR "ENABLE_BLAS=ON, but a LAPACK library was not found")
    endif()

    find_package(BLAS)
    if (BLAS_FOUND)
      set(HAVE_BLAS 1)
    else()
      message(FATAL_ERROR "ENABLE_BLAS=ON, but a BLAS library was not found")
    endif()

  if(ENABLE_CXX)
    set(BPP_GIT_TAG b6c90653cb941fccc7b6905e3919d7cf0cb917a1)
    set(LPP_GIT_TAG 95cc9a5f72e54b76ee32f76bf67fc3c2e7399b06)
    set(SPP_GIT_TAG 6397f52cf11c0dfd82a79698ee198a2fce515d81)
    if(ENABLE_DEV_MODE)
      set(BPP_GIT_TAG master)
      set(LPP_GIT_TAG master)
      set(SPP_GIT_TAG master)
    endif()
    include(FetchContent)
    set( gpu_backend "none" CACHE STRING "GPU backend to use" FORCE)
    if(NOT TARGET blaspp)
      if(ENABLE_OFFLINE_BUILD)
      FetchContent_Declare(
        blaspp
        URL ${DEPS_LOCAL_PATH}/blaspp
      )
      else()
      #set(BUILD_SHARED_LIBS ON CACHE BOOL "Build SHARED libraries" FORCE)
      FetchContent_Declare(
        blaspp
        GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
        GIT_TAG ${BPP_GIT_TAG}
      )
      endif()
      FetchContent_MakeAvailable( blaspp )
    endif()

    if(NOT TARGET lapackpp)
    if(ENABLE_OFFLINE_BUILD)
      FetchContent_Declare(
        lapackpp
        URL ${DEPS_LOCAL_PATH}/lapackpp
      )
      else()
      FetchContent_Declare(
        lapackpp
        GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp.git
        GIT_TAG ${LPP_GIT_TAG}
      )      
      endif()
      FetchContent_MakeAvailable( lapackpp )
    endif()

    if(ENABLE_SCALAPACK)
      if(NOT TARGET scalapackpp::scalapackpp)
        if(ENABLE_OFFLINE_BUILD)
        FetchContent_Declare(
          scalapackpp
          URL ${DEPS_LOCAL_PATH}/scalapackpp
        )
        else()
        FetchContent_Declare(
          scalapackpp
          GIT_REPOSITORY https://github.com/wavefunction91/scalapackpp.git
          GIT_TAG ${SPP_GIT_TAG}
        )
        endif()
        FetchContent_MakeAvailable( scalapackpp )
      endif()
    endif()

    set(_la_cxx_blas blaspp)
    set(_la_cxx_lapack lapackpp)
    set(_la_cxx_scalapack scalapackpp::scalapackpp)
  endif()

else()
    set(HAVE_BLAS 0)
    set(HAVE_LAPACK 0)
    set(HAVE_SCALAPACK 0)
endif()

if(ENABLE_DPCPP)
  set(USE_DPCPP ON)
endif()

if (ENABLE_SCALAPACK)
  set(SCALAPACK_I8 OFF)
  if( "ilp64" IN_LIST LINALG_REQUIRED_COMPONENTS )
    set(SCALAPACK_I8 ON)
  endif()

  # add_definitions(-DHAVE_SCALAPACK)
  if (SCALAPACK_I8)
    add_definitions(-DSCALAPACK_I8)
  endif()
endif()
if (ENABLE_EISPACK)
  add_definitions(-DENABLE_EISPACK)
endif()
# if (ENABLE_FORTRAN)
#   add_definitions(-DENABLE_F77)
# endif()

message(STATUS "HAVE_BLAS: ${HAVE_BLAS}")
message(STATUS "HAVE_LAPACK: ${HAVE_LAPACK}")
message(STATUS "HAVE_SCALAPACK: ${HAVE_SCALAPACK}")

set(linalg_lib )

if (HAVE_BLAS)
  if("${LINALG_VENDOR}" STREQUAL "IntelMKL")
    set(BLA_VENDOR_MKL ON )
    set(BLA_LAPACK_INT       "MKL_INT" )
    set(BLA_LAPACK_COMPLEX8  "MKL_Complex8" )
    set(BLA_LAPACK_COMPLEX16 "MKL_Complex16" )
  elseif("${LINALG_VENDOR}" STREQUAL "IBMESSL")
    set(BLA_VENDOR_ESSL ON)
    set(BLA_LAPACK_INT "int32_t")
    if(GA_BLAS_ILP64)
      set(BLA_LAPACK_INT "int64_t")
    endif()
    set(BLA_LAPACK_COMPLEX8  "std::complex<float>")
    set(BLA_LAPACK_COMPLEX16 "std::complex<double>")
  elseif("${LINALG_VENDOR}" STREQUAL "BLIS")
    set(USE_BLIS ON)
    set(BLA_VENDOR_BLIS ON)
    set(BLA_LAPACK_INT "int32_t")
    if(GA_BLAS_ILP64)
      set(BLA_LAPACK_INT "int64_t")
    endif()
    set(BLA_LAPACK_COMPLEX8  "std::complex<float>")
    set(BLA_LAPACK_COMPLEX16 "std::complex<double>")
  endif()

  CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ga_linalg.h.in
                  ${CMAKE_CURRENT_BINARY_DIR}/ga_linalg.h )
                  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ga_linalg.h
                  DESTINATION include/ga
  )

  list(APPEND linalg_lib BLAS::BLAS ${_la_cxx_blas})
  message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
endif()

if (HAVE_LAPACK)
  list(APPEND linalg_lib LAPACK::LAPACK ${_la_cxx_lapack})
  message(STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
endif()

if (HAVE_SCALAPACK)
  list(APPEND linalg_lib ScaLAPACK::ScaLAPACK ${_la_cxx_scalapack})
  message(STATUS "ScaLAPACK_LIBRARIES: ${ScaLAPACK_LIBRARIES}")
endif()
