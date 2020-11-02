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

if (ENABLE_FORTRAN)
  set(GA_FORTRAN_INTERFACE_C_FILES
    fapi.c
  )
  set(GA_FORTRAN_INTERFACE_H_FILES
    cnames.h
    ${CMAKE_CURRENT_BINARY_DIR}/global.fh
  )
  set (GA_FORTRAN_INTERFACE_F_FILES
    complex.F
  )
endif()

if (ENABLE_SCALAPACK)
  set (GA_FORTRAN_INTERFACE_H_FILES
       ${GA_FORTRAN_INTERFACE_H_FILES} scalapack.fh)
  set (GA_FORTRAN_INTERFACE_F_FILES
       ${GA_FORTRAN_INTERFACE_F_FILES} scalapack.F)
  set (GA_C_SCALAPACK_WRAPPER ga_diag_seqc.c)
endif()

if (ENABLE_FORTRAN)
  #if (HAVE_LAPACK OR ENABLE_EISPACK)
  set (GA_FORTRAN_INTERFACE_F_FILES
        ${GA_FORTRAN_INTERFACE_F_FILES} ga_diag_seq.F)
  #endif()
endif()

set(target_libraries
    ma
    elio
    dra
    eaf
    sf)

set(WAPI_FILES
  ga-wapi.h
  ga-wapidefs.h
)

set(GA_FILES
  base.c
  onesided.c
  collect.c
  ghosts.c
  capi.c
  fapi.c
  datatypes.c
  decomp.c
  DP.c
  elem_alg.c
  ga_diag_seqc.c
  ga_malloc.c
  ga_profile.c
  ga_solve_seq.c
  ga_symmetr.c
  ga_trace.c
  global.nalg.c
  global.npatch.c
  global.periodic.c
  global.util.c
  hsort.scat.c
  iterator.c
  matmul.c
  matrix.c
  nbutil.c
  peigstubs.c
  sclstubs.c
  select.c
  sparse.c
  ${GA_FORTRAN_INTERFACE_C_FILES}
  ${GA_FORTRAN_INTERFACE_F_FILES}
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ga-wapi.h
  COMMAND ${CMAKE_COMMAND} -D INPUT:PATH="${CMAKE_CURRENT_SOURCE_DIR}/ga-papi.h" -D OUTPUT:PATH="${CMAKE_CURRENT_BINARY_DIR}/ga-wapi.h" -P ${PROJECT_SOURCE_DIR}/tools/ga_papi_to_wapi.cmake
  DEPENDS ga-papi.h
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ga-wapidefs.h
  COMMAND ${CMAKE_COMMAND} -D INPUT:PATH="${CMAKE_CURRENT_SOURCE_DIR}/ga-papi.h" -D OUTPUT:PATH="${CMAKE_CURRENT_BINARY_DIR}/ga-wapidefs.h" -P ${PROJECT_SOURCE_DIR}/tools/ga_papi_to_wapidefs.cmake
  DEPENDS ga-papi.h
)

add_library(ga_src OBJECT
  ${CMAKE_CURRENT_BINARY_DIR}/ga-wapi.h
  ${CMAKE_CURRENT_BINARY_DIR}/ga-wapidefs.h
  ${GA_FILES}
)

target_include_directories(ga_src BEFORE PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${PROJECT_BINARY_DIR}/gaf2c
                    ${PROJECT_SOURCE_DIR}/ma
                    ${PROJECT_BINARY_DIR}/ma
                    ${PROJECT_SOURCE_DIR}/comex/src-armci
                    #${PROJECT_SOURCE_DIR}/tcgmsg
                    ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${PROJECT_BINARY_DIR}
)

if(ENABLE_FORTRAN)
  ADD_DEPENDENCIES(ga_src GenerateConfigFH)
endif()

# -------------------------------------------------------------
# Global Arrays header installation
# -------------------------------------------------------------
install(FILES
#  abstract_ops.h
#  base.h
#  cnames.h
  gacommon.h
#  gaconfig.h
  ga.h
  ga-mpi.h
  ga-papi.h
#  ga_profile.h
#  ga-wapidefs.h
  ${CMAKE_CURRENT_BINARY_DIR}/ga-wapi.h
#  global.h
#  globalp.h
#  matmul.h
  ${GA_FORTRAN_INTERFACE_H_FILES}
  DESTINATION include/ga
)
