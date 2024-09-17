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

# -------------------------------------------------------------
# GA++ header installation
# -------------------------------------------------------------

set(GAXX_HEADERS
  src/ga++.h
  src/GAServices.h
  src/GlobalArray.h
  src/init_term.h
  src/PGroup.h
  src/services.h
)

install (FILES
  ${GAXX_HEADERS}
  DESTINATION include/ga
)

list (APPEND GA_HEADER_PATHS ${CMAKE_CURRENT_LIST_DIR}/src)
set (GA_HEADER_PATHS ${GA_HEADER_PATHS} PARENT_SCOPE)

# -------------------------------------------------------------
# GA++ library installation
# -------------------------------------------------------------

add_library(ga++
  src/GAServices.cc
  src/GlobalArray.cc
  src/init_term.cc
  src/overload.cc
  src/PGroup.cc
  src/services.cc
)

set (_gapp_inc_dirs ${CMAKE_CURRENT_BINARY_DIR}
                   ${PROJECT_SOURCE_DIR}/ga++/src
                   ${PROJECT_SOURCE_DIR}/global/src
                   ${PROJECT_SOURCE_DIR}/global/testing
                   ${PROJECT_SOURCE_DIR}/ma
                   ${PROJECT_BINARY_DIR}/ma
                   ${PROJECT_BINARY_DIR}/gaf2c
                   ${PROJECT_BINARY_DIR})

target_include_directories(ga++ BEFORE PRIVATE ${_gapp_inc_dirs})

target_link_libraries(ga++)

install (TARGETS
  ga++
  DESTINATION lib
)

function(gapp_add_parallel_test test_name test_srcs)
  get_filename_component(_test_name_only "${test_name}" NAME)
  ga_add_parallel_test(${test_name} ${test_srcs})
  target_include_directories(${_test_name_only}.x PRIVATE ${_gapp_inc_dirs})
  target_link_libraries(${_test_name_only}.x ga++)  
endfunction()

if(ENABLE_TESTS)
  gapp_add_parallel_test(ga++/elempatch_cpp ${PROJECT_SOURCE_DIR}/ga++/testing/elempatch.cc)
  gapp_add_parallel_test(ga++/mtest_cpp  ${PROJECT_SOURCE_DIR}/ga++/testing/mtest.cc)
  gapp_add_parallel_test(ga++/ntestc_cpp ${PROJECT_SOURCE_DIR}/ga++/testing/ntestc.cc)
  gapp_add_parallel_test(ga++/testc_cpp ${PROJECT_SOURCE_DIR}/ga++/testing/testc.cc)
  gapp_add_parallel_test(ga++/testmult_cpp ${PROJECT_SOURCE_DIR}/ga++/testing/testmult.cc)
  gapp_add_parallel_test(ga++/thread-safe_cpp ${PROJECT_SOURCE_DIR}/ga++/testing/thread-safe.cc)
endif()
