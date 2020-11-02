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
  ga++.h
  GAServices.h
  GlobalArray.h
  init_term.h
  PGroup.h
  services.h
)

install (FILES
  ${GAXX_HEADERS}
  DESTINATION include/ga
)

# -------------------------------------------------------------
# GA++ library installation
# -------------------------------------------------------------

add_library(ga++
  GAServices.cc
  GlobalArray.cc
  init_term.cc
  overload.cc
  PGroup.cc
  services.cc
)

target_include_directories(ga++ BEFORE PRIVATE 
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${PROJECT_SOURCE_DIR}/global/src
                    ${PROJECT_SOURCE_DIR}/global/testing
                    ${PROJECT_SOURCE_DIR}/ma
                    ${PROJECT_BINARY_DIR}/ma
                    ${PROJECT_BINARY_DIR}/gaf2c
                    ${PROJECT_BINARY_DIR})

target_link_libraries(ga++)

install (TARGETS
  ga++
  DESTINATION lib
)
