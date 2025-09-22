#
# module: CMakeLists.txt
# author: Bruce Palmer
# description: CMake build for XGA. Only MPI-based runtimes are supported.
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

cmake_minimum_required(VERSION 3.8)

project(XGA)

find_package(MPI)

set(CMAKE_CXX_STANDARD 17)

include_directories(AFTER ${MPI_INCLUDE_PATH})

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR} 
      ${PROJECT_SOURCE_DIR}/../xga/src
      ${PROJECT_SOURCE_DIR}/../cmx/src-common
      ${PROJECT_SOURCE_DIR}/../cmx/src-mpi-pr
)

option (MPI_PR "use MPI progress ranks protocol for communication" OFF)
message(STATUS "MPI_PR: ${MPI_PR}")
if (MPI_PR)
  message(STATUS "Setting parameters for Progress Ranks runtime")
  set(XGA_BASE
    src/xga_group.cpp
    src/xga_environment.cpp
    src/block_iterator.cpp
    src/xga_base.cpp
    src/xga_onesided.cpp
    src/decomp.cpp
  )
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------------------------------------------------------------
#  Check for RT library
# -------------------------------------------------------------
find_library(LIBRT rt PATHS /lib /usr/lib /usr/local/lib)

#include(comex-utils)

#CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
#                ${CMAKE_CURRENT_BINARY_DIR}/config.h )

# -------------------------------------------------------------
# XGA header installation
# -------------------------------------------------------------

add_library(xga
 ${XGA_BASE}
)

set(XGA_HEADERS
  src/xga_groups.hpp
  src/xga_environment.hpp
  src/xga_interface.hpp
  src/xga_private.hpp
  src/xga_macros.hpp
  src/xga_types.hpp
)

install (FILES
  ${XGA_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# XGA library installation
# -------------------------------------------------------------

#add_library(GlobalArrays::xga ALIAS xga)

target_link_libraries(xga ${MPI_CXX_LIBRARIES} ${LIBRT})

install(TARGETS cmx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# -------------------------------------------------------------
# XGA tests
# -------------------------------------------------------------

add_executable(env_unit_test.x testing/env_unit_test.cpp)
target_link_libraries(env_unit_test.x xga cmx)
add_executable(group_unit_test.x testing/group_unit_test.cpp)
target_link_libraries(group_unit_test.x xga cmx)
add_executable(put_unit_test.x testing/put_unit_test.cpp)
target_link_libraries(put_unit_test.x xga cmx)
add_executable(get_unit_test.x testing/get_unit_test.cpp)
target_link_libraries(get_unit_test.x xga cmx)
add_executable(acc_unit_test.x testing/acc_unit_test.cpp)
target_link_libraries(acc_unit_test.x xga cmx)
add_executable(scatter_unit_test.x testing/scatter_unit_test.cpp)
target_link_libraries(scatter_unit_test.x xga cmx)
add_executable(gather_unit_test.x testing/gather_unit_test.cpp)
target_link_libraries(gather_unit_test.x xga cmx)
add_executable(scatter_acc_unit_test.x testing/scatter_acc_unit_test.cpp)
target_link_libraries(scatter_acc_unit_test.x xga cmx)
add_executable(zero_unit_test.x testing/zero_unit_test.cpp)
target_link_libraries(zero_unit_test.x xga cmx)
add_executable(fill_unit_test.x testing/fill_unit_test.cpp)
target_link_libraries(fill_unit_test.x xga cmx)
add_executable(read_inc_unit_test.x testing/read_inc_unit_test.cpp)
target_link_libraries(read_inc_unit_test.x xga cmx)
