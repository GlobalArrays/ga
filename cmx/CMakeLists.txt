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

cmake_minimum_required(VERSION 3.8)

project(CMX_Runtime)

find_package(MPI)

include_directories(AFTER ${MPI_INCLUDE_PATH})

option (CMX_NETWORK_MPI_PR "use MPI progress ranks protocol for communication" OFF)
#option (CMX_NETWORK_MPI_RMA "use MPI RMA protocols for communication" OFF)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR} src-common)
option (MPI_PR "use MPI progress ranks protocol for communication" OFF)
message(STATUS "MPI_PR: ${MPI_PR}")
if (MPI_PR)
  message(STATUS "Setting parameters for Progress Ranks runtime")
  set(CMX_DEVICE
    src-mpi-pr/p_group.cpp
    src-mpi-pr/shmem.cpp
    src-mpi-pr/node_config.cpp
    src-mpi-pr/reg_cache.cpp
    src-mpi-pr/p_environment.cpp
    src-mpi-pr/p_alloc.cpp
  )
  set (CMX_NETWORK_MPI_PR ON)
  include_directories(AFTER src-mpi-pr)
#elseif (MPI_RMA)
#  set(CMX_DEVICE
#    src-mpi3/comex.c
#    src-mpi3/groups.c
#    src-mpi3/reg_win.c
#  )
#  set (CMX_NETWORK_MPI_RMA ON)
#  include_directories(AFTER src-mpi3)
endif()

# CMX interface
set(CMX_INTERFACE
    src-common/group.cpp
    src-common/environment.cpp
    src-common/alloc.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------------------------------------------------------------
#  Check for RT library
# -------------------------------------------------------------
find_library(LIBRT rt PATHS /lib /usr/lib /usr/local/lib)

#include(comex-utils)

#CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
#                ${CMAKE_CURRENT_BINARY_DIR}/config.h )

# -------------------------------------------------------------
# CMX header installation
# -------------------------------------------------------------

set(CMX_DEVICE_HEADERS
  src-common/cmx.hpp
)

install (FILES
  ${CMX_DEVICE_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# CMX library installation
# -------------------------------------------------------------

add_library(cmx
 ${CMX_DEVICE}
 ${CMX_INTERFACE}
)

#add_library(GlobalArrays::cmx ALIAS cmx)

target_link_libraries(cmx ${MPI_CXX_LIBRARIES} ${LIBRT})

install(TARGETS cmx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# -------------------------------------------------------------
# CMX test programs
# -------------------------------------------------------------
add_executable(unit_env.x testing/unit_env.cpp)
target_link_libraries(unit_env.x cmx)
add_executable(unit_group.x testing/unit_group.cpp)
target_link_libraries(unit_group.x cmx)
add_executable(unit_alloc.x testing/unit_alloc.cpp)
target_link_libraries(unit_alloc.x cmx)
add_executable(unit_contig_put.x testing/unit_contig_put.cpp)
target_link_libraries(unit_contig_put.x cmx)
add_executable(unit_contig_get.x testing/unit_contig_get.cpp)
target_link_libraries(unit_contig_get.x cmx)
add_executable(unit_contig_acc.x testing/unit_contig_acc.cpp)
target_link_libraries(unit_contig_acc.x cmx)
add_executable(unit_strided_put.x testing/unit_strided_put.cpp)
target_link_libraries(unit_strided_put.x cmx)
