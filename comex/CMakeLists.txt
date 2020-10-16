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

option (COMEX_NETWORK_MPI_TS "use MPI 2-sided protocol for communication" OFF)
option (COMEX_NETWORK_MPI_PR "use MPI progress ranks protocol for communication" OFF)
option (COMEX_NETWORK_MPI3 "use MPI RMA protocols for communication" OFF)
option (COMEX_NETWORK_MPI_MT "use MPI multi-threading protocol for communication" OFF)
option (COMEX_NETWORK_MPI_PT "use MPI progress threads protocol for communication" OFF)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR} src-common)
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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(comex-utils)

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/config.h )

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
  DESTINATION include/ga
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
target_compile_definitions(armci_comex PRIVATE HAVE_CONFIG_H)

add_library(armci
  ${ARMCI_FILES}
  ${COMEX_DEVICE}
)
target_compile_definitions(armci PRIVATE HAVE_CONFIG_H)

add_library(GlobalArrays::armci ALIAS armci)

target_link_libraries(armci)

install(TARGETS armci
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_library(comex
 ${COMEX_DEVICE}
)

add_library(GlobalArrays::comex ALIAS comex)

target_link_libraries(comex)

install(TARGETS comex
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
