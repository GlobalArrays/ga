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

include_directories(AFTER src-common)
if (MPI_TS)
  set(COMEX_DEVICE
    src-mpi/comex.c
    src-mpi/groups.c
  )
  include_directories(AFTER src-mpi)
elseif (MPI_PR)
  set(COMEX_DEVICE
    src-mpi-pr/comex.c
    src-mpi-pr/groups.c
    src-mpi-pr/reg_cache.c
  )
  include_directories(AFTER src-mpi-pr)
elseif (MPI3)
  set(COMEX_DEVICE
    src-mpi3/comex.c
    src-mpi3/groups.c
    src-mpi3/reg_win.c
  )
  include_directories(AFTER src-mpi3)
endif()

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
  DESTINATION include
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

add_library(armci
  ${ARMCI_FILES}
  ${COMEX_DEVICE}
)

target_link_libraries(armci)

install (TARGETS
  armci
  DESTINATION lib
)

add_library(comex
 ${COMEX_DEVICE}
)

target_link_libraries(comex)

install (TARGETS
  comex
  DESTINATION lib
)
