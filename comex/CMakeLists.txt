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

# Allow selecting an alternate runtime. When GA_RUNTIME is set to OPEN_SHMEM
# the src-oshmem backend will be used instead of the MPI-based runtime.
set(GA_RUNTIME "MPI" CACHE STRING "Runtime to build: MPI (default) or OPEN_SHMEM")
set_property(CACHE GA_RUNTIME PROPERTY STRINGS MPI OPEN_SHMEM)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR} src-common)
if (GA_RUNTIME STREQUAL "OPEN_SHMEM")
  set(COMEX_DEVICE
    src-oshmem/comex.c
    src-oshmem/groups.c
    src-oshmem/nb.c
  )
  set (COMEX_NETWORK_OSHMEM ON)
  include_directories(AFTER src-oshmem)
  # Allow user to point to an OpenSHMEM installation root. If provided,
  # we'll search under that prefix for headers and libraries.
  option(OSHMEM_ROOT "Path to OpenSHMEM installation root" "")

  # Find headers (shmem.h or openshmem.h) and library (libshmem / libopenshmem).
  if (OSHMEM_ROOT)
    set(_oshmem_hints ${OSHMEM_ROOT}/include ${OSHMEM_ROOT}/usr/include)
    set(_oshmem_lib_hints ${OSHMEM_ROOT}/lib ${OSHMEM_ROOT}/lib64 ${OSHMEM_ROOT}/usr/lib ${OSHMEM_ROOT}/usr/lib64)
  else()
    set(_oshmem_hints /usr/include /usr/local/include)
    set(_oshmem_lib_hints /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/openmpi/lib)
  endif()

  find_path(SHMEM_INCLUDE
    NAMES shmem.h openshmem.h
    HINTS ${_oshmem_hints}
  )

  find_library(SHMEM_LIB
    NAMES oshmem shmem openshmem liboshmem libshmem
    HINTS ${_oshmem_lib_hints}
  )

  if (SHMEM_INCLUDE)
    include_directories(AFTER ${SHMEM_INCLUDE})
    message(STATUS "OpenSHMEM headers found: ${SHMEM_INCLUDE}")
  else()
    message(STATUS "OpenSHMEM headers NOT found; continuing but build may fail until OSHMEM_ROOT is set")
  endif()

  if (NOT SHMEM_LIB)
    message(STATUS "OpenSHMEM library not found by CMake; will try linker name 'oshmem' then 'shmem' at link time")
    # prefer 'oshmem' as the linker name for Open MPI's implementation
    set(SHMEM_LIB oshmem)
  else()
    message(STATUS "OpenSHMEM library found: ${SHMEM_LIB}")
  endif()
  # Export the discovered library to the parent scope so top-level targets
  # (e.g., the 'ga' library) can link against OpenSHMEM.
  set(SHMEM_LIB ${SHMEM_LIB} CACHE STRING "OpenSHMEM library")
  # Append to GA_EXTRA_LIBS in the parent scope so 'ga' picks it up
  set(GA_EXTRA_LIBS ${GA_EXTRA_LIBS} ${SHMEM_LIB} PARENT_SCOPE)
elseif (MPI_TS)
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

list (APPEND GA_HEADER_PATHS ${CMAKE_CURRENT_LIST_DIR}/src-armci ${CMAKE_CURRENT_LIST_DIR}/src-common)
set (GA_HEADER_PATHS ${GA_HEADER_PATHS} PARENT_SCOPE)

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
target_include_directories(armci_comex PRIVATE ${MPI_C_INCLUDE_DIRS})

add_library(armci
  ${ARMCI_FILES}
  ${COMEX_DEVICE}
)
target_compile_definitions(armci PRIVATE HAVE_CONFIG_H)

add_library(GlobalArrays::armci ALIAS armci)

target_include_directories(armci PRIVATE ${MPI_C_INCLUDE_DIRS})
if (GA_RUNTIME STREQUAL "OPEN_SHMEM")
  # Link OpenSHMEM into armci & comex libraries
  target_link_libraries(armci PRIVATE ${SHMEM_LIB})
endif()

if (GA_RUNTIME STREQUAL "OPEN_SHMEM")
  # Also link the top-level 'ga' target if it exists so the final
  # executable link commands get the SHMEM library. If 'ga' is not yet
  # defined, we already appended SHMEM_LIB to GA_EXTRA_LIBS above.
  if (TARGET ga)
    target_link_libraries(ga PRIVATE ${SHMEM_LIB})
  endif()
endif()

install(TARGETS armci
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_library(comex
 ${COMEX_DEVICE}
)

add_library(GlobalArrays::comex ALIAS comex)

target_include_directories(comex PRIVATE ${MPI_C_INCLUDE_DIRS})
if (GA_RUNTIME STREQUAL "OPEN_SHMEM")
  target_link_libraries(comex PRIVATE ${SHMEM_LIB})
endif()

install(TARGETS comex
    EXPORT globalarrays-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
