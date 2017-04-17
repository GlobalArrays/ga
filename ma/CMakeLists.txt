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

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/gaf2c
)

if (NOT MSVC)
set(ctargetlibs ${MPI_C_LIBRARIES} m)
else()
set(ctargetlibs ${MPI_C_LIBRARIES})
endif()

# -------------------------------------------------------------
# MA header installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(FORTRAN_HEADERS maf2c.fh mafdecls.fh)
endif()

set(MA_HEADERS
  ma.h
#  error.h
  macdecls.h
  macommon.h
  matypes.h
#  memcpy.h
  scope.h
  string-util.h
  table.h
  ${FORTRAN_HEADERS}
)

install (FILES
  ${MA_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# MA library installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(FORTRAN_FILES f2c.c maf.F)
endif()

add_library(ma OBJECT
  ma.c
  error.c
  string-util.c
  table.c
  ${FORTRAN_FILES}
)

if (ENABLE_FORTRAN)
  ADD_DEPENDENCIES(ga_src GenerateConfigFH)
endif()

# -------------------------------------------------------------
# Build MA tests
# -------------------------------------------------------------

add_executable(testf.x testf.F)
add_executable(test-coalesce.x test-coalesce.c)
add_executable(test-inquire.x test-inquire.c)
#add_executable(testc.x testc.c)
target_link_libraries(testf.x ga ${ctargetlibs})
target_link_libraries(test-coalesce.x ga ${ctargetlibs})
target_link_libraries(test-inquire.x ga ${ctargetlibs})
#target_link_libraries(testc.x ga ${ctargetlibs})
