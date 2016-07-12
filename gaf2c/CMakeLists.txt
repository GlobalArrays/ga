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
                           ${PROJECT_SOURCE_DIR}/comex/src-armci
)

# -------------------------------------------------------------
# GAF2C header installation
# -------------------------------------------------------------

set(GAF2C_HEADERS
  typesf2c.h
)

install (FILES
  ${GAF2C_HEADERS}
  DESTINATION include
)

# -------------------------------------------------------------
# GAF2C library installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(FORTRAN_FILES f2c.c)
endif()

add_library(gaf2c OBJECT
  gaf2c.c
  testarg.c
  drand.c
  ${FORTRAN_FILES}
)
