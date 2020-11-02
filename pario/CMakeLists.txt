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

# -------------------------------------------------------------
# PARIO header installation
# -------------------------------------------------------------

if (ENABLE_FORTRAN)
  set(DRA_FORTRAN_HEADERS
      dra/dra.fh
      dra/draf2c.h)
  set(DRA_FORTRAN_FILES
      dra/fortran.c)
  set(EAF_FORTRAN_HEADERS
      eaf/eaf.fh)
  set(EAF_FORTRAN_FILES
      eaf/eaf_f2c.c)
  set(SF_FORTRAN_HEADERS
      sf/sf.fh
      sf/sff2c.h)
  set(SF_FORTRAN_FILES
      sf/sf_fortran.c)
endif()

set(DRA_HEADERS
  dra/dra.h
#  dra/buffers.h
#  dra/drap.h
  ${DRA_FORTRAN_HEADERS}
)

set(EAF_HEADERS
  eaf/eaf.h
#  eaf/eafP.h
  ${EAF_FORTRAN_HEADERS}
)

set(ELIO_HEADERS
   elio/elio.h
   elio/chemio.h
#   elio/eliop.h
#   elio/pablo.h
)

set(SF_HEADERS
    sf/sf.h
    sf/coms.h
    ${SF_FORTRAN_HEADERS}
)

install (FILES
  ${DRA_HEADERS}
  ${EAF_HEADERS}
  ${ELIO_HEADERS}
  ${SF_HEADERS}
  DESTINATION include/ga
)

# -------------------------------------------------------------
#  PARIO executable files
# -------------------------------------------------------------

set(DRA_FILES
    dra/disk.arrays.c
    dra/buffers.c
    dra/capi.c
    dra/disk.param.c
    dra/env.c
    dra/global.unsup.c
    dra/patch.util.c
    dra/util.c
    ${DRA_FORTRAN_FILES}
)

set(EAF_FILES
    eaf/eaf.c
    ${EAF_FORTRAN_FILES}
)

set(ELIO_FILES
    elio/elio.c
    elio/stat.c
)

set(SF_FILES
     sf/shared.files.c
     sf/sf_capi.c
     ${SF_FORTRAN_FILES}
)

# -------------------------------------------------------------
# PARIO library installation
# -------------------------------------------------------------

set(PARIO_INC_DIRS  ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/dra
                    ${CMAKE_CURRENT_SOURCE_DIR}/eaf
                    ${CMAKE_CURRENT_SOURCE_DIR}/elio
                    ${CMAKE_CURRENT_SOURCE_DIR}/sf
                    ${PROJECT_SOURCE_DIR}/gaf2c
                    ${PROJECT_SOURCE_DIR}/ma
                    ${PROJECT_SOURCE_DIR}/global/src)

add_library(sf OBJECT ${SF_FILES})
add_library(dra OBJECT ${DRA_FILES})
add_library(eaf OBJECT ${EAF_FILES})
add_library(elio OBJECT ${ELIO_FILES})

target_include_directories(sf   BEFORE PRIVATE ${PARIO_INC_DIRS})
target_include_directories(dra  BEFORE PRIVATE ${PARIO_INC_DIRS})
target_include_directories(eaf  BEFORE PRIVATE ${PARIO_INC_DIRS})
target_include_directories(elio BEFORE PRIVATE ${PARIO_INC_DIRS})
