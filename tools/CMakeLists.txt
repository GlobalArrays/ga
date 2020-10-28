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
# GA tools headers installation
# -------------------------------------------------------------

set(GA_TOOLS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/ga-wprof.h
)

install(FILES
  ${GA_TOOLS_HEADERS}
  DESTINATION include/ga
)

add_library(ga_tools OBJECT
  ga-wapi.c
  ga-wprof.c
)


target_include_directories(ga_tools BEFORE PRIVATE
                    ${PROJECT_SOURCE_DIR}/ma
                    ${PROJECT_BINARY_DIR}/ma
                    ${PROJECT_BINARY_DIR}/gaf2c
                    ${PROJECT_BINARY_DIR}/global/src
                    ${PROJECT_SOURCE_DIR}/global/src
                    ${PROJECT_SOURCE_DIR}/comex/src-armci
                    ${PROJECT_BINARY_DIR})
