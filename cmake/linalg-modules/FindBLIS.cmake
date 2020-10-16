#==================================================================
#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
#   
#   This file is part of cmake-modules. All rights reserved.
#   
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   
#   (1) Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#   (2) Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#   (3) Neither the name of the University of California, Lawrence Berkeley
#   National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#   
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#   
#   You are under no obligation whatsoever to provide any bug fixes, patches, or
#   upgrades to the features, functionality or performance of the source code
#   ("Enhancements") to anyone; however, if you choose to make your Enhancements
#   available either publicly, or directly to Lawrence Berkeley National
#   Laboratory, without imposing a separate written license agreement for such
#   Enhancements, then you hereby grant the following license: a non-exclusive,
#   royalty-free perpetual license to install, use, modify, prepare derivative
#   works, incorporate into other computer software, distribute, and sublicense
#   such enhancements or derivative works thereof, in binary and source code form.
#
#==================================================================

include(CommonFunctions)

# SANITY CHECK
if( "ilp64" IN_LIST BLIS_FIND_COMPONENTS AND "lp64" IN_LIST BLIS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLIS cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( blis_PREFERS_STATIC )
  set( blis_LIBRARY_NAME "libblis.a" )
else()
  set( blis_LIBRARY_NAME "blis" )
endif()

if( NOT blis_PREFIX )
  set( blis_PREFIX ${BLISROOT} $ENV{BLISROOT} )
endif()

find_library( blis_LIBRARY
  NAMES ${blis_LIBRARY_NAME}
  HINTS ${blis_PREFIX}
  PATHS ${blis_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "BLIS Library"
)

find_path( blis_INCLUDE_DIR
  NAMES blis/blis.h
  HINTS ${blis_PREFIX}
  PATHS ${blis_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "BLIS header"
)
  

if( blis_INCLUDE_DIR )
  set( BLIS_INCLUDE_DIR ${blis_INCLUDE_DIR} )
endif()

if( blis_LIBRARY )
  find_package( Threads QUIET )
  set( BLIS_LIBRARIES ${blis_LIBRARY} Threads::Threads "m")
endif()

# check ILP64
if( EXISTS ${BLIS_INCLUDE_DIR}/blis/blis.h )

  set( idxwidth_pattern
  "^#define[\t ]+BLIS_INT_TYPE_SIZE[\t ]+([0-9\\.]+[0-9\\.]+)$"
  )
  file( STRINGS ${BLIS_INCLUDE_DIR}/blis/blis.h blis_idxwidth
        REGEX ${idxwidth_pattern} )

  string( REGEX REPLACE ${idxwidth_pattern} 
          "${BLIS_IDXWIDTH_STRING}\\1"
          BLIS_IDXWIDTH_STRING ${blis_idxwidth} )

  if( ${BLIS_IDXWIDTH_STRING} MATCHES "64" )
    set( BLIS_USES_ILP64 TRUE )
  else()
    set( BLIS_USES_ILP64 FALSE )
  endif()

  unset( idxwidth_pattern      )
  unset( blis_idxwidth        )
  unset( BLIS_IDXWIDTH_STRING )

endif()

# Handle components
if( BLIS_USES_ILP64 )
  set( BLIS_ilp64_FOUND TRUE  )
  set( BLIS_lp64_FOUND  FALSE )
else()
  set( BLIS_ilp64_FOUND FALSE )
  set( BLIS_lp64_FOUND  TRUE  )
endif()

set(BLIS_COMPILE_DEFINITIONS "USE_BLIS")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( BLIS
  REQUIRED_VARS BLIS_LIBRARIES BLIS_INCLUDE_DIR BLIS_COMPILE_DEFINITIONS
#  VERSION_VAR BLIS_VERSION_STRING
  HANDLE_COMPONENTS
)

if( BLIS_FOUND AND NOT TARGET BLIS::blis )

  add_library( BLIS::blis INTERFACE IMPORTED )
  set_target_properties( BLIS::blis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${BLIS_LIBRARIES}"
    INTERFACE_COMPILE_DEFINITIONS "${BLIS_COMPILE_DEFINITIONS}"
  )

endif()
