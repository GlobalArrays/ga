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

cmake_minimum_required( VERSION 3.17.0 ) 

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( CMakeFindDependencyMacro )
include( FindPackageHandleStandardArgs )


include( ${CMAKE_CURRENT_LIST_DIR}/linalg-modules/CommonFunctions.cmake )

# SANITY CHECK
if( "ilp64" IN_LIST LAPACK_FIND_COMPONENTS AND "lp64" IN_LIST LAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "LAPACK cannot link to both ILP64 and LP64 iterfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${LAPACK_FIND_COMPONENTS} )
  if( LAPACK_FIND_REQUIRED_${_comp} )
    list( APPEND LAPACK_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND LAPACK_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

fill_out_prefix( lapack )

if( NOT LAPACK_PREFERENCE_LIST )
  set( LAPACK_PREFERENCE_LIST "ReferenceLAPACK" )
endif()

if( NOT lapack_LIBRARIES )

  if( NOT TARGET BLAS::blas )
    find_dependency( BLAS 
      COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS} 
      OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS} 
    )
  else()
    message( STATUS "LAPACK will use predetermined BLAS::blas" )
  endif()

  message( STATUS "Checking if BLAS::blas contains a LAPACK implementation...")
  # Check if BLAS LINKS to LAPACK
  cmake_push_check_state( RESET )
  set( CMAKE_REQUIRED_LIBRARIES BLAS::blas )
  set( CMAKE_REQUIRED_QUIET     ON         )
  
  check_library_exists( "" dsyev_ "" blas_HAS_DSYEV_UNDERSCORE    )
  check_library_exists( "" dsyev  "" blas_HAS_DSYEV_NO_UNDERSCORE )
  
  cmake_pop_check_state()

  if( blas_HAS_DSYEV_UNDERSCORE OR blas_HAS_DSYEV_NO_UNDERSCORE )
    set( blas_HAS_LAPACK TRUE )
    message( STATUS "Checking if BLAS::blas contains a LAPACK implementation... Yes!")
  else()
    message( STATUS "Checking if BLAS::blas contains a LAPACK implementation... No!")
  endif()

  unset( blas_HAS_DSYEV_UNDERSCORE    )
  unset( blas_HAS_DSYEV_NO_UNDERSCORE )

  if( blas_HAS_LAPACK )
    set( lapack_LIBRARIES       BLAS::blas              )
    set( LAPACK_VENDOR          ${BLAS_VENDOR}          )
    if( BLAS_USES_ILP64 )
      set( LAPACK_ilp64_FOUND TRUE )
    else()
      set( LAPACK_lp64_FOUND TRUE )
    endif()
  else()

    if( BLAS_USES_ILP64 )
      message( FATAL_ERROR "ReferenceLAPACK cannot be compiled ILP64" )
    endif()

    list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/linalg-modules )
    foreach( lapack_type ${LAPACK_PREFERENCE_LIST} )

      string( TOLOWER ${lapack_type} lapack_lower_case )
      #set( ${lapack_lower_case}_PREFIX         ${lapack_PREFIX}         )
      #set( ${lapack_lower_case}_INCLUDE_DIR    ${lapack_INCLUDE_DIR}    )
      #set( ${lapack_lower_case}_LIBRARY_DIR    ${lapack_LIBRARY_DIR}    )
      #set( ${lapack_lower_case}_PREFERS_STATIC ${lapack_PREFERS_STATIC} )
      copy_meta_data( lapack ${lapack_lower_case} )

      find_package( ${lapack_type} 
        COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS} 
        OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS} 
      )

      if( ${lapack_type}_FOUND )

        set( LAPACK_VENDOR ${lapack_type} )

        if( ${lapack_type} MATCHES "ReferenceLAPACK" ) 
          set( lapack_LIBRARIES ReferenceLAPACK::lapack )
        endif()

        # Propagate integers
        if( "ilp64" IN_LIST LAPACK_FIND_COMPONENTS )
          set( LAPACK_ilp64_FOUND ${${lapack_type}_ilp64_FOUND} )
        else()
          set( LAPACK_lp64_FOUND ${${lapack_type}_lp64_FOUND} )
        endif()

        break()

      endif()

    endforeach()

    list(REMOVE_AT CMAKE_MODULE_PATH -1)

    # Append BLAS to LAPACK
    if( lapack_LIBRARIES )
      list( APPEND lapack_LIBRARIES BLAS::blas )
    endif()

  endif()
else()

  message( STATUS "LAPACK LIBRARIES WERE SET BY USER: ${lapack_LIBRARIES}" )

endif()

if( NOT LAPACK_ilp64_FOUND )
  set( LAPACK_ilp64_FOUND FALSE )
endif()
if( NOT LAPACK_lp64_FOUND )
  set( LAPACK_lp64_FOUND FALSE )
endif()


if( LAPACK_ilp64_FOUND )
  set( LAPACK_USES_ILP64 TRUE )
else()
  set( LAPACK_USES_ILP64 FALSE )
endif()



# Handle implicit LAPACK linkage
if( lapack_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( lapack_LIBRARIES )
endif()

# Check function existance and linkage / name mangling
cmake_push_check_state( RESET )
if( lapack_LIBRARIES )
  set( CMAKE_REQUIRED_LIBRARIES ${lapack_LIBRARIES} )
endif()
set( CMAKE_REQUIRED_QUIET ON )

check_library_exists( "" dsyev       "" LAPACK_NO_UNDERSCORE   ) 
check_library_exists( "" dsyev_      "" LAPACK_USES_UNDERSCORE ) 

set( TEST_USES_UNDERSCORE_STR "Performing Test LAPACK_USES_UNDERSCORE" )
set( TEST_NO_UNDERSCORE_STR   "Performing Test LAPACK_NO_UNDERSCORE"   )

message( STATUS  ${TEST_USES_UNDERSCORE_STR} )
if( LAPACK_USES_UNDERSCORE )
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- not found" )
endif()

message( STATUS  ${TEST_NO_UNDERSCORE_STR} )
if( LAPACK_NO_UNDERSCORE )
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- not found" )
endif()

unset( TEST_USES_UNDERSCORE_STR )
unset( TEST_NO_UNDERSCORE_STR )


cmake_pop_check_state()

if( LAPACK_NO_UNDERSCORE OR LAPACK_USES_UNDERSCORE )
  set( LAPACK_LINK_OK TRUE )
endif()


find_package_handle_standard_args( LAPACK
  REQUIRED_VARS LAPACK_LINK_OK
  HANDLE_COMPONENTS
)

if( LAPACK_FOUND AND NOT TARGET LAPACK::lapack )

  set( LAPACK_LIBRARIES ${lapack_LIBRARIES} )
  
  add_library( LAPACK::lapack INTERFACE IMPORTED )
  set_target_properties( LAPACK::lapack PROPERTIES
    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
  )

endif()
