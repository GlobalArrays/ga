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
include( FindPackageHandleStandardArgs )


include( ${CMAKE_CURRENT_LIST_DIR}/linalg-modules/CommonFunctions.cmake )

# SANITY CHECK
if( "ilp64" IN_LIST BLAS_FIND_COMPONENTS AND "lp64" IN_LIST BLAS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLAS cannot link to both ILP64 and LP64 iterfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${BLAS_FIND_COMPONENTS} )
  if( BLAS_FIND_REQUIRED_${_comp} )
    list( APPEND BLAS_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND BLAS_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

fill_out_prefix( blas )

set( SUPPORTED_BLAS_VENDORS "IntelMKL" "IBMESSL" "ReferenceBLAS" ) #"BLIS" "OpenBLAS"
if (NOT "${BLAS_VENDOR}" IN_LIST SUPPORTED_BLAS_VENDORS)
  message(FATAL_ERROR "Unsupported BLAS_VENDOR ${BLAS_VENDOR} specified!!")
endif()

set(BLAS_PREFERENCE_LIST ${BLAS_VENDOR})
# if( NOT BLAS_PREFERENCE_LIST )
#   set( BLAS_PREFERENCE_LIST "IntelMKL" "IBMESSL" "BLIS" "OpenBLAS" "ReferenceBLAS" )
# endif()

if( NOT blas_LIBRARIES )

  list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/linalg-modules )
  foreach( blas_type ${BLAS_PREFERENCE_LIST} )

    string( TOLOWER ${blas_type} blas_lower_case )
    #set( ${blas_lower_case}_PREFIX         ${blas_PREFIX}         )
    #set( ${blas_lower_case}_INCLUDE_DIR    ${blas_INCLUDE_DIR}    )
    #set( ${blas_lower_case}_LIBRARY_DIR    ${blas_LIBRARY_DIR}    )
    #set( ${blas_lower_case}_PREFERS_STATIC ${blas_PREFERS_STATIC} )
    copy_meta_data( blas ${blas_lower_case} )


    find_package( ${blas_type} 
      COMPONENTS          ${BLAS_REQUIRED_COMPONENTS} 
      OPTIONAL_COMPONENTS ${BLAS_OPTIONAL_COMPONENTS} 
    )

    if( ${blas_type}_FOUND )

      set( BLAS_VENDOR ${blas_type} )

      if    ( ${blas_type} MATCHES "IntelMKL" )
        set( blas_LIBRARIES IntelMKL::mkl )
        set(BLAS_INCLUDE_DIRS "${IntelMKL_INCLUDE_DIR}")
        set(BLAS_COMPILE_OPTIONS     "${IntelMKL_C_COMPILE_FLAGS}")
        # set(BLAS_COMPILE_DEFINITIONS "${IntelMKL_COMPILE_DEFINITIONS}")
      elseif( ${blas_type} MATCHES "IBMESSL" )
        set( blas_LIBRARIES IBMESSL::essl )
        set(BLAS_INCLUDE_DIRS "${IBMESSL_INCLUDE_DIR}")
        set(BLAS_COMPILE_OPTIONS     "${IBMESSL_C_COMPILE_FLAGS}")
        # set(BLAS_COMPILE_DEFINITIONS "${IBMESSL_COMPILE_DEFINITIONS}")        
      elseif( ${blas_type} MATCHES "BLIS" )
        set( blas_LIBRARIES BLIS::blis )
      elseif( ${blas_type} MATCHES "OpenBLAS" )
        set( blas_LIBRARIES OpenBLAS::openblas )
      elseif( ${blas_type} MATCHES "ReferenceBLAS" ) 
        set( blas_LIBRARIES ReferenceBLAS::blas )
        set(BLAS_INCLUDE_DIRS "${ReferenceBLAS_INCLUDE_DIR}")
        set(BLAS_COMPILE_OPTIONS     "${ReferenceBLAS_C_COMPILE_FLAGS}")
        # set(BLAS_COMPILE_DEFINITIONS "${ReferenceBLAS_COMPILE_DEFINITIONS}")           
      endif()

      # Propagate integers
      if( "ilp64" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_ilp64_FOUND ${${blas_type}_ilp64_FOUND} )
      else()
        set( BLAS_lp64_FOUND ${${blas_type}_lp64_FOUND} )
      endif()

      # Propagate BLACS / ScaLAPACK
      if( "blacs" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_blacs_FOUND ${${blas_type}_blacs_FOUND} )
      endif()
      if( "scalapack" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_scalapack_FOUND ${${blas_type}_scalapack_FOUND} )
      endif()


      break()

    endif()

  endforeach()

  list(REMOVE_AT CMAKE_MODULE_PATH -1)
endif()

if( NOT BLAS_ilp64_FOUND )
  set( BLAS_ilp64_FOUND FALSE )
endif()
if( NOT BLAS_lp64_FOUND )
  set( BLAS_lp64_FOUND FALSE )
endif()


if( BLAS_ilp64_FOUND )
  set( BLAS_USES_ILP64 TRUE )
else()
  set( BLAS_USES_ILP64 FALSE )
endif()

# Handle implicit BLAS linkage
if( blas_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( blas_LIBRARIES )
endif()

# Check function existance and linkage / name mangling
cmake_push_check_state( RESET )
if( blas_LIBRARIES )
  set( CMAKE_REQUIRED_LIBRARIES ${blas_LIBRARIES} )
endif()
set( CMAKE_REQUIRED_QUIET ON )

check_library_exists( "" dgemm       "" BLAS_NO_UNDERSCORE   ) 
check_library_exists( "" dgemm_      "" BLAS_USES_UNDERSCORE ) 

set( TEST_USES_UNDERSCORE_STR "Performing Test BLAS_USES_UNDERSCORE" )
set( TEST_NO_UNDERSCORE_STR   "Performing Test BLAS_NO_UNDERSCORE"   )

message( STATUS  ${TEST_USES_UNDERSCORE_STR} )
if( BLAS_USES_UNDERSCORE )
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- not found" )
endif()

message( STATUS  ${TEST_NO_UNDERSCORE_STR} )
if( BLAS_NO_UNDERSCORE )
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- not found" )
endif()

unset( TEST_USES_UNDERSCORE_STR )
unset( TEST_NO_UNDERSCORE_STR )


cmake_pop_check_state()

if( BLAS_NO_UNDERSCORE OR BLAS_USES_UNDERSCORE )
  set( BLAS_LINK_OK TRUE )
endif()


find_package_handle_standard_args( BLAS
  REQUIRED_VARS BLAS_LINK_OK
  HANDLE_COMPONENTS
)

if( BLAS_FOUND AND NOT TARGET BLAS::blas )

  set( BLAS_LIBRARIES ${blas_LIBRARIES} )
  
  add_library( BLAS::blas INTERFACE IMPORTED )
  set_target_properties( BLAS::blas PROPERTIES
  INTERFACE_LINK_LIBRARIES      "${BLAS_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
  INTERFACE_COMPILE_OPTIONS     "${BLAS_COMPILE_OPTIONS}"
  # INTERFACE_COMPILE_DEFINITIONS "${BLAS_COMPILE_DEFINITIONS}"    
  )

endif()
