cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( FindPackageHandleStandardArgs )


include( ${CMAKE_CURRENT_LIST_DIR}/util/CommonFunctions.cmake )
include( ${CMAKE_CURRENT_LIST_DIR}/util/BLASUtilities.cmake   )

# SANITY CHECK: Make sure only one integer interface is requested
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

fill_out_prefix( BLAS )

if( NOT BLAS_PREFERENCE_LIST )
  set( BLAS_PREFERENCE_LIST "IntelMKL" "IBMESSL" "BLIS" "OpenBLAS" "ReferenceBLAS" )
endif()

if (NOT "${BLAS_VENDOR}" IN_LIST BLAS_PREFERENCE_LIST)
  message(FATAL_ERROR "Unsupported BLAS_VENDOR ${BLAS_VENDOR} specified!!")
endif()

if( NOT BLAS_LIBRARIES )

  message( STATUS "BLAS_LIBRARIES Not Given: Will Perform Search" )

  foreach( blas_type ${BLAS_VENDOR} )

    copy_meta_data( BLAS ${blas_type} )

    find_package( ${blas_type} 
      COMPONENTS          ${BLAS_REQUIRED_COMPONENTS} 
      OPTIONAL_COMPONENTS ${BLAS_OPTIONAL_COMPONENTS} 
    )

    if( ${blas_type}_FOUND )

      # Propagate Linker / Includes
      set( BLAS_VENDOR              "${blas_type}"                        )
      set( BLAS_LIBRARIES           "${${blas_type}_LIBRARIES}"           )
      set( BLAS_COMPILE_DEFINITIONS "${${blas_type}_COMPILE_DEFINITIONS}" )
      set( BLAS_INCLUDE_DIRS        "${${blas_type}_INCLUDE_DIR}"         )
      set( BLAS_COMPILE_OPTIONS     "${${blas_type}_C_COMPILE_FLAGS}"     )

      # Generic Components
      #set( BLAS_headers_FOUND   ${${blas_type}_headers_FOUND}   )
      set( BLAS_blacs_FOUND     ${${blas_type}_blacs_FOUND}     )
      set( BLAS_scalapack_FOUND ${${blas_type}_scalapack_FOUND} )

      break() # Break from search loop

    endif()

  endforeach()

endif()


# Handle implicit BLAS linkage
if( BLAS_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( BLAS_LIBRARIES )
endif()


# Check if DGEMM exists in proposed BLAS_LIBRARIES
check_dgemm_exists( BLAS_LIBRARIES 
                    BLAS_LINK_OK BLAS_FORTRAN_LOWER BLAS_FORTRAN_UNDERSCORE )


# If BLAS linkage sucessful, check if it is ILP64/LP64
if( BLAS_LINK_OK )

  set( _dgemm_name "dgemm" )
  if( NOT BLAS_FORTRAN_LOWER )
    string( TOUPPER "${_dgemm_name}" _dgemm_name )
  endif()
  if( BLAS_FORTRAN_UNDERSCORE )
    set( _dgemm_name "${_dgemm_name}_" )
  endif()

  check_blas_int( BLAS_LIBRARIES ${_dgemm_name} BLAS_IS_LP64 )
  if( BLAS_IS_LP64 )
    set( BLAS_lp64_FOUND  TRUE  )
    set( BLAS_ilp64_FOUND FALSE )
  else()
    set( BLAS_lp64_FOUND  FALSE )
    set( BLAS_ilp64_FOUND TRUE  )
  endif()

endif()


find_package_handle_standard_args( BLAS
  REQUIRED_VARS BLAS_LINK_OK
  HANDLE_COMPONENTS
)


if( BLAS_FOUND AND NOT TARGET BLAS::BLAS )
  
  add_library( BLAS::BLAS INTERFACE IMPORTED )
  set_target_properties( BLAS::BLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
    INTERFACE_COMPILE_OPTIONS     "${BLAS_COMPILE_OPTIONS}"
    INTERFACE_COMPILE_DEFINITIONS "${BLAS_COMPILE_DEFINITIONS}"
    INTERFACE_LINK_LIBRARIES      "${BLAS_LIBRARIES}"
  )

endif()
