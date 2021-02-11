cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( CMakeFindDependencyMacro )
include( FindPackageHandleStandardArgs )


include( ${CMAKE_CURRENT_LIST_DIR}/util/CommonFunctions.cmake )
include( ${CMAKE_CURRENT_LIST_DIR}/util/LAPACKUtilities.cmake )
include( ${CMAKE_CURRENT_LIST_DIR}/LinAlgModulesMacros.cmake  )

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

fill_out_prefix( LAPACK )

if( NOT LAPACK_PREFERENCE_LIST )
  set( LAPACK_PREFERENCE_LIST "ReferenceLAPACK" )
endif()

if( NOT LAPACK_LIBRARIES )

  # Find BLAS
  if( NOT TARGET BLAS::BLAS )
    copy_meta_data( LAPACK BLAS )	  
    find_dependency( BLAS 
      COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS} 
      OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS} 
    )
  endif()
  
  # Check if BLAS contains a LAPACK linker
  message( STATUS "LAPACK_LIBRARIES Not Given: Checking for LAPACK in BLAS" )
  set( LAPACK_LIBRARIES           ${BLAS_LIBRARIES}           )
  set( LAPACK_INCLUDE_DIRS        ${BLAS_INCLUDE_DIRS}        )
  set( LAPACK_COMPILE_DEFINITIONS ${BLAS_COMPILE_DEFINITIONS} )
  check_dpstrf_exists( LAPACK_LIBRARIES 
    BLAS_HAS_LAPACK LAPACK_FORTRAN_LOWER LAPACK_FORTRAN_UNDERSCORE
  )
  
  
  # If BLAS has a full LAPACK Linker, propagate vars
  if( BLAS_HAS_LAPACK )

    message( STATUS "BLAS Has A Full LAPACK Linker" )
    set( LAPACK_VENDOR          ${BLAS_VENDOR}          )
    set( LAPACK_IS_LP64         ${BLAS_IS_LP64}         )
    set( LAPACK_blacs_FOUND     ${BLAS_blacs_FOUND}     )
    set( LAPACK_scalapack_FOUND ${BLAS_scalapack_FOUND} )
    set( LAPACK_sycl_FOUND      ${BLAS_sycl_FOUND}      )

  # Else find LAPACK installation consistent with BLAS
  else( BLAS_HAS_LAPACK )

    # Ensure proper integer size
    if( BLAS_IS_LP64 AND (NOT "lp64" IN_LIST LAPACK_REQUIRED_COMPONENTS) )
      list( APPEND LAPACK_REQUIRED_COMPONENTS "lp64" )
    elseif( (NOT BLAS_IS_LP64) AND (NOT "ilp64" IN_LIST LAPACK_REQUIRED_COMPONENTS ) )
      list( APPEND LAPACK_REQUIRED_COMPONENTS "ilp64" )
    endif()

    message( STATUS "BLAS Does Not Have A Full LAPACK Linker -- Performing Search" )
    foreach( lapack_type ${LAPACK_PREFERENCE_LIST} )

      copy_meta_data( LAPACK ${lapack_type} )

      find_package( ${lapack_type} 
        COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS} 
        OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS} 
      )

      if( ${lapack_type}_FOUND )

        # Propagate Linker / Includes
        set( LAPACK_VENDOR "${lapack_type}" )

        list( PREPEND LAPACK_LIBRARIES           ${${lapack_type}_LIBRARIES}           )
        list( PREPEND LAPACK_COMPILE_DEFINITIONS ${${lapack_type}_COMPILE_DEFINITIONS} )
        list( PREPEND LAPACK_INCLUDE_DIR         ${${lapack_type}_INCLUDE_DIR}         )

        # Generic Components
        #set( LAPACK_headers_FOUND   ${${lapack_type}_headers_FOUND}   )
        set( LAPACK_blacs_FOUND     ${${lapack_type}_blacs_FOUND}     )
        set( LAPACK_scalapack_FOUND ${${lapack_type}_scalapack_FOUND} )
        set( LAPACK_sycl_FOUND      ${${lapack_type}_sycl_FOUND}      )

        break() # Break from search loop

      endif()

    endforeach()
  endif( BLAS_HAS_LAPACK )

else()
  find_linalg_dependencies( LAPACK_LIBRARIES )
endif()

# Handle implicit LAPACK linkage
if( LAPACK_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( LAPACK_LIBRARIES )
endif()


# Check for LAPACK Linker
if( BLAS_HAS_LAPACK )
  set( LAPACK_LINK_OK TRUE )
else()
  check_dpstrf_exists( LAPACK_LIBRARIES 
    LAPACK_LINK_OK LAPACK_FORTRAN_LOWER LAPACK_FORTRAN_UNDERSCORE
  )
endif()

# If LAPACK linkage sucessful, check if it is ILP64/LP64
if( LAPACK_LINK_OK )

  set( _dsyev_name "dsyev" )
  if( NOT LAPACK_FORTRAN_LOWER )
    string( TOUPPER "${_dsyev_name}" _dsyev_name )
  endif()
  if( LAPACK_FORTRAN_UNDERSCORE )
    set( _dsyev_name "${_dsyev_name}_" )
  endif()

  check_lapack_int( LAPACK_LIBRARIES ${_dsyev_name} LAPACK_IS_LP64 )
  if( LAPACK_IS_LP64 )
    set( LAPACK_lp64_FOUND  TRUE  )
    set( LAPACK_ilp64_FOUND FALSE )
  else()
    set( LAPACK_lp64_FOUND  FALSE )
    set( LAPACK_ilp64_FOUND TRUE  )
    find_dependency( ILP64 )
    list( APPEND LAPACK_COMPILE_OPTIONS "${ILP64_COMPILE_OPTIONS}" )
  endif()

else()

  # Unset everything for safety
  unset( LAPACK_LIBRARIES )
  unset( LAPACK_COMPILE_DEFINITIONS )

endif()




find_package_handle_standard_args( LAPACK
  REQUIRED_VARS LAPACK_LINK_OK
  HANDLE_COMPONENTS
)

# Cache variables
if( LAPACK_FOUND )
  set( LAPACK_VENDOR              "${LAPACK_VENDOR}"              CACHE STRING "LAPACK Vendor"              FORCE )
  set( LAPACK_IS_LP64             "${LAPACK_IS_LP64}"             CACHE STRING "LAPACK LP64 Flag"           FORCE )
  set( LAPACK_LIBRARIES           "${LAPACK_LIBRARIES}"           CACHE STRING "LAPACK Libraries"           FORCE )
  set( LAPACK_COMPILE_DEFINITIONS "${LAPACK_COMPILE_DEFINITIONS}" CACHE STRING "LAPACK Compile Definitions" FORCE )
  set( LAPACK_INCLUDE_DIRS        "${LAPACK_INCLUDE_DIRS}"        CACHE STRING "LAPACK Include Directories" FORCE )
  set( LAPACK_COMPILE_OPTIONS     "${LAPACK_COMPILE_OPTIONS}"     CACHE STRING "LAPACK Compile Options"     FORCE )
endif()

if( LAPACK_FOUND AND NOT TARGET LAPACK::LAPACK )
  
  add_library( LAPACK::LAPACK INTERFACE IMPORTED )
  set_target_properties( LAPACK::LAPACK PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACK_INCLUDE_DIRS}"
    INTERFACE_COMPILE_OPTIONS     "${LAPACK_COMPILE_OPTIONS}"
    INTERFACE_COMPILE_DEFINITIONS "${LAPACK_COMPILE_DEFINITIONS}"
    INTERFACE_LINK_LIBRARIES      "${LAPACK_LIBRARIES}"
  )

endif()
