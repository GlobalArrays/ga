#   FindScaLAPACK.cmake
#
#   Finds the ScaLAPACK library.
#
#   This module will define the following variables:
#   
#     ScaLAPACK_FOUND        - System has found ScaLAPACK installation
#     ScaLAPACK_LIBRARIES    - ScaLAPACK libraries
#
#   This module will export the following targets if SCALAPACK_FOUND
#
#     ScaLAPACK::ScaLAPACK
#
#   Proper usage:
#
#     project( TEST_FIND_SCALAPACK C )
#     find_package( ScaLAPACK )
#
#     if( ScaLAPACK_FOUND )
#       add_executable( test test.cxx )
#       target_link_libraries( test ScaLAPACK::ScaLAPACK )
#     endif()
#
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     ScaLAPACK_PREFIX
#     ScaLAPACK_LIBRARY_DIR
#     ScaLAPACK_LIBRARIES

cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

include( CMakeFindDependencyMacro )

include( ${CMAKE_CURRENT_LIST_DIR}/util/CommonFunctions.cmake    )
include( ${CMAKE_CURRENT_LIST_DIR}/util/ScaLAPACKUtilities.cmake )
include( ${CMAKE_CURRENT_LIST_DIR}/LinAlgModulesMacros.cmake     )


# SANITY CHECK
if( "ilp64" IN_LIST ScaLAPACK_FIND_COMPONENTS AND "lp64" IN_LIST ScaLAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "ScaLAPACK cannot link to both ILP64 and LP64 iterfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${ScaLAPACK_FIND_COMPONENTS} )
  if( ScaLAPACK_FIND_REQUIRED_${_comp} )
    list( APPEND ScaLAPACK_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND ScaLAPACK_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

fill_out_prefix( ScaLAPACK )

if( NOT ScaLAPACK_PREFERENCE_LIST )
  set( ScaLAPACK_PREFERENCE_LIST "ReferenceScaLAPACK" )
endif()


if( NOT ScaLAPACK_LIBRARIES )

  # Find LAPACK
  if( NOT TARGET LAPACK::LAPACK )
    find_dependency( LAPACK 
      COMPONENTS          ${ScaLAPACK_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS ${ScaLAPACK_OPTIONAL_COMPONENTS} scalapack blacs 
    )
  endif()

  # Check if LAPACK contains ScaLAPACK linker (e.g. MKL)
  message( STATUS "ScaLAPACK_LIBRARIES Not Given: Checking for ScaLAPACK in LAPACK" )
  set( ScaLAPACK_LIBRARIES           ${LAPACK_LIBRARIES}           )
  set( ScaLAPACK_INCLUDE_DIR         ${LAPACK_INCLUDE_DIR}         )
  set( ScaLAPACK_COMPILE_DEFINITIONS ${LAPACK_COMPILE_DEFINITIONS} )
  check_pdpotrf_exists( ScaLAPACK_LIBRARIES 
    LAPACK_HAS_ScaLAPACK ScaLAPACK_FORTRAN_LOWER ScaLAPACK_FORTRAN_UNDERSCORE
  )

  # If LAPACK has a full ScaLAPACK Linker, propagate vars
  if( LAPACK_HAS_ScaLAPACK )

    message( STATUS "LAPACK Has A Full ScaLAPACK Linker" )
    set( ScaLAPACK_VENDOR  ${LAPACK_VENDOR}  )
    set( ScaLAPACK_IS_LP64 ${LAPACK_IS_LP64} )

  # Else find ScaLAPACK installation consistent with LAPACK
  else( LAPACK_HAS_ScaLAPACK )

    # Ensure proper integer size
    if( LAPACK_IS_LP64 AND (NOT "lp64" IN_LIST ScaLAPACK_REQUIRED_COMPONENTS) )
      list( APPEND ScaLAPACK_REQUIRED_COMPONENTS "lp64" )
    elseif( (NOT LAPACK_IS_LP64) AND (NOT "ilp64" IN_LIST ScaLAPACK_REQUIRED_COMPONENTS ) )
      list( APPEND ScaLAPACK_REQUIRED_COMPONENTS "ilp64" )
    endif()

    message( STATUS "LAPACK Does Not Have A Full ScaLAPACK Linker -- Performing Search" )
    foreach( scalapack_type ${ScaLAPACK_PREFERENCE_LIST} )

      copy_meta_data( ScaLAPACK ${scalapack_type} )

      find_package( ${scalapack_type} 
        COMPONENTS          ${ScaLAPACK_REQUIRED_COMPONENTS} 
        OPTIONAL_COMPONENTS ${ScaLAPACK_OPTIONAL_COMPONENTS} 
      )

      if( ${scalapack_type}_FOUND )

        # Propagate Linker / Includes
        set( ScaLAPACK_VENDOR "${scalapack_type}" )

        list( PREPEND ScaLAPACK_LIBRARIES           ${${scalapack_type}_LIBRARIES}           )
        list( PREPEND ScaLAPACK_COMPILE_DEFINITIONS ${${scalapack_type}_COMPILE_DEFINITIONS} )
        list( PREPEND ScaLAPACK_INCLUDE_DIR         ${${scalapack_type}_INCLUDE_DIR}         )

        break() # Break from search loop

      endif()

    endforeach()
  endif( LAPACK_HAS_ScaLAPACK )

  else()
    find_linalg_dependencies( ScaLAPACK_LIBRARIES )
endif()

# Handle implicit LAPACK linkage
if( ScaLAPACK_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( ScaLAPACK_LIBRARIES )
endif()


# Check for ScaLAPACK Linker
if( LAPACK_HAS_ScaLAPACK )
  set( ScaLAPACK_LINK_OK TRUE )
else()
  check_pdpotrf_exists( ScaLAPACK_LIBRARIES 
    ScaLAPACK_LINK_OK ScaLAPACK_FORTRAN_LOWER ScaLAPACK_FORTRAN_UNDERSCORE
  )
endif()

# If ScaLAPACK linkage sucessful, check if it is ILP64/LP64
if( ScaLAPACK_LINK_OK )

  # TODO: This requires running an MPI program, pretty dangerous
  #set( _pdpotrf_name "pdpotrf" )
  #if( NOT ScaLAPACK_FORTRAN_LOWER )
  #  string( TOUPPER "${_pdpotrf_name}" _pdpotrf_name )
  #endif()
  #if( ScaLAPACK_FORTRAN_UNDERSCORE )
  #  set( _pdpotrf_name "${_pdpotrf_name}_" )
  #endif()

  #check_scalapack_int( ScaLAPACK_LIBRARIES ${_pdpotrf_name} ScaLAPACK_IS_LP64 )

  # XXX: Unless expressly told otherwise, assume ScaLAPACK is LP64
  if( NOT DEFINED ScaLAPACK_IS_LP64 )
    set( ScaLAPACK_IS_LP64 TRUE )
  endif()

  if( ScaLAPACK_IS_LP64 )
    set( ScaLAPACK_lp64_FOUND  TRUE  )
    set( ScaLAPACK_ilp64_FOUND FALSE )
  else()
    set( ScaLAPACK_lp64_FOUND  FALSE )
    set( ScaLAPACK_ilp64_FOUND TRUE  )
  endif()

else()

  # Unset everything for safety
  unset( ScaLAPACK_LIBRARIES )
  unset( ScaLAPACK_COMPILE_DEFINITIONS )

endif()

find_package_handle_standard_args( ScaLAPACK
  REQUIRED_VARS ScaLAPACK_LINK_OK
  HANDLE_COMPONENTS
)

if( ScaLAPACK_FOUND AND NOT TARGET ScaLAPACK::ScaLAPACK )
  
  add_library( ScaLAPACK::ScaLAPACK INTERFACE IMPORTED )
  set_target_properties( ScaLAPACK::ScaLAPACK PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${ScaLAPACK_COMPILE_DEFINITIONS}"
    INTERFACE_LINK_LIBRARIES      "${ScaLAPACK_LIBRARIES}"
  )

endif()
