cmake_minimum_required( VERSION 3.17 ) # Require CMake 3.17+

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( FindPackageHandleStandardArgs )
include( CMakeFindDependencyMacro )


include( ${CMAKE_CURRENT_LIST_DIR}/util/CommonFunctions.cmake )
include( ${CMAKE_CURRENT_LIST_DIR}/util/BLASUtilities.cmake   )
include( ${CMAKE_CURRENT_LIST_DIR}/LinAlgModulesMacros.cmake  )

# SANITY CHECK: Make sure only one integer interface is requested
if( "ilp64" IN_LIST BLAS_FIND_COMPONENTS AND "lp64" IN_LIST BLAS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLAS cannot link to both ILP64 and LP64 interfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${BLAS_FIND_COMPONENTS} )
  if( BLAS_FIND_REQUIRED_${_comp} )
    list( APPEND BLAS_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND BLAS_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

emulate_kitware_linalg_modules( BLAS )
fill_out_prefix( BLAS )

if( NOT BLAS_PREFERENCE_LIST )
  set( BLAS_PREFERENCE_LIST "IntelMKL" "IBMESSL" "BLIS" "OpenBLAS" "ReferenceBLAS" )
  if( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
    list( PREPEND BLAS_PREFERENCE_LIST "Accelerate" )
  endif()
endif()

if( NOT BLAS_LIBRARIES )

  message( STATUS "BLAS_LIBRARIES Not Given: Will Perform Search" )

  foreach( blas_type ${BLAS_PREFERENCE_LIST} )

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
      set( BLAS_COMPILE_OPTIONS     "${${blas_type}_COMPILE_OPTIONS}"     )

      # Generic Components
      #set( BLAS_headers_FOUND   ${${blas_type}_headers_FOUND}   )
      set( BLAS_sycl_FOUND      ${${blas_type}_sycl_FOUND}      )
      set( BLAS_blacs_FOUND     ${${blas_type}_blacs_FOUND}     )
      set( BLAS_scalapack_FOUND ${${blas_type}_scalapack_FOUND} )

      break() # Break from search loop

    endif()

  endforeach()

else()
  find_linalg_dependencies( BLAS_LIBRARIES )
endif()


# Handle implicit BLAS linkage
if( BLAS_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( BLAS_LIBRARIES )
endif()


# Check if DGEMM exists in proposed BLAS_LIBRARIES
check_fortran_functions_exist( dgemm BLAS BLAS_LIBRARIES
                               BLAS_LINK_OK BLAS_Fortran_LOWER BLAS_Fortran_UNDERSCORE )


# If BLAS linkage successful, check if it is ILP64/LP64
if( BLAS_LINK_OK )

  set( _dgemm_name "dgemm" )
  if( NOT BLAS_Fortran_LOWER )
    string( TOUPPER "${_dgemm_name}" _dgemm_name )
  endif()
  if( BLAS_Fortran_UNDERSCORE )
    set( _dgemm_name "${_dgemm_name}_" )
  endif()

  check_blas_int( BLAS_LIBRARIES ${_dgemm_name} BLAS_IS_LP64 )
  if( BLAS_IS_LP64 )
    set( BLAS_lp64_FOUND  TRUE  )
    set( BLAS_ilp64_FOUND FALSE )
  else()
    set( BLAS_lp64_FOUND  FALSE )
    set( BLAS_ilp64_FOUND TRUE  )
    find_dependency( ILP64 )
    list( APPEND BLAS_COMPILE_OPTIONS "${ILP64_COMPILE_OPTIONS}" )
    foreach ( lang C CXX Fortran )
        if ( DEFINED ILP64_${lang}_COMPILE_OPTIONS )
            list( APPEND BLAS_${lang}_COMPILE_OPTIONS "${ILP64_${lang}_COMPILE_OPTIONS}" )
        endif()
    endforeach()
  endif()

endif()


find_package_handle_standard_args( BLAS
  REQUIRED_VARS BLAS_LINK_OK
  HANDLE_COMPONENTS
)

# Cache variables
if( BLAS_FOUND )
  set( BLAS_VENDOR              "${BLAS_VENDOR}"              CACHE STRING "BLAS Vendor"              FORCE )
  set( BLAS_IS_LP64             "${BLAS_IS_LP64}"             CACHE STRING "BLAS LP64 Flag"           FORCE )
  set( BLAS_LIBRARIES           "${BLAS_LIBRARIES}"           CACHE STRING "BLAS Libraries"           FORCE )
  set( BLAS_COMPILE_DEFINITIONS "${BLAS_COMPILE_DEFINITIONS}" CACHE STRING "BLAS Compile Definitions" FORCE )
  set( BLAS_INCLUDE_DIRS        "${BLAS_INCLUDE_DIRS}"        CACHE STRING "BLAS Include Directories" FORCE )
  set( BLAS_COMPILE_OPTIONS     "${BLAS_COMPILE_OPTIONS}"     CACHE STRING "BLAS Compile Options"     FORCE )
  foreach ( lang C CXX Fortran )
      if ( DEFINED BLAS_${lang}_COMPILE_OPTIONS )
          set( BLAS_${lang}_COMPILE_OPTIONS     "${BLAS_${lang}_COMPILE_OPTIONS}"     CACHE STRING "BLAS Compile Options for Language ${lang}"     FORCE )
      endif()
  endforeach()
endif()

if( BLAS_FOUND AND NOT TARGET BLAS::BLAS )
  
  add_library( BLAS::BLAS INTERFACE IMPORTED )
  set_target_properties( BLAS::BLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
    INTERFACE_COMPILE_OPTIONS     "${BLAS_COMPILE_OPTIONS}"
    INTERFACE_COMPILE_DEFINITIONS "${BLAS_COMPILE_DEFINITIONS}"
    INTERFACE_LINK_LIBRARIES      "${BLAS_LIBRARIES}"
  )

endif()
