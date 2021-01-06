set( ScaLAPACK_UTILITY_CMAKE_FILE_DIR ${CMAKE_CURRENT_LIST_DIR} )

function( check_pdpotrf_exists _libs _link_ok _uses_lower _uses_underscore )

include( ${ScaLAPACK_UTILITY_CMAKE_FILE_DIR}/CommonFunctions.cmake )

set( ${_link_ok} FALSE )
set( ${_uses_lower} )
set( ${_uses_underscore} )

foreach( _uplo LOWER UPPER )

  set( _pdpotrf_name_template "pdpotrf" )
  string( TO${_uplo} ${_pdpotrf_name_template} _pdpotrf_name_uplo )
  
  foreach( _under UNDERSCORE NO_UNDERSCORE )

    set( _item ScaLAPACK_${_uplo}_${_under} )
    if( _under EQUAL "UNDERSCORE" )
      set( _pdpotrf_name "${_pdpotrf_name_uplo}_" )
    else()
      set( _pdpotrf_name "${_pdpotrf_name_uplo}_" )
    endif()

    check_function_exists_w_results( 
      "${${_libs}}" ${_pdpotrf_name} _compile_output _compile_result 
    )

    message( STATUS "Performing Test ${_item}" )
    if( _compile_result )

      message( STATUS "Performing Test ${_item} -- found" )
      set( ${_link_ok} TRUE )
      string( COMPARE EQUAL "${_uplo}"  "LOWER"      ${_uses_lower}      )
      string( COMPARE EQUAL "${_under}" "UNDERSCORE" ${_uses_underscore} )
      break()

    else()

      # Check for Standard Fortran Libraries
      if(NOT STANDARDFORTRAN_LIBRARIES)
        find_dependency(StandardFortran REQUIRED)
        list( APPEND ${_libs} ${STANDARDFORTRAN_LIBRARIES} )
        set( ${_libs} ${${_libs}} PARENT_SCOPE )
      endif()

      if( _compile_output MATCHES "logf" )

        message( STATUS "  * Mising LIBM - Adding TO ScaLAPACK LINKER" )

        list( APPEND ${_libs} "m" )
        set( ${_libs} ${${_libs}} PARENT_SCOPE )

      endif()

      # Recheck Compiliation
      check_function_exists_w_results( 
        "${${_libs}}" ${_pdpotrf_name} _compile_output _compile_result 
      )

      if( _compile_result )
        message( STATUS "Performing Test ${_item} -- found" )
        set( ${_link_ok} TRUE )
        string( COMPARE EQUAL "${_uplo}"  "LOWER"      ${_uses_lower}      )
        string( COMPARE EQUAL "${_under}" "UNDERSCORE" ${_uses_underscore} )
        break()
      else()
        message( STATUS "Performing Test ${_item} -- not found" )
      endif()

    endif()

  endforeach()

  if( ${${_link_ok}} )
    break()
  endif()

  unset( _pdpotrf_name_template )
  unset( _pdpotrf_name_uplo     )
endforeach() 

#cmake_pop_check_state()


set( ${_link_ok}         ${${_link_ok}}         PARENT_SCOPE )
set( ${_uses_lower}      ${${_uses_lower}}      PARENT_SCOPE )
set( ${_uses_underscore} ${${_uses_underscore}} PARENT_SCOPE )

endfunction()




#function( check_scalapack_int _libs _dsyev_name _libs_are_lp64 )
#
#try_run( _run_result _compile_result ${CMAKE_CURRENT_BINARY_DIR}
#  SOURCES        ${ScaLAPACK_UTILITY_CMAKE_FILE_DIR}/scalapack_ilp64_checker.c
#  LINK_LIBRARIES ${${_libs}}
#  COMPILE_DEFINITIONS "-DDSYEV_NAME=${_dsyev_name}"
#  COMPILE_OUTPUT_VARIABLE _compile_output
#  RUN_OUTPUT_VARIABLE     _run_output
#)
#
##message( STATUS ${_run_result} )
#
#if( ${_run_result} EQUAL 0 )
#  set( ${_libs_are_lp64} TRUE PARENT_SCOPE )
#else()
#  set( ${_libs_are_lp64} FALSE PARENT_SCOPE )
#endif()
#
#endfunction() 
