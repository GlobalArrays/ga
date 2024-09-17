set( BLAS_UTILITY_CMAKE_FILE_DIR ${CMAKE_CURRENT_LIST_DIR} )

function( check_blas_int _libs _dgemm_name _libs_are_lp64 )

try_run( _run_result _compile_result ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES        ${BLAS_UTILITY_CMAKE_FILE_DIR}/ilp64_checker.c
  LINK_LIBRARIES ${${_libs}}
  COMPILE_DEFINITIONS "-DDGEMM_NAME=${_dgemm_name}"
  COMPILE_OUTPUT_VARIABLE _compile_output
  RUN_OUTPUT_VARIABLE     _run_output
)

if (NOT _compile_result)
  message(FATAL_ERROR "check_blas_int: try_run failed: _compile_output=${_compile_output}")
endif()

if( _run_output MATCHES "BLAS IS LP64" )
  set( ${_libs_are_lp64} TRUE PARENT_SCOPE )
else()
  set( ${_libs_are_lp64} FALSE PARENT_SCOPE )
endif()

endfunction() 
