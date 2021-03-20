set( MPI_UTILITY_CMAKE_FILE_DIR ${CMAKE_CURRENT_LIST_DIR} )

function( get_mpi_vendor _mpi_libs _mpi_vendor )

if( NOT TARGET ${_mpi_libs} )
  set( _mpi_linker ${${_mpi_libs}} )
else()
  set( _mpi_linker ${_mpi_libs} )
endif()

try_run( _run_result _compile_result ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES        ${MPI_UTILITY_CMAKE_FILE_DIR}/get_mpi_vendor.c
  LINK_LIBRARIES ${_mpi_linker}
  COMPILE_OUTPUT_VARIABLE _compile_output
  RUN_OUTPUT_VARIABLE     _run_output
)

string( TOUPPER "${_run_output}" _run_output )

if( _run_output MATCHES "CRAY" )
  set( ${_mpi_vendor} "CRAY" PARENT_SCOPE )
elseif( _run_output MATCHES "OPEN MPI" )
  set( ${_mpi_vendor} "OPENMPI" PARENT_SCOPE )
elseif( _run_output MATCHES "OPENMPI" )
  set( ${_mpi_vendor} "OPENMPI" PARENT_SCOPE )
elseif( _run_output MATCHES "MVAPICH" )
  set( ${_mpi_vendor} "MVAPICH" PARENT_SCOPE )
elseif( _run_output MATCHES "MPICH" )
  set( ${_mpi_vendor} "MPICH" PARENT_SCOPE )
else()
  set( ${_mpi_vendor} "UNKNOWN" PARENT_SCOPE )
endif()



endfunction()
