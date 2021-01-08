set( LINALG_MACROS_DIR ${CMAKE_CURRENT_LIST_DIR} )

macro( find_linalg_dependencies _libs )
  foreach( _lib ${${_libs}} )
    if (${_lib} MATCHES "OpenMP::OpenMP")
      find_package(OpenMP)
    elseif (${_lib} MATCHES "Threads::Threads")
      find_package(Threads)
    endif()
  endforeach()
endmacro()

function( install_linalg_modules _dest_dir )

set( LINALG_FIND_MODULES
     FindBLAS.cmake
     FindBLIS.cmake
     FindIBMESSL.cmake
     FindIntelMKL.cmake
     FindLAPACK.cmake
     FindOpenBLAS.cmake
     FindReferenceBLAS.cmake
     FindReferenceLAPACK.cmake
     FindReferenceScaLAPACK.cmake
     FindScaLAPACK.cmake
     FindStandardFortran.cmake
     LinAlgModulesMacros.cmake
)

set( LINALG_UTIL_FILES
     util/blis_int_size.c
     util/func_check.c
     util/get_mpi_vendor.c
     util/ilp64_checker.c
     util/lapack_ilp64_checker.c
     util/openblas_int_size.c
     util/BLASUtilities.cmake
     util/CommonFunctions.cmake
     util/IntrospectMPI.cmake
     util/IntrospectOpenMP.cmake
     util/LAPACKUtilities.cmake
     util/ScaLAPACKUtilities.cmake )

list( TRANSFORM LINALG_FIND_MODULES
      PREPEND   ${LINALG_MACROS_DIR}/ )
list( TRANSFORM LINALG_UTIL_FILES
      PREPEND   ${LINALG_MACROS_DIR}/ )

install(
  FILES ${LINALG_FIND_MODULES} ${LINALG_MACROS_DIR}/LICENSE.txt
  DESTINATION ${${_dest_dir}}/linalg-cmake-modules
)

install(
  FILES ${LINALG_UTIL_FILES}
  DESTINATION ${${_dest_dir}}/linalg-cmake-modules/util
)

endfunction()
