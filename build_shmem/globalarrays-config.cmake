cmake_minimum_required( VERSION 3.17 FATAL_ERROR)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was globalarrays-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(ENABLE_BLAS       OFF)
set(ENABLE_CXX        ON)
set(ENABLE_FORTRAN    ON)
set(ENABLE_SCALAPACK  OFF)
set(BUILD_SHARED_LIBS )

set(GA_MPI_LIBS MPI::MPI_C)
if(ENABLE_FORTRAN)
  enable_language(Fortran)
  set(GA_MPI_LIBS MPI::MPI_Fortran)
endif()

if (NOT TARGET Threads::Threads)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
endif()

if(ENABLE_BLAS)
  list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}")
  list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/linalg-cmake-modules")

  set(LINALG_VENDOR BLIS)
  set(LINALG_PREFIX )
  
  set(BLAS_PREFERENCE_LIST      BLIS)
  set(LAPACK_PREFERENCE_LIST    )
  set(ScaLAPACK_PREFERENCE_LIST )

  set(BLIS_PREFERS_STATIC    )
  set(ReferenceLAPACK_PREFERS_STATIC    )
  set(ReferenceScaLAPACK_PREFERS_STATIC )

  set(BLIS_THREAD_LAYER  )
  set(BLAS_REQUIRED_COMPONENTS      )
  set(LAPACK_REQUIRED_COMPONENTS    )
  set(ScaLAPACK_REQUIRED_COMPONENTS )
  set(BLAS_OPTIONAL_COMPONENTS      )
  set(LAPACK_OPTIONAL_COMPONENTS    )
  set(ScaLAPACK_OPTIONAL_COMPONENTS )  

  set(BLAS_PREFIX )
  set(LAPACK_PREFIX )

  set(ENABLE_DPCPP  OFF)
  set(GA_BLAS_ILP64 OFF)

  if(ENABLE_SCALAPACK)
    set(ScaLAPACK_PREFIX )

    if(NOT TARGET ScaLAPACK::ScaLAPACK)
      find_package(ScaLAPACK REQUIRED)
    endif()
  endif()

  if(NOT TARGET LAPACK::LAPACK)
    find_package(LAPACK REQUIRED)
  endif()

  if(NOT TARGET BLAS::BLAS)
    find_package(BLAS REQUIRED)
  endif()

  if(ENABLE_CXX)
    get_filename_component(_ga_la_pp "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
    list(INSERT CMAKE_PREFIX_PATH 0 "${_ga_la_pp}")

    set(use_openmp )

    if(NOT TARGET blaspp)
      find_package(blaspp REQUIRED)
    endif()

    if(NOT TARGET lapackpp)
      find_package(lapackpp REQUIRED)
    endif()

    if(ENABLE_SCALAPACK)
      if(NOT TARGET scalapackpp::scalapackpp)
        find_package(scalapackpp REQUIRED)
      endif()
    endif()

    list(REMOVE_AT CMAKE_PREFIX_PATH 0)
    list(REMOVE_AT CMAKE_MODULE_PATH 0)
    list(REMOVE_AT CMAKE_MODULE_PATH 0)
  endif()
endif()

if(NOT TARGET GlobalArrays::ga)
  include("${CMAKE_CURRENT_LIST_DIR}/globalarrays-targets.cmake")
endif()

if(NOT TARGET MPI::MPI_C)
  find_package(MPI REQUIRED)
endif()


if(NOT TARGET MPI::MPI_Fortran)
  find_package(MPI REQUIRED)
endif()

set(GlobalArrays_FOUND TRUE)
set(GlobalArrays_LIBRARIES GlobalArrays::ga)
set(GlobalArrays_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include/ga" "${MPI_C_INCLUDE_DIRS}")
