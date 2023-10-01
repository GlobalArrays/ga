
#Tmp Fix: segfaults with the presence of -DNDEBUG
if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
    set(__cmbt_upper RELEASE)
  else()
    string( TOUPPER ${CMAKE_BUILD_TYPE} __cmbt_upper )
  endif()
  set(CMAKE_C_FLAGS_${__cmbt_upper} "-O3 -g")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    if(GCCROOT)
      set(__GA_GCC_INSTALL_PREFIX ${GCCROOT})
      set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${GCCROOT}")
    else()
      get_filename_component(__GA_GCC_INSTALL_PREFIX "${CMAKE_Fortran_COMPILER}/../.." ABSOLUTE)
      if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${__GA_GCC_INSTALL_PREFIX}")
      else()
        message(FATAL_ERROR "GCCROOT cmake option not set when using clang compilers. \
                Please set a valid path to the GCC installation.")
      endif()
    endif()
    #Check GCC installation
    if(NOT 
       (EXISTS ${__GA_GCC_INSTALL_PREFIX}/bin AND
        EXISTS ${__GA_GCC_INSTALL_PREFIX}/include AND
        EXISTS ${__GA_GCC_INSTALL_PREFIX}/lib)
      )
      message(FATAL_ERROR "GCC installation path found ${__GA_GCC_INSTALL_PREFIX} seems to be incorrect. \
      Please set the GCCROOT cmake option to the correct GCC installation prefix.")
    endif()
    message(STATUS "GA_GCC_TOOLCHAIN_FLAG: ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GA_GCC_TOOLCHAIN_FLAG}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${GA_GCC_TOOLCHAIN_FLAG}")
  endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_C_COMPILER_VERSION VERSION_EQUAL CMAKE_Fortran_COMPILER_VERSION)
    message(STATUS "Check GNU compiler versions.")
  else()
    message(STATUS "GNU C and Fortran compiler versions do not match")
    message(FATAL_ERROR "GNU Compiler versions provided: gcc: ${CMAKE_C_COMPILER_VERSION}, gfortran version: ${CMAKE_Fortran_COMPILER_VERSION}")
  endif()
endif()

