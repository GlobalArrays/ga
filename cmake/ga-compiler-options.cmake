
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

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
  if(NOT "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
      get_filename_component(__GA_GCC_INSTALL_PREFIX "${CMAKE_Fortran_COMPILER}/../.." ABSOLUTE)
      if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
          set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=/usr/")
        # JBMF: temporary fix for junction  
	#set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${__GA_GCC_INSTALL_PREFIX}")
      else()
          if(GCCROOT)
              set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=/usr/")
              #set(GA_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${GCCROOT}")
          else()
              message(FATAL_ERROR "GCCROOT variable not set when using clang compilers. \
                  The GCCROOT path can be found using the command: \"which gcc\" ")
          endif()
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


