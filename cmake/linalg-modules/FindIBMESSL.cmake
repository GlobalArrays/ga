#==================================================================
#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
#   
#   This file is part of cmake-modules. All rights reserved.
#   
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   
#   (1) Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#   (2) Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#   (3) Neither the name of the University of California, Lawrence Berkeley
#   National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#   
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#   
#   You are under no obligation whatsoever to provide any bug fixes, patches, or
#   upgrades to the features, functionality or performance of the source code
#   ("Enhancements") to anyone; however, if you choose to make your Enhancements
#   available either publicly, or directly to Lawrence Berkeley National
#   Laboratory, without imposing a separate written license agreement for such
#   Enhancements, then you hereby grant the following license: a non-exclusive,
#   royalty-free perpetual license to install, use, modify, prepare derivative
#   works, incorporate into other computer software, distribute, and sublicense
#   such enhancements or derivative works thereof, in binary and source code form.
#
#==================================================================

include(CommonFunctions)

if( "ilp64" IN_LIST IBMESSL_FIND_COMPONENTS AND "lp64" IN_LIST IBMESSL_FIND_COMPONENTS )
  message( FATAL_ERROR "IBMESSL cannot link to both ILP64 and LP64 iterfaces" )
endif()

set( ibmessl_LP64_SERIAL_LIBRARY_NAME  "essl"        )
set( ibmessl_LP64_SMP_LIBRARY_NAME     "esslsmp"     )
set( ibmessl_ILP64_SERIAL_LIBRARY_NAME "essl6464"    )
set( ibmessl_ILP64_SMP_LIBRARY_NAME    "esslsmp6464" )


if( NOT ibmessl_PREFERED_THREAD_LEVEL )
  set( ibmessl_PREFERED_THREAD_LEVEL "smp" )
endif()

if( ibmessl_PREFERED_THREAD_LEVEL MATCHES "smp" )
  set( ibmessl_LP64_LIBRARY_NAME  ${ibmessl_LP64_SMP_LIBRARY_NAME}  )
  set( ibmessl_ILP64_LIBRARY_NAME ${ibmessl_ILP64_SMP_LIBRARY_NAME} )
else()
  set( ibmessl_LP64_LIBRARY_NAME  ${ibmessl_LP64_SERIAL_LIBRARY_NAME}  )
  set( ibmessl_ILP64_LIBRARY_NAME ${ibmessl_ILP64_SERIAL_LIBRARY_NAME} )
endif()

if( NOT ibmessl_PREFIX )
  set( ibmessl_PREFIX ${ESSLROOT} $ENV{ESSLROOT} )
endif()

find_path( ibmessl_INCLUDE_DIR
  NAMES essl.h
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "IBM(R) ESSL header"
)

find_library( ibmessl_LP64_LIBRARY
  NAMES ${ibmessl_LP64_LIBRARY_NAME}
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "IBM(R) ESSL Library"
)

find_library( ibmessl_ILP64_LIBRARY
  NAMES ${ibmessl_LP64_LIBRARY_NAME}
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "IBM(R) ESSL Library"
)

if( ibmessl_ILP64_LIBRARY )
  set( IBMESSL_ilp64_FOUND TRUE )
endif()

if( ibmessl_LP64_LIBRARY )
  set( IBMESSL_lp64_FOUND TRUE )
endif()

if( "ilp64" IN_LIST IBMESSL_FIND_COMPONENTS )
  set( ibmessl_LIBRARY ${ibmessl_ILP64_LIBRARY} )
  set(LAPACK_ILP64 ON)
  # set(IBMESSL_COMPILE_DEFINITIONS "LAPACK_ILP64")
  set(IBMESSL_C_COMPILE_FLAGS        "-m64" )
  set(BLA_LAPACK_INT "int64_t")
  # list(APPEND IBMESSL_COMPILE_DEFINITIONS "BLA_LAPACK_INT=int64_t")
else()
  set( ibmessl_LIBRARY ${ibmessl_LP64_LIBRARY} )
  set(BLA_LAPACK_INT "int32_t")
  # list(APPEND IBMESSL_COMPILE_DEFINITIONS "BLA_LAPACK_INT=int32_t")
endif()



if( ibmessl_INCLUDE_DIR )
  set( IBMESSL_INCLUDE_DIR ${ibmessl_INCLUDE_DIR} )
endif()

if( ibmessl_LIBRARY )
  set( IBMESSL_LIBRARIES ${ibmessl_LIBRARY} )
endif()

# list(APPEND IBMESSL_COMPILE_DEFINITIONS "BLA_LAPACK_COMPLEX8=std::complex<float>")
# list(APPEND IBMESSL_COMPILE_DEFINITIONS "BLA_LAPACK_COMPLEX16=std::complex<double>")
# list(APPEND IBMESSL_COMPILE_DEFINITIONS "BLA_VENDOR_ESSL")

set(BLA_VENDOR_ESSL ON)
set(BLA_LAPACK_COMPLEX8  "std::complex<float>")
set(BLA_LAPACK_COMPLEX16 "std::complex<double>")


include(FindPackageHandleStandardArgs)
is_valid(IBMESSL_C_COMPILE_FLAGS __has_cflags)
if(__has_cflags)
  find_package_handle_standard_args( IBMESSL
    REQUIRED_VARS IBMESSL_LIBRARIES IBMESSL_INCLUDE_DIR
      IBMESSL_C_COMPILE_FLAGS
  #  VERSION_VAR IBMESSL_VERSION_STRING
    HANDLE_COMPONENTS
  )
else()
  find_package_handle_standard_args( IBMESSL
    REQUIRED_VARS IBMESSL_LIBRARIES IBMESSL_INCLUDE_DIR
  #  VERSION_VAR IBMESSL_VERSION_STRING
    HANDLE_COMPONENTS
  )
endif()

if( IBMESSL_FOUND AND NOT TARGET IBMESSL::essl )

  add_library( IBMESSL::essl INTERFACE IMPORTED )
  set_target_properties( IBMESSL::essl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${IBMESSL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${IBMESSL_LIBRARIES}"
    INTERFACE_COMPILE_OPTIONS     "${IBMESSL_C_COMPILE_FLAGS}"
    # INTERFACE_COMPILE_DEFINITIONS "${IBMESSL_COMPILE_DEFINITIONS}"    
  )

endif()
