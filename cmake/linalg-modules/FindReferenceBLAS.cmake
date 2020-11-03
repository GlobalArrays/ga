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

set( referenceblas_LIBRARY_NAME libblis.a blis )
if(BUILD_SHARED_LIBS)
  set( referenceblas_LIBRARY_NAME blis )
endif()

set( referenceblas_IPREFIX "${CMAKE_INSTALL_PREFIX}" )
set( referenceblas_PREFIX ${ReferenceBLASROOT} $ENV{ReferenceBLASROOT} )

find_path( referenceblas_INCLUDE_DIR
  NAMES blis/blis.h
  HINTS ${referenceblas_IPREFIX} ${referenceblas_PREFIX}
  PATHS ${referenceblas_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "Reference BLAS header"
)

find_library( referenceblas_LIBRARY
  NAMES ${referenceblas_LIBRARY_NAME}
  HINTS ${referenceblas_IPREFIX} ${referenceblas_PREFIX}
  PATHS ${referenceblas_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "Reference BLAS Library"
)

# Reference BLAS is always LP64
set( ReferenceBLAS_ilp64_FOUND FALSE )
set( ReferenceBLAS_lp64_FOUND  TRUE  )

if( referenceblas_INCLUDE_DIR )
  set( ReferenceBLAS_INCLUDE_DIR ${referenceblas_INCLUDE_DIR} )
endif()

if( referenceblas_LIBRARY )
  find_package( Threads QUIET )
  set( ReferenceBLAS_LIBRARIES ${referenceblas_LIBRARY} Threads::Threads "m")
endif()

# list(APPEND ReferenceBLAS_COMPILE_DEFINITIONS "BLA_LAPACK_INT=int32_t")
# list(APPEND ReferenceBLAS_COMPILE_DEFINITIONS "BLA_LAPACK_COMPLEX8=std::complex<float>")
# list(APPEND ReferenceBLAS_COMPILE_DEFINITIONS "BLA_LAPACK_COMPLEX16=std::complex<double>")
# list(APPEND ReferenceBLAS_COMPILE_DEFINITIONS "BLA_VENDOR_REFERENCE")
# list(APPEND ReferenceBLAS_COMPILE_DEFINITIONS "USE_BLIS")

set(USE_BLIS ON)
set(BLA_VENDOR_REFERENCE ON)
set(BLA_LAPACK_INT "int32_t")
set(BLA_LAPACK_COMPLEX8  "std::complex<float>")
set(BLA_LAPACK_COMPLEX16 "std::complex<double>")

include(FindPackageHandleStandardArgs)
is_valid(ReferenceBLAS_C_COMPILE_FLAGS __has_cflags)
if(__has_cflags)
  find_package_handle_standard_args( ReferenceBLAS
    REQUIRED_VARS ReferenceBLAS_LIBRARIES ReferenceBLAS_INCLUDE_DIR 
    ReferenceBLAS_C_COMPILE_FLAGS 
    VERSION_VAR ReferenceBLAS_VERSION_STRING
    HANDLE_COMPONENTS
  )
else()
  find_package_handle_standard_args( ReferenceBLAS
    REQUIRED_VARS ReferenceBLAS_LIBRARIES ReferenceBLAS_INCLUDE_DIR   
    VERSION_VAR ReferenceBLAS_VERSION_STRING
    HANDLE_COMPONENTS
  )
endif()

if( ReferenceBLAS_FOUND AND NOT TARGET ReferenceBLAS::blas )

  add_library( ReferenceBLAS::blas INTERFACE IMPORTED )
  set_target_properties( ReferenceBLAS::blas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ReferenceBLAS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${ReferenceBLAS_LIBRARIES}"
    INTERFACE_COMPILE_OPTIONS     "${ReferenceBLAS_C_COMPILE_FLAGS}"
    # INTERFACE_COMPILE_DEFINITIONS "${ReferenceBLAS_COMPILE_DEFINITIONS}"    
  )

endif()

