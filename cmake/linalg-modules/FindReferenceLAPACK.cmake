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

# if( referencelapack_PREFERS_STATIC )
#   set( referencelapack_LIBRARY_NAME "liblapack.a" )
# else()
set( referencelapack_LIBRARY_NAME liblapack.a lapack )
if(BUILD_SHARED_LIBS)
  set( referencelapack_LIBRARY_NAME lapack )
endif()

set( referencelapack_IPREFIX "${CMAKE_INSTALL_PREFIX}" )
set( referencelapack_PREFIX ${ReferenceLAPACKROOT} $ENV{ReferenceLAPACKROOT} )

find_library( referencelapack_LIBRARY
  NAMES ${referencelapack_LIBRARY_NAME}
  HINTS ${referencelapack_IPREFIX} ${referencelapack_PREFIX}
  PATHS ${referencelapack_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceLAPACK Library"
)

find_package(StandardFortran REQUIRED)

#if( referencelapack_INCLUDE_DIR )
#  set( ReferenceLAPACK_INCLUDE_DIR ${referencelapack_INCLUDE_DIR} )
#endif()

if( referencelapack_LIBRARY AND STANDARDFORTRAN_LIBRARIES )
  set( ReferenceLAPACK_LIBRARIES ${referencelapack_LIBRARY} ${STANDARDFORTRAN_LIBRARIES} )
endif()


# Reference LAPACK is always LP64
set( ReferenceLAPACK_ilp64_FOUND FALSE )
set( ReferenceLAPACK_lp64_FOUND  TRUE  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceLAPACK
#  REQUIRED_VARS ReferenceLAPACK_LIBRARIES ReferenceLAPACK_INCLUDE_DIR
  REQUIRED_VARS ReferenceLAPACK_LIBRARIES
#  VERSION_VAR ReferenceLAPACK_VERSION_STRING
  HANDLE_COMPONENTS
)

if( ReferenceLAPACK_FOUND AND NOT TARGET ReferenceLAPACK::lapack )

  add_library( ReferenceLAPACK::lapack INTERFACE IMPORTED )
  set_target_properties( ReferenceLAPACK::lapack PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${ReferenceLAPACK_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${ReferenceLAPACK_LIBRARIES}"
  )

endif()

