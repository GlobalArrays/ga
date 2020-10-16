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

if( openblas_PREFERS_STATIC )
  set( openblas_LIBRARY_NAME "libopenblas.a" )
else()
  set( openblas_LIBRARY_NAME "openblas" )
endif()

if( NOT openblas_PREFIX )
  set( openblas_PREFIX ${OpenBLASROOT} $ENV{OpenBLASROOT} )
endif()

find_library( openblas_LIBRARY
  NAMES ${openblas_LIBRARY_NAME}
  HINTS ${openblas_PREFIX}
  PATHS ${openblas_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "OpenBLAS Library"
)

#if( openblas_INCLUDE_DIR )
#  set( OpenBLAS_INCLUDE_DIR ${openblas_INCLUDE_DIR} )
#endif()

if( openblas_LIBRARY )
  find_package( OpenMP QUIET )
  if( NOT gfortran_LIBRARY )
    find_library( gfortran_LIBRARY 
      NAMES gfortran 
      PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
      DOC "GFortran Library" 
    )
  endif()
  set( OpenBLAS_LIBRARIES ${openblas_LIBRARY} OpenMP::OpenMP_C "m" ${gfortran_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( OpenBLAS
#  REQUIRED_VARS OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIR
  REQUIRED_VARS OpenBLAS_LIBRARIES
#  VERSION_VAR OpenBLAS_VERSION_STRING
  HANDLE_COMPONENTS
)

if( OpenBLAS_FOUND AND NOT TARGET OpenBLAS::openblas )

  add_library( OpenBLAS::openblas INTERFACE IMPORTED )
  set_target_properties( OpenBLAS::openblas PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${OpenBLAS_LIBRARIES}"
  )

endif()
