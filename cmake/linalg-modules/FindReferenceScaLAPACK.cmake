# SANITY CHECK
if( "ilp64" IN_LIST ReferenceScaLAPACK_FIND_COMPONENTS AND "lp64" IN_LIST ReferenceScaLAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "ReferenceScaLAPACK cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( "ilp64" IN_LIST ReferenceScaLAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "ReferenceScaLAPACK ILP64 interface discovery is currently not supported" )
endif()

if( NOT TARGET MPI::MPI_C )
  enable_language(C)
  find_dependency( MPI )
endif()

if( ReferenceScaLAPACK_PREFERS_STATIC )
  set( ReferenceScaLAPACK_LP64_LIBRARY_NAME  "libscalapack.a"   )
  set( ReferenceScaLAPACK_ILP64_LIBRARY_NAME "libscalapack64.a" )
else()
  set( ReferenceScaLAPACK_LP64_LIBRARY_NAME  "scalapack" )
  set( ReferenceScaLAPACK_ILP64_LIBRARY_NAME "scalapack64" )
endif()

if( NOT ReferenceScaLAPACK_PREFIX )
  set( ReferenceScaLAPACK_PREFIX ${ReferenceScaLAPACKROOT} $ENV{ReferenceScaLAPACKROOT} )
endif()

find_library( ReferenceScaLAPACK_LP64_LIBRARIES
  NAMES ${ReferenceScaLAPACK_LP64_LIBRARY_NAME}
  HINTS ${ReferenceScaLAPACK_PREFIX}
  PATHS ${ReferenceScaLAPACK_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceScaLAPACK LP64 Library"
)

if( ReferenceScaLAPACK_LP64_LIBRARIES )
  set( ReferenceScaLAPACK_lp64_FOUND TRUE )
else()
  set( ReferenceScaLAPACK_lp64_FOUND FALSE )
endif()

find_library( ReferenceScaLAPACK_ILP64_LIBRARIES
  NAMES ${ReferenceScaLAPACK_ILP64_LIBRARY_NAME}
  HINTS ${ReferenceScaLAPACK_PREFIX}
  PATHS ${ReferenceScaLAPACK_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceScaLAPACK ILP64 Library"
)

if( ReferenceScaLAPACK_ILP64_LIBRARIES )
  set( ReferenceScaLAPACK_ilp64_FOUND TRUE )
else()
  set( ReferenceScaLAPACK_ilp64_FOUND FALSE )
endif()

# Default to LP64
if( "ilp64" IN_LIST ReferenceScaLAPACK_FIND_COMPONENTS )
  set( ReferenceScaLAPACK_LIBRARIES ${ReferenceScaLAPACK_ILP64_LIBRARIES} )
else()
  set( ReferenceScaLAPACK_LIBRARIES ${ReferenceScaLAPACK_LP64_LIBRARIES} )
endif()

list( APPEND ReferenceScaLAPACK_LIBRARIES MPI::MPI_C )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceScaLAPACK
  REQUIRED_VARS ReferenceScaLAPACK_LIBRARIES
  HANDLE_COMPONENTS
)

#if( ReferenceScaLAPACK_FOUND AND NOT TARGET ReferenceScaLAPACK::ReferenceScaLAPACK )
#
#  add_library( ReferenceScaLAPACK::ReferenceScaLAPACK INTERFACE IMPORTED )
#  set_target_properties( ReferenceScaLAPACK::ReferenceScaLAPACK PROPERTIES
#    INTERFACE_LINK_LIBRARIES      "${ReferenceScaLAPACK_LIBRARIES}"
#  )
#
#endif()
