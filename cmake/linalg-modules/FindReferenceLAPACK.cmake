# SANITY CHECK
if( "ilp64" IN_LIST ReferenceLAPACK_FIND_COMPONENTS AND "lp64" IN_LIST ReferenceLAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "ReferenceLAPACK cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( ReferenceLAPACK_PREFERS_STATIC )
  set( ReferenceLAPACK_LP64_LIBRARY_NAME  "liblapack.a"   )
  set( ReferenceLAPACK_ILP64_LIBRARY_NAME "liblapack64.a" )
else()
  set( ReferenceLAPACK_LP64_LIBRARY_NAME  "lapack" )
  set( ReferenceLAPACK_ILP64_LIBRARY_NAME "lapack64" )
endif()

if( NOT ReferenceLAPACK_PREFIX )
  set( ReferenceLAPACK_PREFIX ${ReferenceLAPACKROOT} $ENV{ReferenceLAPACKROOT} )
endif()

find_library( ReferenceLAPACK_LP64_LIBRARIES
  NAMES ${ReferenceLAPACK_LP64_LIBRARY_NAME}
  HINTS ${ReferenceLAPACK_PREFIX}
  PATHS ${ReferenceLAPACK_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceLAPACK LP64 Library"
)

if( ReferenceLAPACK_LP64_LIBRARIES )
  set( ReferenceLAPACK_lp64_FOUND TRUE )
else()
  set( ReferenceLAPACK_lp64_FOUND FALSE )
endif()

find_library( ReferenceLAPACK_ILP64_LIBRARIES
  NAMES ${ReferenceLAPACK_ILP64_LIBRARY_NAME}
  HINTS ${ReferenceLAPACK_PREFIX}
  PATHS ${ReferenceLAPACK_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceLAPACK ILP64 Library"
)

if( ReferenceLAPACK_ILP64_LIBRARIES )
  set( ReferenceLAPACK_ilp64_FOUND TRUE )
else()
  set( ReferenceLAPACK_ilp64_FOUND FALSE )
endif()

# Default to LP64
if( "ilp64" IN_LIST ReferenceLAPACK_FIND_COMPONENTS )
  set( ReferenceLAPACK_LIBRARIES ${ReferenceLAPACK_ILP64_LIBRARIES} )
else()
  set( ReferenceLAPACK_LIBRARIES ${ReferenceLAPACK_LP64_LIBRARIES} )
endif()

find_package(StandardFortran REQUIRED)

if( STANDARDFORTRAN_LIBRARIES )
  set( ReferenceLAPACK_LIBRARIES ${ReferenceLAPACK_LIBRARIES} ${STANDARDFORTRAN_LIBRARIES} )
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceLAPACK
  REQUIRED_VARS ReferenceLAPACK_LIBRARIES
  HANDLE_COMPONENTS
)

#if( ReferenceLAPACK_FOUND AND NOT TARGET ReferenceLAPACK::ReferenceLAPACK )
#
#  add_library( ReferenceLAPACK::ReferenceLAPACK INTERFACE IMPORTED )
#  set_target_properties( ReferenceLAPACK::ReferenceLAPACK PROPERTIES
#    INTERFACE_LINK_LIBRARIES      "${ReferenceLAPACK_LIBRARIES}"
#  )
#
#endif()
