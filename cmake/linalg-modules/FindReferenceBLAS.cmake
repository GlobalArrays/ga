# SANITY CHECK
if( "ilp64" IN_LIST ReferenceBLAS_FIND_COMPONENTS AND "lp64" IN_LIST ReferenceBLAS_FIND_COMPONENTS )
  message( FATAL_ERROR "ReferenceBLAS cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( ReferenceBLAS_PREFERS_STATIC )
  set( ReferenceBLAS_LP64_LIBRARY_NAME  "libblas.a"   )
  set( ReferenceBLAS_ILP64_LIBRARY_NAME "libblas64.a" )
else()
  set( ReferenceBLAS_LP64_LIBRARY_NAME  "blas" )
  set( ReferenceBLAS_ILP64_LIBRARY_NAME "blas64" )
endif()

find_library( ReferenceBLAS_LP64_LIBRARIES
  NAMES ${ReferenceBLAS_LP64_LIBRARY_NAME}
  HINTS ${ReferenceBLAS_PREFIX}
  PATHS ${ReferenceBLAS_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceBLAS LP64 Library"
)

if( ReferenceBLAS_LP64_LIBRARIES )
  set( ReferenceBLAS_lp64_FOUND TRUE )
else()
  set( ReferenceBLAS_lp64_FOUND FALSE )
endif()

find_library( ReferenceBLAS_ILP64_LIBRARIES
  NAMES ${ReferenceBLAS_ILP64_LIBRARY_NAME}
  HINTS ${ReferenceBLAS_PREFIX}
  PATHS ${ReferenceBLAS_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceBLAS ILP64 Library"
)

if( ReferenceBLAS_ILP64_LIBRARIES )
  set( ReferenceBLAS_ilp64_FOUND TRUE )
else()
  set( ReferenceBLAS_ilp64_FOUND FALSE )
endif()

# Default to LP64
if( "ilp64" IN_LIST ReferenceBLAS_FIND_COMPONENTS )
  set( ReferenceBLAS_LIBRARIES ${ReferenceBLAS_ILP64_LIBRARIES} )
else()
  set( ReferenceBLAS_LIBRARIES ${ReferenceBLAS_LP64_LIBRARIES} )
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceBLAS
  REQUIRED_VARS ReferenceBLAS_LIBRARIES
  HANDLE_COMPONENTS
)

#if( ReferenceBLAS_FOUND AND NOT TARGET ReferenceBLAS::ReferenceBLAS )
#
#  add_library( ReferenceBLAS::ReferenceBLAS INTERFACE IMPORTED )
#  set_target_properties( ReferenceBLAS::ReferenceBLAS PROPERTIES
#    INTERFACE_LINK_LIBRARIES      "${ReferenceBLAS_LIBRARIES}"
#  )
#
#endif()
