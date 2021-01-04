# SANITY CHECK
if( "ilp64" IN_LIST BLIS_FIND_COMPONENTS AND "lp64" IN_LIST BLIS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLIS cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( BLIS_PREFERS_STATIC )
  set( BLIS_LIBRARY_NAME "libblis.a" )
else()
  set( BLIS_LIBRARY_NAME "blis" )
endif()

find_library( BLIS_LIBRARIES
  NAMES ${BLIS_LIBRARY_NAME}
  HINTS ${BLIS_PREFIX}
  PATHS ${BLIS_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "BLIS Library"
)

find_path( BLIS_INCLUDE_DIR
  NAMES blis/blis.h
  HINTS ${BLIS_PREFIX}
  PATHS ${BLIS_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "BLIS header"
)
  
if( BLIS_LIBRARIES )
  find_package( Threads QUIET )
  set( BLIS_LIBRARIES ${BLIS_LIBRARIES} Threads::Threads "m")
endif()

# check ILP64
if( BLIS_INCLUDE_DIR )

  try_run( BLIS_USES_LP64
           _blis_idx_test_compile_result
           ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/util/blis_int_size.c
    CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${BLIS_INCLUDE_DIR}
    COMPILE_OUTPUT_VARIABLE _blis_idx_compile_output
    RUN_OUTPUT_VARIABLE     _blis_idx_run_output
  )

  if( ${BLIS_USES_LP64} EQUAL 0 )
    set( BLIS_USES_LP64 TRUE )
  else()
    set( BLIS_USES_LP64 FALSE )
  endif()

  ## Handle components
  if( BLIS_USES_LP64 )
    set( BLIS_ilp64_FOUND FALSE )
    set( BLIS_lp64_FOUND  TRUE  )
  else()
    set( BLIS_ilp64_FOUND TRUE  )
    set( BLIS_lp64_FOUND  FALSE )
  endif()

endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( BLIS
  REQUIRED_VARS BLIS_LIBRARIES BLIS_INCLUDE_DIR
  HANDLE_COMPONENTS
)

#if( BLIS_FOUND AND NOT TARGET BLIS::BLIS )
#
#  add_library( BLIS::BLIS INTERFACE IMPORTED )
#  set_target_properties( BLIS::BLIS PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
#    INTERFACE_LINK_LIBRARIES      "${BLIS_LIBRARIES}"
#  )
#
#endif()
