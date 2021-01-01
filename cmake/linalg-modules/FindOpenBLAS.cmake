# SANITY CHECK
if( "ilp64" IN_LIST OpenBLAS_FIND_COMPONENTS AND "lp64" IN_LIST OpenBLAS_FIND_COMPONENTS )
  message( FATAL_ERROR "OpenBLAS cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( OpenBLAS_PREFERS_STATIC )
  set( OpenBLAS_LIBRARY_NAME "libopenblas.a" )
else()
  set( OpenBLAS_LIBRARY_NAME "openblas" )
endif()

if( NOT OpenBLAS_PREFIX )
  set( OpenBLAS_PREFIX ${OpenBLASROOT} $ENV{OpenBLASROOT} )
endif()

find_library( OpenBLAS_LIBRARIES
  NAMES ${OpenBLAS_LIBRARY_NAME}
  HINTS ${OpenBLAS_PREFIX}
  PATHS ${OpenBLAS_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} 
  PATH_SUFFIXES lib lib64 lib32
  DOC "OpenBLAS Library"
)

find_path( OpenBLAS_INCLUDE_DIR
  NAMES openblas_config.h
  HINTS ${OpenBLAS_PREFIX}
  PATHS ${OpenBLAS_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "OpenBLAS header"
)
  
#if( OpenBLAS_LIBRARY AND OpenBLAS_PREFERS_STATIC )
#  include( CMakeFindDependency )
#  find_package( Threads QUIET )
#  set( OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY} Threads::Threads "m")
#endif()

# check ILP64
if( OpenBLAS_INCLUDE_DIR )

  try_run( OpenBLAS_USES_LP64
           _openblas_idx_test_compile_result
           ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/util/openblas_int_size.c
    CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${OpenBLAS_INCLUDE_DIR}
    COMPILE_OUTPUT_VARIABLE _openblas_idx_compile_output
    RUN_OUTPUT_VARIABLE     _openblas_idx_run_output
  )

  if( ${OpenBLAS_USES_LP64} EQUAL 0 )
    set( OpenBLAS_USES_LP64 TRUE )
  else()
    set( OpenBLAS_USES_LP64 FALSE )
  endif()

  ## Handle components
  if( OpenBLAS_USES_LP64 )
    set( OpenBLAS_ilp64_FOUND FALSE )
    set( OpenBLAS_lp64_FOUND  TRUE  )
  else()
    set( OpenBLAS_ilp64_FOUND TRUE  )
    set( OpenBLAS_lp64_FOUND  FALSE )
  endif()

endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( OpenBLAS
  REQUIRED_VARS OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIR
  HANDLE_COMPONENTS
)

#if( OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS )
#
#  add_library( OpenBLAS::OpenBLAS INTERFACE IMPORTED )
#  set_target_properties( OpenBLAS::OpenBLAS PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
#    INTERFACE_LINK_LIBRARIES      "${OpenBLAS_LIBRARIES}"
#  )
#
#endif()
