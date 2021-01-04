#    FindIntelMKL.cmake
#
#    Finds Intel(R) MKL
#
#    The module will define the following variables:
#    
#      IntelMKL_FOUND       - Found MKL installation
#      IntelMKL_INCLUDE_DIR - Location of MKL headers (mkl.h)
#      IntelMKL_LIBRARIES   - MKL libraries
#
#    The find behaviour of the module can be influenced by the following
#
#      IntelMKL_PREFERS_STATIC - default OFF
#      IntelMKL_THREAD_LAYER   - ( sequential, openmp, tbb ) default: openmp
#      IntelMKL_OMP_LIBRARY    - ( Intel, GNU, PGI )         default: depends on compiler


include( CMakeFindDependencyMacro )

# SANITY CHECK
if( "ilp64" IN_LIST IntelMKL_FIND_COMPONENTS AND "lp64" IN_LIST IntelMKL_FIND_COMPONENTS )
  message( FATAL_ERROR "IntelMKL cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( "scalapack" IN_LIST IntelMKL_FIND_COMPONENTS AND NOT ("blacs" IN_LIST IntelMKL_FIND_COMPONENTS) )
  list(APPEND IntelMKL_FIND_COMPONENTS "blacs" )
endif()

# MKL lib names
if( IntelMKL_PREFERS_STATIC )
  set( IntelMKL_LP64_LIBRARY_NAME       "libmkl_intel_lp64.a"   )
  set( IntelMKL_ILP64_LIBRARY_NAME      "libmkl_intel_ilp64.a"  )
  set( IntelMKL_SEQUENTIAL_LIBRARY_NAME "libmkl_sequential.a"   )
  set( IntelMKL_OMP_INTEL_LIBRARY_NAME  "libmkl_intel_thread.a" )
  set( IntelMKL_OMP_GNU_LIBRARY_NAME    "libmkl_gnu_thread.a"   )
  set( IntelMKL_OMP_PGI_LIBRARY_NAME    "libmkl_pgi_thread.a"   )
  set( IntelMKL_TBB_LIBRARY_NAME        "libmkl_tbb_thread.a"   )
  set( IntelMKL_CORE_LIBRARY_NAME       "libmkl_core.a"         )
  set( IntelMKL_SYCL_LIBRARY_NAME       "libmkl_sycl.a"         )

  set( IntelMKL_LP64_ScaLAPACK_LIBRARY_NAME  "libmkl_scalapack_lp64.a"  )
  set( IntelMKL_ILP64_ScaLAPACK_LIBRARY_NAME "libmkl_scalapack_ilp64.a" )

  set( IntelMKL_LP64_INTELMPI_BLACS_LIBRARY_NAME  "libmkl_blacs_intelmpi_lp64.a"  )
  set( IntelMKL_LP64_OPENMPI_BLACS_LIBRARY_NAME   "libmkl_blacs_openmpi_lp64.a"   )
  set( IntelMKL_LP64_SGIMPT_BLACS_LIBRARY_NAME    "libmkl_blacs_sgimpt_lp64.a"    )
  set( IntelMKL_ILP64_INTELMPI_BLACS_LIBRARY_NAME "libmkl_blacs_intelmpi_ilp64.a" )
  set( IntelMKL_ILP64_OPENMPI_BLACS_LIBRARY_NAME  "libmkl_blacs_openmpi_ilp64.a"  )
  set( IntelMKL_ILP64_SGIMPT_BLACS_LIBRARY_NAME   "libmkl_blacs_sgimpt_ilp64.a"   )
else()
  set( IntelMKL_LP64_LIBRARY_NAME       "mkl_intel_lp64"   )
  set( IntelMKL_ILP64_LIBRARY_NAME      "mkl_intel_ilp64"  )
  set( IntelMKL_SEQUENTIAL_LIBRARY_NAME "mkl_sequential"   )
  set( IntelMKL_OMP_INTEL_LIBRARY_NAME  "mkl_intel_thread" )
  set( IntelMKL_OMP_GNU_LIBRARY_NAME    "mkl_gnu_thread"   )
  set( IntelMKL_OMP_PGI_LIBRARY_NAME    "mkl_pgi_thread"   )
  set( IntelMKL_TBB_LIBRARY_NAME        "mkl_tbb_thread"   )
  set( IntelMKL_CORE_LIBRARY_NAME       "mkl_core"         )
  set( IntelMKL_SYCL_LIBRARY_NAME       "mkl_sycl"         )

  set( IntelMKL_LP64_ScaLAPACK_LIBRARY_NAME  "mkl_scalapack_lp64"  )
  set( IntelMKL_ILP64_ScaLAPACK_LIBRARY_NAME "mkl_scalapack_ilp64" )

  set( IntelMKL_LP64_INTELMPI_BLACS_LIBRARY_NAME  "mkl_blacs_intelmpi_lp64"  )
  set( IntelMKL_LP64_OPENMPI_BLACS_LIBRARY_NAME   "mkl_blacs_openmpi_lp64"   )
  set( IntelMKL_LP64_SGIMPT_BLACS_LIBRARY_NAME    "mkl_blacs_sgimpt_lp64"    )
  set( IntelMKL_ILP64_INTELMPI_BLACS_LIBRARY_NAME "mkl_blacs_intelmpi_ilp64" )
  set( IntelMKL_ILP64_OPENMPI_BLACS_LIBRARY_NAME  "mkl_blacs_openmpi_ilp64"  )
  set( IntelMKL_ILP64_SGIMPT_BLACS_LIBRARY_NAME   "mkl_blacs_sgimpt_ilp64"   )
endif()


# Defaults
if( NOT IntelMKL_THREAD_LAYER )
  set( IntelMKL_THREAD_LAYER "openmp" )
endif()

if( NOT IntelMKL_MPI_LIBRARY )
  set( IntelMKL_MPI_LIBRARY "intelmpi" )
endif()

if( NOT IntelMKL_PREFIX )
  set( IntelMKL_PREFIX $ENV{MKLROOT} )
endif()



# MKL Threads
if( IntelMKL_THREAD_LAYER MATCHES "sequential" )

  # Sequential
  set( IntelMKL_THREAD_LIBRARY_NAME ${IntelMKL_SEQUENTIAL_LIBRARY_NAME} )

elseif( IntelMKL_THREAD_LAYER MATCHES "tbb" )

  # TBB
  set( IntelMKL_THREAD_LIBRARY_NAME ${IntelMKL_TBB_LIBRARY_NAME} )

else() 

  # OpenMP
  if( NOT IntelMKL_OMP_LIBRARY )

    include( ${CMAKE_CURRENT_LIST_DIR}/util/IntrospectOpenMP.cmake )
    if( NOT TARGET OpenMP::OpenMP_C )
      find_dependency( OpenMP )
    endif()
    check_openmp_is_gnu( OpenMP::OpenMP_C OMP_IS_GNU )

    if( OMP_IS_GNU )
      set( IntelMKL_OMP_LIBRARY "GNU" )
    elseif( CMAKE_C_COMPILER_ID MATCHES "Intel" )
      set( IntelMKL_OMP_LIBRARY "Intel" )
    elseif( CMAKE_C_COMPILER_ID MATCHES "PGI" )
      set( IntelMKL_OMP_LIBRARY "PGI" )
    else()
      message( WARNING 
               "OpenMP Could Not Be Introspected -- Defauting to GNU for MKL" )
      set( IntelMKL_OMP_LIBRARY "GNU" )
    endif()

  endif()



  if( IntelMKL_OMP_LIBRARY MATCHES "Intel" )
    set( IntelMKL_THREAD_LIBRARY_NAME ${IntelMKL_OMP_INTEL_LIBRARY_NAME} )
  elseif( IntelMKL_OMP_LIBRARY MATCHES "PGI" )
    set( IntelMKL_THREAD_LIBRARY_NAME ${IntelMKL_OMP_PGI_LIBRARY_NAME} )
  else() # Default to GNU OpenMP
    set( IntelMKL_THREAD_LIBRARY_NAME ${IntelMKL_OMP_GNU_LIBRARY_NAME} )
  endif()

endif()


# MKL MPI for BLACS
if( "blacs" IN_LIST IntelMKL_FIND_COMPONENTS )

  if( NOT TARGET MPI::MPI_C )
    find_dependency( MPI )
  endif()

  if( NOT IntelMPI_MPI_LIBRARY )
    include( ${CMAKE_CURRENT_LIST_DIR}/util/IntrospectMPI.cmake )
    get_mpi_vendor( MPI::MPI_C MPI_VENDOR )
    if( MPI_VENDOR MATCHES "OPENMPI" )
      set( IntelMKL_MPI_LIBRARY "openmpi" )
    elseif( MPI_VENDOR MATCHES "SGIMPT" )
      set( IntelMKL_MPI_LIBRARY "sgimpt" )
    else() # Default to MPICH ABI
      set( IntelMKL_MPI_LIBRARY "mpich" )
    endif()
  endif()
  


  if( IntelMKL_MPI_LIBRARY MATCHES "openmpi" )
    set( IntelMKL_LP64_BLACS_LIBRARY_NAME  ${IntelMKL_LP64_OPENMPI_BLACS_LIBRARY_NAME}  )
    set( IntelMKL_ILP64_BLACS_LIBRARY_NAME ${IntelMKL_ILP64_OPENMPI_BLACS_LIBRARY_NAME} )
  elseif( IntelMKL_MPI_LIBRARY MATCHES "sgimpt" )
    set( IntelMKL_LP64_BLACS_LIBRARY_NAME  ${IntelMKL_LP64_SGIMPT_BLACS_LIBRARY_NAME}  )
    set( IntelMKL_ILP64_BLACS_LIBRARY_NAME ${IntelMKL_ILP64_SGIMPT_BLACS_LIBRARY_NAME} )
  else() # Default to MPICH ABI
    set( IntelMKL_LP64_BLACS_LIBRARY_NAME  ${IntelMKL_LP64_INTELMPI_BLACS_LIBRARY_NAME}  )
    set( IntelMKL_ILP64_BLACS_LIBRARY_NAME ${IntelMKL_ILP64_INTELMPI_BLACS_LIBRARY_NAME} )
  endif()

endif()


# Header
if( NOT IntelMKL_INCLUDE_DIR )
  find_path( IntelMKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${IntelMKL_PREFIX}
    PATH_SUFFIXES include
    DOC "Intel(R) MKL header"
  )
endif()

find_library( IntelMKL_THREAD_LIBRARY
  NAMES ${IntelMKL_THREAD_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) MKL THREAD Library"
)

find_library( IntelMKL_CORE_LIBRARY
  NAMES ${IntelMKL_CORE_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) MKL CORE Library"
)

# Check version
if( EXISTS ${IntelMKL_INCLUDE_DIR}/mkl_version.h )
  set( version_pattern 
  "^#define[\t ]+__INTEL_MKL(|_MINOR|_UPDATE)__[\t ]+([0-9\\.]+)$"
  )
  file( STRINGS ${IntelMKL_INCLUDE_DIR}/mkl_version.h mkl_version
        REGEX ${version_pattern} )

  foreach( match ${mkl_version} )
  
    if(IntelMKL_VERSION_STRING)
      set(IntelMKL_VERSION_STRING "${IntelMKL_VERSION_STRING}.")
    endif()
  
    string(REGEX REPLACE ${version_pattern} 
      "${IntelMKL_VERSION_STRING}\\2" 
      IntelMKL_VERSION_STRING ${match}
    )
  
    set(IntelMKL_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  
  endforeach()
  
  unset( mkl_version )
  unset( version_pattern )
endif()



# Handle LP64 / ILP64
find_library( IntelMKL_ILP64_LIBRARY
  NAMES ${IntelMKL_ILP64_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) ILP64 MKL Library"
)

find_library( IntelMKL_LP64_LIBRARY
  NAMES ${IntelMKL_LP64_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) LP64 MKL Library"
)

if( IntelMKL_ILP64_LIBRARY )
  set( IntelMKL_ilp64_FOUND TRUE )
else()
  set( IntelMKL_ilp64_FOUND FALSE )
endif()

if( IntelMKL_LP64_LIBRARY )
  set( IntelMKL_lp64_FOUND TRUE )
else()
  set( IntelMKL_lp64_FOUND FALSE )
endif()

# SYCL
if(ENABLE_DPCPP)
  find_library( IntelMKL_SYCL_LIBRARY
    NAMES ${IntelMKL_SYCL_LIBRARY_NAME}
    HINTS ${IntelMKL_PREFIX}
    PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib/intel64 lib/ia32
    DOC "Intel(R) MKL SYCL Library"
  )
endif() 


# BLACS / ScaLAPACK

find_library( IntelMKL_ILP64_BLACS_LIBRARY
  NAMES ${IntelMKL_ILP64_BLACS_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) ILP64 MKL BLACS Library"
)

find_library( IntelMKL_LP64_BLACS_LIBRARY
  NAMES ${IntelMKL_LP64_BLACS_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) LP64 MKL BLACS Library"
)

find_library( IntelMKL_ILP64_ScaLAPACK_LIBRARY
  NAMES ${IntelMKL_ILP64_ScaLAPACK_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) ILP64 MKL ScaLAPACK Library"
)

find_library( IntelMKL_LP64_ScaLAPACK_LIBRARY
  NAMES ${IntelMKL_LP64_ScaLAPACK_LIBRARY_NAME}
  HINTS ${IntelMKL_PREFIX}
  PATHS ${IntelMKL_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib/intel64 lib/ia32
  DOC "Intel(R) LP64 MKL ScaLAPACK Library"
)



# Default to LP64
if( "ilp64" IN_LIST IntelMKL_FIND_COMPONENTS )
  set( IntelMKL_COMPILE_DEFINITIONS "MKL_ILP64" )
  if( CMAKE_C_COMPILER_ID MATCHES "GNU" )
    set( IntelMKL_C_COMPILE_FLAGS        "-m64" )
  endif()
  if( CMAKE_Fortran_COMPILER_ID MATCHES "GNU" )
    set( IntelMKL_Fortran_COMPILE_FLAGS  "-m64" "-fdefault-integer-8" )
  elseif( CMAKE_Fortran_COMPILER_ID MATCHES "Flang" )
    set( IntelMKL_Fortran_COMPILE_FLAGS  "-fdefault-integer-8" )
  elseif( CMAKE_C_COMPILER_ID MATCHES "PGI" )
    set( IntelMKL_Fortran_COMPILE_FLAGS "-i8" )
  endif()
  set( IntelMKL_LIBRARY ${IntelMKL_ILP64_LIBRARY} )

  if( IntelMKL_ILP64_BLACS_LIBRARY )
    set( IntelMKL_BLACS_LIBRARY ${IntelMKL_ILP64_BLACS_LIBRARY} )
    set( IntelMKL_blacs_FOUND TRUE )
  endif()

  if( IntelMKL_ILP64_ScaLAPACK_LIBRARY )
    set( IntelMKL_ScaLAPACK_LIBRARY ${IntelMKL_ILP64_ScaLAPACK_LIBRARY} )
    set( IntelMKL_scalapack_FOUND TRUE )
  endif()

else()
  set( IntelMKL_LIBRARY ${IntelMKL_LP64_LIBRARY} )

  if( IntelMKL_LP64_BLACS_LIBRARY )
    set( IntelMKL_BLACS_LIBRARY ${IntelMKL_LP64_BLACS_LIBRARY} )
    set( IntelMKL_blacs_FOUND TRUE )
  endif()

  if( IntelMKL_LP64_ScaLAPACK_LIBRARY )
    set( IntelMKL_ScaLAPACK_LIBRARY ${IntelMKL_LP64_ScaLAPACK_LIBRARY} )
    set( IntelMKL_scalapack_FOUND TRUE )
  endif()
endif()





# Check if found library is actually static
if( IntelMKL_CORE_LIBRARY MATCHES ".+libmkl_core.a" )
  set( IntelMKL_PREFERS_STATIC TRUE )
endif()




if( IntelMKL_LIBRARY AND IntelMKL_THREAD_LIBRARY AND IntelMKL_CORE_LIBRARY )

  set( IntelMKL_BLAS_LAPACK_LIBRARIES
       ${IntelMKL_LIBRARY} 
       ${IntelMKL_THREAD_LIBRARY} 
       ${IntelMKL_CORE_LIBRARY} )

  if(ENABLE_DPCPP)
    list( APPEND  IntelMKL_BLAS_LAPACK_LIBRARIES ${IntelMKL_SYCL_LIBRARY} )
  endif()

  if( "blacs" IN_LIST IntelMKL_FIND_COMPONENTS )
    set( IntelMKL_BLACS_LIBRARIES 
         ${IntelMKL_BLAS_LAPACK_LIBRARIES} 
         ${IntelMKL_BLACS_LIBRARY} )
  endif()

  if( IntelMKL_PREFERS_STATIC )

    list( PREPEND IntelMKL_BLAS_LAPACK_LIBRARIES "-Wl,--start-group" )
    list( APPEND  IntelMKL_BLAS_LAPACK_LIBRARIES "-Wl,--end-group"   )

    if( IntelMKL_BLACS_LIBRARIES )
      list( PREPEND IntelMKL_BLACS_LIBRARIES "-Wl,--start-group" )
      list( APPEND  IntelMKL_BLACS_LIBRARIES "-Wl,--end-group"   )
    endif()

    if( "scalapack" IN_LIST IntelMKL_FIND_COMPONENTS )
      set( IntelMKL_ScaLAPACK_LIBRARIES 
           ${IntelMKL_ScaLAPACK_LIBRARY} 
           ${IntelMKL_BLACS_LIBRARIES} )
    endif()

  else()

    if( "scalapack" IN_LIST IntelMKL_FIND_COMPONENTS )
      set( IntelMKL_ScaLAPACK_LIBRARIES 
           ${IntelMKL_ScaLAPACK_LIBRARY} 
           ${IntelMKL_BLACS_LIBRARIES} )
    endif()



    list( PREPEND IntelMKL_BLAS_LAPACK_LIBRARIES "-Wl,--no-as-needed" )
    if( IntelMKL_BLACS_LIBRARIES )
      list( PREPEND IntelMKL_BLACS_LIBRARIES "-Wl,--no-as-needed" )
    endif()
    if( IntelMKL_BLACS_LIBRARIES )
      list( PREPEND IntelMKL_ScaLAPACK_LIBRARIES "-Wl,--no-as-needed" )
    endif()

  endif()
    

  if( IntelMKL_THREAD_LAYER MATCHES "openmp" )

    list( APPEND IntelMKL_BLAS_LAPACK_LIBRARIES OpenMP::OpenMP_C )

    if( IntelMKL_BLACS_LIBRARIES )
      list( APPEND IntelMKL_BLACS_LIBRARIES OpenMP::OpenMP_C )
    endif()

    if( IntelMKL_ScaLAPACK_LIBRARIES )
      list( APPEND IntelMKL_ScaLAPACK_LIBRARIES OpenMP::OpenMP_C )
    endif()

  elseif( IntelMKL_THREAD_LAYER MATCHES "tbb" )

    if( NOT TARGET tbb )
      message( FATAL_ERROR "TBB Bindings Not Currently Accessible Through FindIntelMKL" )
      find_dependency( TBB )
    endif()

    list( APPEND IntelMKL_BLAS_LAPACK_LIBRARIES tbb )

    if( IntelMKL_BLACS_LIBRARIES )
      list( APPEND IntelMKL_BLACS_LIBRARIES tbb )
    endif()

    if( IntelMKL_ScaLAPACK_LIBRARIES )
      list( APPEND IntelMKL_ScaLAPACK_LIBRARIES tbb )
    endif()

  endif()


  if( NOT TARGET Threads::Threads )
    find_dependency( Threads )
  endif()

  list( APPEND IntelMKL_BLAS_LAPACK_LIBRARIES "m" "dl" Threads::Threads )

  if( IntelMKL_BLACS_LIBRARIES )
    list( APPEND IntelMKL_BLACS_LIBRARIES "m" "dl" Threads::Threads MPI::MPI_C )
  endif()

  if( IntelMKL_ScaLAPACK_LIBRARIES )
    list( APPEND IntelMKL_ScaLAPACK_LIBRARIES "m" "dl" Threads::Threads MPI::MPI_C )
  endif()




  if( IntelMKL_ScaLAPACK_LIBRARIES )
    set( IntelMKL_LIBRARIES ${IntelMKL_ScaLAPACK_LIBRARIES} )
  elseif( IntelMKL_BLACS_LIBRARIES )
    set( IntelMKL_LIBRARIES ${IntelMKL_BLACS_LIBRARIES} )
  else()
    set( IntelMKL_LIBRARIES ${IntelMKL_BLAS_LAPACK_LIBRARIES} )
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( IntelMKL
  REQUIRED_VARS IntelMKL_LIBRARIES IntelMKL_INCLUDE_DIR
  VERSION_VAR IntelMKL_VERSION_STRING
  HANDLE_COMPONENTS
)

#if( IntelMKL_FOUND AND NOT TARGET IntelMKL::mkl )
#
#  add_library( IntelMKL::mkl INTERFACE IMPORTED )
#  set_target_properties( IntelMKL::mkl PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${IntelMKL_INCLUDE_DIR}"
#    INTERFACE_LINK_LIBRARIES      "${IntelMKL_LIBRARIES}"
#    INTERFACE_COMPILE_OPTIONS     "${IntelMKL_C_COMPILE_FLAGS}"
#    INTERFACE_COMPILE_DEFINITIONS "${IntelMKL_COMPILE_DEFINITIONS}"
#  )
#
#  if( "scalapack" IN_LIST IntelMKL_FIND_COMPONENTS AND NOT scalapack_LIBRARIES )
#    set( scalapack_LIBRARIES IntelMKL::mkl )
#  endif()
#
#  if( "blacs" IN_LIST IntelMKL_FIND_COMPONENTS AND NOT blacs_LIBRARIES )
#    set( blacs_LIBRARIES IntelMKL::mkl )
#  endif()
#
#endif()
