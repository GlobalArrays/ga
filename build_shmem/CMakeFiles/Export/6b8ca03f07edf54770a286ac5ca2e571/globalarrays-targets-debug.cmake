#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GlobalArrays::armci" for configuration "Debug"
set_property(TARGET GlobalArrays::armci APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GlobalArrays::armci PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libarmci.a"
  )

list(APPEND _cmake_import_check_targets GlobalArrays::armci )
list(APPEND _cmake_import_check_files_for_GlobalArrays::armci "${_IMPORT_PREFIX}/lib/libarmci.a" )

# Import target "GlobalArrays::comex" for configuration "Debug"
set_property(TARGET GlobalArrays::comex APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GlobalArrays::comex PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcomex.a"
  )

list(APPEND _cmake_import_check_targets GlobalArrays::comex )
list(APPEND _cmake_import_check_files_for_GlobalArrays::comex "${_IMPORT_PREFIX}/lib/libcomex.a" )

# Import target "GlobalArrays::ga" for configuration "Debug"
set_property(TARGET GlobalArrays::ga APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GlobalArrays::ga PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;Fortran"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libga.a"
  )

list(APPEND _cmake_import_check_targets GlobalArrays::ga )
list(APPEND _cmake_import_check_files_for_GlobalArrays::ga "${_IMPORT_PREFIX}/lib/libga.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
