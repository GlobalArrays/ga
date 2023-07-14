set( ILP64_FOUND TRUE CACHE BOOL "ILP64 Flags Found" FORCE )
set( ILP64_COMPILE_OPTIONS
        # Ensure 64-bit executables for GNU C,CXX,Fortran
        $<$<AND:$<COMPILE_LANGUAGE:CXX,C,Fortran>,$<C_COMPILER_ID:GNU>>:-m64>
        # Make default integers 64-bit for Fortran
        $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<C_COMPILER_ID:Intel,PGI>>:-i8>
        $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<C_COMPILER_ID:GNU,Flang>>:-fdefault-integer-8>
        )
set( ILP64_COMPILE_OPTIONS "${ILP64_COMPILE_OPTIONS}" CACHE STRING "ILP64 compile options" FORCE )

foreach (lang C CXX Fortran)
    if ( NOT DEFINED CMAKE_${lang}_COMPILER_ID )
        continue()
    endif()
    if ( CMAKE_${lang}_COMPILER_ID STREQUAL GNU )
        list( APPEND ILP64_${lang}_COMPILE_OPTIONS -m64 )
    endif()
    if ( lang STREQUAL Fortran )
        if ( CMAKE_Fortran_COMPILER_ID STREQUAL Intel OR CMAKE_Fortran_COMPILER_ID STREQUAL PGI )
            list( APPEND ILP64_${lang}_COMPILE_OPTIONS -i8 )
        endif()
        if ( CMAKE_Fortran_COMPILER_ID STREQUAL GNU OR CMAKE_Fortran_COMPILER_ID STREQUAL Flang )
            list( APPEND ILP64_${lang}_COMPILE_OPTIONS -fdefault-integer-8 )
        endif()
    endif()
    set( ILP64_${lang}_COMPILE_OPTIONS "${ILP64_${lang}_COMPILE_OPTIONS}" CACHE STRING "ILP64 compile options for language ${lang}" FORCE )
endforeach()
