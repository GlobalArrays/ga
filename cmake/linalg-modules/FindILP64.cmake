set( ILP64_COMPILE_OPTIONS
     # Ensure 64-bit executables for GNU C,CXX,Fortran
     $<$<AND:$<COMPILE_LANGUAGE:CXX,C,Fortran>,$<C_COMPILER_ID:GNU>>:-m64>
     # Make default integers 64-bit for Fortran
     $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<C_COMPILER_ID:Intel,PGI>>:-i8>
     $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<C_COMPILER_ID:GNU,Flang>>:-fdefault-integer-8>
)

set( ILP64_FOUND TRUE CACHE BOOL "ILP64 Flags Found" FORCE )
set( ILP64_COMPILE_OPTIONS "${ILP64_COMPILE_OPTIONS}" CACHE STRING "ILP64 Flags" FORCE )
