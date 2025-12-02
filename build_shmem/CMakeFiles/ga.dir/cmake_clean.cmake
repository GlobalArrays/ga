file(REMOVE_RECURSE
  "libga.a"
  "libga.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C Fortran)
  include(CMakeFiles/ga.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
