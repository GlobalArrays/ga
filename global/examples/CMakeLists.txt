

set(__global_example_incs
  ${PROJECT_SOURCE_DIR}/global/src
  ${PROJECT_BINARY_DIR}/global/src
  ${PROJECT_SOURCE_DIR}/ma 
  ${PROJECT_BINARY_DIR}/ma
  ${PROJECT_BINARY_DIR}/gaf2c
  ${PROJECT_SOURCE_DIR}/comex/src-armci
  ${PROJECT_SOURCE_DIR}/LinAlg/lapack+blas
  ${PROJECT_SOURCE_DIR}/global/testing
  ${PROJECT_BINARY_DIR})


ga_add_parallel_test (lennard "lennard-jones/lennard.c")
target_include_directories(lennard.x PRIVATE ${__global_example_incs})

set(boltz_srcs boltzmann/boltzmann.F     
               boltzmann/common      
               boltzmann/equil.F     
               boltzmann/get_patch.F 
               boltzmann/initpar.F   
               boltzmann/printdat.F  
               boltzmann/properties.F
               boltzmann/setup.F     
               boltzmann/timestep.F  
               boltzmann/vorticity.F)

#FIXME: boltz test does not compile in few cases
# add_library(boltz OBJECT ${boltz_srcs})
# target_include_directories(boltz PRIVATE ${__global_example_incs})

# if (ENABLE_FORTRAN)
#   ga_add_parallel_test (boltz boltzmann/main.F)
#   target_include_directories(boltz.x PRIVATE ${__global_example_incs})
#   target_link_libraries(boltz.x boltz)
# endif()
