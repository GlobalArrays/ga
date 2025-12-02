# CMake generated Testfile for 
# Source directory: /home/d3g293/ga-oshmem/global/examples
# Build directory: /home/d3g293/ga-oshmem/build_shmem/global/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(lennard "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/global/examples/lennard.x")
set_tests_properties(lennard PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/global/examples/CMakeLists.txt;15;ga_add_parallel_test;/home/d3g293/ga-oshmem/global/examples/CMakeLists.txt;0;")
