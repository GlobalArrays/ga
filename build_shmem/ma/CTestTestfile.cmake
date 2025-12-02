# CMake generated Testfile for 
# Source directory: /home/d3g293/ga-oshmem/ma
# Build directory: /home/d3g293/ga-oshmem/build_shmem/ma
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ma/testf "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ma/testf.x")
set_tests_properties(ma/testf PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;94;ga_add_parallel_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;0;")
add_test(ma/test-coalesce "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ma/test-coalesce.x")
set_tests_properties(ma/test-coalesce PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;97;ga_add_parallel_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;0;")
add_test(ma/test-inquire "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ma/test-inquire.x")
set_tests_properties(ma/test-inquire PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;98;ga_add_parallel_test;/home/d3g293/ga-oshmem/ma/CMakeLists.txt;0;")
