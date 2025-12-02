# CMake generated Testfile for 
# Source directory: /home/d3g293/ga-oshmem/ga++
# Build directory: /home/d3g293/ga-oshmem/build_shmem/ga++
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ga++/elempatch_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/elempatch_cpp.x")
set_tests_properties(ga++/elempatch_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;95;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
add_test(ga++/mtest_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/mtest_cpp.x")
set_tests_properties(ga++/mtest_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;96;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
add_test(ga++/ntestc_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/ntestc_cpp.x")
set_tests_properties(ga++/ntestc_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;97;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
add_test(ga++/testc_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/testc_cpp.x")
set_tests_properties(ga++/testc_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;98;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
add_test(ga++/testmult_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/testmult_cpp.x")
set_tests_properties(ga++/testmult_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;99;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
add_test(ga++/thread-safe_cpp "/home/d3g293/MPI/openmpi-5.0.8/install/bin/mpiexec" "-n" "4" "/home/d3g293/ga-oshmem/build_shmem/ga++/thread-safe_cpp.x")
set_tests_properties(ga++/thread-safe_cpp PROPERTIES  _BACKTRACE_TRIPLES "/home/d3g293/ga-oshmem/cmake/ga-utils.cmake;77;add_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;89;ga_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;100;gapp_add_parallel_test;/home/d3g293/ga-oshmem/ga++/CMakeLists.txt;0;")
