rm -rf CMake*

cmake -DGA_RUNTIME:STRING="OPEN_SHMEM" \
      -DCMAKE_INSTALL_PREFIX:STRING="/home/d3g293/ga-oshmem/build_oshmem/install"\
      -DCMAKE_BUILD_TYPE:STRING="ReleaseWithDebInfo" \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE \
      -DGA_TEST_NPROCS=5 \
      ..

